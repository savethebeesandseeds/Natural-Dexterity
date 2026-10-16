#define DROID_HEADROOM_TEST_IMPLEMENTATION
#include "../src/light_headroom_cli.cpp"
#include <chrono>

namespace {
using namespace droid_headroom;
void rejected(const auto& action, std::string_view reason) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, std::string(reason));
}
void replace_lines(const fs::path& path, const std::vector<Json>& rows) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    for (const auto& row : rows) append_line(stream, row);
    stream.close();
    require(static_cast<bool>(stream), "test tape replacement failed");
}
void test(const fs::path& output) {
    prepare_output(output);
    const Json marker{{"retained", true}};
    write_json_new(output / "marker.json", marker);
    rejected([&] { prepare_output(output); }, "existing output must be rejected");
    require(read_json(output / "marker.json") == marker, "refusal changed existing evidence");
    const Json scenes = cases();
    require(scenes.size() == 24 && std::count_if(scenes.begin(), scenes.end(), [](const Json& scene) {
        return scene.at("primary") == true;
    }) == 12, "case family differs");
    require(requested("square_2s", 54) == -.15 && requested("square_2s", 55) == .15 &&
        requested("square_4s", 104) == -.15 && requested("square_4s", 105) == .15 &&
        requested("constant_positive", 4) == 0., "warmup or square schedule boundary differs");
    const Json scene = scenes.at(2); // First body/base/lower/left.
    droid::ConstructionWorld initialized, comparison;
    initialize(initialized, scene.at("assembly"), scene.at("sun"));
    comparison.rebuild(scene.at("assembly"));
    comparison.set_sun(scene.at("sun"));
    // Without reset, the queued t=0 sample would still describe the default sun.
    const Json actual = initialized.step(0.);
    const Json old_queue = comparison.step(0.);
    require(actual.at("observation") != old_queue.at("observation"), "initialization did not refresh source acquisition queue");
    const Json trial = run_trial(output, scene, controller_spec("constant_negative"), 30);
    require(trial.at("completed") == true && trial.at("replay").at("exact_full_transition_tape") == true &&
        trial.at("replay").at("exact_request_controller_replay") == true, "short native trial/replay failed");
    std::ifstream input(output / trial.at("trace").at("path").get<std::string>());
    std::vector<Json> rows;
    std::string line;
    double sum = 0., light = 0., imu = 0.;
    while (std::getline(input, line)) {
        rows.push_back(Json::parse(line));
        if (rows.back().at("type") != "step") continue;
        const auto& transition = rows.back().at("transition");
        sum += transition.at("reward").get<double>();
        for (const auto& part : transition.at("reward_components")) {
            if (part.at("family_id") == "ambient_light_v0") light += part.at("transition_reward").get<double>();
            else imu += part.at("transition_reward").get<double>();
        }
    }
    require(sum == trial.at("metrics").at("integrated_sensor_reward") &&
        light == trial.at("metrics").at("integrated_light_reward") &&
        imu == trial.at("metrics").at("integrated_imu_reward"), "metric arithmetic differs from raw native transitions");
    require(trial.at("metrics").at("duration_s") == .6, "native time integrated twice");
    const auto tape_path = output / trial.at("trace").at("path").get<std::string>();
    auto changed = rows;
    changed.at(10)["transition"]["reward"] = changed.at(10).at("transition").at("reward").get<double>() + .001;
    replace_lines(tape_path, changed);
    rejected([&] { (void)replay_trial(output, trial); }, "tampered tape hash must fail");
    rejected([&] { (void)replay_trial(output, trial, false); }, "tampered full transition must fail independently of hash");
    replace_lines(tape_path, rows);
    require(replay_trial(output, trial) == trial.at("replay"), "restored tape did not verify");
    Json bad_metrics = trial;
    bad_metrics["metrics"]["integrated_sensor_reward"] = sum + 1.;
    rejected([&] { (void)replay_trial(output, bad_metrics); }, "tampered summary arithmetic must fail");
    const Json learner = run_trial(output, scenes.at(3), controller_spec("learner", 101), 40);
    require(learner.at("completed") == true && learner.at("replay").at("exact_request_controller_replay") == true,
        "unchanged learner controller replay failed");
    const Json feedback = run_trial(output, scenes.at(3), controller_spec("sensor_search_hold"), 810);
    require(feedback.at("completed") == true && feedback.at("replay").at("exact_request_controller_replay") == true,
        "sensor search and hold controller replay failed");
    Json gate_cases = scenes;
    for (auto& gate_case : gate_cases) {
        gate_case["trials"] = Json::array();
        std::vector<Json> controls;
        for (const auto mode : fixed_modes) controls.push_back(controller_spec(mode));
        controls.push_back(controller_spec("sensor_search_hold"));
        for (const auto seed : seeds) controls.push_back(controller_spec("learner", seed));
        for (const auto& controller : controls) {
            const std::string mode = controller.at("mode");
            const double reward = mode == "quiet" ? 0. : mode == "learner" || mode == "sensor_search_hold" ? 3.21 : 3.2;
            gate_case["trials"].push_back(Json{{"id", mode}, {"controller", controller}, {"safe", true},
                {"metrics", Json{{"integrated_sensor_reward", reward}}},
                {"replay", Json{{"exact_full_transition_tape", true}, {"exact_request_controller_replay", true}}}});
        }
    }
    const Json pass = decisions(gate_cases);
    require(pass.at("headroom_gate").at("passed") == true && pass.at("learner_competence_gate").at("passed") == true &&
        pass.at("cases").at(2).at("fixed_headroom") == false && pass.at("cases").at(2).at("feedback_headroom") == true,
        "strict threshold or distinct feedback witness gate differs");
    gate_cases.at(2).at("trials").at(5)["safe"] = false;
    gate_cases.at(2).at("trials").at(8)["metrics"]["integrated_sensor_reward"] = 3.2;
    const Json fail = decisions(gate_cases);
    require(fail.at("headroom_gate").at("passed") == false && fail.at("learner_competence_gate").at("passed") == false,
        "unsafe witness or exact-threshold learner must fail all-case gates");
    rejected([&] { (void)contained(output, "../escape"); }, "evidence traversal must fail");
}
}
int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output = droid_headroom::fs::temp_directory_path() / ("droid-headroom-test-" + std::to_string(stamp));
    try {
        test(output);
        droid_headroom::fs::remove_all(output);
        std::cout << "native light headroom checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "; preserved test evidence at " << output << '\n';
        return 1;
    }
}
