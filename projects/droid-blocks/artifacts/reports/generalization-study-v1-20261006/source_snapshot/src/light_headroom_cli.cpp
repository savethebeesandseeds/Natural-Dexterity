#include "droid/construction.hpp"
#include "droid/light_session.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Kept separate from the live session and all prior experiment executables.
// The focused test includes this implementation to exercise short tapes without
// exposing an alternative scientific horizon through the command-line interface.
namespace droid_headroom {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr double dt = .02;
constexpr std::size_t horizon = 3200;
constexpr double threshold = 3.2;
constexpr std::string_view schema = "light_headroom_development_v1";
constexpr std::string_view protocol_path = "docs/LIGHT_HEADROOM_EXPERIMENT.md";
constexpr std::array<std::size_t, 4> window_steps{320, 640, 1600, 3200};
constexpr std::array<std::string_view, 5> fixed_modes{
    "quiet", "constant_negative", "constant_positive", "square_2s", "square_4s"};
constexpr std::array<std::uint64_t, 3> seeds{101, 202, 303};

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
Json read_json(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot read " + path.generic_string());
    return Json::parse(stream);
}
fs::path contained(const fs::path& root, const std::string& relative) {
    const fs::path path(relative);
    require(!path.empty() && !path.is_absolute(), "invalid relative evidence path");
    for (const auto& part : path) require(part != "..", "evidence path escapes directory");
    // Evidence is created under a fresh directory; reject links when verifying.
    fs::path current = root;
    for (const auto& part : path) {
        current /= part;
        require(!fs::is_symlink(current), "linked evidence path is unsupported");
    }
    return root / path;
}
void write_json_new(const fs::path& path, const Json& value) {
    require(!fs::exists(path), "refusing to overwrite " + path.generic_string());
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot create " + path.generic_string());
    stream << value.dump() << '\n';
    stream.close();
    require(static_cast<bool>(stream), "cannot finish " + path.generic_string());
}
void prepare_output(const fs::path& output) {
    require(fs::create_directory(output), "output directory already exists; preserve previous evidence");
}
void save_report(const fs::path& output, const Json& report) {
    const fs::path next = output / "progress.next.json";
    std::ofstream stream(next, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(stream), "cannot create partial report");
    stream << report.dump(2) << '\n';
    stream.close();
    require(static_cast<bool>(stream), "cannot finish partial report");
    fs::rename(next, output / "report.json");
}
void initialize(droid::ConstructionWorld& world, const Json& assembly, const Json& sun) {
    world.rebuild(assembly);
    world.set_sun(sun);
    world.reset(); // Include acquisition queues in the reset under the chosen sun.
}
Json cases() {
    Json result = Json::array();
    const std::array<std::array<int, 2>, 3> lengths{{{4, 3}, {3, 4}, {4, 4}}};
    const std::array<std::string, 3> names{"long-upper", "long-lower", "weighted"};
    for (std::size_t body = 0; body < lengths.size(); ++body) {
        for (int hinge : {0, 1}) for (bool lower : {false, true}) for (int side : {-1, 1}) {
            Json assembly = droid::ConstructionWorld::light_assembly();
            assembly["name"] = "The same parts, rebuilt";
            assembly["segments"] = lengths[body];
            assembly["powered_hinge"] = hinge;
            if (body == 2) assembly["blocks"].push_back(Json{{"id", "block-1"},
                {"segment", 1}, {"slot", 0}, {"side", -1}});
            const std::string id = names[body] + "-hinge" + std::to_string(hinge) +
                (lower ? "-lower-" : "-high-") + (side < 0 ? "left" : "right");
            result.push_back(Json{{"id", id}, {"primary", lower}, {"assembly", assembly},
                {"sun", Json{{"position_m", {side * .30, 0., lower ? -.30 : .20}},
                    {"intensity_lux", 1000.}}}});
        }
    }
    return result;
}
Json controller_spec(std::string_view mode, std::uint64_t seed = 0) {
    return Json{{"mode", mode}, {"seed", mode == "learner" ? Json(seed) : Json(nullptr)}};
}
double requested(std::string_view mode, std::size_t step) {
    if (step < 5) return 0.;
    step -= 5;
    if (mode == "quiet") return 0.;
    if (mode == "constant_negative") return -.15;
    if (mode == "constant_positive") return .15;
    if (mode == "square_2s") return (step / 50) % 2 == 0 ? -.15 : .15;
    if (mode == "square_4s") return (step / 100) % 2 == 0 ? -.15 : .15;
    throw std::invalid_argument("unknown fixed controller");
}
struct Control {
    droid::LightControl learner;
    std::string mode;
    double previous{};
    double last_request{};
    std::array<double, 4> scores{};
    std::size_t last_step{}, best_target{};
    static constexpr std::array<double, 4> targets{0., 1.5707963267948966, -1.5707963267948966, 3.1415926535897932};
    explicit Control(const Json& spec) : mode(spec.at("mode").get<std::string>()) {
        if (mode == "learner") learner.reset("learner", spec.at("seed").get<std::uint64_t>());
        else if (mode != "sensor_search_hold") (void)requested(mode, 5);
    }
    double act(const Json& observation, std::size_t step) {
        if (mode == "learner") {
            const double effort = learner.act(observation);
            last_request = learner.diagnostics().at("requested_effort").get<double>();
            return effort;
        }
        last_step = step;
        double request = 0.;
        if (mode == "sensor_search_hold" && step >= 5) {
            // Only local powered-shaft feedback and earlier measured rewards.
            const auto& feedback = observation.at("actuator_feedback").at(0).at("feedback");
            const double angle = feedback.at("position_rad");
            const double speed = feedback.at("velocity_rad_s");
            const std::size_t target = step < 805 ? (step - 5) / 200 : best_target;
            const double difference = targets.at(target) - angle;
            request = std::clamp(.30 * std::atan2(std::sin(difference), std::cos(difference)) - .05 * speed, -.15, .15);
        } else if (mode != "sensor_search_hold") request = requested(mode, step);
        last_request = request;
        previous = droid::LightLearner::bounded_effort(request, previous, observation);
        return previous;
    }
    void observe(const Json& transition) {
        if (mode == "learner") learner.observe(transition);
        else if (mode == "sensor_search_hold" && last_step >= 5 && last_step < 805) {
            const std::size_t index = (last_step - 5) / 200;
            if ((last_step - 5) % 200 >= 150) scores.at(index) += transition.at("reward").get<double>();
            if (last_step == 804) best_target = static_cast<std::size_t>(
                std::max_element(scores.begin(), scores.end()) - scores.begin());
        }
    }
    Json diagnostics() const {
        if (mode == "learner") return learner.diagnostics();
        Json result{{"mode", mode}, {"learning", false}};
        if (mode == "sensor_search_hold") result.update(Json{{"scores", scores},
            {"best_target_index", best_target}, {"held_target_rad", targets.at(best_target)}});
        return result;
    }
};
struct Metrics {
    std::size_t steps{}, vetoes{}, flags{}, invalid_reward{}, invalid_feedback{}, valid_lux_steps{};
    double reward{}, light_reward{}, imu_reward{}, lux_integral{}, energy{};
    double max_current{}, max_temperature{}, max_speed{}, absolute_effort{}, max_effort{};
    void add(double effort, const Json& transition, const Json& physics) {
        require(transition.at("info").at("physics_steps_executed") == 10 &&
            std::abs(transition.at("info").at("elapsed_control_dt_s").get<double>() - dt) < 1e-12 &&
            transition.at("truncated") == false, "native step duration differs");
        ++steps;
        const double value = transition.at("reward").get<double>();
        require(std::isfinite(value), "nonfinite native reward");
        reward += value;
        double parts = 0.;
        for (const auto& component : transition.at("reward_components")) {
            const double part = component.at("transition_reward").get<double>();
            require(std::isfinite(part), "nonfinite reward component");
            parts += part;
            const std::string family = component.at("family_id");
            if (family == "ambient_light_v0") light_reward += part;
            else if (family == "imu_6axis_v0") imu_reward += part;
            else throw std::runtime_error("unexpected reward family");
        }
        require(std::abs(parts - value) < 1e-12, "native reward differs from its parts");
        const auto& observation = transition.at("observation");
        for (const auto& sensor : observation.at("reward_sensors")) {
            if (sensor.at("valid") != true) ++invalid_reward;
            if (sensor.at("family_id") == "ambient_light_v0" && sensor.at("valid") == true) {
                lux_integral += dt * sensor.at("observations").at("illuminance_lux").get<double>();
                ++valid_lux_steps;
            }
        }
        const auto& actuator = observation.at("actuator_feedback").at(0);
        if (actuator.at("valid") != true) ++invalid_feedback;
        const auto& feedback = actuator.at("feedback");
        max_current = std::max(max_current, std::abs(feedback.at("current_a").get<double>()));
        max_temperature = std::max(max_temperature, feedback.at("temperature_c").get<double>());
        max_speed = std::max(max_speed, std::abs(feedback.at("velocity_rad_s").get<double>()));
        if (transition.at("safety").at("veto") == true || transition.at("terminated") == true) ++vetoes;
        if (!transition.at("safety").at("flags").empty() || !feedback.at("fault_flags").empty()) ++flags;
        energy = physics.at("diagnostics").at("electrical_energy_j").get<double>();
        absolute_effort += std::abs(effort);
        max_effort = std::max(max_effort, std::abs(effort));
    }
    bool safe() const { return !vetoes && !flags && !invalid_feedback && !invalid_reward; }
    Json json(double energy_offset = 0.) const {
        return Json{{"steps", steps}, {"duration_s", steps * dt},
            {"integrated_sensor_reward", reward}, {"integrated_light_reward", light_reward},
            {"integrated_imu_reward", imu_reward}, {"mean_sensor_reward_rate", steps ? reward / (steps * dt) : 0.},
            {"sampled_mean_received_lux", steps ? lux_integral / (steps * dt) : 0.},
            {"valid_light_control_endpoints", valid_lux_steps}, {"electrical_energy_j", steps ? energy - energy_offset : 0.},
            {"sampled_max_current_a", max_current}, {"sampled_max_temperature_c", max_temperature},
            {"sampled_max_motor_speed_rad_s", max_speed}, {"mean_absolute_effort", steps ? absolute_effort / steps : 0.},
            {"max_absolute_effort", max_effort}, {"native_veto_steps", vetoes}, {"flagged_steps", flags},
            {"invalid_reward_sensor_samples", invalid_reward}, {"invalid_feedback_samples", invalid_feedback}};
    }
};
std::string window_key(std::size_t step) { return std::to_string(step); }
void append_line(std::ofstream& stream, const Json& value) {
    stream << value.dump() << '\n';
    require(static_cast<bool>(stream), "cannot retain native transition tape");
}

// Verification streams one tape at a time; it never needs to load raw trajectories
// for all cases into memory. Tape replay and controller replay are distinct checks.
Json replay_trial(const fs::path& output, const Json& entry, bool check_hashes = true) {
    const auto tape_path = contained(output, entry.at("trace").at("path"));
    const auto snapshots_path = contained(output, entry.at("snapshots").at("path"));
    if (check_hashes) {
        require(droid::sha256_file(tape_path.string()) == entry.at("trace").at("sha256"), "transition tape hash differs");
        require(droid::sha256_file(snapshots_path.string()) == entry.at("snapshots").at("sha256"), "snapshot hash differs");
    }
    const Json snapshots = read_json(snapshots_path);
    require(snapshots.is_array() && !snapshots.empty(), "snapshot sequence is empty");
    std::ifstream tape(tape_path, std::ios::binary);
    require(static_cast<bool>(tape), "cannot open tape");
    std::string line;
    require(static_cast<bool>(std::getline(tape, line)), "missing initial tape record");
    const Json initial = Json::parse(line);
    require(initial.at("type") == "initial" && initial.at("schema") == "light_headroom_tape_v1", "invalid initial tape record");
    require(initial.at("controller") == entry.at("controller") && initial.at("case_id") == entry.at("case_id"), "trial identity differs");
    require(initial.at("assembly") == entry.at("assembly") && initial.at("sun") == entry.at("sun"), "tape construction or source differs");
    droid::ConstructionWorld world;
    initialize(world, initial.at("assembly"), initial.at("sun"));
    require(world.observation() == initial.at("observation") && world.state() == initial.at("physics"), "initial native state differs");
    require(snapshots.at(0) == Json{{"step", 0}, {"physics", world.state()}}, "initial visible snapshot differs");
    Control control(initial.at("controller"));
    bool controller_matches = true;
    std::string controller_error;
    Metrics metrics, late;
    double late_energy_origin = 0.;
    Json windows = Json::object(), final;
    std::size_t snapshot_index = 1, records = 1;
    while (std::getline(tape, line)) {
        ++records;
        const Json row = Json::parse(line);
        if (row.at("type") == "final") { final = row; break; }
        require(row.at("type") == "step" && row.at("step") == metrics.steps + 1, "tape step ordering differs");
        const double effort = row.at("effort");
        require(std::isfinite(effort), "nonfinite retained effort");
        if (controller_matches) {
            try {
                require(control.act(world.observation(), metrics.steps) == effort, "controller emitted effort differs");
                require(control.last_request == row.at("requested_effort").get<double>(), "controller requested effort differs");
            }
            catch (const std::exception& error) { controller_matches = false; controller_error = error.what(); }
        }
        const Json transition = world.step(effort);
        require(transition == row.at("transition"), "full native transition differs at step " + std::to_string(metrics.steps + 1));
        const Json physics = world.state();
        metrics.add(effort, transition, physics);
        if (metrics.steps == 2400) late_energy_origin = metrics.energy;
        if (metrics.steps > 2400) late.add(effort, transition, physics);
        if (std::find(window_steps.begin(), window_steps.end(), metrics.steps) != window_steps.end())
            windows[window_key(metrics.steps)] = metrics.json();
        if (metrics.steps % 10 == 0) {
            require(snapshot_index < snapshots.size() && snapshots.at(snapshot_index) ==
                Json{{"step", metrics.steps}, {"physics", physics}}, "visible snapshot differs");
            ++snapshot_index;
        }
        if (controller_matches) {
            try { control.observe(transition); }
            catch (const std::exception& error) { controller_error = error.what(); controller_matches = false; }
        }
    }
    require(final.is_object() && !std::getline(tape, line), "missing final record or extra tape records");
    require(records == entry.at("trace").at("records") && metrics.steps == entry.at("completed_steps") &&
        metrics.steps == entry.at("recorded_steps"), "retained tape counts differ");
    require(snapshot_index == snapshots.size() && snapshots.size() == entry.at("snapshots").at("count"), "snapshot counts differ");
    require(final.at("metrics") == metrics.json() && entry.at("metrics") == metrics.json() &&
        entry.at("windows") == windows && entry.at("late_window") ==
        Json{{"start_step", 2400}, {"end_step", 3200}, {"metrics", late.json(late_energy_origin)}}, "recomputed native metrics differ");
    require(final.at("physics") == world.state() && final.at("completed") == entry.at("completed") &&
        final.at("error") == entry.at("error"), "final trial metadata differs");
    require(initial.at("planned_steps") == entry.at("planned_steps") && metrics.steps <= entry.at("planned_steps").get<std::size_t>(), "trial horizon differs");
    if (entry.at("completed") == true) require(metrics.steps == entry.at("planned_steps").get<std::size_t>() && entry.at("error") == "", "false completion claim");
    require(entry.at("safe") == (entry.at("completed") == true && metrics.safe()), "safe trial claim differs");
    if (controller_matches) require(final.at("controller_final") == control.diagnostics(), "final controller state differs");
    return Json{{"exact_full_transition_tape", true}, {"verified_steps", metrics.steps},
        {"exact_request_controller_replay", controller_matches}, {"controller_replay_error", controller_error},
        {"scope", "fresh native world with retained emitted efforts; separate unchanged native controller replay"}};
}

Json run_trial(const fs::path& output, const Json& scene, const Json& controller, std::size_t planned = horizon) {
    const std::string label = controller.at("mode").get<std::string>() +
        (controller.at("seed").is_null() ? "" : "-" + std::to_string(controller.at("seed").get<std::uint64_t>()));
    const std::string id = scene.at("id").get<std::string>() + "--" + label;
    const std::string tape_relative = "traces/" + id + ".jsonl";
    const std::string snapshots_relative = "snapshots/" + id + ".json";
    const fs::path tape_path = contained(output, tape_relative);
    fs::create_directories(tape_path.parent_path());
    require(!fs::exists(tape_path), "trial tape already exists");
    std::ofstream tape(tape_path, std::ios::binary);
    require(static_cast<bool>(tape), "cannot open new trial tape");
    droid::ConstructionWorld world;
    initialize(world, scene.at("assembly"), scene.at("sun"));
    Control control(controller);
    Metrics metrics, late;
    double late_energy_origin = 0.;
    std::size_t recorded_steps = 0;
    Json windows = Json::object(), snapshots = Json::array({Json{{"step", 0}, {"physics", world.state()}}});
    append_line(tape, Json{{"type", "initial"}, {"schema", "light_headroom_tape_v1"},
        {"case_id", scene.at("id")}, {"assembly", scene.at("assembly")}, {"sun", scene.at("sun")},
        {"controller", controller}, {"planned_steps", planned}, {"observation", world.observation()}, {"physics", world.state()}});
    std::string error;
    try {
        while (metrics.steps < planned) {
            const double effort = control.act(world.observation(), metrics.steps);
            const Json transition = world.step(effort);
            append_line(tape, Json{{"type", "step"}, {"step", recorded_steps + 1},
                {"requested_effort", control.last_request}, {"effort", effort}, {"transition", transition}});
            ++recorded_steps;
            const Json physics = world.state();
            metrics.add(effort, transition, physics);
            if (metrics.steps == 2400) late_energy_origin = metrics.energy;
            if (metrics.steps > 2400) late.add(effort, transition, physics);
            if (std::find(window_steps.begin(), window_steps.end(), metrics.steps) != window_steps.end())
                windows[window_key(metrics.steps)] = metrics.json();
            if (metrics.steps % 10 == 0) snapshots.push_back(Json{{"step", metrics.steps}, {"physics", physics}});
            control.observe(transition);
            if (transition.at("terminated") == true || metrics.vetoes || metrics.flags)
                throw std::runtime_error("native safety or feedback stopped this trial");
        }
    } catch (const std::exception& exception) { error = exception.what(); }
    const bool complete = metrics.steps == planned && error.empty();
    append_line(tape, Json{{"type", "final"}, {"completed", complete}, {"error", error},
        {"metrics", metrics.json()}, {"physics", world.state()}, {"controller_final", control.diagnostics()}});
    tape.close();
    require(static_cast<bool>(tape), "cannot finish transition tape");
    write_json_new(contained(output, snapshots_relative), snapshots);
    Json result{{"id", id}, {"case_id", scene.at("id")}, {"controller", controller},
        {"assembly", scene.at("assembly")}, {"sun", scene.at("sun")},
        {"planned_steps", planned}, {"completed_steps", metrics.steps}, {"recorded_steps", recorded_steps}, {"completed", complete},
        {"safe", complete && metrics.safe()}, {"error", error}, {"metrics", metrics.json()}, {"windows", windows},
        {"late_window", Json{{"start_step", 2400}, {"end_step", 3200}, {"metrics", late.json(late_energy_origin)}}},
        {"trace", Json{{"path", tape_relative}, {"sha256", droid::sha256_file(tape_path.string())}, {"records", recorded_steps + 2}}},
        {"snapshots", Json{{"path", snapshots_relative}, {"sha256", droid::sha256_file((output / snapshots_relative).string())}, {"count", snapshots.size()}}}};
    try { result["replay"] = replay_trial(output, result); }
    catch (const std::exception& exception) {
        result["replay"] = Json{{"exact_full_transition_tape", false}, {"exact_request_controller_replay", false}, {"error", exception.what()}};
    }
    return result;
}

void snapshot_sources(const fs::path& output, Json& report) {
    std::set<fs::path> files;
    for (const std::string directory : {"src", "include/droid", "tests"}) {
        for (const auto& item : fs::recursive_directory_iterator(directory)) {
            if (item.is_regular_file() && (item.path().extension() == ".cpp" || item.path().extension() == ".hpp")) files.insert(item.path());
        }
    }
    for (const std::string file : {"CMakeLists.txt", "Dockerfile", "setup.sh", "run.sh",
        "config/module_catalog.json", "models/droid.xml", "config/learning_experiment_v3.json",
        "artifacts/light-search-policy-v3-seed0.json", "docs/LIGHT_HEADROOM_EXPERIMENT.md",
        "tools/audit-light-headroom.cpp", "tools/export-light-headroom.cpp",
        "build/light-headroom-qa-20261004/runtime.json"}) files.insert(file);
    for (const fs::path& file : files) {
        const std::string name = file.generic_string();
        const std::string before = droid::sha256_file(name);
        const fs::path target = contained(output / "source", name);
        fs::create_directories(target.parent_path());
        require(fs::copy_file(file, target), "cannot archive " + name);
        require(before == droid::sha256_file(target.string()) && before == droid::sha256_file(name), "source changed while archived");
        report["source_sha256"][name] = before;
    }
}
bool valid_trial(const Json& trial) {
    return trial.at("safe") == true && trial.at("replay").at("exact_full_transition_tape") == true &&
        trial.at("replay").at("exact_request_controller_replay") == true;
}
Json decisions(const Json& scenes) {
    Json by_case = Json::array();
    std::size_t primary_count = 0, headroom_count = 0;
    std::array<std::size_t, 3> competent{};
    for (const auto& scene : scenes) {
        const Json* quiet = nullptr;
        for (const auto& trial : scene.at("trials")) if (trial.at("controller").at("mode") == "quiet") quiet = &trial;
        require(quiet != nullptr, "case lacks quiet control");
        const double baseline = quiet->at("metrics").at("integrated_sensor_reward");
        Json contrasts = Json::array();
        std::optional<double> best, feedback_delta;
        std::string witness, feedback_witness;
        std::array<bool, 3> learned{};
        for (const auto& trial : scene.at("trials")) {
            const double delta = trial.at("metrics").at("integrated_sensor_reward").get<double>() - baseline;
            const bool safe_pair = valid_trial(*quiet) && valid_trial(trial);
            const std::string mode = trial.at("controller").at("mode");
            contrasts.push_back(Json{{"trial_id", trial.at("id")}, {"safe_pair", safe_pair},
                {"integrated_reward_minus_quiet", delta}, {"reward_rate_minus_quiet", delta / 64.}, {"clears_threshold", safe_pair && delta > threshold}});
            if (safe_pair && mode == "sensor_search_hold") {
                feedback_delta = delta; feedback_witness = trial.at("id");
            }
            if (safe_pair && mode != "quiet" && mode != "learner" && mode != "sensor_search_hold" && (!best || delta > *best)) {
                best = delta; witness = trial.at("id");
            }
            if (mode == "learner") for (std::size_t i = 0; i < seeds.size(); ++i)
                if (trial.at("controller").at("seed") == seeds[i]) learned[i] = safe_pair && delta > threshold;
        }
        const bool fixed_headroom = best && *best > threshold;
        const bool feedback_headroom = feedback_delta && *feedback_delta > threshold;
        const bool headroom = fixed_headroom || feedback_headroom;
        if (scene.at("primary") == true) {
            ++primary_count;
            if (headroom) ++headroom_count;
            for (std::size_t i = 0; i < seeds.size(); ++i) if (learned[i]) ++competent[i];
        }
        by_case.push_back(Json{{"case_id", scene.at("id")}, {"primary", scene.at("primary")},
            {"headroom", headroom}, {"fixed_headroom", fixed_headroom}, {"feedback_headroom", feedback_headroom},
            {"best_safe_fixed_trial", witness.empty() ? Json(nullptr) : Json(witness)},
            {"best_safe_fixed_delta", best ? Json(*best) : Json(nullptr)},
            {"safe_feedback_trial", feedback_witness.empty() ? Json(nullptr) : Json(feedback_witness)},
            {"safe_feedback_delta", feedback_delta ? Json(*feedback_delta) : Json(nullptr)}, {"contrasts", contrasts}});
    }
    Json seed_gates = Json::array();
    bool all_learners = primary_count == 12;
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        seed_gates.push_back(Json{{"seed", seeds[i]}, {"passing_primary_cases", competent[i]}, {"passed", primary_count == 12 && competent[i] == 12}});
        all_learners = all_learners && competent[i] == 12;
    }
    return Json{{"primary_case_count", primary_count}, {"integrated_reward_threshold", threshold},
        {"rate_threshold_reward_per_s", .05}, {"headroom_gate", Json{{"passed", primary_count == 12 && headroom_count == 12}, {"passing_primary_cases", headroom_count}}},
        {"learner_competence_gate", Json{{"passed", all_learners}, {"seeds", seed_gates}}},
        {"selection_scope", "best safe fixed probe is exploratory; sensor_search_hold is a sensor-feedback witness with richer continuous requests than the learner; neither establishes matched superiority, optimality, or transfer"}, {"cases", by_case}};
}
Json manifest() {
    return Json{{"native_dt_s", dt}, {"steps", horizon}, {"duration_s", 64.},
        {"windows_steps", window_steps}, {"fixed_controllers", fixed_modes}, {"learner_seeds", seeds},
        {"fixed_effort_amplitude", .15}, {"initialization", "rebuild, set sun, reset physical state and acquisition queues"},
        {"manual_quiet_steps", 5},
        {"square_2s", "negative first; alternating every 50 native steps (1s)"},
        {"square_4s", "negative first; alternating every 100 native steps (2s)"},
        {"all_commands", "LightLearner::bounded_effort at every native step; unchanged LightControl learner"},
        {"sensor_search_hold", Json{{"targets_rad", Control::targets}, {"dwell_steps", 200}, {"score_final_steps", 50},
            {"startup_steps", 5}, {"servo", "clip(.30*atan2(sin(target-q),cos(target-q))-.05*v,-.15,.15), then shared governor"},
            {"score", "raw integrated native reward in final1s of each4s dwell; first target wins ties"},
            {"after_search", "hold highest scoring target; full16.1s search cost included"}}},
        {"initialization_in_budget", true}, {"cases", cases()}, {"expected_runs", 216},
        {"primary", "lower lamp z=-.30, all12 cases; historical high lamp z=+.20 diagnostic only"},
        {"headroom_rule", "every primary case has a safe fixed nonquiet or sensor-feedback witness strictly >3.2 above paired quiet"},
        {"learner_rule", "each fresh learner seed strictly >3.2 above quiet on every primary case"},
        {"late_window", "final16s: native step endpoints 2401 through3200, after48s"},
        {"snapshot_interval_steps", 10}, {"exposure", "prospective development diagnostic; no frozen-v3 challenge execution"}};
}
void verify_directory(const fs::path& output) {
    const Json report = read_json(output / "report.json");
    require(report.at("schema") == schema && report.at("manifest") == manifest(), "report schema or fixed manifest differs");
    require(report.at("manifest_sha256") == droid::sha256_hex(report.at("manifest").dump()), "manifest checksum differs");
    require(report.at("physics_spec") == droid::ConstructionWorld::light_specification() &&
        report.at("control_spec") == droid::LightControl::specification(), "native runtime specification differs");
    require(!report.at("source_sha256").empty(), "missing source archive");
    for (const auto& [name, digest] : report.at("source_sha256").items())
        require(droid::sha256_file(contained(output / "source", name).string()) == digest, "archived source hash differs: " + name);
    require(report.at("source_sha256").at(std::string(protocol_path)) == report.at("protocol").at("sha256"), "protocol archive differs");
    const Json expected = cases();
    require(report.at("cases").size() <= expected.size(), "too many retained cases");
    std::size_t runs = 0, failures = 0;
    for (std::size_t index = 0; index < report.at("cases").size(); ++index) {
        const auto& scene = report.at("cases").at(index);
        for (const std::string field : {"id", "primary", "assembly", "sun"}) require(scene.at(field) == expected.at(index).at(field), "retained case differs");
        require(scene.at("trials").size() <= 9, "too many retained trials");
        for (std::size_t trial_index = 0; trial_index < scene.at("trials").size(); ++trial_index) {
            const auto& trial = scene.at("trials").at(trial_index);
            const Json wanted = trial_index < 5 ? controller_spec(fixed_modes.at(trial_index)) :
                trial_index == 5 ? controller_spec("sensor_search_hold") : controller_spec("learner", seeds.at(trial_index - 6));
            require(trial.at("controller") == wanted && trial.at("planned_steps") == horizon, "retained controller or horizon differs");
            require(trial.at("case_id") == scene.at("id") && trial.at("assembly") == scene.at("assembly") &&
                trial.at("sun") == scene.at("sun"), "trial case inputs differ");
            const Json replay = replay_trial(output, trial);
            require(replay == trial.at("replay"), "retained replay result differs");
            if (!valid_trial(trial)) ++failures;
            ++runs;
        }
    }
    require(report.at("completed_runs") == runs && report.at("failed_runs") == failures, "report run counts differ");
    if (report.at("complete") == true) {
        require(runs == 216 && report.at("cases").size() == 24, "false complete experiment claim");
        require(report.at("source_sha256_after_collection") == report.at("source_sha256"), "retained final source hashes differ");
        require(report.at("comparison") == decisions(report.at("cases")), "recomputed comparisons differ");
    }
    std::cout << Json{{"verified", true}, {"retained_runs", runs}, {"complete", report.at("complete")}, {"failed_runs", failures}}.dump() << '\n';
}
int run_directory(const fs::path& output) {
    prepare_output(output); // Exclusivity is established before any world or rollout.
    Json report{{"schema", schema}, {"complete", false}, {"completed_runs", 0}, {"failed_runs", 0},
        {"manifest", manifest()}, {"manifest_sha256", droid::sha256_hex(manifest().dump())},
        {"protocol", Json{{"path", protocol_path}, {"sha256", droid::sha256_file(std::string(protocol_path))}}},
        {"physics_spec", droid::ConstructionWorld::light_specification()}, {"control_spec", droid::LightControl::specification()},
        {"source_sha256", Json::object()}, {"cases", Json::array()}};
    snapshot_sources(output, report);
    save_report(output, report);
    for (const auto& scene : cases()) {
        Json stored = scene;
        stored["trials"] = Json::array();
        report["cases"].push_back(stored);
        std::vector<Json> controls;
        for (std::string_view mode : fixed_modes) controls.push_back(controller_spec(mode));
        controls.push_back(controller_spec("sensor_search_hold"));
        for (std::uint64_t seed : seeds) controls.push_back(controller_spec("learner", seed));
        for (const Json& control : controls) {
            Json trial = run_trial(output, scene, control);
            report["completed_runs"] = report.at("completed_runs").get<std::size_t>() + 1;
            if (!valid_trial(trial)) report["failed_runs"] = report.at("failed_runs").get<std::size_t>() + 1;
            std::cout << trial.at("id").get<std::string>() << " reward=" << trial.at("metrics").at("integrated_sensor_reward")
                << " complete=" << trial.at("completed") << " replay=" << trial.at("replay").at("exact_full_transition_tape") << std::endl;
            report["cases"].back()["trials"].push_back(std::move(trial));
            save_report(output, report);
        }
    }
    report["comparison"] = decisions(report.at("cases"));
    report["source_sha256_after_collection"] = Json::object();
    for (const auto& [name, digest] : report.at("source_sha256").items()) {
        const std::string current = droid::sha256_file(name);
        report["source_sha256_after_collection"][name] = current;
        require(current == digest, "source changed during collection; retained partial evidence: " + name);
    }
    report["complete"] = true;
    save_report(output, report);
    // A failed behavioral gate is a completed result, not an execution error.
    return report.at("failed_runs").get<std::size_t>() ? 1 : 0;
}
} // namespace droid_headroom

#ifndef DROID_HEADROOM_TEST_IMPLEMENTATION
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: droid-light-headroom --output NEW_DIRECTORY | --verify EXISTING_DIRECTORY");
        const std::string mode = argv[1];
        if (mode == "--output") return droid_headroom::run_directory(argv[2]);
        if (mode == "--verify") { droid_headroom::verify_directory(argv[2]); return 0; }
        throw std::invalid_argument("Usage: droid-light-headroom --output NEW_DIRECTORY | --verify EXISTING_DIRECTORY");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
#endif
