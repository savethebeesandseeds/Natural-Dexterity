#include "droid/construction.hpp"
#include "droid/light_session.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr double kDt = .02;
constexpr std::size_t kSteps = 6000, kIntervention = 3000;
constexpr std::string_view kReference = "artifacts/reports/light-learning-v2-development-20260905/report.json";
constexpr std::string_view kReferenceSha = "931a26b17b399456bb5461293e382ae9880486f1b38157007500eead65998230";
constexpr std::string_view kProtocol = "docs/LIGHT_ADAPTATION_EXPERIMENT.md";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
Json read_json(const fs::path& file) {
    std::ifstream input(file, std::ios::binary);
    require(static_cast<bool>(input), "cannot read " + file.generic_string());
    return Json::parse(input);
}
Json sun(bool moved) {
    return Json{{"position_m", {moved ? -.30 : .30, 0, .20}}, {"intensity_lux", 1000}};
}
std::string hash(const Json& value) { return droid::sha256_hex(value.dump()); }
bool same_effort(double actual, double expected) {
    return std::bit_cast<std::uint64_t>(actual) == std::bit_cast<std::uint64_t>(expected);
}

struct Metrics {
    double reward{}, light_reward{}, imu_reward{}, lux_integral{}, energy{};
    double max_current{}, max_temperature{}, max_speed{};
    std::size_t steps{}, vetoes{}, flagged{}, unique_light{};
    std::uint64_t last_sequence{};
    void add(const Json& transition, const Json& physical, double energy_delta) {
        ++steps;
        energy += energy_delta;
        reward += transition.at("reward").get<double>();
        for (const auto& component : transition.at("reward_components")) {
            if (component.at("family_id") == "ambient_light_v0") light_reward += component.at("transition_reward").get<double>();
            if (component.at("family_id") == "imu_6axis_v0") imu_reward += component.at("transition_reward").get<double>();
        }
        const auto& feedback = transition.at("observation").at("actuator_feedback").at(0).at("feedback");
        max_current = std::max(max_current, std::abs(feedback.at("current_a").get<double>()));
        max_temperature = std::max(max_temperature, feedback.at("temperature_c").get<double>());
        max_speed = std::max(max_speed, std::abs(feedback.at("velocity_rad_s").get<double>()));
        if (!feedback.at("fault_flags").empty()) ++flagged;
        if (transition.at("safety").at("veto") == true) ++vetoes;
        const auto& light = physical.at("light_sensor");
        if (light.at("valid") == true) {
            lux_integral += kDt * light.at("observations").at("illuminance_lux").get<double>();
            const auto sequence = light.at("sequence").get<std::uint64_t>();
            if (sequence != last_sequence) { ++unique_light; last_sequence = sequence; }
        }
    }
    Json json() const {
        return Json{{"steps", steps}, {"duration_s", steps * kDt},
            {"integrated_sensor_reward", reward}, {"integrated_light_reward", light_reward},
            {"integrated_imu_reward", imu_reward}, {"mean_sensor_reward_rate", steps ? reward / (steps * kDt) : 0},
            {"integrated_received_lux_s", lux_integral}, {"mean_received_lux", steps ? lux_integral / (steps * kDt) : 0},
            {"unique_light_samples", unique_light}, {"electrical_energy_j", energy},
            {"max_current_a", max_current}, {"max_temperature_c", max_temperature},
            {"max_motor_speed_rad_s", max_speed}, {"native_veto_steps", vetoes}, {"feedback_flag_steps", flagged}};
    }
};

struct Run {
    Json result;
    Json observations = Json::array();
    Json transitions = Json::array();
};

Json prefix_snapshot(const droid::ConstructionWorld& world, const droid::LightControl& controller,
                     const Json& efforts, const Run& run) {
    const Json physical = world.state();
    const Json diagnostics = controller.diagnostics();
    return Json{{"steps", efforts.size()}, {"commands_sha256", hash(efforts)},
        {"observation_trace_sha256", hash(run.observations)}, {"transition_trace_sha256", hash(run.transitions)},
        {"physical_snapshot", physical}, {"physical_snapshot_sha256", hash(physical)},
        {"controller_diagnostics", diagnostics},
        {"state_fingerprint_sha256", diagnostics.at("state_fingerprint_sha256")},
        {"parameter_fingerprint_sha256", diagnostics.at("parameter_fingerprint_sha256")}};
}

Run run_branch(const Json& assembly, std::uint64_t seed, bool frozen, const Run* paired_reference) {
    droid::ConstructionWorld world;
    world.rebuild(assembly);
    world.set_sun(sun(false));
    droid::LightControl controller;
    controller.reset("learner", seed);
    Run run;
    Metrics whole, before, after, early, middle, late, before_tail;
    Json efforts = Json::array(), curve = Json::array(), prefix;
    std::string error, frozen_parameter_hash;
    std::size_t steps = 0, frozen_parameter_checks = 0;
    double previous_energy = 0;
    bool parameters_unchanged = true;
    try {
        while (steps < kSteps) {
            if (steps == kIntervention) {
                prefix = prefix_snapshot(world, controller, efforts, run);
                require(prefix.at("controller_diagnostics").at("learning_frozen") == false,
                        "controller was already frozen before intervention");
                if (paired_reference) {
                    const Json& expected = paired_reference->result.at("prefix");
                    require(expected.is_object(), "continued branch lacks a valid intervention snapshot");
                    for (const std::string key : {"commands_sha256", "observation_trace_sha256", "transition_trace_sha256",
                            "physical_snapshot", "state_fingerprint_sha256", "controller_diagnostics"}) {
                        require(prefix.at(key).dump() == expected.at(key).dump(), "paired pre-intervention mismatch: " + key);
                    }
                }
                if (frozen) {
                    frozen_parameter_hash = prefix.at("parameter_fingerprint_sha256").get<std::string>();
                    controller.freeze_learning();
                    require(controller.diagnostics().at("parameter_fingerprint_sha256") == frozen_parameter_hash,
                            "freezing itself changed learned parameters");
                }
                // Keep the pending option, physical state and queued sensor samples.
                world.set_sun(sun(true));
            }
            const double effort = controller.act(world.observation());
            const Json transition = world.step(effort);
            ++steps;
            const Json physical = world.state();
            efforts.push_back(effort);
            run.observations.push_back(transition.at("observation"));
            run.transitions.push_back(transition);
            const double total_energy = physical.at("diagnostics").at("electrical_energy_j");
            const double energy_delta = total_energy - previous_energy;
            previous_energy = total_energy;
            whole.add(transition, physical, energy_delta);
            if (steps <= kIntervention) before.add(transition, physical, energy_delta);
            else after.add(transition, physical, energy_delta);
            if (steps > 2000 && steps <= 3000) before_tail.add(transition, physical, energy_delta);
            if (steps > 3000 && steps <= 4000) early.add(transition, physical, energy_delta);
            if (steps > 4000 && steps <= 5000) middle.add(transition, physical, energy_delta);
            if (steps > 5000) late.add(transition, physical, energy_delta);
            if (steps % 5 == 0) curve.push_back(Json{{"time_s", steps * kDt},
                {"illuminance_lux", physical.at("light_sensor").at("observations").at("illuminance_lux")},
                {"sensor_reward_rate", physical.at("reward").at("rate")}, {"effort", effort},
                {"cumulative_sensor_reward", whole.reward}});
            controller.observe(transition);
            if (paired_reference && steps <= kIntervention) {
                require(same_effort(effort, paired_reference->result.at("efforts").at(steps - 1).get<double>()),
                        "paired prefix command differs at step " + std::to_string(steps));
                require(transition.at("observation").dump() == paired_reference->observations.at(steps - 1).dump(),
                        "paired prefix observation differs at step " + std::to_string(steps));
                require(transition.dump() == paired_reference->transitions.at(steps - 1).dump(),
                        "paired prefix transition/reward differs at step " + std::to_string(steps));
            }
            if (frozen && steps > kIntervention) {
                const Json diagnostics = controller.diagnostics();
                parameters_unchanged = parameters_unchanged && diagnostics.at("learning_frozen") == true &&
                    diagnostics.at("parameter_fingerprint_sha256") == frozen_parameter_hash;
                ++frozen_parameter_checks;
                require(parameters_unchanged, "a frozen parameter changed after intervention");
            }
        }
    } catch (const std::exception& exception) { error = exception.what(); }

    const Json final_diagnostics = controller.diagnostics();
    const Json final_physical = world.state();
    std::string replay_error;
    std::size_t replayed_steps = 0;
    bool replay_matches = false;
    try {
        droid::ConstructionWorld replay;
        replay.rebuild(assembly);
        replay.set_sun(sun(false));
        droid::LightControl replay_controller;
        replay_controller.reset("learner", seed);
        for (std::size_t i = 0; i < efforts.size(); ++i) {
            if (i == kIntervention) {
                require(replay.state().dump() == prefix.at("physical_snapshot").dump(), "replay intervention public physical snapshot differs");
                require(replay_controller.diagnostics().dump() == prefix.at("controller_diagnostics").dump(),
                        "replay intervention controller state differs");
                if (frozen) replay_controller.freeze_learning();
                replay.set_sun(sun(true));
            }
            const double effort = replay_controller.act(replay.observation());
            require(same_effort(effort, efforts.at(i).get<double>()), "replay command differs at step " + std::to_string(i + 1));
            const Json transition = replay.step(effort);
            require(transition.dump() == run.transitions.at(i).dump(), "replay full transition differs at step " + std::to_string(i + 1));
            replay_controller.observe(transition);
            ++replayed_steps;
        }
        require(replay_controller.diagnostics().dump() == final_diagnostics.dump(), "replay final controller fingerprint/state differs");
        require(replay.state().dump() == final_physical.dump(), "replay final public physical snapshot differs");
        replay_matches = error.empty() && replayed_steps == kSteps;
    } catch (const std::exception& exception) { replay_error = exception.what(); }

    run.result = Json{{"branch", frozen ? "frozen" : "continued"}, {"mode", "learner"}, {"seed", seed},
        {"completed", steps == kSteps && error.empty()}, {"completed_steps", steps}, {"error", error},
        {"exact_controller_and_observation_replay", replay_matches},
        {"exact_full_transition_replay", replay_matches},
        {"replay", Json{{"completed_steps", replayed_steps}, {"error", replay_error},
            {"scope", "full 120s commands, complete transitions including reward, and final public physical/controller states"}}},
        {"effort_trace_sha256", hash(efforts)}, {"observation_trace_sha256", hash(run.observations)},
        {"transition_trace_sha256", hash(run.transitions)}, {"prefix", prefix},
        {"frozen_parameters_unchanged", parameters_unchanged}, {"frozen_parameter_checks", frozen_parameter_checks},
        {"frozen_parameter_fingerprint_sha256", frozen_parameter_hash},
        {"whole_run", whole.json()}, {"before_move", before.json()}, {"after_move", after.json()},
        {"first_phase_final_20s", before_tail.json()}, {"early_after_move_20s", early.json()},
        {"middle_after_move_20s", middle.json()}, {"late_after_move_20s", late.json()},
        {"controller_final", final_diagnostics}, {"final_public_physical_snapshot_sha256", hash(final_physical)},
        {"efforts", std::move(efforts)}, {"curve", std::move(curve)}};
    return run;
}

bool metrics_match_reference(const Json& branch, const Json& reference) {
    for (const std::string group : {"whole_run", "before_move", "after_move", "first_phase_final_20s"}) {
        for (const auto& [key, value] : reference.at(group).items()) if (branch.at(group).at(key) != value) return false;
    }
    for (const auto& [key, value] : reference.at("second_phase_final_20s").items())
        if (branch.at("late_after_move_20s").at(key) != value) return false;
    return true;
}

Json pair_result(const Json& assembly, std::size_t body_index, std::uint64_t seed, const Json& reference) {
    std::cout << "body " << body_index << ", seed " << seed << ": continued branch and replay" << std::endl;
    Run continued = run_branch(assembly, seed, false, nullptr);
    std::cout << "body " << body_index << ", seed " << seed << ": frozen branch and replay" << std::endl;
    Run frozen = run_branch(assembly, seed, true, &continued);
    const Json& c = continued.result;
    const Json& f = frozen.result;
    const bool have_prefix = c.at("prefix").is_object() && f.at("prefix").is_object();
    const auto same_prefix = [&](const std::string& key) {
        return have_prefix && c.at("prefix").at(key).dump() == f.at("prefix").at(key).dump();
    };
    Json checks{{"identical_prefix_commands", same_prefix("commands_sha256")},
        {"identical_prefix_observations", same_prefix("observation_trace_sha256")},
        {"identical_prefix_full_transitions", same_prefix("transition_trace_sha256")},
        {"identical_prefix_public_physical_snapshot", same_prefix("physical_snapshot")},
        {"identical_prefix_controller_state_fingerprint", same_prefix("state_fingerprint_sha256")},
        {"identical_prefix_controller_diagnostics", same_prefix("controller_diagnostics")},
        {"continued_matches_archived_v2_commands", hash(c.at("efforts")) == hash(reference.at("efforts"))},
        {"continued_matches_archived_v2_observation_hash", c.at("observation_trace_sha256") == reference.at("observation_trace_sha256")},
        {"continued_matches_archived_v2_return", c.at("whole_run").at("integrated_sensor_reward").dump() == reference.at("whole_run").at("integrated_sensor_reward").dump()},
        {"continued_matches_archived_v2_metrics", metrics_match_reference(c, reference)},
        {"frozen_parameters_unchanged", f.at("frozen_parameters_unchanged") == true && f.at("frozen_parameter_checks") == kSteps - kIntervention},
        {"continued_full_replay", c.at("exact_full_transition_replay") == true},
        {"frozen_full_replay", f.at("exact_full_transition_replay") == true}};
    bool valid = c.at("completed") == true && f.at("completed") == true;
    for (const auto& [name, value] : checks.items()) {
        (void)name;
        valid = valid && value == true;
    }
    const auto difference = [&](const std::string& group, const std::string& field) {
        return c.at(group).at(field).get<double>() - f.at(group).at(field).get<double>();
    };
    Json differences{{"after_move_integrated_sensor_reward", difference("after_move", "integrated_sensor_reward")},
        {"early_after_move_mean_received_lux", difference("early_after_move_20s", "mean_received_lux")},
        {"late_after_move_mean_received_lux", difference("late_after_move_20s", "mean_received_lux")},
        {"after_move_electrical_energy_j", difference("after_move", "electrical_energy_j")},
        {"after_move_native_veto_steps", difference("after_move", "native_veto_steps")},
        {"after_move_feedback_flag_steps", difference("after_move", "feedback_flag_steps")}};
    return Json{{"pair_id", "body-" + std::to_string(body_index) + "-seed-" + std::to_string(seed)},
        {"body_index", body_index}, {"assembly", assembly}, {"assembly_sha256", hash(assembly)}, {"seed", seed},
        {"valid", valid}, {"checks", checks}, {"differences", differences},
        {"continued", std::move(continued.result)}, {"frozen", std::move(frozen.result)}};
}

void save(const fs::path& output, const Json& report) {
    const fs::path next = output / "progress.next.json";
    std::ofstream stream(next, std::ios::binary | std::ios::trunc);
    stream << report.dump(2) << '\n';
    stream.close();
    require(static_cast<bool>(stream), "cannot save adaptation report");
    fs::rename(next, output / "report.json");
}

void snapshot(const fs::path& output, Json& report, const fs::path& relative) {
    require(!relative.is_absolute(), "source snapshot path must be relative");
    for (const auto& part : relative) require(part != "..", "source snapshot escaped project root");
    const std::string name = relative.generic_string();
    if (report.at("source_sha256").contains(name)) return;
    const std::string before = droid::sha256_file(name);
    const fs::path target = output / "source" / relative;
    fs::create_directories(target.parent_path());
    require(fs::copy_file(relative, target), "cannot snapshot " + name);
    require(droid::sha256_file(target.string()) == before && droid::sha256_file(name) == before,
            "source changed while being snapshotted: " + name);
    report["source_sha256"][name] = before;
}

void snapshot_sources(const fs::path& output, Json& report, const Json& reference) {
    std::set<fs::path> files;
    for (const std::string directory : {"src", "include/droid", "tests"}) {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".cpp" || entry.path().extension() == ".hpp")) files.insert(entry.path());
        }
    }
    for (const std::string file : {"CMakeLists.txt", "run.sh", "Dockerfile", "setup.sh",
             "config/module_catalog.json", "models/droid.xml", "tools/summarize-light-adaptation.mjs",
             "config/learning_experiment_v3.json", "artifacts/light-search-policy-v3-seed0.json",
             "docs/LIGHT_ADAPTATION_EXPERIMENT.md", "docs/LIGHT_LEARNING_EXPERIMENT.md"}) files.insert(file);
    if (fs::is_regular_file("build/light-adaptation-qa/runtime.json")) files.insert("build/light-adaptation-qa/runtime.json");
    files.insert(fs::path(kReference));
    const fs::path reference_directory = fs::path(kReference).parent_path();
    for (const auto& [relative, expected] : reference.at("source_sha256").items()) {
        const fs::path archived = reference_directory / "source" / relative;
        require(droid::sha256_file(archived.string()) == expected.get<std::string>(), "archived v2 source hash differs: " + relative);
        files.insert(archived);
    }
    for (const auto& file : files) snapshot(output, report, file);
}
} // namespace

int main(int argc, char** argv) {
    fs::path output;
    Json report;
    bool reserved = false;
    try {
        if (argc != 3 || std::string(argv[1]) != "--output")
            throw std::invalid_argument("Usage: droid-light-adaptation --output NEW_DIRECTORY (from project root)");
        require(droid::sha256_file(std::string(kReference)) == kReferenceSha, "fixed archived v2 report hash differs");
        const Json reference = read_json(fs::path(kReference));
        require(reference.at("complete") == true && reference.at("failed_trials") == 0,
                "archived v2 report is not complete and valid");
        require(reference.at("control_spec").at("learner").at("learner_id") == "history_expected_sarsa_v2",
                "reference is not the declared v2 learner");
        require(reference.at("constructions").size() == 2, "reference does not contain exactly two known bodies");
        output = fs::path(argv[2]);
        require(fs::create_directory(output), "output directory already exists; preserve previous evidence");
        reserved = true;
        report = Json{{"schema", "light_adaptation_v1"}, {"complete", false}, {"finished", false}, {"failed_pairs", 0},
            {"exposure", "Paired mechanistic ablation on six known v2 development prefixes; no new bodies, tuning or held-out claim"},
            {"protocol", Json{{"document", kProtocol}, {"steps", kSteps}, {"control_dt_s", kDt},
                {"intervention_step", kIntervention}, {"intervention_time_s", kIntervention * kDt},
                {"initial_sun", sun(false)}, {"moved_sun", sun(true)}, {"reset_at_intervention", false},
                {"seeds", {1, 2, 3}}, {"body_count", 2}, {"primary_metric", "after_move.integrated_sensor_reward"},
                {"difference_direction", "continued minus frozen"}, {"early_window_s", {60, 80}}, {"late_window_s", {100, 120}},
                {"freeze_scope", "parameter writes only; observation history, eligibility/replay processing, RNG, exploration and pending option continue"},
                {"prefix_scope", "all commands and complete transitions, public physical snapshot and full controller fingerprint before freezing"},
                {"native_state_caveat", "public snapshot is not a direct comparison of every MuJoCo internal; same initialization and full identical prefix reconstruct native state"},
                {"causal_limit", "total effect of continued learning under this moving-sun scenario; no stationary-sun factorial contrast"}}},
            {"reference_report", Json{{"path", kReference}, {"sha256", kReferenceSha}, {"snapshot", "source/" + std::string(kReference)}}},
            {"compiler", __VERSION__}, {"physics_spec", droid::ConstructionWorld::light_specification()},
            {"control_spec", droid::LightControl::specification()}, {"module_catalog", read_json("config/module_catalog.json")},
            {"source_sha256", Json::object()}, {"pairs", Json::array()}};
        snapshot_sources(output, report, reference);
        save(output, report);
        std::size_t failed = 0, body_index = 0;
        for (const auto& construction : reference.at("constructions")) {
            const Json& assembly = construction.at("assembly");
            require(hash(assembly) == construction.at("assembly_sha256"), "reference assembly hash differs");
            for (std::uint64_t seed = 1; seed <= 3; ++seed) {
                const Json* original = nullptr;
                for (const auto& trial : construction.at("trials")) {
                    if (trial.at("mode") == "learner" && trial.at("seed") == seed) {
                        require(original == nullptr, "duplicate archived learner seed");
                        original = &trial;
                    }
                }
                require(original && original->at("completed") == true && original->at("exact_controller_and_observation_replay") == true,
                        "missing valid original v2 learner trial");
                Json pair = pair_result(assembly, body_index, seed, *original);
                if (pair.at("valid") != true) ++failed;
                std::cout << pair.at("pair_id").get<std::string>() << " valid=" << pair.at("valid")
                          << " postmove reward difference=" << pair.at("differences").at("after_move_integrated_sensor_reward") << std::endl;
                report["pairs"].push_back(std::move(pair));
                report["failed_pairs"] = failed;
                save(output, report);
            }
            ++body_index;
        }
        report["finished"] = true;
        report["complete"] = failed == 0 && report.at("pairs").size() == 6;
        save(output, report);
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (reserved && report.is_object()) {
            report["complete"] = false;
            report["fatal_error"] = error.what();
            try { save(output, report); } catch (...) { /* Preserve the existing partial evidence. */ }
        }
        return 2;
    }
}
