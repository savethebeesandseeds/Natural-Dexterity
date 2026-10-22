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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr double kDt = .02;
constexpr std::size_t kSteps = 9000, kFirstMove = 3000, kReturn = 6000;
constexpr std::string_view kReference = "artifacts/reports/light-learning-v2-development-20260905/report.json";
constexpr std::string_view kReferenceSha = "931a26b17b399456bb5461293e382ae9880486f1b38157007500eead65998230";
constexpr std::string_view kProtocol = "docs/LIGHT_RETENTION_EXPERIMENT.md";

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
Json capture_prefix(const droid::ConstructionWorld& world, const droid::LightControl& controller,
                    const Json& efforts, const Run& run) {
    const Json physical = world.state(), diagnostics = controller.diagnostics();
    return Json{{"steps", efforts.size()}, {"commands_sha256", hash(efforts)},
        {"observation_trace_sha256", hash(run.observations)}, {"transition_trace_sha256", hash(run.transitions)},
        {"physical_snapshot", physical}, {"physical_snapshot_sha256", hash(physical)},
        {"controller_diagnostics", diagnostics},
        {"state_fingerprint_sha256", diagnostics.at("state_fingerprint_sha256")},
        {"parameter_fingerprint_sha256", diagnostics.at("parameter_fingerprint_sha256")},
        {"nonparameter_state_fingerprint_sha256", diagnostics.at("nonparameter_state_fingerprint_sha256")}};
}
bool matches_reference_metrics(const Json& branch, const Json& reference) {
    for (const auto& [current, old] : std::vector<std::pair<std::string, std::string>>{
            {"prefix120_metrics", "whole_run"}, {"before_first_move", "before_move"},
            {"middle_condition", "after_move"}, {"first_phase_final_20s", "first_phase_final_20s"},
            {"second_phase_final_20s", "second_phase_final_20s"}}) {
        for (const auto& [key, value] : reference.at(old).items())
            if (branch.at(current).at(key).dump() != value.dump()) return false;
    }
    return true;
}
Run run_branch(const Json& assembly, std::uint64_t seed, const std::string& branch,
               const Json& original, const Run* paired_reference) {
    const bool frozen = branch != "continued", recalled = branch == "recalled_frozen";
    droid::ConstructionWorld world;
    world.rebuild(assembly);
    world.set_sun(sun(false));
    droid::LightControl controller;
    controller.reset("learner", seed);
    std::optional<droid::LightControl::ValueCheckpoint> checkpoint;
    Run run;
    Metrics whole, first120, first, second, returned, early, middle, late, first_tail, second_tail;
    Json efforts = Json::array(), curve = Json::array(), checkpoint60, prefix120, restoration;
    std::string error, frozen_parameter_hash;
    std::size_t steps = 0, frozen_parameter_checks = 0, attempted_step = 0;
    double attempted_effort = 0.0;
    bool attempted_accepted = false;
    double previous_energy = 0;
    bool parameters_unchanged = true;
    try {
        while (steps < kSteps) {
            if (steps == kFirstMove) {
                const Json before = controller.diagnostics();
                checkpoint.emplace(controller.capture_values());
                const Json after = controller.diagnostics();
                require(before.dump() == after.dump(), "capturing values changed controller state");
                checkpoint60 = Json{{"steps", steps}, {"controller_diagnostics", before},
                    {"captured_parameter_fingerprint_sha256", before.at("parameter_fingerprint_sha256")},
                    {"state_fingerprint_before_capture_sha256", before.at("state_fingerprint_sha256")},
                    {"state_fingerprint_after_capture_sha256", after.at("state_fingerprint_sha256")},
                    {"capture_readonly_verified", true}};
                if (paired_reference) require(checkpoint60.dump() == paired_reference->result.at("checkpoint60").dump(),
                    "paired first-A checkpoint differs");
                world.set_sun(sun(true));
            }
            if (steps == kReturn) {
                require(checkpoint.has_value(), "missing first-A value checkpoint");
                prefix120 = capture_prefix(world, controller, efforts, run);
                require(prefix120.at("controller_diagnostics").at("learning_frozen") == false,
                        "learning froze before the return intervention");
                if (paired_reference) require(prefix120.dump() == paired_reference->result.at("prefix120").dump(),
                    "paired 120-second prefix snapshot or controller state differs");
                if (recalled) {
                    const Json before = controller.diagnostics();
                    const std::string physical_before = world.state().dump();
                    controller.restore_values(*checkpoint);
                    const Json after = controller.diagnostics();
                    restoration = Json{
                        {"before_parameter_fingerprint_sha256", before.at("parameter_fingerprint_sha256")},
                        {"after_parameter_fingerprint_sha256", after.at("parameter_fingerprint_sha256")},
                        {"before_nonparameter_state_fingerprint_sha256", before.at("nonparameter_state_fingerprint_sha256")},
                        {"after_nonparameter_state_fingerprint_sha256", after.at("nonparameter_state_fingerprint_sha256")},
                        {"after_restore_controller_diagnostics", after},
                        {"public_physical_state_unchanged", physical_before == world.state().dump()}};
                    require(before.at("nonparameter_state_fingerprint_sha256") == after.at("nonparameter_state_fingerprint_sha256"),
                        "value restoration changed history, counters, random state or other nonparameter state");
                    require(after.at("parameter_fingerprint_sha256") == checkpoint60.at("captured_parameter_fingerprint_sha256"),
                        "restored parameters do not match the first-A checkpoint");
                    require(restoration.at("public_physical_state_unchanged") == true, "restoration changed physical state");
                }
                if (frozen) {
                    frozen_parameter_hash = controller.diagnostics().at("parameter_fingerprint_sha256").get<std::string>();
                    controller.freeze_learning();
                    require(controller.diagnostics().at("parameter_fingerprint_sha256") == frozen_parameter_hash,
                        "freezing changed learned parameters");
                }
                // No option, history, physical state or sensor queue resets.
                world.set_sun(sun(false));
            }
            const double effort = controller.act(world.observation());
            attempted_step = steps + 1;
            attempted_effort = effort;
            attempted_accepted = false;
            const Json transition = world.step(effort);
            ++steps;
            const Json physical = world.state();
            efforts.push_back(effort);
            run.observations.push_back(transition.at("observation"));
            run.transitions.push_back(transition);
            const double energy = physical.at("diagnostics").at("electrical_energy_j");
            const double energy_delta = energy - previous_energy;
            previous_energy = energy;
            whole.add(transition, physical, energy_delta);
            if (steps <= kReturn) first120.add(transition, physical, energy_delta);
            if (steps <= kFirstMove) first.add(transition, physical, energy_delta);
            else if (steps <= kReturn) second.add(transition, physical, energy_delta);
            else returned.add(transition, physical, energy_delta);
            if (steps > 2000 && steps <= 3000) first_tail.add(transition, physical, energy_delta);
            if (steps > 5000 && steps <= 6000) second_tail.add(transition, physical, energy_delta);
            if (steps > 6000 && steps <= 7000) early.add(transition, physical, energy_delta);
            if (steps > 7000 && steps <= 8000) middle.add(transition, physical, energy_delta);
            if (steps > 8000) late.add(transition, physical, energy_delta);
            if (steps % 5 == 0) curve.push_back(Json{{"time_s", steps * kDt},
                {"illuminance_lux", physical.at("light_sensor").at("observations").at("illuminance_lux")},
                {"sensor_reward_rate", physical.at("reward").at("rate")}, {"effort", effort},
                {"cumulative_sensor_reward", whole.reward}});
            controller.observe(transition);
            if (paired_reference && steps <= kReturn) {
                require(same_effort(effort, paired_reference->result.at("efforts").at(steps - 1).get<double>()),
                        "paired prefix command differs at step " + std::to_string(steps));
                require(transition.dump() == paired_reference->transitions.at(steps - 1).dump(),
                        "paired prefix full transition differs at step " + std::to_string(steps));
            }
            if (frozen && steps > kReturn) {
                const Json diagnostics = controller.diagnostics();
                parameters_unchanged = parameters_unchanged && diagnostics.at("learning_frozen") == true &&
                    diagnostics.at("parameter_fingerprint_sha256") == frozen_parameter_hash;
                ++frozen_parameter_checks;
                require(parameters_unchanged, "a frozen parameter changed after returning to A");
            }
            attempted_accepted = true;
        }
    } catch (const std::exception& exception) { error = exception.what(); }
    Json final_diagnostics, final_physical;
    std::vector<std::string> snapshot_errors;
    try { final_diagnostics = controller.diagnostics(); }
    catch (const std::exception& exception) { snapshot_errors.push_back("controller: " + std::string(exception.what())); }
    try { final_physical = world.state(); }
    catch (const std::exception& exception) { snapshot_errors.push_back("public physical state: " + std::string(exception.what())); }
    if (!snapshot_errors.empty() && error.empty()) error = "final state snapshot failed";
    std::string replay_error;
    std::size_t replayed_steps = 0;
    bool replay_matches = false;
    try {
        require(snapshot_errors.empty() && final_diagnostics.is_object() && final_physical.is_object(),
                "final snapshots unavailable; full replay cannot be certified");
        droid::ConstructionWorld replay;
        replay.rebuild(assembly);
        replay.set_sun(sun(false));
        droid::LightControl replay_controller;
        replay_controller.reset("learner", seed);
        std::optional<droid::LightControl::ValueCheckpoint> replay_checkpoint;
        for (std::size_t i = 0; i < efforts.size(); ++i) {
            if (i == kFirstMove) {
                require(replay_controller.diagnostics().dump() == checkpoint60.at("controller_diagnostics").dump(),
                        "replay first-A checkpoint state differs");
                replay_checkpoint.emplace(replay_controller.capture_values());
                require(replay_controller.diagnostics().dump() == checkpoint60.at("controller_diagnostics").dump(),
                        "replay capture changed controller state");
                replay.set_sun(sun(true));
            }
            if (i == kReturn) {
                require(replay.state().dump() == prefix120.at("physical_snapshot").dump(), "replay return public physical snapshot differs");
                require(replay_controller.diagnostics().dump() == prefix120.at("controller_diagnostics").dump(),
                        "replay return controller state differs");
                require(replay_checkpoint.has_value(), "replay lacks the first-A checkpoint");
                if (recalled) {
                    replay_controller.restore_values(*replay_checkpoint);
                    require(replay_controller.diagnostics().dump() == restoration.at("after_restore_controller_diagnostics").dump(),
                            "replay value restoration differs");
                }
                if (frozen) replay_controller.freeze_learning();
                replay.set_sun(sun(false));
            }
            const double effort = replay_controller.act(replay.observation());
            require(same_effort(effort, efforts.at(i).get<double>()), "replay command bits differ at step " + std::to_string(i + 1));
            const Json transition = replay.step(effort);
            require(transition.dump() == run.transitions.at(i).dump(), "replay serialized transition differs at step " + std::to_string(i + 1));
            replay_controller.observe(transition);
            ++replayed_steps;
        }
        require(replay_controller.diagnostics().dump() == final_diagnostics.dump(), "replay final controller state differs");
        require(replay.state().dump() == final_physical.dump(), "replay final public physical state differs");
        replay_matches = error.empty() && replayed_steps == kSteps;
    } catch (const std::exception& exception) { replay_error = exception.what(); }
    run.result = Json{{"branch", branch}, {"mode", "learner"}, {"seed", seed},
        {"completed", steps == kSteps && error.empty()}, {"completed_steps", steps}, {"error", error},
        {"recorded_transition_count", run.transitions.size()}, {"recorded_effort_count", efforts.size()},
        {"last_attempted_action", Json{{"step", attempted_step}, {"effort", attempted_effort}, {"fully_accepted", attempted_accepted}}},
        {"snapshot_errors", snapshot_errors}, {"final_controller_snapshot_available", final_diagnostics.is_object()},
        {"final_public_physical_snapshot_available", final_physical.is_object()},
        {"exact_controller_and_observation_replay", replay_matches}, {"exact_full_transition_replay", replay_matches},
        {"replay", Json{{"completed_steps", replayed_steps}, {"error", replay_error},
            {"scope", "all 180s command bits, serialized full transitions, checkpoint/restore events and final states"}}},
        {"effort_trace_sha256", hash(efforts)}, {"observation_trace_sha256", hash(run.observations)},
        {"transition_trace_sha256", hash(run.transitions)}, {"checkpoint60", checkpoint60}, {"prefix120", prefix120},
        {"restoration", restoration}, {"frozen_parameters_unchanged", parameters_unchanged},
        {"frozen_parameter_checks", frozen_parameter_checks}, {"frozen_parameter_fingerprint_sha256", frozen_parameter_hash},
        {"whole_run", whole.json()}, {"prefix120_metrics", first120.json()}, {"before_first_move", first.json()},
        {"middle_condition", second.json()}, {"after_return", returned.json()},
        {"first_phase_final_20s", first_tail.json()}, {"second_phase_final_20s", second_tail.json()},
        {"early_after_return_20s", early.json()}, {"middle_after_return_20s", middle.json()}, {"late_after_return_20s", late.json()},
        {"controller_final", final_diagnostics}, {"final_public_physical_snapshot_sha256", final_physical.is_object() ? Json(hash(final_physical)) : Json(nullptr)},
        {"efforts", std::move(efforts)}, {"curve", std::move(curve)}};
    run.result["reference_compatibility"] = Json{
        {"commands", prefix120.is_object() && prefix120.at("commands_sha256") == hash(original.at("efforts"))},
        {"observations", prefix120.is_object() && prefix120.at("observation_trace_sha256") == original.at("observation_trace_sha256")},
        {"return", first120.json().at("integrated_sensor_reward").dump() == original.at("whole_run").at("integrated_sensor_reward").dump()},
        {"metrics", matches_reference_metrics(run.result, original)}};
    return run;
}
Json case_result(const Json& assembly, std::size_t body_index, std::uint64_t seed, const Json& original) {
    std::cout << "body " << body_index << ", seed " << seed << ": continued and replay" << std::endl;
    Run continued = run_branch(assembly, seed, "continued", original, nullptr);
    std::cout << "body " << body_index << ", seed " << seed << ": current frozen and replay" << std::endl;
    Run current = run_branch(assembly, seed, "current_frozen", original, &continued);
    std::cout << "body " << body_index << ", seed " << seed << ": recalled frozen and replay" << std::endl;
    Run recalled = run_branch(assembly, seed, "recalled_frozen", original, &continued);
    const Json& c = continued.result;
    const Json& f = current.result;
    const Json& r = recalled.result;
    const bool have_prefix = c.at("prefix120").is_object() && f.at("prefix120").is_object() && r.at("prefix120").is_object();
    const auto same_prefix = [&](const std::string& key) {
        return have_prefix && c.at("prefix120").at(key).dump() == f.at("prefix120").at(key).dump() &&
            c.at("prefix120").at(key).dump() == r.at("prefix120").at(key).dump();
    };
    bool references_match = true;
    for (const Json* branch : {&c, &f, &r})
        for (const auto& [key, passed] : branch->at("reference_compatibility").items()) { (void)key; references_match = references_match && passed == true; }
    const bool restored = r.at("restoration").is_object() &&
        r.at("restoration").at("before_nonparameter_state_fingerprint_sha256") == r.at("restoration").at("after_nonparameter_state_fingerprint_sha256") &&
        r.at("restoration").at("after_parameter_fingerprint_sha256") == r.at("checkpoint60").at("captured_parameter_fingerprint_sha256") &&
        r.at("restoration").at("public_physical_state_unchanged") == true;
    Json checks{{"identical_prefix_commands", same_prefix("commands_sha256")},
        {"identical_prefix_observations", same_prefix("observation_trace_sha256")},
        {"identical_prefix_full_transitions", same_prefix("transition_trace_sha256")},
        {"identical_prefix_public_physical_snapshot", same_prefix("physical_snapshot")},
        {"identical_prefix_controller_state_fingerprint", same_prefix("state_fingerprint_sha256")},
        {"identical_prefix_controller_diagnostics", same_prefix("controller_diagnostics")},
        {"identical_first_A_checkpoint", c.at("checkpoint60").is_object() && c.at("checkpoint60").dump() == f.at("checkpoint60").dump() && c.at("checkpoint60").dump() == r.at("checkpoint60").dump()},
        {"all_prefixes_match_archived_v2", references_match},
        {"restoration_changes_only_values", restored},
        {"current_frozen_parameters_unchanged", f.at("frozen_parameters_unchanged") == true && f.at("frozen_parameter_checks") == 3000},
        {"recalled_frozen_parameters_unchanged", r.at("frozen_parameters_unchanged") == true && r.at("frozen_parameter_checks") == 3000},
        {"continued_full_replay", c.at("exact_full_transition_replay") == true},
        {"current_frozen_full_replay", f.at("exact_full_transition_replay") == true},
        {"recalled_frozen_full_replay", r.at("exact_full_transition_replay") == true}};
    bool valid = c.at("completed") == true && f.at("completed") == true && r.at("completed") == true;
    for (const auto& [name, passed] : checks.items()) { (void)name; valid = valid && passed == true; }
    const auto diff = [](const Json& a, const Json& b, const std::string& group, const std::string& field) {
        return a.at(group).at(field).get<double>() - b.at(group).at(field).get<double>();
    };
    Json differences{{"primary_early_recalled_minus_current_reward", diff(r, f, "early_after_return_20s", "integrated_sensor_reward")},
        {"full_return_recalled_minus_current_reward", diff(r, f, "after_return", "integrated_sensor_reward")},
        {"early_lux_recalled_minus_current", diff(r, f, "early_after_return_20s", "mean_received_lux")},
        {"late_lux_recalled_minus_current", diff(r, f, "late_after_return_20s", "mean_received_lux")},
        {"energy_recalled_minus_current", diff(r, f, "after_return", "electrical_energy_j")},
        {"full_return_continued_minus_current_reward", diff(c, f, "after_return", "integrated_sensor_reward")},
        {"full_return_continued_minus_recalled_reward", diff(c, r, "after_return", "integrated_sensor_reward")}};
    return Json{{"case_id", "body-" + std::to_string(body_index) + "-seed-" + std::to_string(seed)},
        {"body_index", body_index}, {"assembly", assembly}, {"assembly_sha256", hash(assembly)}, {"seed", seed},
        {"valid", valid}, {"checks", checks}, {"differences", differences},
        {"continued", std::move(continued.result)}, {"current_frozen", std::move(current.result)}, {"recalled_frozen", std::move(recalled.result)}};
}
void save(const fs::path& output, const Json& report) {
    const fs::path next = output / "progress.next.json";
    std::ofstream stream(next, std::ios::binary | std::ios::trunc);
    stream << report.dump(2) << '\n';
    stream.close();
    require(static_cast<bool>(stream), "cannot save retention report");
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
             "config/module_catalog.json", "models/droid.xml", "tools/summarize-light-retention.mjs",
             "config/learning_experiment_v3.json", "artifacts/light-search-policy-v3-seed0.json",
             "docs/LIGHT_RETENTION_EXPERIMENT.md", "docs/LIGHT_LEARNING_EXPERIMENT.md"}) files.insert(file);
    if (fs::is_regular_file("build/light-retention-qa/runtime.json")) files.insert("build/light-retention-qa/runtime.json");
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
            throw std::invalid_argument("Usage: droid-light-retention --output NEW_DIRECTORY (from project root)");
        constexpr std::string_view protocol_sha = "d3a8daf76154004c1610d0d15e6b60983782ebfa081ef1739c3341977fd44af9";
        require(droid::sha256_file(std::string(kProtocol)) == protocol_sha, "predeclared retention protocol hash differs");
        require(droid::sha256_file(std::string(kReference)) == kReferenceSha, "fixed archived v2 report hash differs");
        const Json reference = read_json(fs::path(kReference));
        require(reference.at("complete") == true && reference.at("failed_trials") == 0, "archived v2 reference is invalid");
        require(reference.at("control_spec").at("learner").at("learner_id") == "history_expected_sarsa_v2",
                "reference learner differs from declared v2");
        require(reference.at("constructions").size() == 2, "reference must contain the two known bodies");
        output = fs::path(argv[2]);
        require(fs::create_directory(output), "output directory already exists; preserve previous evidence");
        reserved = true;
        report = Json{{"schema", "light_retention_v1"}, {"complete", false}, {"finished", false}, {"failed_cases", 0},
            {"exposure", "Six known v2 prefixes, three branches each; no new bodies, tuning, context recognition or held-out claim"},
            {"protocol", Json{{"document", kProtocol}, {"document_sha256", protocol_sha}, {"steps", kSteps}, {"control_dt_s", kDt},
                {"first_move_step", kFirstMove}, {"return_step", kReturn}, {"first_move_time_s", 60}, {"return_time_s", 120},
                {"initial_sun", sun(false)}, {"middle_sun", sun(true)}, {"returned_sun", sun(false)},
                {"reset_at_interventions", false}, {"seeds", {1, 2, 3}}, {"body_count", 2},
                {"branches", {"continued", "current_frozen", "recalled_frozen"}},
                {"primary_metric", "recalled_frozen minus current_frozen integrated SUM sensor reward over120-140s"},
                {"early_window_s", {120, 140}}, {"late_window_s", {160, 180}},
                {"checkpoint_scope", "168 action-value parameters captured at60s; only these restored at120s in recalled_frozen"},
                {"nonparameter_scope", "physical state, sensor queues, history, replay, eligibility, RNG, counters and unfinished option are preserved"},
                {"freeze_scope", "both comparison branches suppress parameter writes after120s; continued branch keeps updating"},
                {"control_spec_scope", "runner owns180s; inherited comparison budget text and untouched browser LightSession describe120s, while LightControl has no horizon stop"},
                {"native_state_caveat", "public snapshots are not complete MuJoCo checkpoints; identical deterministic full prefixes reconstruct native state"},
                {"causal_limit", "saved-checkpoint evaluation of retained value competence; checkpoint selection is external evaluation machinery, not autonomous context recognition"}}},
            {"reference_report", Json{{"path", kReference}, {"sha256", kReferenceSha}, {"snapshot", "source/" + std::string(kReference)}}},
            {"compiler", __VERSION__}, {"physics_spec", droid::ConstructionWorld::light_specification()},
            {"control_spec", droid::LightControl::specification()}, {"module_catalog", read_json("config/module_catalog.json")},
            {"source_sha256", Json::object()}, {"cases", Json::array()}};
        snapshot_sources(output, report, reference);
        require(report.at("source_sha256").at(std::string(kProtocol)) == protocol_sha, "protocol changed before its snapshot");
        save(output, report);
        std::size_t failed = 0, body_index = 0;
        for (const auto& construction : reference.at("constructions")) {
            const Json& assembly = construction.at("assembly");
            require(hash(assembly) == construction.at("assembly_sha256"), "reference assembly hash differs");
            for (std::uint64_t seed = 1; seed <= 3; ++seed) {
                const Json* original = nullptr;
                for (const auto& trial : construction.at("trials")) {
                    if (trial.at("mode") == "learner" && trial.at("seed") == seed) {
                        require(original == nullptr, "duplicate reference learner seed");
                        original = &trial;
                    }
                }
                require(original && original->at("completed") == true && original->at("exact_controller_and_observation_replay") == true,
                        "missing valid reference learner trial");
                Json result = case_result(assembly, body_index, seed, *original);
                if (result.at("valid") != true) ++failed;
                std::cout << result.at("case_id").get<std::string>() << " valid=" << result.at("valid")
                          << " early recalled-current reward=" << result.at("differences").at("primary_early_recalled_minus_current_reward") << std::endl;
                report["cases"].push_back(std::move(result));
                report["failed_cases"] = failed;
                save(output, report);
            }
            ++body_index;
        }
        report["finished"] = true;
        report["complete"] = failed == 0 && report.at("cases").size() == 6;
        save(output, report);
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (reserved && report.is_object()) {
            report["complete"] = false;
            report["fatal_error"] = error.what();
            try { save(output, report); } catch (...) { }
        }
        return 2;
    }
}