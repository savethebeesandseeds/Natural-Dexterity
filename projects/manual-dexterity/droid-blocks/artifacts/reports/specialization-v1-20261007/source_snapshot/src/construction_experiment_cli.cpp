#include "droid/construction.hpp"
#include "droid/motion_explorer.hpp"
#include "droid/policy.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
double magnitude(const Json& vector) {
    double sum = 0;
    for (const auto& component : vector) sum += std::pow(component.get<double>(), 2);
    return std::sqrt(sum);
}
Json run_trial(const Json& assembly, const Json& pattern) {
    droid::ConstructionWorld world;
    world.rebuild(assembly);
    droid::MotionExplorer explorer;
    const std::string id = pattern.at("pattern_id").get<std::string>();
    const bool zero = id == "zero";
    if (!zero) explorer.reset(id);
    const Json initial = world.state();
    Json trace = Json::array(), efforts = Json::array(), poses = Json::array();
    double reward = 0, max_current = 0, max_temperature = 0, max_shaft_speed = 0;
    double max_passive_speed = 0, max_gyro = 0, max_force = 0;
    std::size_t steps = 0, vetoes = 0, flagged_steps = 0;
    bool terminated = false;
    std::string error;
    try {
        for (; steps < 300;) {
            const double effort = zero ? 0.0 : explorer.act(world.observation());
            const Json transition = world.step(effort);
            ++steps;
            const Json obs = transition.at("observation");
            const Json physical = world.state();
            efforts.push_back(effort);
            trace.push_back(obs);
            if (steps % 10 == 0) poses.push_back(Json{{"time_s", steps * .02}, {"tip_position_m", physical.at("tip_position_m")}});
            reward += transition.at("reward").get<double>();
            if (transition.at("safety").at("veto") == true) ++vetoes;
            const Json feedback = obs.at("actuator_feedback").at(0).at("feedback");
            if (!feedback.at("fault_flags").empty()) ++flagged_steps;
            max_current = std::max(max_current, std::abs(feedback.at("current_a").get<double>()));
            max_temperature = std::max(max_temperature, feedback.at("temperature_c").get<double>());
            max_shaft_speed = std::max(max_shaft_speed, std::abs(feedback.at("velocity_rad_s").get<double>()));
            max_passive_speed = std::max(max_passive_speed,
                std::abs(physical.at("diagnostics").at("passive_velocity_rad_s").get<double>()));
            const Json sensor = obs.at("reward_sensors").at(0);
            if (sensor.at("valid") == true) {
                max_gyro = std::max(max_gyro, magnitude(sensor.at("observations").at("angular_velocity_rad_s")));
                max_force = std::max(max_force, magnitude(sensor.at("observations").at("specific_force_m_s2")));
            }
            terminated = transition.at("terminated").get<bool>();
            if (terminated || vetoes) throw std::runtime_error("native safety stopped trial");
            if (!zero) explorer.observe(obs);
        }
    } catch (const std::exception& exception) { error = exception.what(); }
    // Exact command replay is a separate deterministic check, not another search.
    bool exact_replay = true;
    std::string replay_error;
    try {
        droid::ConstructionWorld replay;
        replay.rebuild(assembly);
        for (std::size_t i = 0; i < efforts.size(); ++i) {
            const Json result = replay.step(efforts.at(i).get<double>());
            if (result.at("observation") != trace.at(i)) { exact_replay = false; break; }
        }
    } catch (const std::exception& exception) {
        exact_replay = false;
        replay_error = exception.what();
    }
    return Json{{"pattern", pattern}, {"completed_steps", steps}, {"completed", steps == 300 && error.empty()},
        {"error", error}, {"terminated", terminated}, {"native_veto_steps", vetoes},
        {"feedback_flag_steps", flagged_steps}, {"exact_observation_replay", exact_replay}, {"replay_error", replay_error},
        {"observation_trace_sha256", droid::sha256_hex(trace.dump())}, {"efforts", efforts},
        {"initial_physics", initial.at("diagnostics")}, {"final_physics", world.state().at("diagnostics")},
        {"max_current_a", max_current}, {"max_temperature_c", max_temperature},
        {"max_motor_speed_rad_s", max_shaft_speed}, {"max_passive_speed_rad_s", max_passive_speed},
        {"max_gyro_rad_s", max_gyro}, {"max_specific_force_m_s2", max_force},
        {"integrated_sensor_reward", reward}, {"diagnostic_tip_samples", poses},
        {"motion", zero || !error.empty() ? Json(nullptr) : explorer.result()},
        {"final_observation", world.observation()}};
}
void save(const std::filesystem::path& directory, const Json& report) {
    const auto pending = directory / "progress.next.json";
    std::ofstream stream(pending, std::ios::binary | std::ios::trunc);
    stream << report.dump(2) << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("unable to save construction report");
    std::filesystem::rename(pending, directory / "report.json");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--output")
            throw std::invalid_argument("Usage: droid-construction-experiment --output NEW_DIRECTORY");
        const std::filesystem::path output(argv[2]);
        if (!std::filesystem::create_directory(output))
            throw std::invalid_argument("output directory already exists; preserve previous evidence");
        Json report{{"schema", "construction_development_report_v1"},
            {"exposure", "known development constructions; no frozen-v3 challenge exposure"},
            {"protocol", droid::ConstructionWorld::specification()},
            {"patterns", droid::MotionExplorer::patterns()}, {"explorer_spec", droid::motion_explorer_spec()},
            {"reward_optimization", false}, {"source_sha256", Json::object()}, {"constructions", Json::array()}};
        for (const std::string file : {"src/construction.cpp", "include/droid/construction.hpp",
             "src/motion_explorer.cpp", "include/droid/motion_explorer.hpp", "src/construction_session.cpp",
             "src/construction_experiment_cli.cpp", "config/module_catalog.json", "models/droid.xml",
             "config/learning_experiment_v3.json", "artifacts/light-search-policy-v3-seed0.json"})
            report["source_sha256"][file] = droid::sha256_file(file);
        Json short_body = droid::ConstructionWorld::default_assembly();
        short_body["name"] = "Short arm";
        Json long_body = short_body;
        long_body["name"] = "Long arm";
        long_body["segments"] = Json::array({3, 5});
        Json near_weight = long_body;
        near_weight["name"] = "Weight near the hinge";
        near_weight["blocks"] = Json::array({Json{{"id", "block-1"}, {"segment", 1}, {"slot", 0}, {"side", 1}}});
        Json tip_weight = near_weight;
        tip_weight["name"] = "Same weight at the tip";
        tip_weight["blocks"][0]["slot"] = 4;
        Json patterns = droid::MotionExplorer::patterns();
        patterns.push_back(Json{{"pattern_id", "zero"}, {"label", "No motor effort"}, {"duration_s", 6.0}});
        save(output, report);
        std::size_t failures = 0;
        for (const Json& assembly : {short_body, long_body, near_weight, tip_weight}) {
            report["constructions"].push_back(Json{{"assembly", assembly}, {"assembly_sha256", droid::sha256_hex(assembly.dump())}, {"trials", Json::array()}});
            for (const Json& pattern : patterns) {
                Json trial = run_trial(assembly, pattern);
                if (trial.at("completed") != true || trial.at("exact_observation_replay") != true) ++failures;
                std::cout << assembly.at("name").get<std::string>() << " / "
                          << pattern.at("pattern_id").get<std::string>() << ": "
                          << trial.at("completed_steps") << " steps; " << trial.at("error") << std::endl;
                report["constructions"].back()["trials"].push_back(std::move(trial));
                save(output, report);
            }

            save(output, report);
        }
        report["failed_trials"] = failures;
        report["complete"] = true;
        save(output, report);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
