#include "droid/body_controller.hpp"
#include "droid/environment.hpp"
#include "droid/playground.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

void save_progress(const std::filesystem::path& directory, const Json& report) {
    // The directory is reserved exclusively before any rollout. Preserve a
    // previous complete checkpoint if writing the next checkpoint fails.
    const auto temporary = directory / "progress.next.json";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream << report.dump(2) << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("unable to retain development progress");
    std::filesystem::rename(temporary, directory / "report.json");
}

double light(const Json& observation) {
    for (const auto& sensor : observation.at("reward_sensors")) {
        if (sensor.at("family_id") == "ambient_light_v0" && sensor.at("valid") == true)
            return sensor.at("observations").at("illuminance_lux").get<double>();
    }
    return -1.0;
}

Json run_episode(droid::DroidEnvironment& environment, const std::string& variant,
                 const std::string& side, const std::string& strategy) {
    const auto ids = environment.spec().at("action").at("motor_ids").get<std::vector<std::string>>();
    const auto reset = environment.reset("transmission_" + variant + "_v1",
        side == "left" ? droid::kPlaygroundLeftSeed : droid::kPlaygroundRightSeed,
        false, "body_discovery_1d_v1");
    Json observation = reset.at("observation");
    droid::BodyProbeController controller;
    controller.reset(ids, strategy == "zero" ? "discover" : strategy);
    std::map<std::string, double> zeros;
    for (const auto& id : ids) zeros[id] = 0.0;
    double initial = -1.0, final = -1.0, peak_lux = 0.0, reward = 0.0;
    double effort_sum = 0.0, max_effort = 0.0, max_current = 0.0, max_temperature = 0.0;
    double peak_shaft_speed = 0.0;
    std::size_t steps = 0, vetoes = 0;
    std::size_t feedback_flag_steps = 0;
    bool terminated = false, truncated = false;
    std::string controller_error;
    Json samples = Json::array();
    for (; steps < droid::kBodyPlaygroundMaxSteps;) {
        // Complete controller boundary: observation and opaque IDs only.
        std::map<std::string, double> actions;
        try {
            actions = strategy == "zero" ? zeros : controller.act(observation, ids);
        } catch (const std::exception& exception) {
            controller_error = exception.what();
            break;
        }
        const Json transition = environment.step(actions, droid::kPlaygroundControlDtS);
        observation = transition.at("observation");
        ++steps;
        final = light(observation);
        if (final >= 0.0) {
            if (initial < 0.0) initial = final;
            peak_lux = std::max(peak_lux, final);
        }
        reward += transition.at("reward").get<double>();
        for (const auto& [id, effort] : actions) {
            (void)id;
            effort_sum += std::abs(effort);
            max_effort = std::max(max_effort, std::abs(effort));
        }
        const Json state = environment.visualization_state();
        if (state.at("safety").at("veto") == true) ++vetoes;
        for (const auto& value : state.at("actuators").at("current_a"))
            max_current = std::max(max_current, std::abs(value.get<double>()));
        for (const auto& value : state.at("actuators").at("temperature_c"))
            max_temperature = std::max(max_temperature, value.get<double>());
        for (const auto& value : state.at("actuators").at("shaft_velocity_rad_s"))
            peak_shaft_speed = std::max(peak_shaft_speed, std::abs(value.get<double>()));
        bool flagged = false;
        for (const auto& motor : observation.at("actuator_feedback"))
            flagged = flagged || !motor.at("feedback").at("fault_flags").empty();
        if (flagged) ++feedback_flag_steps;
        if (steps % 25 == 0 || steps == 1) samples.push_back(Json{
            {"step", steps}, {"illuminance_lux", final < 0.0 ? Json(nullptr) : Json(final)},
            {"controller", strategy == "zero" ? Json(nullptr) : controller.diagnostics()},
        });
        terminated = transition.at("terminated").get<bool>();
        truncated = transition.at("truncated").get<bool>();
        if (terminated || truncated) break;
    }
    return Json{
        {"variant", variant}, {"side", side}, {"strategy", strategy},
        {"episode_status", !controller_error.empty() ? "controller_stopped" : terminated || truncated ? "environment_stopped" : "completed"},
        {"controller_error", controller_error.empty() ? Json(nullptr) : Json(controller_error)},
        {"steps", steps}, {"simulated_seconds", steps * droid::kPlaygroundControlDtS},
        {"initial_illuminance_lux", initial}, {"final_illuminance_lux", final},
        {"peak_illuminance_lux", peak_lux}, {"summed_sensor_reward_integral", reward},
        {"mean_absolute_effort", steps == 0 ? 0.0 : effort_sum / static_cast<double>(steps * ids.size())},
        {"max_absolute_effort", max_effort}, {"peak_current_a", max_current},
        {"peak_temperature_c", max_temperature}, {"safety_veto_steps", vetoes},
        {"peak_absolute_shaft_velocity_rad_s", peak_shaft_speed},
        {"feedback_flag_steps", feedback_flag_steps},
        {"terminated", terminated}, {"truncated", truncated},
        {"final_controller", strategy == "zero" ? Json(nullptr) : controller.diagnostics()},
        {"controller_samples", samples}, {"final_visualization", environment.visualization_state()},
        {"final_observation", observation},
    };
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--output-dir") {
            std::cerr << "Usage: droid-body-experiment --output-dir NEW_DIRECTORY\n";
            return 2;
        }
        const std::filesystem::path directory(argv[2]);
        std::ifstream input("config/body_experiment_v1.json");
        if (!input) throw std::runtime_error("missing development protocol");
        const Json protocol = Json::parse(input);
        if (protocol.at("id") != "transmission_discovery_v1" ||
            protocol.at("status") != "development_only" ||
            protocol.at("parent_weights_used") != false ||
            protocol.at("historical_parent_artifact_sha256") != droid::kPlaygroundArtifactSha256 ||
            protocol.at("episode_profile") != "body_discovery_1d_v1" ||
            protocol.at("assembly_ids") != Json::array({"transmission_normal_v1", "transmission_reversed_v1", "transmission_disconnected_v1"}) ||
            protocol.at("strategies") != Json::array({"discover", "together", "zero"}) ||
            protocol.at("worlds").size() != 2 ||
            protocol.at("worlds").at(0).at("side") != "left" ||
            protocol.at("worlds").at(1).at("side") != "right" ||
            protocol.at("light_sensor_period_s") != droid::body_controller_spec().at("sensor_sample_period_s") ||
            protocol.at("light_sensor_latency_s") != droid::body_controller_spec().at("sensor_latency_s") ||
            protocol.at("control_dt_s") != droid::kPlaygroundControlDtS ||
            protocol.at("max_control_steps") != droid::kBodyPlaygroundMaxSteps ||
            protocol.at("worlds").at(0).at("seed_decimal") != std::to_string(droid::kPlaygroundLeftSeed) ||
            protocol.at("worlds").at(1).at("seed_decimal") != std::to_string(droid::kPlaygroundRightSeed))
            throw std::runtime_error("development protocol differs from runner");
        droid::DroidEnvironment environment;
        const Json mechanics = environment.body_experiment_spec();
        if (mechanics.at("episode_profile") != protocol.at("episode_profile") ||
            mechanics.at("supported_assembly_ids") != protocol.at("assembly_ids") ||
            mechanics.at("light_timing").at("sample_period_s") != protocol.at("light_sensor_period_s") ||
            mechanics.at("light_timing").at("latency_s") != protocol.at("light_sensor_latency_s"))
            throw std::runtime_error("declared protocol differs from native body mechanics");
        Json inputs = Json::object();
        for (const std::string path : {"config/body_experiment_v1.json", "models/droid.xml",
            "config/module_catalog.json", "src/simulation.cpp", "src/environment.cpp",
            "src/body_controller.cpp", "include/droid/body_controller.hpp", "src/body_experiment_cli.cpp",
            "artifacts/light-search-policy-v3-seed0.json", "config/learning_experiment_v3.json"})
            inputs[path] = droid::sha256_file(path);
        Json report{
            {"schema", "droid-blocks.body-development-report.v1"},
            {"status", "running"}, {"protocol", protocol}, {"source_sha256", inputs},
            {"runtime", environment.health()}, {"legacy_environment_spec", environment.spec()},
            {"body_environment_spec", mechanics},
            {"controller_spec", droid::body_controller_spec()},
            {"test_rollouts_executed", false}, {"official_acceptance", nullptr},
            {"episodes", Json::array()},
        };
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("output directory already exists; preserved without overwrite");
        save_progress(directory, report);
        try {
            for (const std::string variant : {"normal", "reversed", "disconnected"})
                for (const std::string side : {"left", "right"})
                    for (const std::string strategy : {"discover", "together", "zero"}) {
                        report["current_episode"] = Json{{"variant", variant}, {"side", side}, {"strategy", strategy}};
                        save_progress(directory, report);
                        Json episode = run_episode(environment, variant, side, strategy);
                        std::cout << variant << ' ' << side << ' ' << strategy << ": "
                            << episode.at("initial_illuminance_lux") << " -> "
                            << episode.at("final_illuminance_lux") << " lx; steps="
                            << episode.at("steps") << "; " << episode.at("episode_status") << '\n' << std::flush;
                        report["episodes"].push_back(std::move(episode));
                        save_progress(directory, report);
                    }
            report["status"] = "completed";
            report["current_episode"] = nullptr;
            save_progress(directory, report);
        } catch (const std::exception& exception) {
            report["status"] = "error";
            report["error"] = exception.what();
            save_progress(directory, report);
            throw;
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
