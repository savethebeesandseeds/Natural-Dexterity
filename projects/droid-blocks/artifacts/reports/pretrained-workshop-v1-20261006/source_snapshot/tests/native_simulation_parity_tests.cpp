#include "droid/simulation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Json = nlohmann::json;

constexpr double kKinematicTolerance = 2.0e-5;
constexpr double kTelemetryTolerance = 5.0e-6;
constexpr double kRewardTolerance = 2.0e-6;

[[nodiscard]] std::filesystem::path default_model_path() {
#if defined(DROID_TEST_MODEL_PATH)
    return DROID_TEST_MODEL_PATH;
#elif defined(DROID_MODEL_PATH)
    return DROID_MODEL_PATH;
#else
    return "models/droid.xml";
#endif
}

[[nodiscard]] std::filesystem::path default_fixture_path() {
#if defined(DROID_GOLDEN_TRAJECTORY_PATH)
    return DROID_GOLDEN_TRAJECTORY_PATH;
#elif defined(DROID_TEST_FIXTURE_PATH)
    return DROID_TEST_FIXTURE_PATH;
#else
    return "tests/golden_trajectory.json";
#endif
}

[[nodiscard]] Json load_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open fixture: " + path.string());
    }
    Json result;
    try {
        input >> result;
    } catch (const Json::exception& error) {
        throw std::runtime_error(
            "invalid JSON fixture " + path.string() + ": " + error.what());
    }
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& field,
    std::size_t step) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "step " + std::to_string(step) + " field " + field +
            " expected " + std::to_string(expected) + " but got " +
            std::to_string(actual) + " (tolerance " +
            std::to_string(tolerance) + ")");
    }
}

void require_vector_near(
    const Json& actual,
    const Json& expected,
    double tolerance,
    std::string_view field,
    std::size_t step) {
    require(
        actual.is_array() && expected.is_array(),
        "step " + std::to_string(step) + " field " + std::string(field) +
            " must be an array");
    require(
        actual.size() == expected.size(),
        "step " + std::to_string(step) + " field " + std::string(field) +
            " has the wrong length");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_near(
            actual.at(index).get<double>(),
            expected.at(index).get<double>(),
            tolerance,
            std::string(field) + "[" + std::to_string(index) + "]",
            step);
    }
}

void compare_checkpoint(const Json& state, const Json& expected) {
    const std::size_t step = expected.at("step").get<std::size_t>();
    require(
        state.at("tick").get<std::size_t>() == step,
        "state tick does not match checkpoint " + std::to_string(step));
    require(!state.at("running").get<bool>(), "probe simulation must stay paused");
    require(
        state.at("world").at("episode_profile") == "fixed_demo_v0" &&
            state.at("world").at("light_source_position") ==
                Json::array({16.0, 0.0, 0.0}),
        "unprofiled simulation reset must retain the canonical fixed world");

    require_near(
        state.at("sim_time").get<double>(),
        expected.at("time").get<double>(),
        kTelemetryTolerance,
        "sim_time",
        step);
    require_vector_near(
        state.at("robot").at("position"),
        expected.at("position"),
        kKinematicTolerance,
        "robot.position",
        step);
    require_near(
        state.at("robot").at("speed").get<double>(),
        expected.at("speed").get<double>(),
        kKinematicTolerance,
        "robot.speed",
        step);
    require_near(
        state.at("robot").at("distance").get<double>(),
        expected.at("distance").get<double>(),
        kKinematicTolerance,
        "robot.distance",
        step);
    require_near(
        state.at("reward").at("total_rate").get<double>(),
        expected.at("reward_rate").get<double>(),
        kRewardTolerance,
        "reward.total_rate",
        step);
    require_near(
        state.at("reward").at("cumulative").get<double>(),
        expected.at("cumulative_reward").get<double>(),
        kRewardTolerance,
        "reward.cumulative",
        step);

    const Json& actuators = state.at("actuators");
    require_vector_near(
        actuators.at("commands"),
        expected.at("commands"),
        kKinematicTolerance,
        "actuators.commands",
        step);
    require_vector_near(
        actuators.at("current_a"),
        expected.at("current_a"),
        kTelemetryTolerance,
        "actuators.current_a",
        step);
    require_vector_near(
        actuators.at("temperature_c"),
        expected.at("temperature_c"),
        kTelemetryTolerance,
        "actuators.temperature_c",
        step);
    require_vector_near(
        actuators.at("stuck_score"),
        expected.at("stuck_score"),
        kTelemetryTolerance,
        "actuators.stuck_score",
        step);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path model_path =
            argc > 1 ? std::filesystem::path(argv[1]) : default_model_path();
        const std::filesystem::path fixture_path =
            argc > 2 ? std::filesystem::path(argv[2]) : default_fixture_path();
        if (argc > 3) {
            throw std::invalid_argument(
                "usage: native_simulation_parity_tests [model.xml] "
                "[golden_trajectory.json]");
        }

        const Json checkpoints = load_json(fixture_path);
        require(
            checkpoints.is_array() && !checkpoints.empty(),
            "golden trajectory must be a non-empty array");

        droid::DroidSimulation simulation(model_path, false);
        simulation.reset();
        std::size_t previous_step = 0;
        for (const Json& checkpoint : checkpoints) {
            require(
                checkpoint.is_object() && checkpoint.contains("step"),
                "each golden checkpoint must be an object with a step");
            const std::size_t step = checkpoint.at("step").get<std::size_t>();
            require(
                step > previous_step,
                "golden checkpoint steps must be strictly increasing");
            simulation.advance_steps(step - previous_step);
            compare_checkpoint(simulation.state(), checkpoint);
            std::cout << "[PASS] native parity at step " << step << '\n';
            previous_step = step;
        }

        std::cout << checkpoints.size()
                  << " native simulation checkpoints passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] native simulation parity: " << error.what()
                  << '\n';
        return 1;
    }
}
