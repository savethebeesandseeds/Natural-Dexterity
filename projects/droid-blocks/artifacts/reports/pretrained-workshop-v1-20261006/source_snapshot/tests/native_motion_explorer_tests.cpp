#include "droid/motion_explorer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

void close(double actual, double expected, std::string_view message) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 1e-10, message);
}

template <typename Function>
void require_throws(Function&& function, std::string_view message) {
    try { std::invoke(std::forward<Function>(function)); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(std::string(message));
}

// The declared sensor clock has100Hz capture and6ms delivery; the controller
// sees the newest available reading every20ms. Constant physical vectors are
// intentional: descriptor tests must measure inputs, not pattern identities.
[[nodiscard]] Json frame(std::size_t step, double shaft_velocity = 0.0,
                         std::string motor_id = "opaque-motor",
                         std::string sensor_id = "opaque-imu") {
    const double now = static_cast<double>(step) * 0.02;
    const double sampled = now - 0.01;
    const bool valid = step > 0;
    return Json{{"schema_version", "construction_observation_v1"},
        {"reward_sensors", Json::array({Json{
            {"module_id", sensor_id}, {"family_id", "imu_6axis_v0"}, {"valid", valid},
            {"sequence", static_cast<std::uint64_t>(2 * step)},
            {"sample_time_s", valid ? Json(sampled) : Json(nullptr)},
            {"delivered_time_s", valid ? Json(sampled + 0.006) : Json(nullptr)},
            {"age_s", valid ? Json(0.01) : Json(nullptr)},
            {"sample_period_s", 0.01}, {"latency_s", 0.006},
            {"observations", Json{
                {"specific_force_m_s2", valid ? Json::array({0.0, 0.0, 9.81}) : Json::array({0.0, 0.0, 0.0})},
                {"angular_velocity_rad_s", valid ? Json::array({3.0, 4.0, 0.0}) : Json::array({0.0, 0.0, 0.0})}}}}})},
        {"actuator_feedback", Json::array({Json{
            {"module_id", motor_id}, {"sku_id", "rotary_dc_gearmotor_v0"}, {"valid", true},
            {"feedback", Json{{"position_rad", 0.0}, {"velocity_rad_s", shaft_velocity},
                {"current_a", 0.0}, {"bus_voltage_v", 7.4}, {"temperature_c", 25.0},
                {"output_torque_nm", 0.0}, {"load_impedance_nm_s_per_rad", 0.0},
                {"stuck_score", 0.0}, {"stuck", false}, {"fault_flags", Json::array()}}}}})}};
}

[[nodiscard]] Json held_frame(std::size_t step) {
    Json result = frame(step);
    Json& sensor = result["reward_sensors"][0];
    sensor["sequence"] = std::uint64_t{2};
    sensor["sample_time_s"] = 0.01;
    sensor["delivered_time_s"] = 0.016;
    sensor["age_s"] = static_cast<double>(step) * 0.02 - 0.01;
    return result;
}

[[nodiscard]] Json run_pattern(std::string pattern_id, bool renamed = false) {
    droid::MotionExplorer explorer;
    explorer.reset(std::move(pattern_id));
    double previous = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    for (std::size_t step = 0; step < droid::kMotionExplorerMaxSteps; ++step) {
        require(!explorer.complete(), "trial completed before its common horizon");
        const auto current = renamed ? frame(step, 0.0, "totally-different-ID", "renamed-eye") : frame(step);
        const double effort = explorer.act(current);
        require(std::abs(effort) <= 0.200000000001 && std::abs(effort - previous) <= 0.100000000001,
                "effort or slew bound exceeded");
        if (step < 5) require(effort == 0.0, "initial quiet interval moved the motor");
        if (step >= 290) require(effort == 0.0, "final coasting did not settle commands to zero");
        require(explorer.result().at("efforts").size() == step,
                "unobserved transition was prematurely recorded");
        explorer.observe(renamed ? frame(step + 1, 0.0, "totally-different-ID", "renamed-eye") : frame(step + 1));
        previous = effort;
        minimum = std::min(minimum, effort);
        maximum = std::max(maximum, effort);
    }
    require(explorer.complete(), "common horizon did not complete");
    require(minimum < -0.02 && maximum > 0.02, "pattern never exercised both motor signs");
    const Json result = explorer.result();
    require(result.at("status") == "completed" && result.at("steps") == 300 &&
                result.at("efforts").size() == 300, "completed result omitted transitions");
    close(result.at("elapsed_s"), 6.0, "trial duration differs across patterns");
    require(result.at("unique_sample_count") == 300, "sample count does not follow delivered sequences");
    close(result.at("first_sample_time_s"), 0.01, "first captured time wrong");
    close(result.at("last_sample_time_s"), 5.99, "last captured time wrong");
    close(result.at("observed_sample_span_s"), 5.98, "observed sample span wrong");
    close(result.at("gyro_rms_rad_s"), 5.0, "gyro RMS did not measure physical vector magnitude");
    close(result.at("gyro_peak_rad_s"), 5.0, "gyro peak wrong");
    close(result.at("specific_force_rms_m_s2"), 9.81, "gravity was silently removed from specific force");
    close(result.at("specific_force_peak_m_s2"), 9.81, "specific-force peak wrong");
    close(result.at("specific_force_variation_m_s2"), 0.0, "constant force should have zero variation");
    require(result.at("specific_force_includes_gravity") == true, "specific-force meaning missing");
    require_throws([&] { (void)explorer.act(frame(300)); }, "completed trial accepted another act");
    return result;
}

void test_equal_horizons_replay_and_identity() {
    const Json patterns = droid::MotionExplorer::patterns();
    require(patterns.is_array() && patterns.size() == 5, "declared repertoire changed unexpectedly");
    std::set<std::string> trajectories;
    for (const Json& pattern : patterns) {
        require(pattern.at("duration_s") == 6.0 && pattern.at("max_steps") == 300,
                "pattern budgets differ");
        const std::string id = pattern.at("id");
        const Json first = run_pattern(id);
        const Json replayed = run_pattern(id);
        const Json renamed = run_pattern(id, true);
        require(first == replayed, "same pattern and physical observations did not replay exactly");
        require(first == renamed, "opaque module ID spelling changed the physical experiment");
        trajectories.insert(first.at("efforts").dump());
    }
    require(trajectories.size() == patterns.size(), "two declared patterns emitted identical effort sequences");
}

void test_lifecycle_and_atomic_failure() {
    droid::MotionExplorer explorer;
    require(!explorer.complete(), "uninitialized explorer is complete");
    require_throws([&] { (void)explorer.act(frame(0)); }, "reset was not required");
    require_throws([&] { explorer.reset("secret-best-motion"); }, "unknown pattern accepted");
    explorer.reset("paired_taps");
    const Json initial = explorer.diagnostics();
    require_throws([&] { explorer.observe(frame(1)); }, "observe without an act accepted");
    require(explorer.diagnostics() == initial, "invalid observe mutated fresh state");
    (void)explorer.act(frame(0));
    const Json pending = explorer.diagnostics();
    require_throws([&] { (void)explorer.act(frame(0)); }, "two acts without a physics observation accepted");
    require(explorer.diagnostics() == pending, "duplicate act mutated pending state");
    Json bad = frame(1);
    bad["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"voltage_limited"});
    require_throws([&] { explorer.observe(bad); }, "flagged poststep transition accepted");
    require(explorer.diagnostics() == pending && explorer.result().at("efforts").empty(),
            "failed poststep observation committed effort or sensor memory");
    explorer.observe(frame(1));
    require(explorer.result().at("efforts").size() == 1, "valid poststep failed to commit");
    const Json observed = explorer.diagnostics();
    require_throws([&] { explorer.observe(frame(1)); }, "duplicate poststep observation accepted");
    require(explorer.diagnostics() == observed, "duplicate observe changed state");
    explorer.reset("paired_taps");
    require(explorer.diagnostics() == initial && explorer.result().at("efforts").empty(),
            "reset retained previous trial state");
}

void test_held_sample_deduplication_and_staleness() {
    droid::MotionExplorer explorer;
    explorer.reset("sway_medium");
    (void)explorer.act(frame(0));
    explorer.observe(frame(1));
    const Json prior = explorer.diagnostics();
    Json altered = frame(1);
    altered["reward_sensors"][0]["observations"]["angular_velocity_rad_s"][0] = 100.0;
    require_throws([&] { (void)explorer.act(altered); }, "same-sequence altered sample accepted");
    require(explorer.diagnostics() == prior, "altered held sample changed memory");
    for (std::size_t step = 1; step < 3; ++step) {
        (void)explorer.act(held_frame(step));
        explorer.observe(held_frame(step + 1));
    }
    require(explorer.result().at("unique_sample_count") == 1 && explorer.result().at("steps") == 3,
            "held readings counted as independent physical samples");
    (void)explorer.act(held_frame(3));
    const Json pending = explorer.diagnostics();
    require_throws([&] { explorer.observe(held_frame(4)); }, "stale IMU sample accepted");
    require(explorer.diagnostics() == pending && explorer.result().at("unique_sample_count") == 1,
            "stale observation changed experiment state");
}

void test_physical_boundary_validation() {
    droid::MotionExplorer explorer;
    explorer.reset("sway_slow");
    (void)explorer.act(frame(0));
    explorer.observe(frame(1));
    const Json valid = frame(1);
    const auto reject = [&](const std::function<void(Json&)>& mutation) {
        Json bad = valid;
        mutation(bad);
        const Json before = explorer.diagnostics();
        require_throws([&] { (void)explorer.act(bad); }, "malformed or privileged observation accepted");
        require(explorer.diagnostics() == before, "invalid observation mutated explorer state");
    };
    reject([](Json& value) { value["body_shape"] = "long-link"; });
    reject([](Json& value) { value["reward"] = 1.0; });
    reject([](Json& value) { value["world_pose"] = Json::array({0, 0, 0}); });
    reject([](Json& value) { value["schema_version"] = "privileged_state_v0"; });
    reject([](Json& value) { value["reward_sensors"][0]["observations"]["world_orientation"] = Json::array({1, 0, 0, 0}); });
    reject([](Json& value) { value["reward_sensors"][0]["reward_rate"] = 1.0; });
    reject([](Json& value) { value["actuator_feedback"][0]["feedback"]["optimal_pattern"] = "paired_taps"; });
    reject([](Json& value) { value["reward_sensors"].clear(); });
    reject([](Json& value) { value["actuator_feedback"].clear(); });
    reject([](Json& value) { value["reward_sensors"].push_back(value["reward_sensors"][0]); });
    reject([](Json& value) { value["reward_sensors"][0]["valid"] = false; });
    reject([](Json& value) { value["actuator_feedback"][0]["valid"] = false; });
    reject([](Json& value) { value["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"overspeed"}); });
    reject([](Json& value) { value["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"voltage_limited"}); });
    reject([](Json& value) { value["actuator_feedback"][0]["feedback"]["velocity_rad_s"] = std::numeric_limits<double>::quiet_NaN(); });
    reject([](Json& value) { value["reward_sensors"][0]["observations"]["specific_force_m_s2"][1] = std::numeric_limits<double>::infinity(); });
    reject([](Json& value) { value["reward_sensors"][0]["observations"]["angular_velocity_rad_s"] = Json::array({1, 2}); });
    reject([](Json& value) { value["reward_sensors"][0]["sequence"] = -1; });
    reject([](Json& value) { value["reward_sensors"][0]["sample_time_s"] = 100.0; });
    reject([](Json& value) { value["reward_sensors"][0]["age_s"] = 0.1; });
    reject([](Json& value) { value["reward_sensors"][0]["sample_period_s"] = 0.02; });
    reject([](Json& value) { value["reward_sensors"][0]["latency_s"] = 0.0; });
    reject([](Json& value) { value["reward_sensors"][0]["module_id"] = "new-imu-without-reset"; });
    reject([](Json& value) { value["actuator_feedback"][0]["module_id"] = "new-motor-without-reset"; });
    reject([](Json& value) { value["reward_sensors"][0]["sequence"] = std::uint64_t{3}; });

    explorer.reset("sway_slow");
    (void)explorer.act(frame(0));
    Json invalid_first = frame(0);
    require_throws([&] { explorer.observe(invalid_first); }, "missing first IMU delivery accepted after latency");
}

void test_governor_and_sensor_descriptors() {
    droid::MotionExplorer explorer;
    explorer.reset("paired_taps");
    for (std::size_t step = 0; step < 5; ++step) {
        (void)explorer.act(frame(step));
        explorer.observe(frame(step + 1));
    }
    close(explorer.act(frame(5, 3.0)), 0.0, "governor drove further into the shaft target");
    require(explorer.diagnostics().at("speed_limited") == true, "governor intervention missing");
    explorer.observe(frame(6, 3.0));
    close(explorer.act(frame(6, -3.0)), 0.1, "counter-motion command did not preserve slew");
    explorer.observe(frame(7, -3.0));
    close(explorer.act(frame(7, -3.0)), 0.2, "direction-sensitive governor unnecessarily blocked counter-motion");
    explorer.observe(frame(8, -3.0));
    close(explorer.act(frame(8, 10.0)), 0.1, "governor changed the declared slew policy");
    explorer.observe(frame(9, 10.0));
    close(explorer.act(frame(9, 10.0)), 0.0, "governor failed to coast at high shaft velocity");
    explorer.observe(frame(10, 10.0));

    explorer.reset("paired_taps");
    (void)explorer.act(frame(0));
    Json first = frame(1);
    first["reward_sensors"][0]["observations"]["specific_force_m_s2"] = Json::array({0.0, 0.0, 8.0});
    first["reward_sensors"][0]["observations"]["angular_velocity_rad_s"] = Json::array({0.0, 0.0, 3.0});
    explorer.observe(first);
    (void)explorer.act(first);
    Json second = frame(2);
    second["reward_sensors"][0]["observations"]["specific_force_m_s2"] = Json::array({0.0, 0.0, 12.0});
    second["reward_sensors"][0]["observations"]["angular_velocity_rad_s"] = Json::array({0.0, 0.0, 4.0});
    explorer.observe(second);
    const Json result = explorer.result();
    close(result.at("gyro_rms_rad_s"), std::sqrt(12.5), "gyro RMS estimator wrong");
    close(result.at("gyro_peak_rad_s"), 4.0, "gyro peak estimator wrong");
    close(result.at("specific_force_rms_m_s2"), std::sqrt(104.0), "force RMS estimator wrong");
    close(result.at("specific_force_peak_m_s2"), 12.0, "force peak estimator wrong");
    close(result.at("specific_force_variation_m_s2"), 2.0, "force variation estimator wrong");
    require(result.at("unique_sample_count") == 2, "sensor estimator count wrong");
    require(droid::motion_explorer_spec().at("trained_parameters") == false, "fixed-pattern provenance missing");
}
}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"equal horizons, replay and opaque identity", test_equal_horizons_replay_and_identity},
        {"lifecycle and atomic failure", test_lifecycle_and_atomic_failure},
        {"held samples and staleness", test_held_sample_deduplication_and_staleness},
        {"physical observation boundary", test_physical_boundary_validation},
        {"shaft governor and physical descriptors", test_governor_and_sensor_descriptors}};
    for (const auto& [name, test] : tests) {
        try { test(); }
        catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
        std::cout << "PASS " << name << '\n';
    }
    std::cout << tests.size() << " native motion explorer tests passed\n";
    return 0;
}
