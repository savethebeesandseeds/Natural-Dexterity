#include "droid/construction.hpp"
#include "droid/module_catalog.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}
double vector_distance(const Json& first, const Json& second) {
    double squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double difference = first.at(i).get<double>() - second.at(i).get<double>();
        squared += difference * difference;
    }
    return std::sqrt(squared);
}
const Json& imu(const Json& observation) {
    return observation.at("reward_sensors").at(0);
}
bool contains_key(const Json& value, const std::set<std::string>& forbidden) {
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            if (forbidden.contains(key) || contains_key(child, forbidden)) return true;
        }
    } else if (value.is_array()) {
        for (const Json& child : value) if (contains_key(child, forbidden)) return true;
    }
    return false;
}
Json block(std::string id, int segment, int slot, int side) {
    return Json{{"id", std::move(id)}, {"segment", segment}, {"slot", slot}, {"side", side}};
}

void test_generated_physics_and_contract() {
    droid::ConstructionWorld world;
    const Json state = world.state();
    require(world.assembly() == droid::ConstructionWorld::default_assembly(),
            "default assembly must be explicit and reproducible");
    require(state.at("geometry").size() == 7, "six real beam cells and one sensor must exist");
    near(state.at("diagnostics").at("total_mass_kg"), 0.11, 1e-9,
         "cell and sensor physical mass must be realized");
    near(state.at("tip_position_m").at(2), -0.24, 1e-9, "two three-stud links must hang down");
    require(state.at("diagnostics").at("actuator_count") == 1 &&
            state.at("diagnostics").at("degrees_of_freedom") == 2,
            "base motor and passive hinge must be separate physical coordinates");
    const Json observation = world.observation();
    require(observation.at("schema_version") == "construction_observation_v1",
            "construction observation must have its own version");
    require(observation.at("actuator_feedback").size() == 1 &&
            observation.at("actuator_feedback").at(0).at("module_id") == "motor-0001",
            "controller may observe only the powered motor's local feedback");
    require(!contains_key(observation, {"assembly", "tip_position_m", "quaternion",
        "orientation", "passive_angle_rad", "passive_velocity_rad_s", "mass", "reward", "reward_rate"}),
        "policy observation must not disclose structure, world pose, passive encoder or reward");
    require(!imu(observation).at("valid").get<bool>() &&
            imu(observation).at("sequence") == 0 &&
            imu(observation).at("sample_time_s").is_null(),
            "IMU must start invalid until first physical delivery");
    for (const Json& geometry : state.at("geometry")) {
        require(geometry.at("position_m").size() == 3 && geometry.at("size_m").size() == 3 &&
                geometry.at("quaternion").size() == 4,
                "renderer must receive actual geometry transforms and dimensions");
    }
}

void test_passive_coupling() {
    droid::ConstructionWorld powered;
    droid::ConstructionWorld quiet;
    double maximum_difference = 0.0;
    double maximum_passive_speed = 0.0;
    for (int step = 0; step < 100; ++step) {
        const double effort = 0.04 * std::sin(2.0 * 3.14159265358979323846 * 0.5 * step * 0.02);
        const Json transition = powered.step(effort);
        (void)quiet.step(0.0);
        require(!transition.at("terminated").get<bool>(), "small coupled-motion probe must remain within safety limits");
        const Json actual = powered.state();
        const Json reference = quiet.state();
        maximum_difference = std::max(maximum_difference,
            vector_distance(actual.at("tip_position_m"), reference.at("tip_position_m")));
        maximum_passive_speed = std::max(maximum_passive_speed,
            std::abs(actual.at("diagnostics").at("passive_velocity_rad_s").get<double>()));
        near(actual.at("diagnostics").at("passive_actuator_force_nm"), 0.0, 1e-12,
             "passive hinge must never receive hidden actuator force");
    }
    require(maximum_difference > 0.003, "base motor must cause a measurable coupled change at the tip");
    require(maximum_passive_speed > 0.05, "truly passive hinge must move through physical coupling");
}

void test_mass_and_mount_changes() {
    droid::ConstructionWorld near_load;
    droid::ConstructionWorld far_load;
    Json first = droid::ConstructionWorld::default_assembly();
    first["segments"] = Json::array({3, 5});
    first["blocks"] = Json::array({block("weight", 1, 0, 1)});
    Json second = first;
    second["blocks"].at(0)["slot"] = 4;
    near_load.rebuild(first);
    far_load.rebuild(second);
    const Json near_state = near_load.state();
    const Json far_state = far_load.state();
    near(near_state.at("diagnostics").at("total_mass_kg"), 0.152, 1e-9,
         "longer real beam and attached block must add expected mass");
    near(far_state.at("diagnostics").at("total_mass_kg"), 0.152, 1e-9,
         "moving the same block must preserve physical mass");
    require(near_state.at("diagnostics").at("segment_inertias_kg_m2") !=
            far_state.at("diagnostics").at("segment_inertias_kg_m2"),
            "moving a real block must change segment inertia");
    double maximum_difference = 0.0;
    for (int step = 0; step < 80; ++step) {
        const double effort = step < 20 ? 0.04 : -0.015;
        (void)near_load.step(effort);
        (void)far_load.step(effort);
        maximum_difference = std::max(maximum_difference,
            vector_distance(near_load.state().at("tip_position_m"), far_load.state().at("tip_position_m")));
    }
    require(maximum_difference > 1e-4, "same command through differently placed mass must alter actual motion");

    Json sensor_on_other_segment = first;
    sensor_on_other_segment["sensor"] = Json{{"segment", 0}, {"slot", 0}, {"side", -1}};
    near_load.rebuild(sensor_on_other_segment);
    require(near_load.state().at("sensor").at("position_m") !=
            near_state.at("sensor").at("position_m"), "sensor mounting must move its actual MuJoCo site");
    require(near_load.state().at("diagnostics").at("segment_masses_kg") !=
            near_state.at("diagnostics").at("segment_masses_kg"), "sensor mass must move with its mount");
}

void test_invalid_rebuild_is_transactional() {
    droid::ConstructionWorld world;
    (void)world.step(0.01);
    const Json original = world.assembly();
    const Json before = world.state();
    const Json observation = world.observation();
    std::vector<Json> invalid;
    auto candidate = original; candidate["schema"] = "other"; invalid.push_back(candidate);
    candidate = original; candidate["extra"] = true; invalid.push_back(candidate);
    candidate = original; candidate.erase("blocks"); invalid.push_back(candidate);
    candidate = original; candidate["segments"] = Json::array({3, 7}); invalid.push_back(candidate);
    candidate = original; candidate["segments"] = Json::array({3, 2.5}); invalid.push_back(candidate);
    candidate = original; candidate["sensor"]["side"] = 0; invalid.push_back(candidate);
    candidate = original; candidate["sensor"]["slot"] = 3; invalid.push_back(candidate);
    candidate = original; candidate["sensor"]["segment"] = 2; invalid.push_back(candidate);
    candidate = original; candidate["name"] = std::string(61, 'x'); invalid.push_back(candidate);
    candidate = original; candidate["blocks"] = Json::array({block("overlap", 1, 2, -1)}); invalid.push_back(candidate);
    candidate = original; candidate["blocks"] = Json::array({block("same", 0, 0, 1), block("same", 0, 1, 1)}); invalid.push_back(candidate);
    for (const std::string reserved : {"imu-0001", "motor-0001", "segment-0-cell-0"}) {
        candidate = original;
        candidate["blocks"] = Json::array({block(reserved, 0, 0, 1)});
        invalid.push_back(candidate);
    }
    candidate = original; candidate["blocks"] = Json::array({block("a", 0, 0, 1), block("b", 0, 0, 1)}); invalid.push_back(candidate);
    candidate = original; candidate["blocks"] = Json::array({block("a", 0, 0, 1)});
    candidate["blocks"].at(0)["side"] = true; invalid.push_back(candidate);
    candidate = original; candidate["blocks"] = Json::array();
    for (int i = 0; i < 13; ++i) candidate["blocks"].push_back(block(std::to_string(i), 0, 0, 1));
    invalid.push_back(candidate);
    for (const Json& assembly : invalid) {
        bool rejected = false;
        try { world.rebuild(assembly); } catch (const std::exception&) { rejected = true; }
        require(rejected, "malformed, overlapping or unsupported assembly must be rejected");
        require(world.assembly() == original && world.state() == before && world.observation() == observation,
                "rejected rebuild must preserve active mechanics, time, sensor queue and motor state");
    }
    for (const double effort : {std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::infinity(), 1.001, -1.001,
         std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()}) {
        bool rejected = false;
        try { (void)world.step(effort); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && world.state() == before && world.observation() == observation,
                "non-finite or out-of-range command must fail without advancing or changing state");
    }
}

void test_causal_sensors_reward_and_replay() {
    droid::ConstructionWorld world;
    const Json initial_state = world.state();
    const Json initial_observation = world.observation();
    std::vector<Json> transitions;
    std::vector<Json> states;
    bool delayed_differs_from_instantaneous = false;
    for (int step = 1; step <= 120; ++step) {
        const double effort = 0.025 * std::sin(step * 0.08);
        const Json transition = world.step(effort);
        const Json state = world.state();
        const Json& sample = imu(transition.at("observation"));
        require(sample.at("valid").get<bool>() && sample.at("sequence") == step * 2,
                "each control endpoint must expose the latest delivered, never future IMU sample");
        near(sample.at("sample_time_s"), step * 0.02 - 0.01, 1e-9, "sample clock must be 100Hz");
        near(sample.at("delivered_time_s"), step * 0.02 - 0.004, 1e-9, "IMU latency must round up to6ms");
        near(sample.at("age_s"), 0.01, 1e-9, "held sample age must reflect actual acquisition time");
        near(state.at("elapsed_s"), step * 0.02, 1e-9, "step must advance exactly20ms");
        const double expected_rate = droid::sensor_reward_rate("imu_6axis_v0", sample.at("observations"));
        near(state.at("reward").at("rate"), expected_rate, 1e-12,
             "mounted IMU must retain catalog reward semantics");
        near(transition.at("reward_components").at(0).at("transition_reward"),
             transition.at("reward"), 1e-12, "exactly the mounted IMU must vote");
        const Json& instantaneous = state.at("diagnostics").at("instantaneous_imu");
        delayed_differs_from_instantaneous = delayed_differs_from_instantaneous ||
            sample.at("observations") != instantaneous;
        transitions.push_back(transition);
        states.push_back(state);
    }
    require(delayed_differs_from_instantaneous,
            "moving fixture must expose delayed sensing rather than fresh instantaneous vectors");
    world.reset();
    require(world.state() == initial_state && world.observation() == initial_observation,
            "reset must clear physical state, current, thermal state, memory and sensor queue");
    for (int step = 1; step <= 120; ++step) {
        require(world.step(0.025 * std::sin(step * 0.08)) == transitions.at(step - 1),
                "identical assembly and effort tape must replay exact observations/rewards/safety");
        require(world.state() == states.at(step - 1), "replay must reproduce actual passive motion and geometry");
    }
}
}  // namespace

int main() {
    try {
        test_generated_physics_and_contract();
        test_passive_coupling();
        test_mass_and_mount_changes();
        test_invalid_rebuild_is_transactional();
        test_causal_sensors_reward_and_replay();
        std::cout << "PASS construction generation, passive coupling, physical rebuilding, temporal IMU, reward and replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL construction: " << error.what() << '\n';
        return 1;
    }
}
