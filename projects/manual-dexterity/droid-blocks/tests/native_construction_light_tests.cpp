#include "droid/construction.hpp"
#include "droid/module_catalog.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        message + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}
const Json& sample(const Json& observation, const std::string& id) {
    for (const Json& value : observation.at("reward_sensors")) {
        if (value.at("module_id") == id) return value;
    }
    throw std::runtime_error("missing sensor " + id);
}
const Json& light(const Json& transition) {
    return sample(transition.at("observation"), "light-0001");
}
double lux(const Json& transition) {
    return light(transition).at("observations").at("illuminance_lux");
}
bool forbidden_key(const Json& value) {
    static const std::set<std::string> forbidden{
        "sun", "assembly", "position_m", "direction_m", "quaternion", "geometry",
        "passive_angle_rad", "passive_velocity_rad_s", "reward", "reward_rate"};
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            if (forbidden.contains(key) || forbidden_key(child)) return true;
        }
    } else if (value.is_array()) {
        for (const Json& child : value) if (forbidden_key(child)) return true;
    }
    return false;
}
Json sun(double x, double z, double intensity = 1000.0) {
    return Json{{"position_m", Json::array({x, 0.0, z})}, {"intensity_lux", intensity}};
}

void test_mechanics_and_orientation() {
    droid::ConstructionWorld upward;
    droid::ConstructionWorld downward;
    Json assembly = droid::ConstructionWorld::light_assembly();
    upward.rebuild(assembly);
    assembly["light"]["face"] = -1;
    downward.rebuild(assembly);
    const Json initial = upward.state();
    require(initial.at("schema") == "construction_state_v2" &&
            upward.observation().at("schema_version") == "construction_observation_v2",
            "light kit must use explicit successor state and observation versions");
    require(initial.at("geometry").size() == 8,
            "mounted light must add a real physical box to the six cells and IMU");
    near(initial.at("diagnostics").at("total_mass_kg"), 0.13, 1e-9,
         "light cube must contribute twenty grams");
    near(initial.at("light_sensor").at("position_m").at(0), 0.038, 1e-9,
         "light must be on its authored side socket");
    near(initial.at("light_sensor").at("position_m").at(2), -0.202, 1e-9,
         "optical site must lie on the upward cube face");
    require(initial.at("light_sensor").at("direction_m") == Json::array({0.0, 0.0, 1.0}),
            "positive face must point along actual mounted local +Z");
    require(downward.state().at("light_sensor").at("direction_m") == Json::array({0.0, 0.0, -1.0}),
            "negative face must reverse the physical optical normal");
    const double dx = 0.30 - 0.038;
    const double dz = 0.20 + 0.202;
    const double distance = std::hypot(dx, dz);
    const double expected = 1000.0 * (dz / distance) / (1.0 + std::pow(distance / 0.5, 2));
    const Json up_transition = upward.step(0.0);
    const Json down_transition = downward.step(0.0);
    near(lux(up_transition), expected, 1e-6,
         "mounted hemisphere and declared distance falloff must determine illuminance");
    near(lux(down_transition), 0.0, 1e-12, "back-facing sensor must not see sun through its back hemisphere");
    require(!forbidden_key(upward.observation()),
            "light observation must not expose world pose, source or reward");
    for (int step = 0; step < 40; ++step) {
        const double effort = 0.03 * std::sin(step * 0.1);
        (void)upward.step(effort);
        (void)downward.step(effort);
        const Json up_state = upward.state();
        const Json down_state = downward.state();
        require(up_state.at("geometry") == down_state.at("geometry") &&
                up_state.at("tip_position_m") == down_state.at("tip_position_m"),
                "flipping only an optical face must preserve mass and physical trajectory");
    }
    assembly = droid::ConstructionWorld::light_assembly();
    assembly["light"] = Json{{"segment", 0}, {"slot", 0}, {"side", 1}, {"face", 1}};
    upward.rebuild(assembly);
    const Json moved = upward.state();
    near(moved.at("diagnostics").at("total_mass_kg"), 0.13, 1e-9,
         "moving the light must preserve its mass");
    require(moved.at("diagnostics").at("segment_masses_kg") !=
            initial.at("diagnostics").at("segment_masses_kg"),
            "moving the light must relocate actual body mass");
    require(moved.at("light_sensor").at("position_m") != initial.at("light_sensor").at("position_m"),
            "moving its socket must move the actual mounted optical site");
}

void test_clock_move_and_reset() {
    droid::ConstructionWorld world;
    world.rebuild(droid::ConstructionWorld::light_assembly());
    const Json before = world.state();
    const Json observation = world.observation();
    const double acquired_lux = before.at("diagnostics").at("instantaneous_light_lux");
    require(!before.at("light_sensor").at("valid").get<bool>() &&
            before.at("light_sensor").at("sequence") == 0 &&
            before.at("light_sensor").at("sample_time_s").is_null(),
            "reset light must wait for its acquisition delivery");
    world.set_sun(sun(0.3, 0.2, 0.0));
    require(world.observation() == observation && world.state().at("elapsed_s") == 0.0,
            "moving sun must not alter already acquired observations or reset/advance time");
    Json transition;
    for (int step = 1; step <= 5; ++step) {
        transition = world.step(0.0);
        require(light(transition).at("sequence") == 1, "first reading must be held for its full sample interval");
        near(lux(transition), acquired_lux, 1e-6,
             "sample acquired before sun changed must retain its original illuminance");
        near(light(transition).at("sample_time_s"), 0.0, 1e-12, "first acquisition timestamp");
        near(light(transition).at("delivered_time_s"), 0.02, 1e-12, "catalog twenty-millisecond latency");
    }
    const Json at_move = world.state();
    const Json at_move_observation = world.observation();
    world.set_sun(sun(0.3, 0.2));
    require(world.state().at("elapsed_s") == at_move.at("elapsed_s") &&
            world.state().at("tip_position_m") == at_move.at("tip_position_m") &&
            world.observation() == at_move_observation,
            "running sun update must preserve mechanics and the delivered sample");
    for (int step = 6; step <= 10; ++step) {
        transition = world.step(0.0);
        require(light(transition).at("sequence") == 2, "dark sample must remain held until next delivery");
        near(lux(transition), 0.0, 1e-12,
             "dark reading already queued at0.10s must still arrive after sun changes again");
    }
    transition = world.step(0.0);
    require(light(transition).at("sequence") == 3, "next acquisition must observe the new sun");
    near(lux(transition), acquired_lux, 1e-6, "new sun must become visible through a later actual sample");
    near(world.state().at("elapsed_s"), 0.22, 1e-12, "both sun changes must preserve continuous time");
    near(light(transition).at("sample_time_s"), 0.2, 1e-12, "new sun acquisition time");
    near(light(transition).at("age_s"), 0.02, 1e-12, "delayed sample age");

    const Json selected_sun = sun(-0.25, 0.4, 650.0);
    world.set_sun(selected_sun);
    world.reset();
    require(world.state().at("sun") == selected_sun && world.state().at("elapsed_s") == 0,
            "reset must preserve the deliberately selected sun");
    require(!world.state().at("light_sensor").at("valid").get<bool>(),
            "reset must discard old pending and delivered samples");
    world.rebuild(droid::ConstructionWorld::light_assembly());
    require(world.state().at("sun") == selected_sun, "rebuilding the light body must retain the selected sun");
}

void test_reward_and_replay() {
    droid::ConstructionWorld world;
    world.rebuild(droid::ConstructionWorld::light_assembly());
    std::vector<Json> transitions;
    std::vector<Json> states;
    double cumulative = 0.0;
    for (int step = 0; step < 80; ++step) {
        if (step == 25) world.set_sun(sun(-0.4, 0.3, 800.0));
        const Json transition = world.step(0.02 * std::sin(step * 0.12));
        const Json state = world.state();
        const Json& components = transition.at("reward_components");
        require(components.size() == 2 && components.at(0).at("module_id") == "imu-0001" &&
                components.at(1).at("module_id") == "light-0001",
                "each mounted sensor must have exactly one catalog reward vote");
        const Json& imu_sample = sample(transition.at("observation"), "imu-0001");
        const Json& light_sample = sample(transition.at("observation"), "light-0001");
        const double imu_rate = droid::sensor_reward_rate("imu_6axis_v0", imu_sample.at("observations"));
        const double light_rate = droid::sensor_reward_rate("ambient_light_v0", light_sample.at("observations"));
        near(components.at(0).at("reward_rate"), imu_rate, 1e-12, "unchanged catalog IMU vote");
        near(components.at(1).at("reward_rate"), light_rate, 1e-12, "unchanged catalog ambient-light vote");
        near(state.at("reward").at("rate"), imu_rate + light_rate, 1e-12, "displayed rate must sum sensor votes");
        near(components.at(0).at("transition_reward").get<double>() +
             components.at(1).at("transition_reward").get<double>(), transition.at("reward"),
             1e-12, "integrated sensor components must sum to actual transition reward");
        cumulative += transition.at("reward").get<double>();
        near(state.at("reward").at("cumulative"), cumulative, 1e-12, "cumulative reward must contain both sensors");
        if (step == 0) {
            near(components.at(1).at("transition_reward"), -0.018 + 0.002 * light_rate, 1e-12,
                 "light must use missing reward until its first20ms delivery");
        }
        require(!forbidden_key(transition.at("observation")), "privileged light geometry must stay outside observation");
        transitions.push_back(transition);
        states.push_back(state);
    }
    world.set_sun(sun(0.3, 0.2));
    world.reset();
    for (int step = 0; step < 80; ++step) {
        if (step == 25) world.set_sun(sun(-0.4, 0.3, 800.0));
        require(world.step(0.02 * std::sin(step * 0.12)) == transitions.at(step),
                "effort plus sun-move schedule must replay exact observations, reward and safety");
        require(world.state() == states.at(step), "scheduled lighting replay must reproduce actual geometry");
    }
}

void test_transactional_rejections_and_v1_compatibility() {
    droid::ConstructionWorld world;
    world.rebuild(droid::ConstructionWorld::light_assembly());
    (void)world.step(0.01);
    const Json state = world.state();
    const Json observation = world.observation();
    std::vector<Json> bad_suns;
    Json candidate = sun(0.3, 0.2); candidate["extra"] = true; bad_suns.push_back(candidate);
    candidate = sun(2.01, 0.2); bad_suns.push_back(candidate);
    candidate = sun(0.3, -2.01); bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2); candidate["position_m"].at(1) = 0.001; bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2, -0.1); bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2, 1000.1); bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2); candidate["intensity_lux"] = std::numeric_limits<double>::infinity(); bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2); candidate["position_m"].at(0) = std::numeric_limits<double>::quiet_NaN(); bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2); candidate["intensity_lux"] = true; bad_suns.push_back(candidate);
    candidate = sun(0.3, 0.2); candidate["position_m"] = Json::array({0.3, 0.2}); bad_suns.push_back(candidate);
    for (const Json& bad_sun : bad_suns) {
        bool rejected = false;
        try { world.set_sun(bad_sun); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && world.state() == state && world.observation() == observation,
                "invalid sun updates must preserve the entire world and sample queues");
    }
    std::vector<Json> bad_assemblies;
    candidate = world.assembly(); candidate["light"]["face"] = 0; bad_assemblies.push_back(candidate);
    candidate = world.assembly(); candidate["light"]["side"] = -1; bad_assemblies.push_back(candidate);
    candidate = world.assembly(); candidate["light"]["slot"] = 3; bad_assemblies.push_back(candidate);
    candidate = world.assembly(); candidate["light"]["face"] = 1.0; bad_assemblies.push_back(candidate);
    candidate = world.assembly(); candidate["blocks"] = Json::array({Json{{"id", "weight"}, {"segment", 1}, {"slot", 2}, {"side", 1}}}); bad_assemblies.push_back(candidate);
    candidate = world.assembly(); candidate["blocks"] = Json::array({Json{{"id", "light-0001"}, {"segment", 0}, {"slot", 0}, {"side", 1}}}); bad_assemblies.push_back(candidate);
    for (const Json& bad_assembly : bad_assemblies) {
        bool rejected = false;
        try { world.rebuild(bad_assembly); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && world.state() == state && world.observation() == observation,
                "invalid light sockets, face or ID must reject without replacing active physics");
    }
    droid::ConstructionWorld reference;
    world.rebuild(droid::ConstructionWorld::default_assembly());
    require(world.state() == reference.state() && world.observation() == reference.observation(),
            "rebuilding old kit must restore identical legacy state and schema");
    require(!world.state().contains("sun") && !world.state().contains("light_sensor") &&
            world.observation().at("reward_sensors").size() == 1,
            "v1 must not acquire light or source metadata");
    for (int step = 0; step < 50; ++step) {
        const double effort = 0.02 * std::sin(step * 0.07);
        require(world.step(effort) == reference.step(effort) && world.state() == reference.state(),
                "restored v1 mechanics and rewards must match untouched v1 instance exactly");
    }
    const Json legacy_state = world.state();
    bool rejected = false;
    try { world.set_sun(sun(0.3, 0.2)); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && world.state() == legacy_state, "v1 must reject sun control transactionally");
    Json old_valid = droid::ConstructionWorld::default_assembly();
    old_valid["blocks"] = Json::array({Json{{"id", "light-0001"}, {"segment", 0}, {"slot", 0}, {"side", 1}}});
    world.rebuild(old_valid);
    require(world.assembly() == old_valid, "new reserved light ID must not retroactively reject old v1 assemblies");
}
}  // namespace

int main() {
    try {
        test_mechanics_and_orientation();
        test_clock_move_and_reset();
        test_reward_and_replay();
        test_transactional_rejections_and_v1_compatibility();
        std::cout << "PASS mounted light physics, hemisphere sensing, delayed sun changes, reward, replay and v1 preservation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL construction light: " << error.what() << '\n';
        return 1;
    }
}
