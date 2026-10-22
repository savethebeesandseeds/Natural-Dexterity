#include "droid/construction.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;
void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void near(double actual, double expected, double tolerance, std::string_view message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}
double request(int step) {
    if (step < 5) return 0.0;
    return .025 * std::sin(.13 * (step - 5)) + (step < 55 ? .01 : -.008);
}
Json legacy_assembly(int fixture) {
    Json assembly = fixture < 2 ? droid::ConstructionWorld::default_assembly() :
                                 droid::ConstructionWorld::light_assembly();
    if (fixture == 1) {
        assembly["segments"] = {2, 5};
        assembly["sensor"] = Json{{"segment", 0}, {"slot", 1}, {"side", -1}};
        assembly["blocks"] = Json::array({Json{{"id", "compatibility-weight"},
            {"segment", 1}, {"slot", 4}, {"side", 1}}});
    }
    if (fixture == 3) {
        assembly["segments"] = {4, 3};
        assembly["blocks"] = Json::array({Json{{"id", "compatibility-weight"},
            {"segment", 0}, {"slot", 2}, {"side", 1}}});
    }
    return assembly;
}
Json legacy_fingerprints() {
    Json hashes = Json::array();
    for (int fixture = 0; fixture < 4; ++fixture) {
        droid::ConstructionWorld world;
        world.rebuild(legacy_assembly(fixture));
        Json trace{{"initial_state", world.state()}, {"initial_observation", world.observation()},
                   {"steps", Json::array()}};
        for (int step = 0; step < 120; ++step) {
            if (fixture >= 2 && step == 60)
                world.set_sun(Json{{"position_m", {-.30, 0.0, .20}}, {"intensity_lux", 1000.0}});
            const Json transition = world.step(request(step));
            require(transition.at("terminated") == false, "legacy compatibility probe reached safety stop");
            trace["steps"].push_back(Json{{"request", request(step)}, {"transition", transition},
                                         {"state", world.state()}});
        }
        world.reset();
        trace["reset_state"] = world.state();
        trace["reset_observation"] = world.observation();
        hashes.push_back(droid::sha256_hex(trace.dump()));
    }
    return hashes;
}
void test_frozen_legacy_bytes() {
    // Captured before changing construction.cpp, with GCC12.2/MuJoCo3.12 and
    // the existing catalog. Each digest covers initial observation/state, all
    // 120 complete transitions AND physical states, and the final reset.
    const Json expected = Json::array({
        "f1318ef101d022972858e76e8b1ec6ef85b81e5890adafefd54dd02a748b8a63",
        "44195e3f37d5377a0d28029539d6fb4b97f0595adbee3fb2804526e8e29f1180",
        "b6d1785881538bb5c1eedf1f6d465fbb72ef2e115b02b4753b59eaef71937440",
        "cbedce1372bffa5c7dc095cc29bd40501ba9521b8bcec3e26946d55c2acbd358"});
    require(legacy_fingerprints() == expected, "omitted powered_hinge changed frozen legacy state or transition bytes");
}
void test_explicit_zero_equivalence() {
    require(!droid::ConstructionWorld::default_assembly().contains("powered_hinge") &&
            !droid::ConstructionWorld::light_assembly().contains("powered_hinge"),
            "legacy default assembly bytes gained an optional field");
    for (int fixture = 0; fixture < 4; ++fixture) {
        droid::ConstructionWorld legacy, explicit_zero;
        Json assembly = legacy_assembly(fixture);
        legacy.rebuild(assembly);
        assembly["powered_hinge"] = 0;
        explicit_zero.rebuild(assembly);
        require(explicit_zero.assembly() == assembly, "explicit base selection was not retained in assembly metadata");
        require(legacy.observation().dump() == explicit_zero.observation().dump(),
                "explicit base selection changed initial observation bytes");
        for (int step = 0; step < 80; ++step) {
            require(legacy.step(request(step)).dump() == explicit_zero.step(request(step)).dump(),
                    "explicit base selection changed transition bytes");
            Json explicit_state = explicit_zero.state();
            explicit_state["assembly"].erase("powered_hinge");
            require(explicit_state.dump() == legacy.state().dump(),
                    "explicit base selection changed physical state beyond its assembly field");
        }
    }
}
double angle_of(const Json& segment) {
    const auto& start = segment.at("start_m");
    const auto& end = segment.at("end_m");
    return std::atan2(-(end.at(0).get<double>() - start.at(0).get<double>()),
                      -(end.at(2).get<double>() - start.at(2).get<double>()));
}
void angle_near(double actual, double expected, std::string_view message) {
    near(std::remainder(actual - expected, 2 * std::acos(-1.0)), 0.0, 2e-7, message);
}
bool forbidden(const Json& value) {
    if (value.is_object()) for (const auto& [key, child] : value.items()) {
        if (key == "powered_hinge" || key == "assembly" || key == "joints" ||
            key == "geometry" || key == "passive_angle_rad" || key == "passive_velocity_rad_s" ||
            key == "quaternion" || key == "sun" || forbidden(child)) return true;
    }
    if (value.is_array()) for (const auto& child : value) if (forbidden(child)) return true;
    return false;
}
void test_elbow_binding_and_coupling() {
    for (const bool with_light : {false, true}) {
        Json assembly = with_light ? droid::ConstructionWorld::light_assembly() :
                                    droid::ConstructionWorld::default_assembly();
        // A proximal IMU observes the now-passive base rather than sharing the
        // powered elbow coordinate, making an incorrect encoder binding visible.
        assembly["sensor"] = Json{{"segment", 0}, {"slot", 1}, {"side", -1}};
        droid::ConstructionWorld base, elbow, quiet;
        base.rebuild(assembly);
        assembly["powered_hinge"] = 1;
        elbow.rebuild(assembly);quiet.rebuild(assembly);
        const Json initial = elbow.state();
        require(initial.at("joints").at("motor_m") == initial.at("segments").at(1).at("start_m"),
                "elbow motor marker is not at the elbow joint");
        require(initial.at("joints").at("passive_m") == Json::array({0.0, 0.0, 0.0}),
                "elbow layout did not leave the anchored hinge passive");
        require(initial.at("diagnostics").at("total_mass_kg") == base.state().at("diagnostics").at("total_mass_kg") &&
                initial.at("diagnostics").at("segment_inertias_kg_m2") == base.state().at("diagnostics").at("segment_inertias_kg_m2"),
                "moving the drive changed passive geometry or mass instead of joint actuation");
        double max_motor_passive_difference = 0, max_driven_quiet_difference = 0, max_layout_difference = 0;
        for (int step = 0; step < 120; ++step) {
            const Json transition = elbow.step(request(step));
            const Json base_transition = base.step(request(step));
            (void)quiet.step(0.0);
            require(transition.at("terminated") == false && base_transition.at("terminated") == false,
                    "small actuation comparison unexpectedly reached safety stop");
            const Json state = elbow.state(), base_state = base.state(), quiet_state = quiet.state();
            const Json& feedback = transition.at("observation").at("actuator_feedback").at(0);
            require(feedback.at("module_id") == "motor-0001" && feedback.at("sku_id") == "rotary_dc_gearmotor_v0",
                    "moving the drive changed its local motor identity");
            require(!forbidden(transition.at("observation")), "actuation layout leaked into policy observations");
            require(transition.at("observation").at("actuator_feedback").size() == 1 &&
                    transition.at("observation").at("reward_sensors").size() == (with_light ? 2 : 1),
                    "actuation swap changed the actuator or reward sensor interface");
            const double base_angle = angle_of(state.at("segments").at(0));
            const double elbow_angle = angle_of(state.at("segments").at(1)) - base_angle;
            angle_near(feedback.at("feedback").at("position_rad"), elbow_angle,
                       "motor feedback is not the actual relative elbow coordinate");
            angle_near(state.at("diagnostics").at("passive_angle_rad"), base_angle,
                       "passive coordinate did not move to the base hinge");
            angle_near(state.at("diagnostics").at("motor_angle_rad"), elbow_angle,
                       "public motor diagnostic does not address the elbow hinge");
            near(feedback.at("feedback").at("velocity_rad_s"),
                 state.at("diagnostics").at("motor_velocity_rad_s"), 1e-9,
                 "local motor velocity and selected joint disagree");
            near(state.at("diagnostics").at("instantaneous_imu").at("angular_velocity_rad_s").at(1),
                 state.at("diagnostics").at("passive_velocity_rad_s"), 1e-9,
                 "proximal IMU should measure the unpowered base's angular velocity");
            near(state.at("diagnostics").at("passive_actuator_force_nm"), 0, 0,
                 "the unpowered base received hidden generalized actuator force");
            require(state.at("joints").at("motor_m") == state.at("segments").at(1).at("start_m"),
                    "moving elbow motor marker failed to follow physical geometry");
            max_motor_passive_difference = std::max(max_motor_passive_difference,
                std::abs(feedback.at("feedback").at("position_rad").get<double>() -
                         state.at("diagnostics").at("passive_angle_rad").get<double>()));
            max_driven_quiet_difference = std::max(max_driven_quiet_difference,
                std::abs(state.at("diagnostics").at("passive_angle_rad").get<double>() -
                         quiet_state.at("diagnostics").at("passive_angle_rad").get<double>()));
            max_layout_difference = std::max(max_layout_difference,
                std::abs(state.at("tip_position_m").at(0).get<double>() - base_state.at("tip_position_m").at(0).get<double>()));
        }
        require(max_motor_passive_difference > .01, "binding test did not separate elbow and base encoder coordinates");
        require(max_driven_quiet_difference > .005, "elbow drive did not move the passive base through coupling");
        require(max_layout_difference > .003, "the same command did not distinguish base and elbow actuation dynamics");
    }
}
void test_transactional_layout_validation() {
    for (const bool with_light : {false, true}) {
        droid::ConstructionWorld world;
        Json assembly = with_light ? droid::ConstructionWorld::light_assembly() :
                                    droid::ConstructionWorld::default_assembly();
        assembly["powered_hinge"] = 1;world.rebuild(assembly);(void)world.step(.015);
        const std::string before = world.state().dump(), observation = world.observation().dump();
        for (const Json& value : Json::array({-1, 2, 0.0, 1.0, true, false, nullptr, "1", Json::array({1}), Json::object()})) {
            Json bad = assembly;bad["powered_hinge"] = value;bool rejected = false;
            try { world.rebuild(bad); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid powered_hinge value accepted");
            require(world.state().dump() == before && world.observation().dump() == observation,
                    "invalid layout changed mechanics, queued sensors, current or reward");
        }
        Json unknown = assembly;unknown["powered_joint"] = 0;bool rejected = false;
        try { world.rebuild(unknown); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && world.state().dump() == before, "optional field weakened unknown-field validation");
        for (const double effort : {1.01, -1.01, std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity()}) {
            rejected = false;try { (void)world.step(effort); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && world.state().dump() == before && world.observation().dump() == observation,
                    "swapped layout bypassed finite command/safety boundary");
        }
    }
}
void test_elbow_sensor_clock_reward_and_replay() {
    droid::ConstructionWorld world;
    Json assembly = droid::ConstructionWorld::light_assembly();assembly["powered_hinge"] = 1;
    world.rebuild(assembly);
    const Json initial = world.state(), initial_observation = world.observation();
    std::vector<std::string> transitions, states;
    for (int step = 0; step < 120; ++step) {
        const auto transition = world.step(request(step));
        const auto state = world.state();
        const auto& sensors = transition.at("observation").at("reward_sensors");
        const auto imu = std::find_if(sensors.begin(), sensors.end(), [](const Json& s) {return s.at("family_id") == "imu_6axis_v0";});
        const auto light = std::find_if(sensors.begin(), sensors.end(), [](const Json& s) {return s.at("family_id") == "ambient_light_v0";});
        require(imu != sensors.end() && light != sensors.end() && imu->at("valid") == true && light->at("valid") == true,
                "elbow layout lost delayed local reward sensors");
        near(imu->at("age_s"), .01, 1e-9, "IMU acquisition/delivery clock changed with actuation layout");
        require(imu->at("sequence") == 2 * (step + 1), "IMU sequence changed with actuation layout");
        if (step % 5 == 0) {
            near(light->at("age_s"), .02, 1e-9, "light acquisition/delivery clock changed with actuation layout");
            require(light->at("sequence") == step / 5 + 1, "light sequence changed with actuation layout");
        }
        near(transition.at("reward"), transition.at("reward_components").at(0).at("transition_reward").get<double>() +
             transition.at("reward_components").at(1).at("transition_reward").get<double>(), 1e-12,
             "elbow layout changed the integrated sensor reward sum");
        require(transition.at("safety").at("veto") == false && transition.at("safety").at("flags").empty(),
                "bounded elbow replay acquired a safety flag");
        transitions.push_back(transition.dump());states.push_back(state.dump());
    }
    world.reset();require(world.state().dump() == initial.dump() && world.observation().dump() == initial_observation.dump(),
                         "elbow reset retained physical or sensor/motor memory");
    for (int step = 0; step < 120; ++step) {
        require(world.step(request(step)).dump() == transitions[step], "elbow transition replay is not byte-exact");
        require(world.state().dump() == states[step], "elbow physical state replay is not byte-exact");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--legacy-fingerprints") {
        std::cout << legacy_fingerprints().dump(2) << '\n';
        return 0;
    }
    const std::vector<std::pair<std::string_view, std::function<void()>>> tests{
        {"frozen legacy bytes", test_frozen_legacy_bytes},
        {"explicit base selection equivalence", test_explicit_zero_equivalence},
        {"elbow motor binding and passive coupling", test_elbow_binding_and_coupling},
        {"transactional layout and command validation", test_transactional_layout_validation},
        {"elbow sensor clocks, reward and exact replay", test_elbow_sensor_clock_reward_and_replay}};
    for (const auto& [name, test] : tests) {
        try {test();std::cout << "PASS " << name << '\n';}
        catch (const std::exception& error) {std::cerr << "FAIL " << name << ": " << error.what() << '\n';return 1;}
    }
    std::cout << tests.size() << " construction actuation test groups passed\n";
    return 0;
}
