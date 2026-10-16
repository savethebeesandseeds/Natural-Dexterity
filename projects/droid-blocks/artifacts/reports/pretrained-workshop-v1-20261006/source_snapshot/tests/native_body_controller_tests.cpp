#include "droid/body_controller.hpp"
#include "droid/environment.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
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

template <typename Function>
void require_throws(Function&& function, std::string_view message) {
    try { std::invoke(std::forward<Function>(function)); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(std::string(message));
}

// A deliberately tiny causal fixture, not a simulator-quality or development
// world claim. Unknown signed effects map efforts to physical lux changes.
// Readings are captured every .1 s and delivered .04 s later, then held.
struct PhysicalFixture {
    struct Pending { std::size_t capture_step; std::uint64_t sequence; double lux; };
    std::vector<std::string> ids;
    std::vector<double> effects;
    std::size_t step{0};
    std::uint64_t sequence{0};
    double lux{100.0};
    bool saturated{false};
    bool has_delivered{false};
    Pending delivered{};
    std::deque<Pending> pending{};

    [[nodiscard]] Json observation() {
        if (step % 5 == 0) pending.push_back({step, ++sequence, lux});
        while (!pending.empty() && pending.front().capture_step + 2 <= step) {
            delivered = pending.front();
            pending.pop_front();
            has_delivered = true;
        }
        const double time = static_cast<double>(step) * 0.02;
        const double sampled = static_cast<double>(delivered.capture_step) * 0.02;
        Json sensors = Json::array({Json{
            {"module_id", "opaque-eye"}, {"family_id", "ambient_light_v0"},
            {"valid", has_delivered},
            {"observations", Json{{"illuminance_lux", has_delivered ? delivered.lux : 0.0},
                                   {"saturated", saturated}}},
            {"sequence", has_delivered ? delivered.sequence : std::uint64_t{0}},
            {"sample_time_s", has_delivered ? Json(sampled) : Json(nullptr)},
            {"delivered_time_s", has_delivered ? Json(sampled + 0.04) : Json(nullptr)},
            {"age_s", has_delivered ? Json(time - sampled) : Json(nullptr)},
            {"sample_period_s", 0.1}, {"latency_s", 0.04}},
            Json{{"module_id", "opaque-touch"}, {"family_id", "touch_force_v0"},
                 {"valid", true}, {"observations", Json{{"contact", false}, {"normal_force_n", 0.0}}}}});
        Json actuators = Json::array();
        for (const auto& id : ids) {
            actuators.push_back(Json{{"module_id", id}, {"sku_id", "rotary_dc_gearmotor_v0"},
                {"valid", true}, {"feedback", Json{
                    {"position_rad", 0.0}, {"velocity_rad_s", 0.0}, {"current_a", 0.0},
                    {"bus_voltage_v", 7.4}, {"temperature_c", 25.0},
                    {"output_torque_nm", 0.0}, {"load_impedance_nm_s_per_rad", 0.0},
                    {"stuck_score", 0.0}, {"stuck", false}, {"fault_flags", Json::array()}}}});
        }
        return Json{{"reward_sensors", std::move(sensors)}, {"actuator_feedback", std::move(actuators)}};
    }

    void advance(const std::map<std::string, double>& actions) {
        for (std::size_t index = 0; index < ids.size(); ++index) {
            lux += 12.0 * effects.at(index) * actions.at(ids[index]) * 0.02;
        }
        ++step;
    }
};

void run(droid::BodyProbeController& controller, PhysicalFixture& fixture, std::size_t steps) {
    std::map<std::string, double> previous;
    for (const auto& id : fixture.ids) previous[id] = 0.0;
    for (std::size_t count = 0; count < steps; ++count) {
        const auto actions = controller.act(fixture.observation(), fixture.ids);
        for (const auto& [id, effort] : actions) {
            require(std::abs(effort) <= 0.350000000001, "effort limit exceeded");
            if (count > 0) require(std::abs(effort - previous.at(id)) <= 0.100000000001,
                                   "slew limit exceeded");
        }
        fixture.advance(actions);
        previous = actions;
    }
}

void test_paired_identification_and_matched_together() {
    for (const auto& effects : {std::vector<double>{1.0, 1.0},
                                std::vector<double>{1.0, -1.0},
                                std::vector<double>{1.0, 0.0},
                                std::vector<double>{-1.0, 1.0, 0.0, -1.0}}) {
        std::vector<std::string> ids;
        for (std::size_t index = 0; index < effects.size(); ++index) {
            ids.push_back("opaque-" + std::to_string(index));
        }
        PhysicalFixture fixture{ids, effects};
        droid::BodyProbeController controller;
        controller.reset(ids);
        run(controller, fixture, 900);
        const Json diagnostics = controller.diagnostics();
        require(diagnostics.at("phase") == "seeking", "discovery did not finish");
        require(diagnostics.at("evidence_units") == "lx/s", "evidence units missing");
        for (std::size_t index = 0; index < effects.size(); ++index) {
            const Json& motor = diagnostics.at("motors").at(index);
            const int expected = effects[index] > 0.0 ? 1 : effects[index] < 0.0 ? -1 : 0;
            require(motor.at("preference") == expected, "wrong measured local preference");
            require(motor.at("confidence") == (expected == 0 ? "uncertain" : "measured"),
                    "incorrect evidence label");
            require(std::abs(motor.at("evidence_lux").get<double>() - 1.8 * effects[index]) < 1e-8,
                    "paired slope evidence has wrong physical units or sign");
            require(motor.at("positive_samples").get<std::size_t>() >= 3 &&
                        motor.at("positive_samples").get<std::size_t>() <= 7 &&
                        motor.at("negative_samples").get<std::size_t>() >= 3 &&
                        motor.at("negative_samples").get<std::size_t>() <= 7,
                    "held values were counted as independent measurements");
            if (expected == 0) require(motor.at("effort") == 0.0,
                                        "uncertain motor should remain unpowered");
        }
    }
    const std::vector<std::string> ids{"a", "b"};
    for (const auto& effects : {std::vector<double>{1.0, 1.0}, std::vector<double>{1.0, -1.0}}) {
        PhysicalFixture fixture{ids, effects};
        droid::BodyProbeController together;
        together.reset(ids, "together");
        run(together, fixture, 400);
        const Json state = together.diagnostics();
        const int expected = effects[1] > 0.0 ? 1 : 0;
        for (const Json& motor : state.at("motors")) {
            require(motor.at("preference") == expected, "shared paired probe result incorrect");
        }
        require(state.at("phase") == "seeking", "together comparison did not finish");
    }
}

void test_id_relabel_and_collection_order() {
    // Renaming reverses the lexical order. Enumeration, not spelling, carries
    // the deliberate probe schedule; observations and act IDs may be reordered.
    PhysicalFixture first{{"zulu", "alpha"}, {1.0, -1.0}};
    PhysicalFixture renamed{{"one", "two"}, {1.0, -1.0}};
    droid::BodyProbeController a;
    droid::BodyProbeController b;
    a.reset(first.ids);
    b.reset(renamed.ids);
    for (std::size_t step = 0; step < 700; ++step) {
        const auto actions_a = a.act(first.observation(), first.ids);
        Json other = renamed.observation();
        std::reverse(other.at("actuator_feedback").begin(), other.at("actuator_feedback").end());
        std::reverse(other.at("reward_sensors").begin(), other.at("reward_sensors").end());
        std::vector<std::string> reversed_ids = renamed.ids;
        std::reverse(reversed_ids.begin(), reversed_ids.end());
        const auto actions_b = b.act(other, reversed_ids);
        require(actions_a.at("zulu") == actions_b.at("one") &&
                    actions_a.at("alpha") == actions_b.at("two"),
                "ID spelling or collection order changed behavior");
        require(a.diagnostics().at("phase") == b.diagnostics().at("phase"),
                "renaming changed the phase");
        first.advance(actions_a);
        renamed.advance(actions_b);
    }
}

void test_timing_and_held_samples() {
    PhysicalFixture fixture{{"motor"}, {1.0}};
    droid::BodyProbeController controller;
    controller.reset(fixture.ids);
    for (std::size_t count = 0; count < 2; ++count) {
        const Json observation = fixture.observation();
        require(!observation.at("reward_sensors").at(0).at("valid").get<bool>(),
                "fixture delivered a sample early");
        const auto actions = controller.act(observation, fixture.ids);
        require(actions.at("motor") == 0.0, "initial latency window should coast");
        fixture.advance(actions);
    }
    Json delivered = fixture.observation();
    require(delivered.at("reward_sensors").at(0).at("sample_time_s") == 0.0 &&
                delivered.at("reward_sensors").at(0).at("delivered_time_s") == 0.04,
            "fixture capture/delivery mismatch");
    fixture.advance(controller.act(delivered, fixture.ids));
    Json held = fixture.observation();
    Json changed = held;
    changed["reward_sensors"][0]["observations"]["illuminance_lux"] = 1000.0;
    const Json before = controller.diagnostics();
    require_throws([&] { (void)controller.act(changed, fixture.ids); },
                   "same-sequence altered sample was accepted");
    require(controller.diagnostics() == before, "rejected sample mutated controller memory");
    fixture.advance(controller.act(held, fixture.ids));
    run(controller, fixture, 300);

    const Json valid = fixture.observation();
    const auto reject_change = [&](const std::function<void(Json&)>& mutate) {
        Json bad = valid;
        mutate(bad);
        const Json prior = controller.diagnostics();
        require_throws([&] { (void)controller.act(bad, fixture.ids); }, "bad sensor timing accepted");
        require(controller.diagnostics() == prior, "failed call mutated state");
    };
    reject_change([](Json& bad) { bad["reward_sensors"][0]["age_s"] = 0.6; });
    reject_change([](Json& bad) { bad["reward_sensors"][0]["sample_time_s"] = 9999.0; });
    reject_change([](Json& bad) { bad["reward_sensors"][0]["sequence"] = 0; });
    reject_change([](Json& bad) { bad["reward_sensors"][0]["valid"] = false; });
    reject_change([](Json& bad) { bad["reward_sensors"][0]["latency_s"] = 0.0; });
    reject_change([](Json& bad) { bad["reward_sensors"][0]["sample_period_s"] = 0.02; });
    reject_change([](Json& bad) {
        auto& light = bad["reward_sensors"][0];
        light["sequence"] = light["sequence"].get<std::uint64_t>() - 1;
        light["sample_time_s"] = light["sample_time_s"].get<double>() - 0.1;
        light["delivered_time_s"] = light["delivered_time_s"].get<double>() - 0.1;
        light["age_s"] = light["age_s"].get<double>() + 0.1;
    });
    fixture.advance(controller.act(valid, fixture.ids));
}

void test_observation_boundary_and_fail_closed() {
    PhysicalFixture fixture{{"motor"}, {1.0}};
    droid::BodyProbeController controller;
    require_throws([&] { (void)controller.act(fixture.observation(), fixture.ids); },
                   "inference before reset accepted");
    // Fresh fixture because observation() advances the fixture's sample queue.
    fixture = PhysicalFixture{{"motor"}, {1.0}};
    controller.reset(fixture.ids);
    run(controller, fixture, 20);
    const Json valid = fixture.observation();
    const auto reject = [&](const std::function<void(Json&)>& mutate) {
        Json bad = valid;
        mutate(bad);
        const Json prior = controller.diagnostics();
        require_throws([&] { (void)controller.act(bad, fixture.ids); },
                       "malformed or privileged observation accepted");
        require(controller.diagnostics() == prior, "failed observation changed memory");
    };
    reject([](Json& bad) { bad["world_pose"] = Json::array({0, 0, 0}); });
    reject([](Json& bad) { bad["seed"] = 42; });
    reject([](Json& bad) { bad["reward_sensors"][0]["body_variant"] = "reversed"; });
    reject([](Json& bad) { bad["reward_sensors"][0]["observations"]["source_x"] = 20; });
    reject([](Json& bad) { bad["actuator_feedback"][0]["feedback"]["reward"] = 1; });
    reject([](Json& bad) { bad["actuator_feedback"][0]["valid"] = false; });
    reject([](Json& bad) { bad["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"overspeed"}); });
    reject([](Json& bad) { bad["actuator_feedback"].clear(); });
    reject([](Json& bad) { bad["reward_sensors"].erase(bad["reward_sensors"].begin()); });
    reject([](Json& bad) { bad["reward_sensors"][0]["observations"]["illuminance_lux"] = -1.0; });
    reject([](Json& bad) { bad["actuator_feedback"][0]["feedback"]["current_a"] = std::numeric_limits<double>::infinity(); });
    reject([](Json& bad) { bad["reward_sensors"].push_back(bad["reward_sensors"][0]); });
    const std::vector<std::string> other_ids{"renamed-without-reset"};
    require_throws([&] { (void)controller.act(valid, other_ids); }, "unannounced topology change accepted");
    const std::vector<std::string> duplicate_ids{"x", "x"};
    require_throws([&] { controller.reset(duplicate_ids); }, "duplicate reset IDs accepted");
    require_throws([&] { controller.reset(fixture.ids, "secret-optimal-policy"); }, "unknown strategy accepted");
}

void test_reset_and_uninformative_light() {
    PhysicalFixture fixture{{"motor"}, {1.0}};
    droid::BodyProbeController controller;
    controller.reset(fixture.ids);
    const Json fresh = controller.diagnostics();
    run(controller, fixture, 250);
    require(controller.diagnostics().at("motors").at(0).at("preference") == 1,
            "fixture did not learn before reset");
    controller.reset(fixture.ids);
    require(controller.diagnostics() == fresh, "reset retained body-specific experience");
    fixture = PhysicalFixture{{"motor"}, {1.0}};
    fixture.saturated = true;
    run(controller, fixture, 250);
    const Json motor = controller.diagnostics().at("motors").at(0);
    require(motor.at("confidence") == "uncertain" && motor.at("preference") == 0 &&
                motor.at("effort") == 0.0,
            "saturation was treated as informative transmission evidence");
    require(droid::body_controller_spec().at("trained_parameters") == false,
            "handwritten provenance missing");
}

void test_unloaded_shaft_both_probe_signs() {
    // Actual successor mechanics, with both published development worlds.
    // This checks the prospective governor against the failure mechanism:
    // the open shaft must complete both probes without voltage limiting.
    for (const std::uint64_t seed : {6106049681611768608ULL, 6990021049862345368ULL}) {
        droid::DroidEnvironment environment("models/droid.xml", false);
        const auto ids = environment.spec().at("action").at("motor_ids")
            .get<std::vector<std::string>>();
        Json observation = environment.reset("transmission_disconnected_v1", seed,
            false, "body_discovery_1d_v1").at("observation");
        droid::BodyProbeController controller;
        controller.reset(ids);
        double minimum_velocity = 0.0;
        double maximum_velocity = 0.0;
        for (std::size_t step = 0; step < 180; ++step) {
            const auto actions = controller.act(observation, ids);
            const Json transition = environment.step(actions, 0.02);
            require(!transition.at("terminated").get<bool>() &&
                        !transition.at("truncated").get<bool>(),
                    "governed unloaded-shaft probe triggered environment termination");
            observation = transition.at("observation");
            for (const Json& feedback : observation.at("actuator_feedback")) {
                require(feedback.at("feedback").at("fault_flags").empty(),
                        "governed probe emitted a motor feedback flag");
                if (feedback.at("module_id") == ids.front()) {
                    const double speed = feedback.at("feedback").at("velocity_rad_s");
                    minimum_velocity = std::min(minimum_velocity, speed);
                    maximum_velocity = std::max(maximum_velocity, speed);
                }
            }
        }
        require(minimum_velocity < -1.0 && maximum_velocity > 1.0,
                "unloaded shaft did not exercise both probe directions");
        std::cout << "GOVERNOR target_rad_s=" << droid::kBodyControllerShaftTargetRadS
                  << " seed=" << seed << " minimum_rad_s=" << minimum_velocity
                  << " maximum_rad_s=" << maximum_velocity << '\n';
        require(minimum_velocity > -4.5 && maximum_velocity < 4.5,
                "sampled governor overshot the test's declared development envelope: minimum=" +
                    std::to_string(minimum_velocity) + ", maximum=" + std::to_string(maximum_velocity));
        const Json motor = controller.diagnostics().at("motors").at(0);
        require(motor.at("preference") == 0 && motor.at("confidence") == "uncertain",
                "open shaft generated an unjustified light-response preference");
    }
}
}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"paired identification and matched together", test_paired_identification_and_matched_together},
        {"ID relabel and collection order", test_id_relabel_and_collection_order},
        {"timing and held samples", test_timing_and_held_samples},
        {"observation boundary and fail closed", test_observation_boundary_and_fail_closed},
        {"reset and uninformative light", test_reset_and_uninformative_light},
        {"unloaded shaft both probe signs", test_unloaded_shaft_both_probe_signs}};
    for (const auto& [name, test] : tests) {
        try { test(); }
        catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
        std::cout << "PASS " << name << '\n';
    }
    std::cout << tests.size() << " native body controller tests passed\n";
    return 0;
}
