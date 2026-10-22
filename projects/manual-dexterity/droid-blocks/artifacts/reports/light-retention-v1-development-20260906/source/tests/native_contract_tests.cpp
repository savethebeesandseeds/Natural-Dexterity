#include "droid/module_catalog.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using droid::ActuatorFeedbackInput;
using droid::Json;
using droid::MotorState;
using droid::RegisteredSensorSample;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual));
    }
}

template <typename Function>
void require_invalid(Function&& function, const std::string& message) {
    try {
        std::invoke(std::forward<Function>(function));
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

bool has_flag(const std::vector<std::string>& flags, const std::string& flag) {
    return std::find(flags.begin(), flags.end(), flag) != flags.end();
}

class Suite {
  public:
    template <typename Function>
    void run(const std::string& name, Function&& function) {
        ++cases_;
        try {
            std::invoke(std::forward<Function>(function));
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures_;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }

    [[nodiscard]] int finish() const {
        if (cases_ != 26) {
            std::cerr << "[FAIL] suite registered " << cases_
                      << " cases, expected exactly 26\n";
            return 2;
        }
        std::cout << cases_ - failures_ << '/' << cases_ << " cases passed\n";
        return failures_ == 0 ? 0 : 1;
    }

  private:
    int cases_{0};
    int failures_{0};
};

}  // namespace

int main() {
    Suite suite;

    suite.run("catalog is JSON serializable and isolated", [] {
        auto first = droid::catalog_json();
        const auto encoded = first.dump();
        require(
            encoded.find("\"schema_version\":\"0.1.0\"") != std::string::npos,
            "schema version missing from encoded catalog");
        first["schema_version"] = "mutated";
        require(
            droid::catalog_json().at("schema_version") == "0.1.0",
            "default catalog was mutated through a copy");
    });

    suite.run("exactly one motor and five reward sensor families", [] {
        const auto& catalog = droid::catalog_json();
        const auto& motors = catalog.at("motor_skus");
        require(motors.size() == 1, "unexpected motor SKU count");
        require(
            motors.contains("rotary_dc_gearmotor_v0"),
            "reference motor SKU missing");
        require(
            catalog.at("sensor_families").size() == 5,
            "unexpected reward sensor family count");
    });

    suite.run("actuator feedback is structurally excluded from reward", [] {
        const auto& catalog = droid::catalog_json();
        const auto& feedback = catalog.at("motor_skus")
                                   .at("rotary_dc_gearmotor_v0")
                                   .at("actuator_feedback");
        require(feedback.at("reward_vote") == false, "feedback casts a vote");
        require(!feedback.contains("reward"), "feedback has a reward contract");
        const auto& families = catalog.at("sensor_families");
        require(
            !families.contains("motor_proprioception_v0") &&
                !families.contains("motor_electrical_thermal_v0"),
            "motor feedback leaked into reward sensor families");
    });

    suite.run("passive parts have no active semantics", [] {
        const auto& passive = droid::catalog_json().at("passive_part_contract");
        require(
            passive.at("rule").get<std::string>().find(
                "emit no observation, action, or reward") != std::string::npos,
            "passive rule changed");
        const auto forbidden = passive.at("policy_forbidden_semantics");
        const auto families = passive.at("families");
        require(
            std::find(forbidden.begin(), forbidden.end(), "wheel role") !=
                forbidden.end(),
            "wheel role is not forbidden");
        require(
            std::find(families.begin(), families.end(), "gear") != families.end(),
            "gear passive family missing");
    });

    suite.run("danger helpers cover boundaries", [] {
        require_near(droid::danger_high(2.0, 2.0, 4.0), 0.0, 0.0, "high soft");
        require_near(droid::danger_high(3.0, 2.0, 4.0), 0.5, 0.0, "high mid");
        require_near(droid::danger_high(5.0, 2.0, 4.0), 1.0, 0.0, "high hard");
        require_near(droid::danger_low(4.0, 4.0, 2.0), 0.0, 0.0, "low soft");
        require_near(droid::danger_low(3.0, 4.0, 2.0), 0.5, 0.0, "low mid");
        require_near(droid::danger_low(1.0, 4.0, 2.0), 1.0, 0.0, "low hard");
        require_near(
            droid::danger_magnitude(-3.0, 2.0, 4.0),
            0.5,
            0.0,
            "magnitude");
    });

    suite.run("invalid danger intervals are rejected", [] {
        require_invalid(
            [] { (void)droid::danger_high(0.0, 1.0, 1.0); },
            "danger_high accepted an empty interval");
        require_invalid(
            [] { (void)droid::danger_low(0.0, 1.0, 1.0); },
            "danger_low accepted an empty interval");
    });

    suite.run("transition reward is literal sum and duplicates vote", [] {
        const std::vector<RegisteredSensorSample> base_samples{
            {"light-a", "ambient_light_v0", 0.8},
            {"touch-a", "touch_force_v0", -0.2},
        };
        auto duplicate_samples = base_samples;
        duplicate_samples.push_back({"light-b", "ambient_light_v0", 0.8});
        const auto base = droid::transition_reward(0.25, base_samples);
        const auto duplicated = droid::transition_reward(0.25, duplicate_samples);
        require_near(base, 0.15, 1e-15, "base sum");
        require_near(duplicated, base + 0.25 * 0.8, 1e-15, "duplicate vote");
    });

    suite.run("transition reward is not clipped or averaged", [] {
        const std::vector<RegisteredSensorSample> samples{
            {"light-0", "ambient_light_v0", 1.0},
            {"light-1", "ambient_light_v0", 1.0},
            {"light-2", "ambient_light_v0", 1.0},
        };
        require_near(
            droid::transition_reward(2.0, samples),
            6.0,
            0.0,
            "reward was clipped or averaged");
    });

    suite.run("absent catalog families do not vote", [] {
        const std::vector<RegisteredSensorSample> samples;
        require_near(
            droid::transition_reward(1.0, samples),
            0.0,
            0.0,
            "empty registry had imaginary votes");
    });

    suite.run("registered stale and invalid sensors use missing reward", [] {
        const std::vector<RegisteredSensorSample> stale{
            {"light-a", "ambient_light_v0", 1.0, true, true},
        };
        const std::vector<RegisteredSensorSample> invalid{
            {"touch-a", "touch_force_v0", std::nullopt, false, false},
        };
        require_near(
            droid::transition_reward(0.5, stale),
            -0.5,
            0.0,
            "stale registration did not use missing_reward");
        require_near(
            droid::transition_reward(0.25, invalid),
            -0.25,
            0.0,
            "invalid registration did not use missing_reward");
    });

    suite.run("actuator feedback cannot be counted as reward", [] {
        const std::vector<RegisteredSensorSample> feedback{
            {"motor-a", "rotary_dc_gearmotor_v0", 1.0},
        };
        require_invalid(
            [&] { (void)droid::transition_reward(1.0, feedback); },
            "motor feedback was accepted as a reward vote");
    });

    suite.run("same registered instance cannot be double counted", [] {
        const std::vector<RegisteredSensorSample> samples{
            {"light-a", "ambient_light_v0", 0.5},
            {"light-a", "ambient_light_v0", 0.5},
        };
        require_invalid(
            [&] { (void)droid::transition_reward(1.0, samples); },
            "duplicate module_id was accepted");
    });

    suite.run("nonfinite transition reward is rejected", [] {
        const std::vector<RegisteredSensorSample> samples{
            {"light-a",
             "ambient_light_v0",
             std::numeric_limits<double>::quiet_NaN()},
        };
        require_invalid(
            [&] { (void)droid::transition_reward(0.1, samples); },
            "NaN reward rate was accepted");
    });

    suite.run("all sensor rewards are bounded", [] {
        const std::vector<std::pair<std::string, Json>> observations{
            {"power_v0",
             {{"bus_voltage_v", 1e9},
              {"bus_current_a", 1e9},
              {"energy_fraction", -1e9},
              {"temperature_c", 25.0},
              {"charging", false}}},
            {"imu_6axis_v0",
             {{"specific_force_m_s2", {1e9, 0.0, 0.0}},
              {"angular_velocity_rad_s", {0.0, 1e9, 0.0}}}},
            {"touch_force_v0", {{"contact", true}, {"normal_force_n", 1e9}}},
            {"proximity_tof_v0",
             {{"distance_m", -1e9}, {"measurement_valid", true}}},
            {"ambient_light_v0",
             {{"illuminance_lux", 1e100}, {"saturated", true}}},
        };
        for (const auto& [family, observation] : observations) {
            const auto reward = droid::sensor_reward_rate(family, observation);
            require(
                reward >= -1.0 && reward <= 1.0,
                family + " returned an out-of-bounds reward");
        }
    });

    suite.run("nominal IMU has no authored upright preference", [] {
        const Json observation{
            {"specific_force_m_s2", {9.81, 0.0, 0.0}},
            {"angular_velocity_rad_s", {0.0, 0.0, 0.0}},
        };
        require_near(
            droid::sensor_reward_rate("imu_6axis_v0", observation),
            0.0,
            0.0,
            "nominal sideways gravity was penalized");
    });

    suite.run("light reward is monotonic, positive, and golden", [] {
        const auto dark = droid::sensor_reward_rate(
            "ambient_light_v0",
            {{"illuminance_lux", 0.0}, {"saturated", false}});
        const auto dim = droid::sensor_reward_rate(
            "ambient_light_v0",
            {{"illuminance_lux", 10.0}, {"saturated", false}});
        const auto bright = droid::sensor_reward_rate(
            "ambient_light_v0",
            {{"illuminance_lux", 1000.0}, {"saturated", false}});
        require_near(dark, 0.0, 0.0, "dark reward");
        require(dark < dim && dim < bright, "light reward is not monotonic");
        require_near(dim, 0.34708067508455476, 1e-15, "dim golden reward");
        require_near(bright, 1.0, 0.0, "bright reward");
    });

    suite.run("invalid proximity measurement uses negative reward", [] {
        const auto reward = droid::sensor_reward_rate(
            "proximity_tof_v0",
            {{"distance_m", 1.0}, {"measurement_valid", false}});
        require_near(reward, -1.0, 0.0, "invalid proximity reward");
    });

    suite.run("unknown sensor family and reward parameter are rejected", [] {
        require_invalid(
            [] { (void)droid::sensor_reward_rate("imaginary", Json::object()); },
            "unknown sensor family was accepted");
        require_invalid(
            [] {
                (void)droid::sensor_reward_rate(
                    "ambient_light_v0",
                    {{"illuminance_lux", 1.0}},
                    {{"not_a_parameter", 1.0}});
            },
            "unknown reward parameter was accepted");
    });

    suite.run("feedback is serializable, deduplicated, and has no reward", [] {
        ActuatorFeedbackInput input;
        input.position_rad = 7.0;
        input.velocity_rad_s = 0.0;
        input.current_a = 1.0;
        input.bus_voltage_v = 7.4;
        input.temperature_c = 30.0;
        input.output_torque_nm = 0.3;
        input.commanded_effort = 0.8;
        input.previous_stuck_score = 0.0;
        input.dt_s = 1.0;
        input.fault_flags = {"current_limited", "current_limited"};
        const auto feedback = droid::actuator_feedback_sample(input);
        const Json encoded = feedback;
        (void)encoded.dump();
        require(!encoded.contains("reward"), "feedback has reward");
        require(!encoded.contains("reward_rate"), "feedback has reward_rate");
        require_near(
            feedback.position_rad,
            0.7168146928204135,
            1e-15,
            "wrapped position golden value");
        require_near(
            feedback.stuck_score,
            0.8646647167633873,
            1e-15,
            "stuck score golden value");
        require(feedback.stuck, "stuck flag not asserted");
        require(has_flag(feedback.fault_flags, "stuck"), "stuck fault missing");
        require(
            std::count(
                feedback.fault_flags.begin(),
                feedback.fault_flags.end(),
                "current_limited") == 1,
            "fault flags were not deduplicated");
        require_near(
            feedback.load_impedance_nm_s_per_rad,
            3.0,
            1e-15,
            "impedance golden value");
    });

    suite.run("moving motor reduces prior stuck score", [] {
        ActuatorFeedbackInput input;
        input.velocity_rad_s = 2.0;
        input.current_a = 0.5;
        input.bus_voltage_v = 7.4;
        input.temperature_c = 30.0;
        input.output_torque_nm = 0.2;
        input.commanded_effort = 0.8;
        input.previous_stuck_score = 0.9;
        input.dt_s = 0.5;
        const auto feedback = droid::actuator_feedback_sample(input);
        require(feedback.stuck_score < 0.9, "moving motor did not decay stuck score");
        require(!feedback.stuck, "moving motor remained stuck");
    });

    suite.run("motor step is pure and matches positive-torque golden values", [] {
        const MotorState initial{0.0, 25.0};
        const auto result = droid::step_motor_current_thermal(
            initial,
            0.5,
            0.0,
            7.4,
            0.002);
        require_near(initial.current_a, 0.0, 0.0, "input current mutated");
        require_near(initial.temperature_c, 25.0, 0.0, "input temperature mutated");
        require_near(
            result.state.current_a,
            0.22658655865252272,
            1e-15,
            "motor current golden value");
        require_near(
            result.output_torque_nm,
            0.15294592709045282,
            1e-15,
            "motor torque golden value");
        require_near(
            result.state.temperature_c,
            25.000020536587424,
            1e-14,
            "motor temperature golden value");
        const Json encoded = result;
        (void)encoded.dump();
    });

    suite.run("motor action is clamped without changing requested value", [] {
        const auto result = droid::step_motor_current_thermal(
            MotorState{},
            4.0,
            0.0,
            7.4,
            0.002);
        require_near(result.requested_effort, 4.0, 0.0, "requested effort changed");
        require_near(result.applied_effort, 1.0, 0.0, "effort not clamped");
        require(has_flag(result.safety.flags, "action_clamped"), "clamp flag missing");
    });

    suite.run("overtemperature is a noncompensable veto", [] {
        const auto result = droid::step_motor_current_thermal(
            MotorState{0.0, 85.0},
            1.0,
            0.0,
            7.4,
            0.002);
        require(result.safety.veto, "overtemperature did not veto");
        require_near(result.applied_effort, 0.0, 0.0, "veto still applied effort");
        require(
            has_flag(result.safety.flags, "overtemperature"),
            "overtemperature flag missing");
    });

    suite.run("unsafe voltage and overspeed are vetoes", [] {
        const auto low_voltage = droid::step_motor_current_thermal(
            MotorState{},
            1.0,
            0.0,
            5.0,
            0.002);
        const auto overspeed = droid::step_motor_current_thermal(
            MotorState{},
            1.0,
            25.0,
            7.4,
            0.002);
        require(low_voltage.safety.veto, "unsafe voltage did not veto");
        require(
            has_flag(low_voltage.safety.flags, "unsafe_bus_voltage"),
            "unsafe voltage flag missing");
        require(overspeed.safety.veto, "overspeed did not veto");
        require(
            has_flag(overspeed.safety.flags, "overspeed"),
            "overspeed flag missing");
    });

    suite.run("temperature derates available current", [] {
        const auto cool = droid::step_motor_current_thermal(
            MotorState{0.0, 25.0},
            1.0,
            0.0,
            7.4,
            0.002);
        const auto warm = droid::step_motor_current_thermal(
            MotorState{0.0, 72.5},
            1.0,
            0.0,
            7.4,
            0.002);
        require(
            warm.available_current_a < cool.available_current_a,
            "warm motor was not derated");
        require_near(warm.thermal_derate, 0.5, 0.0, "thermal derate golden value");
    });

    suite.run("invalid motor and feedback arguments are rejected", [] {
        require_invalid(
            [] {
                (void)droid::step_motor_current_thermal(
                    MotorState{},
                    0.0,
                    0.0,
                    7.4,
                    0.0);
            },
            "zero motor dt was accepted");
        require_invalid(
            [] {
                (void)droid::step_motor_current_thermal(
                    MotorState{},
                    std::numeric_limits<double>::quiet_NaN(),
                    0.0,
                    7.4,
                    0.002);
            },
            "NaN effort was accepted");
        require_invalid(
            [] {
                ActuatorFeedbackInput input;
                input.dt_s = 0.0;
                (void)droid::actuator_feedback_sample(input);
            },
            "zero feedback dt was accepted");
    });

    return suite.finish();
}
