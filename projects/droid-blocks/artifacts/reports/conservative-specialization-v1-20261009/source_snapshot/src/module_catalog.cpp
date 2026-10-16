#include "droid/module_catalog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace droid {
namespace {

[[nodiscard]] double finite(std::string_view name, double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string{name} + " must be a finite scalar");
    }
    return value;
}

[[nodiscard]] double json_number(
    const Json& object,
    std::string_view key,
    std::string_view prefix = {}) {
    if (!object.is_object()) {
        throw std::invalid_argument(
            prefix.empty() ? "value must be an object"
                           : std::string{prefix} + " must be an object");
    }
    const auto iterator = object.find(std::string{key});
    if (iterator == object.end() || !iterator->is_number()) {
        const auto label = prefix.empty()
                               ? std::string{key}
                               : std::string{prefix} + "." + std::string{key};
        throw std::invalid_argument(label + " must be a finite scalar");
    }
    const auto value = iterator->get<double>();
    const auto label = prefix.empty()
                           ? std::string{key}
                           : std::string{prefix} + "." + std::string{key};
    return finite(label, value);
}

[[nodiscard]] double positive(const Json& values, std::string_view name) {
    const auto value = json_number(values, name);
    if (value <= 0.0) {
        throw std::invalid_argument(
            std::string{name} + " must be greater than zero");
    }
    return value;
}

[[nodiscard]] Json merged_parameters(
    const Json& defaults,
    const Json& overrides,
    std::string_view context) {
    if (!defaults.is_object() || !overrides.is_object()) {
        throw std::invalid_argument(std::string{context} + " parameters must be objects");
    }
    auto merged = defaults;
    for (auto iterator = overrides.begin(); iterator != overrides.end(); ++iterator) {
        if (!defaults.contains(iterator.key())) {
            throw std::invalid_argument(
                "unknown " + std::string{context} + " parameter: " + iterator.key());
        }
        merged[iterator.key()] = iterator.value();
    }
    return merged;
}

[[nodiscard]] const Json& reward_families(const Json& catalog) {
    const auto iterator = catalog.find("sensor_families");
    if (iterator == catalog.end() || !iterator->is_object()) {
        throw std::invalid_argument("catalog.sensor_families must be an object");
    }
    return *iterator;
}

[[nodiscard]] const Json& reference_motor(const Json& catalog) {
    const auto motors = catalog.find("motor_skus");
    if (motors == catalog.end() || !motors->is_object()) {
        throw std::invalid_argument("catalog.motor_skus must be an object");
    }
    const auto motor = motors->find(std::string{kReferenceMotorSku});
    if (motor == motors->end() || !motor->is_object()) {
        throw std::invalid_argument("catalog is missing the reference motor SKU");
    }
    return *motor;
}

[[nodiscard]] const Json& object_member(
    const Json& object,
    std::string_view key,
    std::string_view description) {
    const auto iterator = object.find(std::string{key});
    if (iterator == object.end() || !iterator->is_object()) {
        throw std::invalid_argument(std::string{description} + " must be an object");
    }
    return *iterator;
}

[[nodiscard]] bool optional_bool(
    const Json& object,
    std::string_view key,
    bool fallback) {
    const auto iterator = object.find(std::string{key});
    if (iterator == object.end()) {
        return fallback;
    }
    if (!iterator->is_boolean()) {
        throw std::invalid_argument(std::string{key} + " must be boolean");
    }
    return iterator->get<bool>();
}

[[nodiscard]] std::array<double, 3> vector3(
    const Json& object,
    std::string_view key) {
    const auto iterator = object.find(std::string{key});
    if (iterator == object.end() || !iterator->is_array() || iterator->size() != 3) {
        throw std::invalid_argument(
            std::string{key} + " must contain exactly three finite scalars");
    }
    std::array<double, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (!(*iterator)[index].is_number()) {
            throw std::invalid_argument(
                std::string{key} + " must contain exactly three finite scalars");
        }
        result[index] = finite(key, (*iterator)[index].get<double>());
    }
    return result;
}

[[nodiscard]] double compensated_sum(std::span<const double> values) {
    // Neumaier's variant is deterministic and retains small addends when the
    // running sum has a much larger magnitude.
    double sum = 0.0;
    double compensation = 0.0;
    for (const auto value : values) {
        const auto next = sum + value;
        if (std::abs(sum) >= std::abs(value)) {
            compensation += (sum - next) + value;
        } else {
            compensation += (value - next) + sum;
        }
        sum = next;
    }
    return sum + compensation;
}

[[nodiscard]] bool close_enough(double first, double second) {
    const auto tolerance = std::max(
        1e-12,
        1e-12 * std::max(std::abs(first), std::abs(second)));
    return std::abs(first - second) <= tolerance;
}

[[nodiscard]] const Json& family_by_id(
    std::string_view family_id,
    const Json& catalog) {
    const auto& families = reward_families(catalog);
    const auto iterator = families.find(std::string{family_id});
    if (iterator == families.end() || !iterator->is_object()) {
        throw std::invalid_argument(
            "unknown reward sensor family: " + std::string{family_id});
    }
    return *iterator;
}

[[nodiscard]] double reward_power(const Json& observation, const Json& parameters) {
    const auto voltage_v = json_number(observation, "bus_voltage_v");
    const auto current_a = json_number(observation, "bus_current_a");
    const auto signed_power_w = voltage_v * current_a;
    if (!std::isfinite(signed_power_w)) {
        throw std::invalid_argument("power observation overflowed a finite scalar");
    }
    const auto power_term = clamp(
        -signed_power_w / positive(parameters, "power_scale_w"),
        -1.0,
        1.0);
    const auto low_energy_danger = danger_low(
        json_number(observation, "energy_fraction"),
        json_number(parameters, "energy_soft_fraction"),
        json_number(parameters, "energy_hard_fraction"));
    return json_number(parameters, "power_weight") * power_term -
           json_number(parameters, "low_energy_weight") * low_energy_danger;
}

[[nodiscard]] double reward_imu(const Json& observation, const Json& parameters) {
    const auto acceleration = vector3(observation, "specific_force_m_s2");
    const auto angular_velocity = vector3(observation, "angular_velocity_rad_s");
    std::array<double, 3> acceleration_squared{};
    std::array<double, 3> angular_squared{};
    for (std::size_t index = 0; index < 3; ++index) {
        acceleration_squared[index] = acceleration[index] * acceleration[index];
        angular_squared[index] = angular_velocity[index] * angular_velocity[index];
    }
    const auto acceleration_norm = std::sqrt(compensated_sum(acceleration_squared));
    const auto angular_speed = std::sqrt(compensated_sum(angular_squared));
    const auto acceleration_danger = danger_high(
        std::abs(acceleration_norm - json_number(parameters, "gravity_m_s2")),
        json_number(parameters, "acceleration_deviation_soft_m_s2"),
        json_number(parameters, "acceleration_deviation_hard_m_s2"));
    const auto angular_danger = danger_high(
        angular_speed,
        json_number(parameters, "angular_speed_soft_rad_s"),
        json_number(parameters, "angular_speed_hard_rad_s"));
    return -std::max(acceleration_danger, angular_danger);
}

[[nodiscard]] double reward_touch(const Json& observation, const Json& parameters) {
    const auto force_n = std::max(0.0, json_number(observation, "normal_force_n"));
    return -danger_high(
        force_n,
        json_number(parameters, "force_soft_n"),
        json_number(parameters, "force_hard_n"));
}

[[nodiscard]] double reward_proximity(const Json& observation, const Json& parameters) {
    if (!optional_bool(observation, "measurement_valid", true)) {
        return -1.0;
    }
    return -danger_low(
        json_number(observation, "distance_m"),
        json_number(parameters, "distance_soft_m"),
        json_number(parameters, "distance_hard_m"));
}

[[nodiscard]] double reward_ambient_light(
    const Json& observation,
    const Json& parameters) {
    const auto illuminance_lux =
        std::max(0.0, json_number(observation, "illuminance_lux"));
    const auto saturation_lux = positive(parameters, "saturation_lux");
    return clamp(
        std::log1p(illuminance_lux) / std::log1p(saturation_lux),
        0.0,
        1.0);
}

}  // namespace

Json load_catalog(const std::filesystem::path& path) {
    std::ifstream stream{path};
    if (!stream) {
        throw std::runtime_error(
            "unable to open module catalog: " + path.string());
    }
    Json catalog;
    try {
        stream >> catalog;
    } catch (const Json::exception& error) {
        throw std::invalid_argument(
            "invalid module catalog JSON in " + path.string() + ": " + error.what());
    }
    validate_catalog(catalog);
    return catalog;
}

const Json& catalog_json() {
    static const Json catalog = [] {
        std::vector<std::filesystem::path> candidates;
        if (const auto* environment_path = std::getenv("DROID_MODULE_CATALOG_PATH")) {
            if (*environment_path != '\0') {
                candidates.emplace_back(environment_path);
            }
        }
#ifdef DROID_MODULE_CATALOG_PATH
        candidates.emplace_back(DROID_MODULE_CATALOG_PATH);
#endif
        candidates.emplace_back("config/module_catalog.json");
        candidates.emplace_back("../config/module_catalog.json");
        candidates.emplace_back("/workspace/config/module_catalog.json");

        for (const auto& candidate : candidates) {
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error) && !error) {
                return load_catalog(candidate);
            }
        }
        std::ostringstream message;
        message << "module catalog not found; checked";
        for (const auto& candidate : candidates) {
            message << ' ' << candidate.string();
        }
        throw std::runtime_error(message.str());
    }();
    return catalog;
}

void validate_catalog(const Json& catalog) {
    if (!catalog.is_object()) {
        throw std::invalid_argument("module catalog must be an object");
    }
    const auto schema = catalog.find("schema_version");
    if (schema == catalog.end() || !schema->is_string() ||
        schema->get<std::string>() != kModuleSchemaVersion) {
        throw std::invalid_argument("module catalog schema_version must be 0.1.0");
    }
    const auto motors = catalog.find("motor_skus");
    if (motors == catalog.end() || !motors->is_object() || motors->size() != 1 ||
        !motors->contains(std::string{kReferenceMotorSku})) {
        throw std::invalid_argument(
            "module catalog must contain exactly rotary_dc_gearmotor_v0");
    }
    const auto& motor = reference_motor(catalog);
    const auto& feedback = object_member(
        motor,
        "actuator_feedback",
        "reference motor actuator_feedback");
    const auto reward_vote = feedback.find("reward_vote");
    if (reward_vote == feedback.end() || !reward_vote->is_boolean() ||
        reward_vote->get<bool>() || feedback.contains("reward")) {
        throw std::invalid_argument("actuator feedback must be excluded from reward");
    }
    (void)object_member(motor, "physics", "reference motor physics");
    (void)object_member(feedback, "stuck_estimator", "stuck estimator");

    const auto& families = reward_families(catalog);
    if (families.size() != 5) {
        throw std::invalid_argument(
            "module catalog must contain exactly five reward sensor families");
    }
    constexpr std::array<std::string_view, 5> required_families{
        "power_v0",
        "imu_6axis_v0",
        "touch_force_v0",
        "proximity_tof_v0",
        "ambient_light_v0",
    };
    for (const auto family_id : required_families) {
        const auto& family = family_by_id(family_id, catalog);
        (void)finite("missing_reward", json_number(family, "missing_reward"));
        const auto& reward = object_member(family, "reward", "sensor reward");
        const auto function = reward.find("function_id");
        if (function == reward.end() || !function->is_string() ||
            function->get<std::string>() != family_id) {
            throw std::invalid_argument(
                "sensor reward function_id must match its versioned family ID");
        }
        (void)object_member(reward, "parameters", "sensor reward parameters");
    }
}

double clamp(double value, double lower, double upper) {
    (void)finite("value", value);
    (void)finite("lower", lower);
    (void)finite("upper", upper);
    if (lower > upper) {
        throw std::invalid_argument("lower must be less than or equal to upper");
    }
    return std::min(upper, std::max(lower, value));
}

double bounded_reward(double value) {
    return clamp(value, kRewardRateMinimum, kRewardRateMaximum);
}

double danger_high(double value, double soft, double hard) {
    (void)finite("value", value);
    (void)finite("soft", soft);
    (void)finite("hard", hard);
    if (!(soft < hard)) {
        throw std::invalid_argument("danger_high requires soft < hard");
    }
    return clamp((value - soft) / (hard - soft), 0.0, 1.0);
}

double danger_low(double value, double soft, double hard) {
    (void)finite("value", value);
    (void)finite("soft", soft);
    (void)finite("hard", hard);
    if (!(hard < soft)) {
        throw std::invalid_argument("danger_low requires hard < soft");
    }
    return clamp((soft - value) / (soft - hard), 0.0, 1.0);
}

double danger_magnitude(
    double value,
    double soft_magnitude,
    double hard_magnitude) {
    return danger_high(
        std::abs(finite("value", value)),
        soft_magnitude,
        hard_magnitude);
}

double transition_reward(
    double dt_s,
    std::span<const RegisteredSensorSample> registered_sensor_samples,
    const Json& catalog) {
    (void)finite("dt_s", dt_s);
    if (dt_s < 0.0) {
        throw std::invalid_argument("dt_s must be non-negative");
    }

    std::unordered_set<std::string> module_ids;
    std::vector<double> reward_rates;
    reward_rates.reserve(registered_sensor_samples.size());
    for (const auto& sample : registered_sensor_samples) {
        if (sample.module_id.empty()) {
            throw std::invalid_argument(
                "each registered sensor sample needs a non-empty module_id");
        }
        if (!module_ids.insert(sample.module_id).second) {
            throw std::invalid_argument(
                "duplicate sample for registered sensor " + sample.module_id);
        }
        const auto& family = family_by_id(sample.family_id, catalog);
        if (sample.stale || !sample.valid) {
            reward_rates.push_back(json_number(family, "missing_reward"));
            continue;
        }
        if (!sample.reward_rate.has_value()) {
            throw std::invalid_argument(
                "fresh registered sensor sample needs reward_rate");
        }
        const auto rate = finite("sensor reward rate", *sample.reward_rate);
        if (rate < kRewardRateMinimum || rate > kRewardRateMaximum) {
            throw std::invalid_argument(
                "reward_rate for " + sample.module_id + " must be within [-1, 1]");
        }
        reward_rates.push_back(rate);
    }
    return dt_s * compensated_sum(reward_rates);
}

double sensor_reward_rate(
    std::string_view family_id,
    const Json& observation,
    const Json& parameter_overrides,
    const Json& catalog) {
    if (!observation.is_object()) {
        throw std::invalid_argument("sensor observation must be an object");
    }
    const auto& family = family_by_id(family_id, catalog);
    const auto& reward = object_member(family, "reward", "sensor reward");
    const auto& defaults = object_member(
        reward,
        "parameters",
        "sensor reward parameters");
    const auto parameters = merged_parameters(defaults, parameter_overrides, "reward");

    double reward_rate = 0.0;
    if (family_id == "power_v0") {
        reward_rate = reward_power(observation, parameters);
    } else if (family_id == "imu_6axis_v0") {
        reward_rate = reward_imu(observation, parameters);
    } else if (family_id == "touch_force_v0") {
        reward_rate = reward_touch(observation, parameters);
    } else if (family_id == "proximity_tof_v0") {
        reward_rate = reward_proximity(observation, parameters);
    } else if (family_id == "ambient_light_v0") {
        reward_rate = reward_ambient_light(observation, parameters);
    } else {
        throw std::invalid_argument(
            "no native reward implementation for family: " + std::string{family_id});
    }
    return bounded_reward(reward_rate);
}

MotorStepResult step_motor_current_thermal(
    const MotorState& state,
    double effort,
    double output_velocity_rad_s,
    double bus_voltage_v,
    double dt_s,
    std::optional<double> ambient_temperature_c,
    const Json& parameter_overrides,
    const Json& catalog) {
    const auto& motor = reference_motor(catalog);
    const auto& defaults = object_member(motor, "physics", "motor physics");
    const auto parameters = merged_parameters(defaults, parameter_overrides, "motor");

    const auto current_a = finite("state.current_a", state.current_a);
    const auto temperature_c = finite("state.temperature_c", state.temperature_c);
    (void)finite("effort", effort);
    (void)finite("output_velocity_rad_s", output_velocity_rad_s);
    (void)finite("bus_voltage_v", bus_voltage_v);
    (void)finite("dt_s", dt_s);
    if (dt_s <= 0.0) {
        throw std::invalid_argument("dt_s must be greater than zero");
    }
    if (bus_voltage_v < 0.0) {
        throw std::invalid_argument("bus_voltage_v must be non-negative");
    }
    const auto ambient_c = finite(
        "ambient_temperature_c",
        ambient_temperature_c.value_or(
            json_number(parameters, "reference_ambient_c")));

    const auto resistance = positive(parameters, "winding_resistance_ohm");
    const auto gear_ratio = positive(parameters, "gear_ratio_motor_to_output");
    const auto efficiency = clamp(
        json_number(parameters, "gear_efficiency"),
        0.0,
        1.0);
    const auto torque_constant = positive(parameters, "torque_constant_nm_per_a");
    const auto back_emf_constant = positive(parameters, "back_emf_v_s_per_rad");
    const auto peak_current = positive(parameters, "peak_current_a");
    const auto current_time_constant =
        positive(parameters, "current_loop_time_constant_s");
    const auto thermal_capacity = positive(parameters, "thermal_capacity_j_per_c");
    const auto thermal_resistance =
        positive(parameters, "thermal_resistance_c_per_w");
    const auto warning_temperature = json_number(parameters, "thermal_warning_c");
    const auto cutoff_temperature = json_number(parameters, "thermal_cutoff_c");
    if (!(warning_temperature < cutoff_temperature)) {
        throw std::invalid_argument(
            "thermal_warning_c must be below thermal_cutoff_c");
    }

    MotorStepResult result;
    result.requested_effort = effort;
    const auto clipped_effort = clamp(effort, -1.0, 1.0);
    if (clipped_effort != effort) {
        result.safety.flags.emplace_back("action_clamped");
    }

    if (temperature_c <= warning_temperature) {
        result.thermal_derate = 1.0;
    } else if (temperature_c >= cutoff_temperature) {
        result.thermal_derate = 0.0;
    } else {
        result.thermal_derate =
            (cutoff_temperature - temperature_c) /
            (cutoff_temperature - warning_temperature);
    }

    const auto unsafe_voltage =
        bus_voltage_v < json_number(parameters, "minimum_bus_voltage_v");
    const auto overspeed =
        std::abs(output_velocity_rad_s) >=
        json_number(parameters, "maximum_output_speed_rad_s");
    const auto overtemperature = temperature_c >= cutoff_temperature;
    result.safety.veto = unsafe_voltage || overspeed || overtemperature;
    if (unsafe_voltage) {
        result.safety.flags.emplace_back("unsafe_bus_voltage");
    }
    if (overspeed) {
        result.safety.flags.emplace_back("overspeed");
    }
    if (overtemperature) {
        result.safety.flags.emplace_back("overtemperature");
    }

    result.applied_effort = result.safety.veto ? 0.0 : clipped_effort;
    result.available_current_a = peak_current * result.thermal_derate;
    const auto target_current_a =
        result.applied_effort * result.available_current_a;

    const auto motor_velocity_rad_s = gear_ratio * output_velocity_rad_s;
    result.back_emf_v = back_emf_constant * motor_velocity_rad_s;
    const auto voltage_current_low =
        (-bus_voltage_v - result.back_emf_v) / resistance;
    const auto voltage_current_high =
        (bus_voltage_v - result.back_emf_v) / resistance;
    result.target_current_a = clamp(
        target_current_a,
        std::min(voltage_current_low, voltage_current_high),
        std::max(voltage_current_low, voltage_current_high));
    if (!close_enough(target_current_a, result.target_current_a)) {
        result.safety.flags.emplace_back("voltage_limited");
    }

    const auto response = 1.0 - std::exp(-dt_s / current_time_constant);
    result.state.current_a = current_a +
                             response * (result.target_current_a - current_a);
    result.state.current_a =
        clamp(result.state.current_a, -peak_current, peak_current);

    const auto electromagnetic_torque_nm =
        efficiency * gear_ratio * torque_constant * result.state.current_a;
    const auto smoothing = positive(parameters, "friction_smoothing_rad_s");
    const auto friction_torque_nm =
        json_number(parameters, "output_viscous_friction_nm_s_per_rad") *
            output_velocity_rad_s +
        json_number(parameters, "output_coulomb_friction_nm") *
            std::tanh(output_velocity_rad_s / smoothing);
    result.output_torque_nm = electromagnetic_torque_nm - friction_torque_nm;

    result.copper_loss_w =
        result.state.current_a * result.state.current_a * resistance;
    const auto cooling_w =
        (temperature_c - ambient_c) / thermal_resistance;
    result.state.temperature_c = temperature_c +
                                 (result.copper_loss_w - cooling_w) * dt_s /
                                     thermal_capacity;
    if (result.state.temperature_c >= cutoff_temperature && !overtemperature) {
        result.safety.flags.emplace_back("thermal_cutoff_reached");
    }

    result.applied_voltage_v = clamp(
        result.back_emf_v + resistance * result.state.current_a,
        -bus_voltage_v,
        bus_voltage_v);
    result.electrical_power_w =
        result.applied_voltage_v * result.state.current_a;
    return result;
}

ActuatorFeedback actuator_feedback_sample(
    const ActuatorFeedbackInput& input,
    const Json& estimator_parameter_overrides,
    const Json& catalog) {
    const auto& motor = reference_motor(catalog);
    const auto& feedback_contract = object_member(
        motor,
        "actuator_feedback",
        "motor actuator_feedback");
    const auto& defaults = object_member(
        feedback_contract,
        "stuck_estimator",
        "stuck estimator");
    const auto estimator = merged_parameters(
        defaults,
        estimator_parameter_overrides,
        "stuck-estimator");

    ActuatorFeedback result;
    const auto position_rad = finite("position_rad", input.position_rad);
    result.velocity_rad_s = finite("velocity_rad_s", input.velocity_rad_s);
    result.current_a = finite("current_a", input.current_a);
    result.bus_voltage_v = finite("bus_voltage_v", input.bus_voltage_v);
    result.temperature_c = finite("temperature_c", input.temperature_c);
    result.output_torque_nm = finite("output_torque_nm", input.output_torque_nm);
    const auto commanded_effort =
        finite("commanded_effort", input.commanded_effort);
    const auto previous_stuck_score =
        clamp(input.previous_stuck_score, 0.0, 1.0);
    const auto dt_s = finite("dt_s", input.dt_s);
    if (dt_s <= 0.0) {
        throw std::invalid_argument("dt_s must be greater than zero");
    }

    const auto effort_threshold = positive(estimator, "effort_threshold");
    const auto velocity_threshold =
        positive(estimator, "velocity_threshold_rad_s");
    const auto time_constant = positive(estimator, "time_constant_s");
    const auto assert_score =
        clamp(json_number(estimator, "assert_score"), 0.0, 1.0);
    const auto velocity_floor =
        positive(estimator, "impedance_velocity_floor_rad_s");

    const auto stuck_evidence =
        std::abs(commanded_effort) >= effort_threshold &&
        std::abs(result.velocity_rad_s) <= velocity_threshold
            ? 1.0
            : 0.0;
    const auto response = 1.0 - std::exp(-dt_s / time_constant);
    result.stuck_score = previous_stuck_score +
                         response * (stuck_evidence - previous_stuck_score);
    result.stuck_score = clamp(result.stuck_score, 0.0, 1.0);
    result.stuck = result.stuck_score >= assert_score;
    result.load_impedance_nm_s_per_rad =
        std::abs(result.output_torque_nm) /
        std::max(std::abs(result.velocity_rad_s), velocity_floor);
    result.position_rad = std::atan2(std::sin(position_rad), std::cos(position_rad));

    std::unordered_set<std::string> seen_flags;
    result.fault_flags.reserve(input.fault_flags.size() + 1);
    for (const auto& flag : input.fault_flags) {
        if (flag.empty()) {
            throw std::invalid_argument(
                "fault_flags must contain non-empty strings");
        }
        if (seen_flags.insert(flag).second) {
            result.fault_flags.push_back(flag);
        }
    }
    if (result.stuck && seen_flags.insert("stuck").second) {
        result.fault_flags.emplace_back("stuck");
    }
    return result;
}

void to_json(Json& json, const MotorState& value) {
    json = Json{{"current_a", value.current_a},
                {"temperature_c", value.temperature_c}};
}

void to_json(Json& json, const SafetyStatus& value) {
    json = Json{{"veto", value.veto}, {"flags", value.flags}};
}

void to_json(Json& json, const MotorStepResult& value) {
    json = Json{
        {"state", value.state},
        {"requested_effort", value.requested_effort},
        {"applied_effort", value.applied_effort},
        {"available_current_a", value.available_current_a},
        {"target_current_a", value.target_current_a},
        {"back_emf_v", value.back_emf_v},
        {"applied_voltage_v", value.applied_voltage_v},
        {"output_torque_nm", value.output_torque_nm},
        {"electrical_power_w", value.electrical_power_w},
        {"copper_loss_w", value.copper_loss_w},
        {"thermal_derate", value.thermal_derate},
        {"safety", value.safety},
    };
}

void to_json(Json& json, const ActuatorFeedback& value) {
    json = Json{
        {"position_rad", value.position_rad},
        {"velocity_rad_s", value.velocity_rad_s},
        {"current_a", value.current_a},
        {"bus_voltage_v", value.bus_voltage_v},
        {"temperature_c", value.temperature_c},
        {"output_torque_nm", value.output_torque_nm},
        {"load_impedance_nm_s_per_rad", value.load_impedance_nm_s_per_rad},
        {"stuck_score", value.stuck_score},
        {"stuck", value.stuck},
        {"fault_flags", value.fault_flags},
    };
}

}  // namespace droid
