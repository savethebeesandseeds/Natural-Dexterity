#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace droid {

using Json = nlohmann::json;

inline constexpr std::string_view kModuleSchemaVersion{"0.1.0"};
inline constexpr std::string_view kReferenceMotorSku{"rotary_dc_gearmotor_v0"};
inline constexpr double kRewardRateMinimum = -1.0;
inline constexpr double kRewardRateMaximum = 1.0;

// The default catalog is loaded once. DROID_MODULE_CATALOG_PATH may be set at
// compile time; otherwise config/module_catalog.json is resolved at runtime.
[[nodiscard]] const Json& catalog_json();
[[nodiscard]] Json load_catalog(const std::filesystem::path& path);
void validate_catalog(const Json& catalog);

[[nodiscard]] double clamp(double value, double lower, double upper);
[[nodiscard]] double bounded_reward(double value);
[[nodiscard]] double danger_high(double value, double soft, double hard);
[[nodiscard]] double danger_low(double value, double soft, double hard);
[[nodiscard]] double danger_magnitude(
    double value,
    double soft_magnitude,
    double hard_magnitude);

struct RegisteredSensorSample {
    std::string module_id;
    std::string family_id;
    std::optional<double> reward_rate;
    bool valid{true};
    bool stale{false};
};

// Only registered reward sensors vote. Stale/invalid registrations use their
// family's missing_reward; absent families contribute nothing.
[[nodiscard]] double transition_reward(
    double dt_s,
    std::span<const RegisteredSensorSample> registered_sensor_samples,
    const Json& catalog = catalog_json());

[[nodiscard]] double sensor_reward_rate(
    std::string_view family_id,
    const Json& observation,
    const Json& parameter_overrides = Json::object(),
    const Json& catalog = catalog_json());

struct MotorState {
    double current_a{0.0};
    double temperature_c{25.0};
};

struct SafetyStatus {
    bool veto{false};
    std::vector<std::string> flags;
};

struct MotorStepResult {
    MotorState state;
    double requested_effort{0.0};
    double applied_effort{0.0};
    double available_current_a{0.0};
    double target_current_a{0.0};
    double back_emf_v{0.0};
    double applied_voltage_v{0.0};
    double output_torque_nm{0.0};
    double electrical_power_w{0.0};
    double copper_loss_w{0.0};
    double thermal_derate{1.0};
    SafetyStatus safety;
};

[[nodiscard]] MotorStepResult step_motor_current_thermal(
    const MotorState& state,
    double effort,
    double output_velocity_rad_s,
    double bus_voltage_v,
    double dt_s,
    std::optional<double> ambient_temperature_c = std::nullopt,
    const Json& parameter_overrides = Json::object(),
    const Json& catalog = catalog_json());

struct ActuatorFeedbackInput {
    double position_rad{0.0};
    double velocity_rad_s{0.0};
    double current_a{0.0};
    double bus_voltage_v{0.0};
    double temperature_c{25.0};
    double output_torque_nm{0.0};
    double commanded_effort{0.0};
    double previous_stuck_score{0.0};
    double dt_s{0.02};
    std::vector<std::string> fault_flags;
};

struct ActuatorFeedback {
    double position_rad{0.0};
    double velocity_rad_s{0.0};
    double current_a{0.0};
    double bus_voltage_v{0.0};
    double temperature_c{25.0};
    double output_torque_nm{0.0};
    double load_impedance_nm_s_per_rad{0.0};
    double stuck_score{0.0};
    bool stuck{false};
    std::vector<std::string> fault_flags;
};

// Actuator feedback is deliberately a separate type with no reward field.
[[nodiscard]] ActuatorFeedback actuator_feedback_sample(
    const ActuatorFeedbackInput& input,
    const Json& estimator_parameter_overrides = Json::object(),
    const Json& catalog = catalog_json());

void to_json(Json& json, const MotorState& value);
void to_json(Json& json, const SafetyStatus& value);
void to_json(Json& json, const MotorStepResult& value);
void to_json(Json& json, const ActuatorFeedback& value);

}  // namespace droid
