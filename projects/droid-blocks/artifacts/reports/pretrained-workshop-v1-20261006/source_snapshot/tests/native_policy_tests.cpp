#include "droid/policy.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void require_close(
    double actual,
    double expected,
    double tolerance,
    std::string_view message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(message) + ": expected " +
            std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

template <typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
    try {
        std::invoke(std::forward<Callable>(callable));
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

[[nodiscard]] std::string digest(char digit) {
    return std::string(64, digit);
}

[[nodiscard]] droid::PolicyArtifactData artifact_for(
    std::vector<std::string> motor_ids = {
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"}) {
    droid::PolicyArtifactData artifact;
    artifact.compatibility.environment_api_version = "agent_environment_v1";
    artifact.compatibility.assembly_id = "demo_rover_v0";
    artifact.compatibility.motor_ids = std::move(motor_ids);
    artifact.compatibility.control_dt_s = 0.02;
    artifact.compatibility.model_sha256 = digest('1');
    artifact.compatibility.module_catalog_sha256 = digest('2');
    artifact.compatibility.environment_spec_sha256 = digest('3');
    artifact.compatibility.feature_schema_sha256 =
        droid::policy_feature_schema_sha256();
    artifact.training.protocol_manifest_sha256 = digest('7');
    artifact.training.train_seeds_sha256 = digest('4');
    artifact.training.validation_seeds_sha256 = digest('5');
    artifact.training.test_seeds_sha256 = digest('6');
    artifact.parameters.fill(0.0);
    artifact.parameters[droid::kPolicyCemMaskV1ActiveParameterIndices[0]] =
        -0.051;
    artifact.parameters[droid::kPolicyCemMaskV1ActiveParameterIndices[1]] =
        0.034;
    artifact.parameters[droid::kPolicyCemMaskV1ActiveParameterIndices[2]] =
        0.017;
    return artifact;
}

[[nodiscard]] Json feedback(
    double offset,
    bool valid = true,
    bool fault = false) {
    return Json{
        {"valid", valid},
        {"feedback",
         Json{
             {"position_rad", 0.1 + offset},
             {"velocity_rad_s", 1.0 + offset},
             {"current_a", 0.2 + offset * 0.01},
             {"bus_voltage_v", 7.4},
             {"temperature_c", 25.5 + offset},
             {"output_torque_nm", 0.1 + offset * 0.01},
             {"load_impedance_nm_s_per_rad", 0.3 + offset},
             {"stuck_score", 0.02},
             {"stuck", false},
             {"fault_flags",
              fault ? Json::array({"driver_fault"}) : Json::array()},
         }},
    };
}

[[nodiscard]] Json observation_for(
    const std::vector<std::string>& motor_ids,
    bool reverse_arrays = false,
    std::string sensor_prefix = "sensor",
    std::string missing_motor = {},
    std::string invalid_motor = {},
    std::string fault_motor = {},
    bool include_sensors = true,
    double light_lux = 100.0) {
    Json sensors = Json::array();
    if (include_sensors) {
        sensors.push_back(Json{
            {"module_id", sensor_prefix + "-light"},
            {"family_id", "ambient_light_v0"},
            {"valid", true},
            {"observations",
             Json{{"illuminance_lux", light_lux}, {"saturated", false}}},
        });
        sensors.push_back(Json{
            {"module_id", sensor_prefix + "-touch-b"},
            {"family_id", "touch_force_v0"},
            {"valid", true},
            {"observations", Json{{"contact", true}, {"normal_force_n", 2.0}}},
        });
        sensors.push_back(Json{
            {"module_id", sensor_prefix + "-touch-a"},
            {"family_id", "touch_force_v0"},
            {"valid", true},
            {"observations", Json{{"contact", false}, {"normal_force_n", 0.5}}},
        });
    }

    Json actuators = Json::array();
    for (std::size_t index = 0; index < motor_ids.size(); ++index) {
        const std::string& motor_id = motor_ids[index];
        if (motor_id == missing_motor) {
            continue;
        }
        Json sample = feedback(
            static_cast<double>(index),
            motor_id != invalid_motor,
            motor_id == fault_motor);
        sample["module_id"] = motor_id;
        sample["sku_id"] = "rotary_dc_gearmotor_v0";
        actuators.push_back(std::move(sample));
    }
    if (reverse_arrays) {
        std::reverse(sensors.begin(), sensors.end());
        std::reverse(actuators.begin(), actuators.end());
    }
    return Json{
        {"reward_sensors", std::move(sensors)},
        {"actuator_feedback", std::move(actuators)},
    };
}

void set_all_motor_velocities(Json& observation, double velocity_rad_s) {
    for (Json& sample : observation.at("actuator_feedback")) {
        sample.at("feedback").at("velocity_rad_s") = velocity_rad_s;
    }
}

[[nodiscard]] double normalized_light(double lux) {
    return std::log1p(lux) / std::log1p(1000.0);
}

[[nodiscard]] double light_after_normalized_delta(
    double initial_lux,
    double normalized_delta) {
    return std::expm1(
        (normalized_light(initial_lux) + normalized_delta) *
        std::log1p(1000.0));
}

[[nodiscard]] double bias_for_effort(double effort, double effort_limit = 0.6) {
    return std::atanh(effort / effort_limit);
}

void test_sha_and_binary64() {
    require(
        droid::sha256_hex("") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 empty-string vector mismatch");
    require(
        droid::sha256_hex("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc vector mismatch");
    for (const double value : {0.0, -0.0, 0.02, -17.25, 1.0e100}) {
        const std::string encoded = droid::encode_binary64_hex(value);
        require(encoded.size() == 18, "binary64 encoding is not fixed width");
        require(
            std::bit_cast<std::uint64_t>(droid::decode_binary64_hex(encoded)) ==
                std::bit_cast<std::uint64_t>(value),
            "binary64 round trip changed bits");
    }
    require_throws(
        [] { (void)droid::decode_binary64_hex("0x3FF0000000000000"); },
        "uppercase binary64 encoding was accepted");
    require_throws(
        [] {
            (void)droid::encode_binary64_hex(
                std::numeric_limits<double>::infinity());
        },
        "non-finite binary64 value was accepted");
}

void test_artifact_integrity_and_compatibility() {
    const droid::PolicyArtifactData source = artifact_for();
    const Json document = droid::policy_artifact_json(source);
    require(
        source.training.algorithm_id == droid::kPolicyTrainingAlgorithmId &&
            document.at("training").at("algorithm_id") ==
                droid::kPolicyTrainingAlgorithmId,
        "policy provenance did not use the canonical training algorithm ID");
    const droid::PolicyArtifactData parsed =
        droid::parse_policy_artifact(document);
    require(parsed.self_sha256 == document.at("self_sha256"),
            "artifact self hash was not retained");
    require(
        droid::policy_artifact_json(parsed).dump() == document.dump(),
        "strict artifact round trip changed canonical JSON");
    droid::require_policy_compatible(parsed, source.compatibility);

    Json tampered = document;
    tampered["parameters"][0] = droid::encode_binary64_hex(7.0);
    require_throws(
        [&] { (void)droid::parse_policy_artifact(tampered); },
        "tampered artifact passed its self hash");
    Json unknown = document;
    unknown["unexpected"] = true;
    Json payload = unknown;
    payload.erase("self_sha256");
    unknown["self_sha256"] = droid::sha256_hex(payload.dump());
    require_throws(
        [&] { (void)droid::parse_policy_artifact(unknown); },
        "artifact with an unknown field was accepted");

    for (const std::string_view rejected_algorithm :
         {"cem_diagonal_v1", "unknown_training_algorithm_v9"}) {
        Json incompatible = document;
        incompatible["training"]["algorithm_id"] = rejected_algorithm;
        Json incompatible_payload = incompatible;
        incompatible_payload.erase("self_sha256");
        incompatible["self_sha256"] =
            droid::sha256_hex(incompatible_payload.dump());
        require_throws(
            [&] { (void)droid::parse_policy_artifact(incompatible); },
            "artifact with an unsupported training algorithm was accepted");
    }

    droid::PolicyCompatibility changed = source.compatibility;
    changed.control_dt_s = 0.04;
    require_throws(
        [&] { droid::require_policy_compatible(parsed, changed); },
        "control interval mismatch was accepted");
    changed = source.compatibility;
    changed.motor_ids.pop_back();
    require_throws(
        [&] { droid::require_policy_compatible(parsed, changed); },
        "topology mismatch was accepted");
    changed = source.compatibility;
    changed.model_sha256 = digest('a');
    require_throws(
        [&] { droid::require_policy_compatible(parsed, changed); },
        "model digest mismatch was accepted");
}

void test_cem_mask_artifact_eligibility() {
    const droid::PolicyArtifactData source = artifact_for();
    require(
        droid::policy_parameters_conform_to_cem_mask_v1(source.parameters),
        "canonical masked artifact fixture was ineligible");

    const Json document = droid::policy_artifact_json(source);
    const auto parse_rehashed = [](Json changed) {
        Json payload = changed;
        payload.erase("self_sha256");
        changed["self_sha256"] = droid::sha256_hex(payload.dump());
        return droid::parse_policy_artifact(changed);
    };

    Json nonzero_inactive = document;
    nonzero_inactive["parameters"][0] =
        droid::encode_binary64_hex(0.125);
    const droid::PolicyArtifactData parsed_nonzero =
        parse_rehashed(std::move(nonzero_inactive));
    require(
        !droid::policy_parameters_conform_to_cem_mask_v1(
            parsed_nonzero.parameters),
        "rehashed artifact with a nonzero inactive parameter was eligible");

    Json negative_zero_inactive = document;
    negative_zero_inactive["parameters"][1] =
        droid::encode_binary64_hex(-0.0);
    const droid::PolicyArtifactData parsed_negative_zero =
        parse_rehashed(std::move(negative_zero_inactive));
    require(
        !droid::policy_parameters_conform_to_cem_mask_v1(
            parsed_negative_zero.parameters),
        "rehashed artifact with negative-zero inactive parameter was eligible");

    std::array<double, droid::kPolicyFeatureCount> nonfinite_active =
        source.parameters;
    nonfinite_active[droid::kPolicyCemMaskV1ActiveParameterIndices.front()] =
        std::numeric_limits<double>::infinity();
    require(
        !droid::policy_parameters_conform_to_cem_mask_v1(nonfinite_active),
        "non-finite active parameter was eligible");
}

void test_v2_schema_and_parameter_shape() {
    require(droid::kPolicyArchitectureId == "shared_linear_memory_v2",
            "policy architecture ID is not version 2");
    require(droid::kPolicyEncoderId == "module_set_encoder_v2",
            "policy encoder ID is not version 2");
    require(droid::kPolicyFeatureCount == 33,
            "version 2 policy does not have 33 parameters");
    require(
        std::bit_cast<std::uint64_t>(droid::kPolicyAmbientDeltaScale) ==
            std::bit_cast<std::uint64_t>(8192.0),
        "ambient delta scale is not exact binary64 8192");

    const Json schema = droid::policy_feature_schema();
    require(
        schema.at("schema_version") ==
            "droid-blocks.policy-feature-schema.v2",
        "feature schema version is not v2");
    require(schema.at("architecture_id") == droid::kPolicyArchitectureId &&
                schema.at("encoder_id") == droid::kPolicyEncoderId,
            "feature schema does not bind the v2 architecture and encoder");
    require(schema.at("feature_count") == 33 &&
                schema.at("features").size() == 33,
            "feature schema did not publish all 33 features");
    require(
        schema.at("features").at(9).at("name") ==
            "ambient_mean_log_illuminance_delta_scaled" &&
            schema.at("features").at(31).at("name") ==
                "scaled_ambient_delta_x_previous_local_effort" &&
            schema.at("features").at(32).at("name") ==
                "scaled_ambient_delta_x_local_velocity_fraction",
        "feature schema indexes do not preserve v1 order plus v2 terminals");
    require(
        schema.at("normalization")
                .at("ambient_light_delta")
                .at("scale") == droid::kPolicyAmbientDeltaScale,
        "feature schema omitted the executable ambient delta scale");
    require(
        schema.at("interaction_features").contains(
            "scaled_ambient_delta_x_previous_local_effort") &&
            schema.at("interaction_features").contains(
                "scaled_ambient_delta_x_local_velocity_fraction"),
        "feature schema omitted v2 interaction semantics");

    const Json artifact = droid::policy_artifact_json(artifact_for());
    require(artifact.at("architecture").at("parameter_count") == 33 &&
                artifact.at("parameters").size() == 33,
            "serialized artifact did not use the v2 parameter shape");
    require(
        artifact.at("architecture").at("id") ==
                "shared_linear_memory_v2" &&
            artifact.at("architecture").at("encoder_id") ==
                "module_set_encoder_v2",
        "serialized artifact did not bind v2 architecture IDs");

    Json mislabeled_v1 = artifact;
    mislabeled_v1["architecture"]["id"] = "shared_linear_memory_v1";
    Json payload = mislabeled_v1;
    payload.erase("self_sha256");
    mislabeled_v1["self_sha256"] = droid::sha256_hex(payload.dump());
    require_throws(
        [&] { (void)droid::parse_policy_artifact(mislabeled_v1); },
        "v1 architecture label was accepted as a v2 artifact");
}

void test_scaled_ambient_delta() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    constexpr double initial_lux = 100.0;
    constexpr double unit_delta = 1.0 / droid::kPolicyAmbientDeltaScale;

    droid::PolicyArtifactData artifact = artifact_for(motors);
    artifact.parameters.fill(0.0);
    artifact.parameters[9] = 1.0;
    artifact.slew_limit = 1.2;
    droid::SharedLinearMemoryPolicy policy(artifact);

    Json initial = observation_for(motors, false, "sensor", {}, {}, {}, true,
                                   initial_lux);
    set_all_motor_velocities(initial, 0.0);
    require(policy.act(initial, motors).at(motors.front()) == 0.0,
            "reset observation fabricated a light delta");

    Json half_step = observation_for(
        motors,
        false,
        "sensor",
        {},
        {},
        {},
        true,
        light_after_normalized_delta(initial_lux, 0.5 * unit_delta));
    set_all_motor_velocities(half_step, 0.0);
    require_close(
        policy.act(half_step, motors).at(motors.front()),
        0.6 * std::tanh(0.5),
        1e-11,
        "ambient delta did not receive the exact 8192 scale");

    policy.reset(motors);
    (void)policy.act(initial, motors);
    Json clipped = observation_for(
        motors,
        false,
        "sensor",
        {},
        {},
        {},
        true,
        light_after_normalized_delta(initial_lux, 2.0 * unit_delta));
    set_all_motor_velocities(clipped, 0.0);
    require_close(
        policy.act(clipped, motors).at(motors.front()),
        0.6 * std::tanh(1.0),
        1e-11,
        "positive scaled ambient delta was not clipped at one");

    policy.reset(motors);
    (void)policy.act(initial, motors);
    Json negative = observation_for(
        motors,
        false,
        "sensor",
        {},
        {},
        {},
        true,
        light_after_normalized_delta(initial_lux, -2.0 * unit_delta));
    set_all_motor_velocities(negative, 0.0);
    require_close(
        policy.act(negative, motors).at(motors.front()),
        -0.6 * std::tanh(1.0),
        1e-11,
        "negative scaled ambient delta was not clipped at minus one");
}

void test_directional_interaction_truth_tables() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    constexpr double initial_lux = 100.0;
    constexpr double unit_delta = 1.0 / droid::kPolicyAmbientDeltaScale;

    const auto command_case = [&](double previous_effort, double delta_sign) {
        droid::PolicyArtifactData artifact = artifact_for(motors);
        artifact.parameters.fill(0.0);
        artifact.parameters[30] = bias_for_effort(previous_effort);
        artifact.parameters[31] = 10.0;
        artifact.slew_limit = 1.2;
        droid::SharedLinearMemoryPolicy policy(artifact);
        Json initial = observation_for(
            motors, false, "sensor", {}, {}, {}, true, initial_lux);
        set_all_motor_velocities(initial, 0.0);
        require_close(
            policy.act(initial, motors).at(motors.front()),
            previous_effort,
            1e-12,
            "truth-table fixture did not establish previous effort");
        Json changed = observation_for(
            motors,
            false,
            "sensor",
            {},
            {},
            {},
            true,
            light_after_normalized_delta(
                initial_lux, delta_sign * unit_delta));
        set_all_motor_velocities(changed, 0.0);
        return policy.act(changed, motors).at(motors.front());
    };

    require(command_case(+0.4, +1.0) > 0.0 &&
                command_case(+0.4, -1.0) < 0.0 &&
                command_case(-0.4, +1.0) < 0.0 &&
                command_case(-0.4, -1.0) > 0.0,
            "delta-times-command feature failed its four-corner truth table");

    const auto motion_case = [&](double velocity_rad_s, double delta_sign) {
        droid::PolicyArtifactData artifact = artifact_for(motors);
        artifact.parameters.fill(0.0);
        artifact.parameters[32] = 10.0;
        artifact.slew_limit = 1.2;
        droid::SharedLinearMemoryPolicy policy(artifact);
        Json initial = observation_for(
            motors, false, "sensor", {}, {}, {}, true, initial_lux);
        set_all_motor_velocities(initial, 0.0);
        (void)policy.act(initial, motors);
        Json changed = observation_for(
            motors,
            false,
            "sensor",
            {},
            {},
            {},
            true,
            light_after_normalized_delta(
                initial_lux, delta_sign * unit_delta));
        set_all_motor_velocities(changed, velocity_rad_s);
        return policy.act(changed, motors).at(motors.front());
    };

    require(motion_case(+5.0, +1.0) > 0.0 &&
                motion_case(+5.0, -1.0) < 0.0 &&
                motion_case(-5.0, +1.0) < 0.0 &&
                motion_case(-5.0, -1.0) > 0.0,
            "delta-times-velocity feature failed its four-corner truth table");
    require(motion_case(0.0, +1.0) == 0.0,
            "motion interaction fabricated evidence at zero velocity");
    require(command_case(+0.4, +1.0) > 0.0,
            "command interaction was unavailable at zero velocity");
}

void test_motion_interaction_can_resolve_reversal_inertia() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    constexpr double initial_lux = 100.0;
    constexpr double unit_delta = 1.0 / droid::kPolicyAmbientDeltaScale;
    droid::PolicyArtifactData artifact = artifact_for(motors);
    artifact.parameters.fill(0.0);
    artifact.parameters[30] = bias_for_effort(0.4);
    artifact.parameters[31] = 1.0;
    artifact.parameters[32] = 8.0;
    artifact.slew_limit = 1.2;
    droid::SharedLinearMemoryPolicy policy(artifact);

    Json initial = observation_for(
        motors, false, "sensor", {}, {}, {}, true, initial_lux);
    set_all_motor_velocities(initial, 0.0);
    require_close(
        policy.act(initial, motors).at(motors.front()),
        0.4,
        1e-12,
        "inertia fixture did not establish its probe command");

    const double first_lower_lux =
        light_after_normalized_delta(initial_lux, -unit_delta);
    Json first_lower = observation_for(
        motors, false, "sensor", {}, {}, {}, true, first_lower_lux);
    set_all_motor_velocities(first_lower, +5.0);
    const double reversed = policy.act(first_lower, motors).at(motors.front());
    require(reversed < 0.0,
            "worsening light did not reverse the probe command");

    Json still_lower = observation_for(
        motors,
        false,
        "sensor",
        {},
        {},
        {},
        true,
        light_after_normalized_delta(first_lower_lux, -unit_delta));
    set_all_motor_velocities(still_lower, +5.0);
    require(policy.act(still_lower, motors).at(motors.front()) < 0.0,
            "old-sign shaft velocity could not outweigh the newly reversed "
            "command during inertia");
}

void test_ambient_delta_requires_consecutive_valid_samples() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    constexpr double initial_lux = 100.0;
    constexpr double unit_delta = 1.0 / droid::kPolicyAmbientDeltaScale;
    droid::PolicyArtifactData artifact = artifact_for(motors);
    artifact.parameters.fill(0.0);
    artifact.parameters[9] = 5.0;
    artifact.parameters[30] = bias_for_effort(0.2);
    artifact.parameters[31] = 5.0;
    artifact.parameters[32] = 5.0;
    artifact.slew_limit = 1.2;
    droid::SharedLinearMemoryPolicy policy(artifact);

    Json valid = observation_for(
        motors, false, "sensor", {}, {}, {}, true, initial_lux);
    set_all_motor_velocities(valid, +5.0);
    require_close(
        policy.act(valid, motors).at(motors.front()),
        0.2,
        1e-12,
        "validity fixture did not establish its bias effort");

    Json invalid = observation_for(
        motors, false, "sensor", {}, {}, {}, true, 900.0);
    invalid.at("reward_sensors").at(0).at("valid") = false;
    set_all_motor_velocities(invalid, +5.0);
    require_close(
        policy.act(invalid, motors).at(motors.front()),
        0.2,
        1e-12,
        "invalid ambient sample fabricated a gradient while touch remained valid");

    Json recovered = observation_for(
        motors, false, "sensor", {}, {}, {}, true, 200.0);
    set_all_motor_velocities(recovered, +5.0);
    require_close(
        policy.act(recovered, motors).at(motors.front()),
        0.2,
        1e-12,
        "first ambient sample after an invalid gap bridged the gap");

    Json consecutive = observation_for(
        motors,
        false,
        "sensor",
        {},
        {},
        {},
        true,
        light_after_normalized_delta(200.0, unit_delta));
    set_all_motor_velocities(consecutive, +5.0);
    require(
        policy.act(consecutive, motors).at(motors.front()) > 0.5,
        "consecutive valid ambient samples did not restore gradient evidence");

    policy.reset(motors);
    (void)policy.act(valid, motors);
    Json missing = observation_for(
        motors, false, "sensor", {}, {}, {}, true, 900.0);
    missing.at("reward_sensors").erase(
        missing.at("reward_sensors").begin());
    set_all_motor_velocities(missing, +5.0);
    require_close(
        policy.act(missing, motors).at(motors.front()),
        0.2,
        1e-12,
        "missing ambient sample fabricated a gradient while touch remained valid");
    require_close(
        policy.act(recovered, motors).at(motors.front()),
        0.2,
        1e-12,
        "first ambient sample after a missing gap bridged the gap");
}

void test_exclusive_artifact_save() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("droid-policy-test-" + std::to_string(suffix));
    const std::filesystem::path path = directory / "policy.json";
    std::filesystem::create_directory(directory);
    try {
        const droid::PolicyArtifactData source = artifact_for();
        droid::save_policy_artifact_exclusive(path, source);
        const droid::PolicyArtifactData loaded = droid::load_policy_artifact(path);
        require(
            droid::policy_artifact_json(loaded).dump() ==
                droid::policy_artifact_json(source).dump(),
            "saved artifact did not load exactly");
        require_throws(
            [&] { droid::save_policy_artifact_exclusive(path, source); },
            "artifact save overwrote an existing path");
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        std::filesystem::remove(directory, ignored);
        throw;
    }
    std::filesystem::remove(path);
    std::filesystem::remove(directory);
}

void test_encoder_order_and_id_invariance() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    droid::PolicyArtifactData local_feedback_policy = artifact_for(motors);
    local_feedback_policy.parameters.fill(0.0);
    local_feedback_policy.parameters[19] = 0.75;
    local_feedback_policy.parameters[20] = -0.35;
    local_feedback_policy.parameters[25] = 0.5;
    local_feedback_policy.slew_limit = 0.6;
    droid::SharedLinearMemoryPolicy first(local_feedback_policy);
    droid::SharedLinearMemoryPolicy reordered(local_feedback_policy);
    std::vector<std::string> reverse_motors = motors;
    std::reverse(reverse_motors.begin(), reverse_motors.end());
    first.reset(motors);
    reordered.reset(reverse_motors);
    const auto expected = first.act(
        observation_for(motors, false, "alpha"), motors);
    const auto actual = reordered.act(
        observation_for(motors, true, "renamed-sensor"), reverse_motors);
    require(expected == actual,
            "array order or reward-sensor IDs changed policy actions");
    require(expected.at(motors.front()) != expected.at(motors.back()),
            "order-invariance fixture did not exercise local motor feedback");

    const std::vector<std::string> renamed{
        "opaque-a", "opaque-b", "opaque-c", "opaque-d"};
    droid::PolicyArtifactData renamed_artifact = local_feedback_policy;
    renamed_artifact.compatibility.motor_ids = renamed;
    droid::SharedLinearMemoryPolicy renamed_policy(renamed_artifact);
    renamed_policy.reset(renamed);
    const auto renamed_actions = renamed_policy.act(
        observation_for(renamed, true, "completely-different"), renamed);
    for (std::size_t index = 0; index < motors.size(); ++index) {
        require(
            expected.at(motors[index]) == renamed_actions.at(renamed[index]),
            "opaque motor ID renaming changed corresponding action");
    }
}

void test_memory_limits_and_fail_closed_behavior() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    droid::PolicyArtifactData biased = artifact_for(motors);
    biased.parameters.fill(0.0);
    biased.parameters[30] = 100.0;
    droid::SharedLinearMemoryPolicy policy(biased);
    const Json observation = observation_for(motors);
    const auto first = policy.act(observation, motors);
    const auto second = policy.act(observation, motors);
    require(first.at(motors.front()) == 0.1,
            "first action did not obey slew limit");
    require(second.at(motors.front()) == 0.2,
            "one-step effort memory did not advance slew limit");
    for (int index = 0; index < 10; ++index) {
        (void)policy.act(observation, motors);
    }
    const auto capped = policy.act(observation, motors);
    require(capped.at(motors.front()) <= 0.6,
            "policy exceeded frozen effort cap");

    policy.reset(motors);
    const auto no_sensors = policy.act(
        observation_for(motors, false, "sensor", {}, {}, {}, false), motors);
    require(
        std::all_of(no_sensors.begin(), no_sensors.end(), [](const auto& entry) {
            return entry.second == 0.0;
        }),
        "policy moved without a valid reward sensor");

    policy.reset(motors);
    const auto missing = policy.act(
        observation_for(motors, false, "sensor", motors[1]), motors);
    require(missing.at(motors[1]) == 0.0,
            "missing actuator feedback did not force zero effort");
    require(missing.at(motors[0]) != 0.0,
            "one missing actuator unnecessarily disabled a healthy actuator");

    policy.reset(motors);
    const auto invalid = policy.act(
        observation_for(motors, false, "sensor", {}, motors[2]), motors);
    require(invalid.at(motors[2]) == 0.0,
            "invalid actuator feedback did not force zero effort");
    policy.reset(motors);
    const auto faulted = policy.act(
        observation_for(motors, false, "sensor", {}, {}, motors[3]), motors);
    require(faulted.at(motors[3]) == 0.0,
            "actuator fault did not force zero effort");
}

void test_strict_observation_contract() {
    const std::vector<std::string> motors{
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
    droid::SharedLinearMemoryPolicy policy(artifact_for(motors));
    Json extra_top_level = observation_for(motors);
    extra_top_level["world_position"] = Json::array({1.0, 2.0, 3.0});
    require_throws(
        [&] { (void)policy.act(extra_top_level, motors); },
        "privileged/unknown top-level observation field was accepted");
    Json unknown_family = observation_for(motors);
    unknown_family["reward_sensors"][0]["family_id"] = "future_sensor_v9";
    require_throws(
        [&] { (void)policy.act(unknown_family, motors); },
        "untrained reward-sensor family was silently ignored");
    Json duplicate = observation_for(motors);
    duplicate["reward_sensors"].push_back(duplicate["reward_sensors"][0]);
    require_throws(
        [&] { (void)policy.act(duplicate, motors); },
        "duplicate reward-sensor ID was accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string_view, std::function<void()>>> tests{
        {"SHA-256 and binary64 vectors", test_sha_and_binary64},
        {"artifact integrity and compatibility",
         test_artifact_integrity_and_compatibility},
        {"CEM mask artifact eligibility", test_cem_mask_artifact_eligibility},
        {"v2 schema and parameter shape", test_v2_schema_and_parameter_shape},
        {"exclusive artifact save", test_exclusive_artifact_save},
        {"encoder order and ID invariance", test_encoder_order_and_id_invariance},
        {"scaled ambient delta", test_scaled_ambient_delta},
        {"directional interaction truth tables",
         test_directional_interaction_truth_tables},
        {"motion interaction reversal inertia",
         test_motion_interaction_can_resolve_reversal_inertia},
        {"ambient delta consecutive-validity gate",
         test_ambient_delta_requires_consecutive_valid_samples},
        {"memory, limits, and fail-closed behavior",
         test_memory_limits_and_fail_closed_behavior},
        {"strict observation contract", test_strict_observation_contract},
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            ++passed;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " native policy tests passed\n";
    return 0;
}
