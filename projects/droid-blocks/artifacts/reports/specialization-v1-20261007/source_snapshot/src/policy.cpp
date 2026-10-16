#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace droid {
namespace {

using Json = nlohmann::json;

constexpr double kLightSaturationLux{1000.0};
constexpr double kTouchHardForceN{25.0};
constexpr double kMaximumMotorSpeedRadS{20.0};
constexpr double kPeakMotorCurrentA{2.5};
constexpr double kMinimumBusVoltageV{6.0};
constexpr double kNominalBusVoltageV{7.4};
constexpr double kReferenceAmbientC{25.0};
constexpr double kThermalCutoffC{80.0};
constexpr double kPeakOutputTorqueNm{1.6875};
constexpr double kImpedanceScaleNmSPerRad{16.875};

[[nodiscard]] double clamp(double value, double lower, double upper) noexcept {
    return std::max(lower, std::min(value, upper));
}

void require_exact_keys(
    const Json& object,
    std::initializer_list<std::string_view> expected,
    std::string_view context) {
    if (!object.is_object()) {
        throw std::invalid_argument(std::string(context) + " must be an object");
    }
    std::set<std::string> keys;
    for (const std::string_view key : expected) {
        keys.emplace(key);
    }
    if (object.size() != keys.size()) {
        throw std::invalid_argument(
            std::string(context) + " has missing or unknown fields");
    }
    for (const auto& [key, unused] : object.items()) {
        (void)unused;
        if (!keys.contains(key)) {
            throw std::invalid_argument(
                std::string(context) + " has unknown field '" + key + "'");
        }
    }
}

[[nodiscard]] std::string required_string(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.contains(key) || !object.at(key).is_string() ||
        object.at(key).get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " must be a non-empty string");
    }
    return object.at(key).get<std::string>();
}

[[nodiscard]] bool required_bool(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.contains(key) || !object.at(key).is_boolean()) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " must be Boolean");
    }
    return object.at(key).get<bool>();
}

[[nodiscard]] double required_number(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " must be numeric");
    }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " must be finite");
    }
    return value;
}

[[nodiscard]] std::size_t required_size(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.contains(key) || !object.at(key).is_number_unsigned()) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " must be unsigned");
    }
    const auto value = object.at(key).get<std::uint64_t>();
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint64_t required_u64(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.contains(key) || !object.at(key).is_number_unsigned()) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " must be unsigned");
    }
    return object.at(key).get<std::uint64_t>();
}

[[nodiscard]] bool valid_sha256(std::string_view digest) noexcept {
    if (digest.size() != 64) {
        return false;
    }
    return std::all_of(digest.begin(), digest.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
    });
}

void require_sha256(std::string_view digest, std::string_view context) {
    if (!valid_sha256(digest)) {
        throw std::invalid_argument(
            std::string(context) + " must be 64 lowercase hexadecimal digits");
    }
}

[[nodiscard]] bool same_bits(double first, double second) noexcept {
    return std::bit_cast<std::uint64_t>(first) ==
        std::bit_cast<std::uint64_t>(second);
}

class Sha256 final {
public:
    [[nodiscard]] static std::string digest(std::string_view input) {
        std::vector<std::uint8_t> bytes(input.begin(), input.end());
        const std::uint64_t bit_length =
            static_cast<std::uint64_t>(bytes.size()) * 8U;
        bytes.push_back(0x80U);
        while (bytes.size() % 64U != 56U) {
            bytes.push_back(0U);
        }
        for (int shift = 56; shift >= 0; shift -= 8) {
            bytes.push_back(static_cast<std::uint8_t>(bit_length >> shift));
        }

        std::array<std::uint32_t, 8> state{
            0x6a09e667U,
            0xbb67ae85U,
            0x3c6ef372U,
            0xa54ff53aU,
            0x510e527fU,
            0x9b05688cU,
            0x1f83d9abU,
            0x5be0cd19U,
        };
        for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
            transform(bytes.data() + offset, state);
        }

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t word : state) {
            output << std::setw(8) << word;
        }
        return output.str();
    }

private:
    [[nodiscard]] static constexpr std::uint32_t rotate_right(
        std::uint32_t value,
        unsigned int shift) noexcept {
        return (value >> shift) | (value << (32U - shift));
    }

    static void transform(
        const std::uint8_t* block,
        std::array<std::uint32_t, 8>& state) {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };

        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const std::size_t base = index * 4U;
            schedule[index] =
                (static_cast<std::uint32_t>(block[base]) << 24U) |
                (static_cast<std::uint32_t>(block[base + 1U]) << 16U) |
                (static_cast<std::uint32_t>(block[base + 2U]) << 8U) |
                static_cast<std::uint32_t>(block[base + 3U]);
        }
        for (std::size_t index = 16U; index < schedule.size(); ++index) {
            const std::uint32_t s0 =
                rotate_right(schedule[index - 15U], 7U) ^
                rotate_right(schedule[index - 15U], 18U) ^
                (schedule[index - 15U] >> 3U);
            const std::uint32_t s1 =
                rotate_right(schedule[index - 2U], 17U) ^
                rotate_right(schedule[index - 2U], 19U) ^
                (schedule[index - 2U] >> 10U);
            schedule[index] = schedule[index - 16U] + s0 +
                schedule[index - 7U] + s1;
        }

        std::uint32_t a = state[0];
        std::uint32_t b = state[1];
        std::uint32_t c = state[2];
        std::uint32_t d = state[3];
        std::uint32_t e = state[4];
        std::uint32_t f = state[5];
        std::uint32_t g = state[6];
        std::uint32_t h = state[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const std::uint32_t s1 =
                rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                rotate_right(e, 25U);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 =
                h + s1 + choice + constants[index] + schedule[index];
            const std::uint32_t s0 =
                rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
};

[[nodiscard]] std::vector<std::string> normalized_motor_ids(
    std::span<const std::string> motor_ids) {
    std::vector<std::string> normalized(motor_ids.begin(), motor_ids.end());
    if (normalized.empty() ||
        std::any_of(normalized.begin(), normalized.end(), [](const auto& id) {
            return id.empty();
        })) {
        throw std::invalid_argument("policy motor IDs must be non-empty");
    }
    std::sort(normalized.begin(), normalized.end());
    if (std::adjacent_find(normalized.begin(), normalized.end()) !=
        normalized.end()) {
        throw std::invalid_argument("policy motor IDs must be unique");
    }
    return normalized;
}

[[nodiscard]] double sorted_mean(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    long double sum = 0.0L;
    long double correction = 0.0L;
    for (const double value : values) {
        const long double converted = value;
        const long double tentative = sum + converted;
        if (std::abs(sum) >= std::abs(converted)) {
            correction += (sum - tentative) + converted;
        } else {
            correction += (converted - tentative) + sum;
        }
        sum = tentative;
    }
    return static_cast<double>(
        (sum + correction) / static_cast<long double>(values.size()));
}

[[nodiscard]] double maximum_or_zero(const std::vector<double>& values) {
    return values.empty() ? 0.0 :
        *std::max_element(values.begin(), values.end());
}

struct SensorEncoding {
    std::array<double, 9> features{};
    std::array<double, 3> temporal{};
    bool any_valid{false};
    bool ambient_valid{false};
};

[[nodiscard]] SensorEncoding encode_reward_sensors(const Json& samples) {
    if (!samples.is_array()) {
        throw std::invalid_argument(
            "policy observation.reward_sensors must be an array");
    }
    std::set<std::string> module_ids;
    std::size_t ambient_count = 0;
    std::size_t ambient_valid = 0;
    std::vector<double> ambient_light;
    std::vector<double> ambient_saturated;
    std::size_t touch_count = 0;
    std::size_t touch_valid = 0;
    std::vector<double> touch_contact;
    std::vector<double> touch_force;

    for (const Json& sample : samples) {
        require_exact_keys(
            sample,
            {"module_id", "family_id", "valid", "observations"},
            "reward sensor sample");
        const std::string module_id = required_string(
            sample, "module_id", "reward sensor sample");
        if (!module_ids.emplace(module_id).second) {
            throw std::invalid_argument(
                "duplicate reward sensor module_id '" + module_id + "'");
        }
        const std::string family_id = required_string(
            sample, "family_id", "reward sensor sample");
        const bool valid = required_bool(sample, "valid", "reward sensor sample");
        const Json& observations = sample.at("observations");

        if (family_id == "ambient_light_v0") {
            ++ambient_count;
            require_exact_keys(
                observations,
                {"illuminance_lux", "saturated"},
                "ambient_light_v0 observations");
            const double lux = required_number(
                observations, "illuminance_lux", "ambient_light_v0 observations");
            if (lux < 0.0) {
                throw std::invalid_argument(
                    "ambient_light_v0 illuminance_lux must be nonnegative");
            }
            const bool saturated = required_bool(
                observations, "saturated", "ambient_light_v0 observations");
            if (valid) {
                ++ambient_valid;
                ambient_light.push_back(clamp(
                    std::log1p(lux) / std::log1p(kLightSaturationLux),
                    0.0,
                    1.0));
                ambient_saturated.push_back(saturated ? 1.0 : 0.0);
            }
        } else if (family_id == "touch_force_v0") {
            ++touch_count;
            require_exact_keys(
                observations,
                {"contact", "normal_force_n"},
                "touch_force_v0 observations");
            const bool contact = required_bool(
                observations, "contact", "touch_force_v0 observations");
            const double force = required_number(
                observations, "normal_force_n", "touch_force_v0 observations");
            if (force < 0.0) {
                throw std::invalid_argument(
                    "touch_force_v0 normal_force_n must be nonnegative");
            }
            if (valid) {
                ++touch_valid;
                touch_contact.push_back(contact ? 1.0 : 0.0);
                touch_force.push_back(clamp(force / kTouchHardForceN, 0.0, 1.0));
            }
        } else {
            throw std::invalid_argument(
                "unsupported reward sensor family '" + family_id + "'");
        }
    }

    SensorEncoding result;
    result.features[0] = ambient_count == 0 ? 0.0 :
        static_cast<double>(ambient_count) /
            static_cast<double>(ambient_count + 1U);
    result.features[1] = ambient_count == 0 ? 0.0 :
        static_cast<double>(ambient_valid) / static_cast<double>(ambient_count);
    result.features[2] = sorted_mean(ambient_light);
    result.features[3] = sorted_mean(ambient_saturated);
    result.features[4] = touch_count == 0 ? 0.0 :
        static_cast<double>(touch_count) /
            static_cast<double>(touch_count + 1U);
    result.features[5] = touch_count == 0 ? 0.0 :
        static_cast<double>(touch_valid) / static_cast<double>(touch_count);
    result.features[6] = sorted_mean(touch_contact);
    result.features[7] = sorted_mean(touch_force);
    result.features[8] = maximum_or_zero(touch_force);
    result.temporal = {
        result.features[2], result.features[6], result.features[7]};
    result.any_valid = ambient_valid + touch_valid > 0;
    result.ambient_valid = ambient_valid > 0;
    return result;
}

struct ActuatorEncoding {
    bool present{false};
    bool valid{false};
    bool fault{false};
    bool stuck{false};
    double position{0.0};
    double velocity{0.0};
    double current{0.0};
    double bus_voltage{0.0};
    double temperature{0.0};
    double torque{0.0};
    double impedance{0.0};
    double stuck_score{0.0};
};

[[nodiscard]] std::map<std::string, ActuatorEncoding> encode_actuators(
    const Json& samples,
    const std::vector<std::string>& motor_ids) {
    if (!samples.is_array()) {
        throw std::invalid_argument(
            "policy observation.actuator_feedback must be an array");
    }
    const std::set<std::string> expected(motor_ids.begin(), motor_ids.end());
    std::map<std::string, ActuatorEncoding> result;
    for (const Json& sample : samples) {
        require_exact_keys(
            sample,
            {"module_id", "sku_id", "valid", "feedback"},
            "actuator feedback sample");
        const std::string module_id = required_string(
            sample, "module_id", "actuator feedback sample");
        if (!expected.contains(module_id)) {
            throw std::invalid_argument(
                "actuator feedback has unknown module_id '" + module_id + "'");
        }
        if (result.contains(module_id)) {
            throw std::invalid_argument(
                "duplicate actuator feedback module_id '" + module_id + "'");
        }
        const std::string sku_id = required_string(
            sample, "sku_id", "actuator feedback sample");
        if (sku_id != kPolicyReferenceMotorSku) {
            throw std::invalid_argument(
                "unsupported actuator feedback sku_id '" + sku_id + "'");
        }
        const bool valid = required_bool(
            sample, "valid", "actuator feedback sample");
        const Json& feedback = sample.at("feedback");
        require_exact_keys(
            feedback,
            {"position_rad",
             "velocity_rad_s",
             "current_a",
             "bus_voltage_v",
             "temperature_c",
             "output_torque_nm",
             "load_impedance_nm_s_per_rad",
             "stuck_score",
             "stuck",
             "fault_flags"},
            "actuator feedback fields");
        if (!feedback.at("fault_flags").is_array()) {
            throw std::invalid_argument(
                "actuator feedback fields.fault_flags must be an array");
        }
        bool fault = false;
        std::set<std::string> fault_flags;
        for (const Json& flag : feedback.at("fault_flags")) {
            if (!flag.is_string() || flag.get_ref<const std::string&>().empty()) {
                throw std::invalid_argument(
                    "actuator fault flags must be non-empty strings");
            }
            if (!fault_flags.emplace(flag.get<std::string>()).second) {
                throw std::invalid_argument("actuator fault flags must be unique");
            }
            fault = true;
        }

        const double position = required_number(
            feedback, "position_rad", "actuator feedback fields");
        const double velocity = required_number(
            feedback, "velocity_rad_s", "actuator feedback fields");
        const double current = required_number(
            feedback, "current_a", "actuator feedback fields");
        const double bus_voltage = required_number(
            feedback, "bus_voltage_v", "actuator feedback fields");
        const double temperature = required_number(
            feedback, "temperature_c", "actuator feedback fields");
        const double torque = required_number(
            feedback, "output_torque_nm", "actuator feedback fields");
        const double impedance = required_number(
            feedback,
            "load_impedance_nm_s_per_rad",
            "actuator feedback fields");
        const double stuck_score = required_number(
            feedback, "stuck_score", "actuator feedback fields");
        const bool stuck = required_bool(
            feedback, "stuck", "actuator feedback fields");
        if (impedance < 0.0 || stuck_score < 0.0 || stuck_score > 1.0) {
            throw std::invalid_argument(
                "actuator impedance/stuck_score lies outside its contract");
        }

        result.emplace(
            module_id,
            ActuatorEncoding{
                .present = true,
                .valid = valid,
                .fault = fault,
                .stuck = stuck,
                .position = clamp(position / std::numbers::pi, -1.0, 1.0),
                .velocity = clamp(velocity / kMaximumMotorSpeedRadS, -1.0, 1.0),
                .current = clamp(current / kPeakMotorCurrentA, -1.0, 1.0),
                .bus_voltage = clamp(
                    2.0 * (bus_voltage - kMinimumBusVoltageV) /
                            (kNominalBusVoltageV - kMinimumBusVoltageV) -
                        1.0,
                    -1.0,
                    1.0),
                .temperature = clamp(
                    (temperature - kReferenceAmbientC) /
                        (kThermalCutoffC - kReferenceAmbientC),
                    0.0,
                    1.0),
                .torque = clamp(torque / kPeakOutputTorqueNm, -1.0, 1.0),
                .impedance = clamp(
                    std::log1p(impedance) /
                        std::log1p(kImpedanceScaleNmSPerRad),
                    0.0,
                    1.0),
                .stuck_score = stuck_score,
            });
    }
    for (const std::string& motor_id : motor_ids) {
        result.try_emplace(motor_id);
    }
    return result;
}

[[nodiscard]] std::array<double, 6> pooled_actuator_features(
    const std::map<std::string, ActuatorEncoding>& actuators,
    std::size_t motor_count) {
    std::vector<double> velocities;
    std::vector<double> absolute_velocities;
    std::vector<double> currents;
    std::vector<double> temperatures;
    std::vector<double> stuck_scores;
    std::size_t fault_or_invalid = 0;
    for (const auto& [unused, actuator] : actuators) {
        (void)unused;
        if (!actuator.present || !actuator.valid || actuator.fault) {
            ++fault_or_invalid;
        }
        if (!actuator.present || !actuator.valid) {
            continue;
        }
        velocities.push_back(actuator.velocity);
        absolute_velocities.push_back(std::abs(actuator.velocity));
        currents.push_back(actuator.current);
        temperatures.push_back(actuator.temperature);
        stuck_scores.push_back(actuator.stuck_score);
    }
    return {
        sorted_mean(velocities),
        sorted_mean(absolute_velocities),
        sorted_mean(currents),
        maximum_or_zero(temperatures),
        sorted_mean(stuck_scores),
        motor_count == 0 ? 0.0 :
            static_cast<double>(fault_or_invalid) /
                static_cast<double>(motor_count),
    };
}

[[nodiscard]] Json artifact_payload_json(const PolicyArtifactData& artifact) {
    Json parameters = Json::array();
    for (const double value : artifact.parameters) {
        parameters.push_back(encode_binary64_hex(value));
    }
    Json motor_ids = artifact.compatibility.motor_ids;
    return Json{
        {"schema_version", kPolicyArtifactSchemaVersion},
        {"architecture",
         Json{
             {"id", kPolicyArchitectureId},
             {"encoder_id", kPolicyEncoderId},
             {"activation_id", kPolicyActivationId},
             {"parameter_encoding", kPolicyParameterEncoding},
             {"parameter_count", kPolicyFeatureCount},
             {"effort_limit", encode_binary64_hex(artifact.effort_limit)},
             {"slew_limit", encode_binary64_hex(artifact.slew_limit)},
         }},
        {"compatibility",
         Json{
             {"environment_api_version",
              artifact.compatibility.environment_api_version},
             {"assembly_id", artifact.compatibility.assembly_id},
             {"motor_ids", std::move(motor_ids)},
             {"control_dt_s",
              encode_binary64_hex(artifact.compatibility.control_dt_s)},
             {"model_sha256", artifact.compatibility.model_sha256},
             {"module_catalog_sha256",
              artifact.compatibility.module_catalog_sha256},
             {"environment_spec_sha256",
              artifact.compatibility.environment_spec_sha256},
             {"feature_schema_sha256",
              artifact.compatibility.feature_schema_sha256},
         }},
        {"parameters", std::move(parameters)},
        {"training",
         Json{
             {"algorithm_id", artifact.training.algorithm_id},
             {"scenario_profile_id", artifact.training.scenario_profile_id},
             {"seed_derivation_id", artifact.training.seed_derivation_id},
             {"protocol_manifest_sha256",
              artifact.training.protocol_manifest_sha256},
             {"seed_split_base_seed", artifact.training.seed_split_base_seed},
             {"optimizer_seed", artifact.training.optimizer_seed},
             {"generations", artifact.training.generations},
             {"population_size", artifact.training.population_size},
             {"elite_count", artifact.training.elite_count},
             {"train_batch_size", artifact.training.train_batch_size},
             {"max_control_steps", artifact.training.max_control_steps},
             {"initial_stddev",
              encode_binary64_hex(artifact.training.initial_stddev)},
             {"minimum_stddev",
              encode_binary64_hex(artifact.training.minimum_stddev)},
             {"update_rate", encode_binary64_hex(artifact.training.update_rate)},
             {"selected_generation", artifact.training.selected_generation},
             {"train_seed_count", artifact.training.train_seed_count},
             {"validation_seed_count",
              artifact.training.validation_seed_count},
             {"test_seed_count", artifact.training.test_seed_count},
             {"train_seeds_sha256", artifact.training.train_seeds_sha256},
             {"validation_seeds_sha256",
              artifact.training.validation_seeds_sha256},
             {"test_seeds_sha256", artifact.training.test_seeds_sha256},
         }},
    };
}

void validate_compatibility_fields(const PolicyCompatibility& value) {
    if (value.environment_api_version.empty() || value.assembly_id.empty()) {
        throw std::invalid_argument(
            "policy compatibility identifiers must be non-empty");
    }
    (void)normalized_motor_ids(value.motor_ids);
    if (!std::isfinite(value.control_dt_s) || value.control_dt_s <= 0.0) {
        throw std::invalid_argument(
            "policy compatibility control_dt_s must be finite and positive");
    }
    require_sha256(value.model_sha256, "compatibility.model_sha256");
    require_sha256(
        value.module_catalog_sha256,
        "compatibility.module_catalog_sha256");
    require_sha256(
        value.environment_spec_sha256,
        "compatibility.environment_spec_sha256");
    require_sha256(
        value.feature_schema_sha256,
        "compatibility.feature_schema_sha256");
    if (value.feature_schema_sha256 != policy_feature_schema_sha256()) {
        throw std::invalid_argument(
            "policy compatibility feature schema digest is not executable v2");
    }
}

void validate_artifact_data(const PolicyArtifactData& artifact) {
    validate_compatibility_fields(artifact.compatibility);
    if (artifact.training.algorithm_id != kPolicyTrainingAlgorithmId ||
        artifact.training.scenario_profile_id != "light_search_1d_v1" ||
        artifact.training.seed_derivation_id !=
            "balanced_light_side_splitmix64_v1") {
        throw std::invalid_argument("unsupported policy training algorithm");
    }
    if (artifact.training.generations == 0 ||
        artifact.training.population_size < 2 ||
        artifact.training.elite_count == 0 ||
        artifact.training.elite_count > artifact.training.population_size ||
        artifact.training.train_batch_size == 0 ||
        artifact.training.train_batch_size > artifact.training.train_seed_count ||
        artifact.training.train_batch_size % 2U != 0U ||
        artifact.training.train_seed_count %
                artifact.training.train_batch_size !=
            0U ||
        artifact.training.max_control_steps == 0 ||
        artifact.training.selected_generation > artifact.training.generations ||
        artifact.training.train_seed_count == 0 ||
        artifact.training.validation_seed_count == 0 ||
        artifact.training.test_seed_count == 0 ||
        artifact.training.train_seed_count % 2U != 0U ||
        artifact.training.validation_seed_count % 2U != 0U ||
        artifact.training.test_seed_count % 2U != 0U ||
        !std::isfinite(artifact.training.initial_stddev) ||
        artifact.training.initial_stddev <= 0.0 ||
        !std::isfinite(artifact.training.minimum_stddev) ||
        artifact.training.minimum_stddev <= 0.0 ||
        artifact.training.minimum_stddev > artifact.training.initial_stddev ||
        !std::isfinite(artifact.training.update_rate) ||
        artifact.training.update_rate <= 0.0 ||
        artifact.training.update_rate > 1.0) {
        throw std::invalid_argument("invalid policy training provenance counts");
    }
    require_sha256(
        artifact.training.protocol_manifest_sha256,
        "training.protocol_manifest_sha256");
    require_sha256(
        artifact.training.train_seeds_sha256,
        "training.train_seeds_sha256");
    require_sha256(
        artifact.training.validation_seeds_sha256,
        "training.validation_seeds_sha256");
    require_sha256(
        artifact.training.test_seeds_sha256,
        "training.test_seeds_sha256");
    if (!std::isfinite(artifact.effort_limit) || artifact.effort_limit <= 0.0 ||
        artifact.effort_limit > 1.0 || !std::isfinite(artifact.slew_limit) ||
        artifact.slew_limit <= 0.0 || artifact.slew_limit > 2.0) {
        throw std::invalid_argument("invalid policy effort or slew limit");
    }
    for (const double value : artifact.parameters) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("policy parameters must be finite");
        }
    }
}

[[nodiscard]] std::filesystem::path unique_sibling(
    const std::filesystem::path& destination) {
    static std::atomic<std::uint64_t> sequence{0};
    const auto stamp = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    for (unsigned int attempt = 0; attempt < 100U; ++attempt) {
        std::filesystem::path candidate = destination;
        candidate += ".tmp." + std::to_string(stamp) + "." +
            std::to_string(sequence.fetch_add(1)) + "." +
            std::to_string(attempt);
        std::error_code error;
        if (!std::filesystem::exists(candidate, error) && !error) {
            return candidate;
        }
    }
    throw std::runtime_error("unable to reserve temporary policy artifact path");
}

}  // namespace

Json policy_feature_schema() {
    static constexpr std::array<std::string_view, kPolicyFeatureCount> names{
        "ambient_instance_count_squashed",
        "ambient_valid_fraction",
        "ambient_mean_log_illuminance",
        "ambient_saturated_fraction",
        "touch_instance_count_squashed",
        "touch_valid_fraction",
        "touch_contact_fraction",
        "touch_mean_force_fraction",
        "touch_max_force_fraction",
        "ambient_mean_log_illuminance_delta_scaled",
        "touch_contact_fraction_delta",
        "touch_mean_force_fraction_delta",
        "actuator_mean_velocity_fraction",
        "actuator_mean_abs_velocity_fraction",
        "actuator_mean_current_fraction",
        "actuator_max_temperature_fraction",
        "actuator_mean_stuck_score",
        "actuator_fault_or_invalid_fraction",
        "local_valid",
        "local_wrapped_position_fraction",
        "local_velocity_fraction",
        "local_current_fraction",
        "local_bus_voltage_fraction",
        "local_temperature_fraction",
        "local_output_torque_fraction",
        "local_log_impedance_fraction",
        "local_stuck_score",
        "local_stuck",
        "local_fault",
        "previous_local_effort",
        "bias",
        "scaled_ambient_delta_x_previous_local_effort",
        "scaled_ambient_delta_x_local_velocity_fraction",
    };
    Json features = Json::array();
    for (std::size_t index = 0; index < names.size(); ++index) {
        features.push_back(Json{{"index", index}, {"name", names[index]}});
    }
    return Json{
        {"schema_version", "droid-blocks.policy-feature-schema.v2"},
        {"encoder_id", kPolicyEncoderId},
        {"architecture_id", kPolicyArchitectureId},
        {"feature_count", kPolicyFeatureCount},
        {"supported_reward_sensor_families",
         Json::array({"ambient_light_v0", "touch_force_v0"})},
        {"supported_actuator_skus", Json::array({kPolicyReferenceMotorSku})},
        {"collection_reduction",
         "numeric-value-sorted compensated means; IDs are never features"},
        {"missing_optional_sensor", "zero values with explicit count/valid masks"},
        {"ambient_delta_validity",
         "zero unless ambient light is valid in consecutive observations; "
         "missing or invalid ambient light clears its temporal continuity"},
        {"missing_or_invalid_actuator", "zero effort for that actuator"},
        {"normalization",
         Json{
             {"ambient_light_lux", "clip(log1p(x)/log1p(1000),0,1)"},
             {"ambient_light_delta",
              Json{{"definition",
                    "clip(8192*(current_normalized_ambient-previous_normalized_ambient),-1,1)"},
                   {"scale", kPolicyAmbientDeltaScale},
                   {"first_observation", 0.0}}},
             {"touch_force_n", "clip(x/25,0,1)"},
             {"position_rad", "clip(x/pi,-1,1)"},
             {"velocity_rad_s", "clip(x/20,-1,1)"},
             {"current_a", "clip(x/2.5,-1,1)"},
             {"bus_voltage_v", "clip(2*(x-6)/(7.4-6)-1,-1,1)"},
             {"temperature_c", "clip((x-25)/(80-25),0,1)"},
             {"output_torque_nm", "clip(x/1.6875,-1,1)"},
             {"load_impedance_nm_s_per_rad",
              "clip(log1p(x)/log1p(16.875),0,1)"},
         }},
        {"interaction_features",
         Json{
             {"scaled_ambient_delta_x_previous_local_effort",
              "scaled ambient delta * previous commanded effort for this motor"},
             {"scaled_ambient_delta_x_local_velocity_fraction",
              "scaled ambient delta * current normalized shaft velocity for this motor"},
         }},
        {"features", std::move(features)},
    };
}

std::string policy_feature_schema_sha256() {
    return sha256_hex(policy_feature_schema().dump());
}

std::string sha256_hex(std::string_view bytes) {
    return Sha256::digest(bytes);
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open file for SHA-256: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.eof() && input.fail()) {
        throw std::runtime_error("unable to read file for SHA-256: " + path.string());
    }
    return sha256_hex(buffer.str());
}

std::string encode_binary64_hex(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("binary64 artifact value must be finite");
    }
    std::ostringstream output;
    output << "0x" << std::hex << std::nouppercase << std::setfill('0')
           << std::setw(16) << std::bit_cast<std::uint64_t>(value);
    return output.str();
}

double decode_binary64_hex(std::string_view encoded) {
    if (encoded.size() != 18 || encoded[0] != '0' || encoded[1] != 'x') {
        throw std::invalid_argument(
            "binary64 value must use 0x plus exactly 16 lowercase hex digits");
    }
    std::uint64_t bits = 0;
    for (std::size_t index = 2; index < encoded.size(); ++index) {
        const char character = encoded[index];
        std::uint64_t digit = 0;
        if (character >= '0' && character <= '9') {
            digit = static_cast<std::uint64_t>(character - '0');
        } else if (character >= 'a' && character <= 'f') {
            digit = static_cast<std::uint64_t>(character - 'a' + 10);
        } else {
            throw std::invalid_argument(
                "binary64 value contains a non-lowercase-hex digit");
        }
        bits = (bits << 4U) | digit;
    }
    const double value = std::bit_cast<double>(bits);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("binary64 artifact value must be finite");
    }
    return value;
}

bool policy_parameters_conform_to_cem_mask_v1(
    std::span<const double> parameters) noexcept {
    if (parameters.size() != kPolicyFeatureCount) {
        return false;
    }
    constexpr std::uint64_t kPositiveZeroBits =
        std::bit_cast<std::uint64_t>(0.0);
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const bool active = std::find(
                                kPolicyCemMaskV1ActiveParameterIndices.begin(),
                                kPolicyCemMaskV1ActiveParameterIndices.end(),
                                index) !=
            kPolicyCemMaskV1ActiveParameterIndices.end();
        if (active) {
            if (!std::isfinite(parameters[index])) {
                return false;
            }
        } else if (
            std::bit_cast<std::uint64_t>(parameters[index]) !=
            kPositiveZeroBits) {
            return false;
        }
    }
    return true;
}

Json policy_artifact_json(const PolicyArtifactData& artifact) {
    validate_artifact_data(artifact);
    Json document = artifact_payload_json(artifact);
    document["self_sha256"] = sha256_hex(document.dump());
    return document;
}

PolicyArtifactData parse_policy_artifact(const Json& document) {
    require_exact_keys(
        document,
        {"schema_version",
         "architecture",
         "compatibility",
         "parameters",
         "training",
         "self_sha256"},
        "policy artifact");
    if (required_string(document, "schema_version", "policy artifact") !=
        kPolicyArtifactSchemaVersion) {
        throw std::invalid_argument("unsupported policy artifact schema_version");
    }
    const std::string recorded_hash = required_string(
        document, "self_sha256", "policy artifact");
    require_sha256(recorded_hash, "policy artifact.self_sha256");
    Json payload = document;
    payload.erase("self_sha256");
    if (sha256_hex(payload.dump()) != recorded_hash) {
        throw std::invalid_argument("policy artifact self SHA-256 mismatch");
    }

    const Json& architecture = document.at("architecture");
    require_exact_keys(
        architecture,
        {"id",
         "encoder_id",
         "activation_id",
         "parameter_encoding",
         "parameter_count",
         "effort_limit",
         "slew_limit"},
        "policy artifact.architecture");
    if (required_string(architecture, "id", "policy artifact.architecture") !=
            kPolicyArchitectureId ||
        required_string(
            architecture, "encoder_id", "policy artifact.architecture") !=
            kPolicyEncoderId ||
        required_string(
            architecture, "activation_id", "policy artifact.architecture") !=
            kPolicyActivationId ||
        required_string(
            architecture,
            "parameter_encoding",
            "policy artifact.architecture") != kPolicyParameterEncoding ||
        required_size(
            architecture, "parameter_count", "policy artifact.architecture") !=
            kPolicyFeatureCount) {
        throw std::invalid_argument("unsupported policy artifact architecture");
    }

    PolicyArtifactData result;
    result.effort_limit = decode_binary64_hex(required_string(
        architecture, "effort_limit", "policy artifact.architecture"));
    result.slew_limit = decode_binary64_hex(required_string(
        architecture, "slew_limit", "policy artifact.architecture"));

    const Json& compatibility = document.at("compatibility");
    require_exact_keys(
        compatibility,
        {"environment_api_version",
         "assembly_id",
         "motor_ids",
         "control_dt_s",
         "model_sha256",
         "module_catalog_sha256",
         "environment_spec_sha256",
         "feature_schema_sha256"},
        "policy artifact.compatibility");
    result.compatibility.environment_api_version = required_string(
        compatibility,
        "environment_api_version",
        "policy artifact.compatibility");
    result.compatibility.assembly_id = required_string(
        compatibility, "assembly_id", "policy artifact.compatibility");
    if (!compatibility.at("motor_ids").is_array()) {
        throw std::invalid_argument(
            "policy artifact.compatibility.motor_ids must be an array");
    }
    for (const Json& id : compatibility.at("motor_ids")) {
        if (!id.is_string() || id.get_ref<const std::string&>().empty()) {
            throw std::invalid_argument(
                "policy artifact motor IDs must be non-empty strings");
        }
        result.compatibility.motor_ids.push_back(id.get<std::string>());
    }
    result.compatibility.control_dt_s = decode_binary64_hex(required_string(
        compatibility, "control_dt_s", "policy artifact.compatibility"));
    result.compatibility.model_sha256 = required_string(
        compatibility, "model_sha256", "policy artifact.compatibility");
    result.compatibility.module_catalog_sha256 = required_string(
        compatibility,
        "module_catalog_sha256",
        "policy artifact.compatibility");
    result.compatibility.environment_spec_sha256 = required_string(
        compatibility,
        "environment_spec_sha256",
        "policy artifact.compatibility");
    result.compatibility.feature_schema_sha256 = required_string(
        compatibility,
        "feature_schema_sha256",
        "policy artifact.compatibility");

    const Json& parameters = document.at("parameters");
    if (!parameters.is_array() || parameters.size() != kPolicyFeatureCount) {
        throw std::invalid_argument(
            "policy artifact parameters have the wrong dimension");
    }
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        if (!parameters.at(index).is_string()) {
            throw std::invalid_argument(
                "policy artifact parameters must be binary64 strings");
        }
        result.parameters[index] = decode_binary64_hex(
            parameters.at(index).get_ref<const std::string&>());
    }

    const Json& training = document.at("training");
    require_exact_keys(
        training,
        {"algorithm_id",
         "scenario_profile_id",
         "seed_derivation_id",
         "protocol_manifest_sha256",
         "seed_split_base_seed",
         "optimizer_seed",
         "generations",
         "population_size",
         "elite_count",
         "train_batch_size",
         "max_control_steps",
         "initial_stddev",
         "minimum_stddev",
         "update_rate",
         "selected_generation",
         "train_seed_count",
         "validation_seed_count",
         "test_seed_count",
         "train_seeds_sha256",
         "validation_seeds_sha256",
         "test_seeds_sha256"},
        "policy artifact.training");
    result.training.algorithm_id = required_string(
        training, "algorithm_id", "policy artifact.training");
    result.training.scenario_profile_id = required_string(
        training, "scenario_profile_id", "policy artifact.training");
    result.training.seed_derivation_id = required_string(
        training, "seed_derivation_id", "policy artifact.training");
    result.training.protocol_manifest_sha256 = required_string(
        training, "protocol_manifest_sha256", "policy artifact.training");
    result.training.seed_split_base_seed = required_u64(
        training, "seed_split_base_seed", "policy artifact.training");
    result.training.optimizer_seed = required_u64(
        training, "optimizer_seed", "policy artifact.training");
    result.training.generations = required_size(
        training, "generations", "policy artifact.training");
    result.training.population_size = required_size(
        training, "population_size", "policy artifact.training");
    result.training.elite_count = required_size(
        training, "elite_count", "policy artifact.training");
    result.training.train_batch_size = required_size(
        training, "train_batch_size", "policy artifact.training");
    result.training.max_control_steps = required_size(
        training, "max_control_steps", "policy artifact.training");
    result.training.initial_stddev = decode_binary64_hex(required_string(
        training, "initial_stddev", "policy artifact.training"));
    result.training.minimum_stddev = decode_binary64_hex(required_string(
        training, "minimum_stddev", "policy artifact.training"));
    result.training.update_rate = decode_binary64_hex(required_string(
        training, "update_rate", "policy artifact.training"));
    result.training.selected_generation = required_size(
        training, "selected_generation", "policy artifact.training");
    result.training.train_seed_count = required_size(
        training, "train_seed_count", "policy artifact.training");
    result.training.validation_seed_count = required_size(
        training, "validation_seed_count", "policy artifact.training");
    result.training.test_seed_count = required_size(
        training, "test_seed_count", "policy artifact.training");
    result.training.train_seeds_sha256 = required_string(
        training, "train_seeds_sha256", "policy artifact.training");
    result.training.validation_seeds_sha256 = required_string(
        training, "validation_seeds_sha256", "policy artifact.training");
    result.training.test_seeds_sha256 = required_string(
        training, "test_seeds_sha256", "policy artifact.training");
    result.self_sha256 = recorded_hash;
    validate_artifact_data(result);
    return result;
}

PolicyArtifactData load_policy_artifact(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open policy artifact: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.eof() && input.fail()) {
        throw std::runtime_error("unable to read policy artifact: " + path.string());
    }
    try {
        return parse_policy_artifact(Json::parse(buffer.str()));
    } catch (const Json::exception& error) {
        throw std::invalid_argument(
            "invalid policy artifact JSON: " + std::string(error.what()));
    }
}

void save_policy_artifact_exclusive(
    const std::filesystem::path& path,
    const PolicyArtifactData& artifact) {
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument("policy artifact destination must name a file");
    }
    const std::filesystem::path parent = path.has_parent_path()
        ? path.parent_path()
        : std::filesystem::current_path();
    if (!std::filesystem::is_directory(parent)) {
        throw std::runtime_error(
            "policy artifact destination directory does not exist: " +
            parent.string());
    }
    if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite existing policy artifact: " + path.string());
    }
    const Json document = policy_artifact_json(artifact);
    const std::filesystem::path temporary = unique_sibling(path);
    try {
        {
            std::ofstream output(
                temporary, std::ios::binary | std::ios::out | std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "unable to open temporary policy artifact: " +
                    temporary.string());
            }
            output << document.dump(2) << '\n';
            output.flush();
            if (!output) {
                throw std::runtime_error(
                    "unable to write temporary policy artifact: " +
                    temporary.string());
            }
        }
        std::error_code error;
        std::filesystem::create_hard_link(temporary, path, error);
        if (error) {
            if (std::filesystem::exists(path)) {
                throw std::runtime_error(
                    "refusing to overwrite existing policy artifact: " +
                    path.string());
            }
            throw std::runtime_error(
                "unable to install policy artifact: " + error.message());
        }
        std::filesystem::remove(temporary, error);
        if (error) {
            throw std::runtime_error(
                "policy artifact saved but temporary link removal failed: " +
                error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void require_policy_compatible(
    const PolicyArtifactData& artifact,
    const PolicyCompatibility& actual) {
    validate_artifact_data(artifact);
    const std::string computed_self = policy_artifact_json(artifact)
                                          .at("self_sha256")
                                          .get<std::string>();
    if (!artifact.self_sha256.empty() &&
        artifact.self_sha256 != computed_self) {
        throw std::invalid_argument("policy artifact in-memory self SHA-256 mismatch");
    }
    validate_compatibility_fields(actual);
    const PolicyCompatibility& expected = artifact.compatibility;
    if (expected.environment_api_version != actual.environment_api_version ||
        expected.assembly_id != actual.assembly_id ||
        normalized_motor_ids(expected.motor_ids) !=
            normalized_motor_ids(actual.motor_ids) ||
        !same_bits(expected.control_dt_s, actual.control_dt_s) ||
        expected.model_sha256 != actual.model_sha256 ||
        expected.module_catalog_sha256 != actual.module_catalog_sha256 ||
        expected.environment_spec_sha256 != actual.environment_spec_sha256 ||
        expected.feature_schema_sha256 != actual.feature_schema_sha256) {
        throw std::invalid_argument(
            "policy artifact is incompatible with the current runtime");
    }
}

SharedLinearMemoryPolicy::SharedLinearMemoryPolicy(PolicyArtifactData artifact)
    : artifact_(std::move(artifact)) {
    validate_artifact_data(artifact_);
    const std::string computed_self = policy_artifact_json(artifact_)
                                          .at("self_sha256")
                                          .get<std::string>();
    if (!artifact_.self_sha256.empty() &&
        artifact_.self_sha256 != computed_self) {
        throw std::invalid_argument("policy artifact in-memory self SHA-256 mismatch");
    }
    artifact_.self_sha256 = computed_self;
    reset(artifact_.compatibility.motor_ids);
}

void SharedLinearMemoryPolicy::reset(std::span<const std::string> motor_ids) {
    const std::vector<std::string> normalized = normalized_motor_ids(motor_ids);
    if (normalized != normalized_motor_ids(artifact_.compatibility.motor_ids)) {
        throw std::invalid_argument(
            "policy reset motor topology does not match frozen artifact");
    }
    motor_ids_ = normalized;
    previous_efforts_.clear();
    for (const std::string& motor_id : motor_ids_) {
        previous_efforts_.emplace(motor_id, 0.0);
    }
    previous_sensor_summary_.fill(0.0);
    has_previous_sensor_summary_ = false;
    previous_ambient_valid_ = false;
}

std::map<std::string, double> SharedLinearMemoryPolicy::act(
    const Json& policy_safe_observation,
    std::span<const std::string> motor_ids) {
    require_exact_keys(
        policy_safe_observation,
        {"reward_sensors", "actuator_feedback"},
        "policy observation");
    const std::vector<std::string> normalized = normalized_motor_ids(motor_ids);
    if (normalized != motor_ids_) {
        throw std::invalid_argument(
            "policy action motor topology changed without reset");
    }

    const SensorEncoding sensors = encode_reward_sensors(
        policy_safe_observation.at("reward_sensors"));
    const auto actuators = encode_actuators(
        policy_safe_observation.at("actuator_feedback"), motor_ids_);
    const auto pooled = pooled_actuator_features(actuators, motor_ids_.size());
    std::array<double, 3> deltas{};
    if (has_previous_sensor_summary_) {
        if (sensors.ambient_valid && previous_ambient_valid_) {
            deltas[0] = clamp(
                kPolicyAmbientDeltaScale *
                    (sensors.temporal[0] - previous_sensor_summary_[0]),
                -1.0,
                1.0);
        }
        for (std::size_t index = 1; index < deltas.size(); ++index) {
            deltas[index] = clamp(
                sensors.temporal[index] - previous_sensor_summary_[index],
                -1.0,
                1.0);
        }
    }

    std::map<std::string, double> actions;
    for (const std::string& motor_id : motor_ids_) {
        const ActuatorEncoding& actuator = actuators.at(motor_id);
        const double previous_effort = previous_efforts_.at(motor_id);
        std::array<double, kPolicyFeatureCount> features{};
        std::copy(sensors.features.begin(), sensors.features.end(), features.begin());
        std::copy(deltas.begin(), deltas.end(), features.begin() + 9);
        std::copy(pooled.begin(), pooled.end(), features.begin() + 12);
        features[18] = actuator.present && actuator.valid ? 1.0 : 0.0;
        features[19] = actuator.position;
        features[20] = actuator.velocity;
        features[21] = actuator.current;
        features[22] = actuator.bus_voltage;
        features[23] = actuator.temperature;
        features[24] = actuator.torque;
        features[25] = actuator.impedance;
        features[26] = actuator.stuck_score;
        features[27] = actuator.stuck ? 1.0 : 0.0;
        features[28] = actuator.fault ? 1.0 : 0.0;
        features[29] = previous_effort;
        features[30] = 1.0;
        features[31] = deltas[0] * previous_effort;
        features[32] = deltas[0] * actuator.velocity;

        long double activation = 0.0L;
        for (std::size_t index = 0; index < features.size(); ++index) {
            activation += static_cast<long double>(artifact_.parameters[index]) *
                static_cast<long double>(features[index]);
        }
        double effort = artifact_.effort_limit *
            std::tanh(static_cast<double>(activation));
        effort = clamp(
            effort,
            previous_effort - artifact_.slew_limit,
            previous_effort + artifact_.slew_limit);
        effort = clamp(
            effort, -artifact_.effort_limit, artifact_.effort_limit);
        if (!sensors.any_valid || !actuator.present || !actuator.valid ||
            actuator.fault) {
            effort = 0.0;
        }
        if (effort == 0.0) {
            effort = 0.0;
        }
        actions.emplace(motor_id, effort);
    }

    previous_sensor_summary_ = sensors.temporal;
    has_previous_sensor_summary_ = true;
    previous_ambient_valid_ = sensors.ambient_valid;
    previous_efforts_ = actions;
    return actions;
}

const PolicyArtifactData& SharedLinearMemoryPolicy::artifact() const noexcept {
    return artifact_;
}

}  // namespace droid
