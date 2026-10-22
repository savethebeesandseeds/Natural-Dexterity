#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kPolicyArtifactSchemaVersion{
    "droid-blocks.policy-artifact.v1"};
inline constexpr std::string_view kPolicyArchitectureId{
    "shared_linear_memory_v2"};
inline constexpr std::string_view kPolicyEncoderId{
    "module_set_encoder_v2"};
inline constexpr std::string_view kPolicyActivationId{"tanh_v1"};
inline constexpr std::string_view kPolicyParameterEncoding{
    "ieee754-binary64-hex-v1"};
inline constexpr std::string_view kPolicyReferenceMotorSku{
    "rotary_dc_gearmotor_v0"};
inline constexpr std::string_view kPolicyTrainingAlgorithmId{
    "cem_masked_best_sample_v1"};

inline constexpr std::size_t kPolicyFeatureCount{33};
inline constexpr std::array<std::size_t, 3>
    kPolicyCemMaskV1ActiveParameterIndices{29U, 30U, 32U};
inline constexpr std::array<double, 3> kPolicyCemMaskV1ParameterScales{
    2.0, 0.25, 64.0};
// Exact power-of-two gain for the one-control-step change in normalized
// ambient illuminance. This brings the observed ~1e-4 signal into an O(1)
// range without platform-dependent decimal division.
inline constexpr double kPolicyAmbientDeltaScale{8192.0};
inline constexpr double kDefaultLearnedEffortLimit{0.6};
inline constexpr double kDefaultLearnedSlewLimit{0.1};

struct PolicyCompatibility {
    std::string environment_api_version;
    std::string assembly_id;
    std::vector<std::string> motor_ids;
    double control_dt_s{0.02};
    std::string model_sha256;
    std::string module_catalog_sha256;
    std::string environment_spec_sha256;
    std::string feature_schema_sha256;
};

struct PolicyTrainingProvenance {
    std::string algorithm_id{std::string(kPolicyTrainingAlgorithmId)};
    std::string scenario_profile_id{"light_search_1d_v1"};
    std::string seed_derivation_id{"balanced_light_side_splitmix64_v1"};
    std::string protocol_manifest_sha256;
    std::uint64_t seed_split_base_seed{0};
    std::uint64_t optimizer_seed{0};
    std::size_t generations{40};
    std::size_t population_size{32};
    std::size_t elite_count{8};
    std::size_t train_batch_size{8};
    std::size_t max_control_steps{1'000};
    double initial_stddev{1.0};
    double minimum_stddev{0.1};
    double update_rate{0.25};
    std::size_t selected_generation{0};
    std::size_t train_seed_count{32};
    std::size_t validation_seed_count{16};
    std::size_t test_seed_count{64};
    std::string train_seeds_sha256;
    std::string validation_seeds_sha256;
    std::string test_seeds_sha256;
};

struct PolicyArtifactData {
    PolicyCompatibility compatibility;
    PolicyTrainingProvenance training;
    std::array<double, kPolicyFeatureCount> parameters{};
    double effort_limit{kDefaultLearnedEffortLimit};
    double slew_limit{kDefaultLearnedSlewLimit};
    // Canonical-document integrity checksum. This detects accidental or
    // post-freeze mutation; it is deliberately not an authenticity proof.
    std::string self_sha256;
};

// The official v3 experiment searches only the three published physical
// parameters above. Inactive entries must retain the exact binary64 +0 bit
// pattern; this is deliberately narrower than general artifact validity.
[[nodiscard]] bool policy_parameters_conform_to_cem_mask_v1(
    std::span<const double> parameters) noexcept;

// The feature schema is executable provenance. Its canonical JSON digest is
// embedded in every artifact, so normalization or feature-order changes cannot
// silently reinterpret frozen parameters.
[[nodiscard]] nlohmann::json policy_feature_schema();
[[nodiscard]] std::string policy_feature_schema_sha256();

[[nodiscard]] std::string sha256_hex(std::string_view bytes);
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);
[[nodiscard]] std::string encode_binary64_hex(double value);
[[nodiscard]] double decode_binary64_hex(std::string_view encoded);

[[nodiscard]] nlohmann::json policy_artifact_json(
    const PolicyArtifactData& artifact);
[[nodiscard]] PolicyArtifactData parse_policy_artifact(
    const nlohmann::json& document);
[[nodiscard]] PolicyArtifactData load_policy_artifact(
    const std::filesystem::path& path);
void save_policy_artifact_exclusive(
    const std::filesystem::path& path,
    const PolicyArtifactData& artifact);

// Validate a frozen artifact against the actual runtime before inference.
// Every value is exact; topology, timing, or content changes fail closed.
void require_policy_compatible(
    const PolicyArtifactData& artifact,
    const PolicyCompatibility& actual);

class SharedLinearMemoryPolicy final {
public:
    explicit SharedLinearMemoryPolicy(PolicyArtifactData artifact);

    // Clears one-step memory and establishes the opaque action topology.
    void reset(std::span<const std::string> motor_ids);

    // This is intentionally the complete inference boundary: policy-safe
    // observation plus opaque motor IDs. Reward, lifecycle, info, seeds,
    // visualization state, and simulator internals cannot be supplied.
    [[nodiscard]] std::map<std::string, double> act(
        const nlohmann::json& policy_safe_observation,
        std::span<const std::string> motor_ids);

    [[nodiscard]] const PolicyArtifactData& artifact() const noexcept;

private:
    PolicyArtifactData artifact_;
    std::vector<std::string> motor_ids_;
    std::map<std::string, double> previous_efforts_;
    std::array<double, 3> previous_sensor_summary_{};
    bool has_previous_sensor_summary_{false};
    bool previous_ambient_valid_{false};
};

}  // namespace droid
