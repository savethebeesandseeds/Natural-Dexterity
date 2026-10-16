#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kPolicyEvaluationSchemaVersion{
    "droid-blocks.policy-evaluation.v1"};

// Generic deterministic rollout configuration. When
// explicit_environment_seeds is non-empty, it must contain exactly `episodes`
// values and replaces the derived environment stream. The policy seed remains
// independently derived from base_seed and the episode index.
struct PolicyEvaluationOptions {
    std::string assembly_id{"demo_rover_v0"};
    std::string episode_profile{"fixed_demo_v0"};
    std::size_t episodes{8};
    std::size_t max_control_steps{1'000};
    std::uint64_t base_seed{0};
    double control_dt_s{0.02};
    std::vector<std::uint64_t> explicit_environment_seeds;
};

// metadata is copied beneath the standard policy fields in the evaluation
// document. It may contain policy-specific provenance, but must not redefine
// schema_version, id, deterministic, uses_observation, or uses_policy_seed.
struct PolicyEvaluationDescriptor {
    std::string schema_version{"droid-blocks.policy.v1"};
    std::string id;
    bool deterministic{true};
    bool uses_observation{true};
    bool uses_policy_seed{false};
    nlohmann::json metadata{nlohmann::json::object()};
};

// The callback's argument list is the complete inference boundary. It receives
// only the current policy-safe observation, opaque motor IDs, zero-based
// control-step index, and the policy-only seed. In particular it cannot receive
// reset/step info, environment seed, reward, safety, or visualization state.
using PolicyActionCallback = std::function<std::map<std::string, double>(
    const nlohmann::json& policy_safe_observation,
    const std::vector<std::string>& opaque_motor_ids,
    std::size_t control_step,
    std::uint64_t policy_seed)>;

[[nodiscard]] std::uint64_t policy_evaluation_environment_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept;
[[nodiscard]] std::uint64_t policy_evaluation_policy_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept;

// Runs worker-free episodes exclusively through DroidEnvironment's public
// spec/reset/step boundary. The deterministic result contains no clock,
// timestamp, throughput, RSS, raw observation, or privileged simulation state.
[[nodiscard]] nlohmann::json evaluate_policy(
    const std::filesystem::path& model_path,
    const PolicyEvaluationDescriptor& policy,
    const PolicyActionCallback& action_callback,
    const PolicyEvaluationOptions& options = {});

}  // namespace droid
