#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kBaselineEvaluationSchemaVersion{
    "droid-blocks.baseline-evaluation.v2"};

enum class BaselinePolicyKind {
    zero,
    seeded_random,
};

struct BaselineEvaluationOptions {
    std::string assembly_id{"demo_rover_v0"};
    std::string episode_profile{"fixed_demo_v0"};
    std::size_t episodes{8};
    std::size_t max_control_steps{1'000};
    std::uint64_t base_seed{0};
    double control_dt_s{0.02};
    double random_effort_limit{0.6};
    std::size_t random_hold_steps{5};
};

[[nodiscard]] std::string_view baseline_policy_id(
    BaselinePolicyKind policy) noexcept;
[[nodiscard]] BaselinePolicyKind parse_baseline_policy(
    std::string_view policy_id);

// Stable, independently domain-separated streams. Environment seeds do not
// depend on the selected policy, so evaluations with the same options are
// paired episode-for-episode.
[[nodiscard]] std::uint64_t baseline_environment_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept;
[[nodiscard]] std::uint64_t baseline_policy_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept;

// Both baseline policies are deliberately observation-free. The seeded
// random policy uses a specified integer mixer and direct 53-bit conversion;
// it never relies on implementation-defined standard random distributions.
[[nodiscard]] std::map<std::string, double> make_baseline_actions(
    BaselinePolicyKind policy,
    const std::vector<std::string>& motor_ids,
    std::size_t control_step,
    std::uint64_t policy_seed,
    const BaselineEvaluationOptions& options);

// Run deterministic, worker-free episodes exclusively through the public
// DroidEnvironment spec/reset/step boundary. The returned document contains
// no clock, timestamp, throughput, RSS, or privileged simulation state.
[[nodiscard]] nlohmann::json evaluate_baseline(
    const std::filesystem::path& model_path,
    BaselinePolicyKind policy,
    const BaselineEvaluationOptions& options = {});

}  // namespace droid
