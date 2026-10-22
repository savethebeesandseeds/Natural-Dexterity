#pragma once

#include "droid/evaluation.hpp"
#include "droid/policy.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kLearningExperimentSchemaVersion{
    "droid-blocks.learning-experiment.v3"};
inline constexpr std::string_view kCemAlgorithmId{
    kPolicyTrainingAlgorithmId};
inline constexpr std::string_view kCemSearchSpaceId{
    "ambient_velocity_extremum_3d_v1"};
inline constexpr std::string_view kCemInactiveParameterPolicy{
    "positive_zero"};
inline constexpr std::string_view kCemCheckpointRule{
    "train_batch_best_sample_then_validation_v1"};
inline constexpr std::string_view kCemSamplerParameterKey{
    "physical_parameter_index"};
inline constexpr std::string_view kCemPopulationSamplingRule{
    "all_candidates_stochastic_no_mean_injection_v1"};
inline constexpr const auto& kCemActiveParameterIndices =
    kPolicyCemMaskV1ActiveParameterIndices;
inline constexpr const auto& kCemParameterScales =
    kPolicyCemMaskV1ParameterScales;

struct MirroredSeedSplits {
    std::uint64_t base_seed{0};
    std::vector<std::uint64_t> train;
    std::vector<std::uint64_t> validation;
    std::vector<std::uint64_t> test;
};

[[nodiscard]] MirroredSeedSplits make_mirrored_seed_splits(
    std::uint64_t base_seed = 0,
    std::size_t train_pairs = 16,
    std::size_t validation_pairs = 8,
    std::size_t test_pairs = 32);
[[nodiscard]] bool light_search_seed_has_positive_side(
    std::uint64_t seed) noexcept;
[[nodiscard]] std::uint32_t light_search_realization_key(
    std::uint64_t seed) noexcept;
[[nodiscard]] std::string seed_list_sha256(
    std::span<const std::uint64_t> seeds);

struct PolicyScore {
    std::uint64_t invalid_samples{0};
    std::uint64_t terminated_episodes{0};
    std::uint64_t safety_vetoes{0};
    // These are paired, per-environment-seed reward-rate differences from
    // zero control, reduced separately over each hidden light side. The
    // policy never receives the side, seed, zero outcome, reward, or score.
    double negative_x_mean_reward_rate_gain_vs_zero{0.0};
    double positive_x_mean_reward_rate_gain_vs_zero{0.0};
    // Absolute mean return remains the final deterministic tie-breaker after
    // safety, worst-side gain, and fixed-weight balanced gain.
    double mean_return{0.0};
};

[[nodiscard]] bool policy_score_better(
    const PolicyScore& candidate,
    const PolicyScore& incumbent) noexcept;

struct CemOptions {
    std::size_t generations{40};
    std::size_t population_size{32};
    std::size_t elite_count{8};
    // Standard deviations are in the three-dimensional latent search space.
    // The fixed physical mapping is published in the executable protocol.
    double initial_stddev{1.0};
    double minimum_stddev{0.1};
    double update_rate{0.25};
    std::uint64_t optimizer_seed{0};
};

struct CemTrainingResult {
    std::array<double, kPolicyFeatureCount> parameters{};
    PolicyScore validation_score;
    std::size_t selected_generation{0};
    nlohmann::json deterministic_history;
};

// All candidates in a generation receive the exact same balanced seed span.
// The callback must populate both paired-zero side gains and reduce episode
// results in the supplied order.
using ParameterBatchEvaluator = std::function<PolicyScore(
    std::span<const double> parameters,
    std::span<const std::uint64_t> environment_seeds)>;

[[nodiscard]] CemTrainingResult train_cem_diagonal(
    const CemOptions& options,
    const std::vector<std::vector<std::uint64_t>>& common_train_batches,
    std::span<const std::uint64_t> validation_seeds,
    const ParameterBatchEvaluator& evaluator,
    std::size_t candidate_evaluation_workers = 1);

struct PolicyTrainingRequest {
    PolicyCompatibility compatibility;
    CemOptions optimizer;
    MirroredSeedSplits seed_splits;
    std::string episode_profile{"light_search_1d_v1"};
    std::string protocol_manifest_sha256;
    std::size_t train_batch_size{8};
    std::size_t max_control_steps{1'000};
    // Execution-only throughput setting. It does not alter candidate samples,
    // reductions, the statistical protocol, or artifact identity. Evaluators
    // must support concurrent calls when this is greater than one.
    std::size_t candidate_evaluation_workers{1};
    double effort_limit{kDefaultLearnedEffortLimit};
    double slew_limit{kDefaultLearnedSlewLimit};
};

struct PolicyTrainingResult {
    PolicyArtifactData artifact;
    nlohmann::json deterministic_report;
};

[[nodiscard]] PolicyTrainingResult train_frozen_policy(
    const PolicyTrainingRequest& request,
    const ParameterBatchEvaluator& evaluator);

// Native environment integration used by the CLI. Candidate policy callbacks
// receive only the generic evaluator's policy-safe observation boundary.
[[nodiscard]] PolicyTrainingResult train_frozen_policy_in_environment(
    const std::filesystem::path& model_path,
    const PolicyTrainingRequest& request);

[[nodiscard]] nlohmann::json evaluate_frozen_policy(
    const std::filesystem::path& model_path,
    const PolicyArtifactData& artifact,
    const PolicyCompatibility& actual_runtime,
    const PolicyEvaluationOptions& options);

struct EpisodeOutcome {
    std::uint64_t environment_seed{0};
    double episode_return{0.0};
    double mean_reward_rate{0.0};
    bool terminated{false};
    std::uint64_t safety_vetoes{0};
    std::uint64_t invalid_samples{0};
};

struct HeldoutScenarioOutcome {
    EpisodeOutcome learned;
    EpisodeOutcome zero;
    std::vector<EpisodeOutcome> random;
};

struct AcceptanceOptions {
    std::size_t expected_scenarios{64};
    std::size_t random_streams_per_scenario{4};
    std::uint64_t random_control_base_seed{0x52414e4443544c31ULL};
    double random_effort_limit{0.6};
    std::size_t random_hold_steps{5};
    double random_slew_limit_per_control_step{0.1};
    std::size_t bootstrap_resamples{10'000};
    std::uint64_t bootstrap_seed{0x504f4c4943594556ULL};
    double practical_reward_rate_gain{0.01};
    std::size_t minimum_wins{48};
    std::size_t minimum_wins_per_side{24};
};

// Report-only scientific-governance metadata. This is deliberately outside
// the experiment manifest: it constrains interpretation and future dataset
// classification without changing the frozen v3 decision rule.
[[nodiscard]] nlohmann::json v3_heldout_report_governance();

// Computes the preregistered held-out decision from already executed frozen
// rollouts. The scenario is the statistical unit; random streams are averaged
// within each scenario before any comparison or resampling.
[[nodiscard]] nlohmann::json assess_heldout_policy(
    std::span<const HeldoutScenarioOutcome> scenarios,
    const PolicyArtifactData& artifact,
    const AcceptanceOptions& options = {});

// Executes the frozen artifact and both controls through evaluate_policy(),
// then applies assess_heldout_policy(). explicit_environment_seeds must hold
// the untouched balanced test split.
[[nodiscard]] nlohmann::json evaluate_frozen_heldout(
    const std::filesystem::path& model_path,
    const PolicyArtifactData& artifact,
    const PolicyCompatibility& actual_runtime,
    PolicyEvaluationOptions options,
    const AcceptanceOptions& acceptance = {});

[[nodiscard]] nlohmann::json default_learning_experiment_manifest();
[[nodiscard]] std::string learning_experiment_manifest_sha256();

}  // namespace droid
