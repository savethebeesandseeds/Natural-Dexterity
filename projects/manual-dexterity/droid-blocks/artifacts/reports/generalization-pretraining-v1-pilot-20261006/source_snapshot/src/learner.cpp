#include "droid/learner.hpp"

#include "droid/baseline.hpp"
#include "droid/simulation.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace droid {
namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kTrainSplitTag{0x545241494e5f5631ULL};
constexpr std::uint64_t kValidationSplitTag{0x56414c49445f5631ULL};
constexpr std::uint64_t kTestSplitTag{0x544553545f5f5631ULL};
constexpr std::uint64_t kCandidateSamplerTag{0x43454d53414d5031ULL};
constexpr std::uint64_t kBootstrapTag{0x424f4f5453545031ULL};
constexpr std::uint64_t kRandomControlTag{0x52414e444f4d5631ULL};
constexpr double kTwoToMinus53{0x1.0p-53};

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

[[nodiscard]] double uniform_unit(std::uint64_t key) noexcept {
    return static_cast<double>(mix64(key) >> 11U) * kTwoToMinus53;
}

[[nodiscard]] double irwin_hall_normal(
    std::uint64_t optimizer_seed,
    std::size_t generation,
    std::size_t candidate,
    std::size_t parameter) noexcept {
    // Sum(U[0,1), 12) - 6 has exactly unit variance and approximates a normal
    // without relying on any standard-library distribution or libm transform.
    double sum = 0.0;
    for (std::size_t draw = 0; draw < 12; ++draw) {
        const std::uint64_t key = optimizer_seed ^ kCandidateSamplerTag ^
            mix64(static_cast<std::uint64_t>(generation)) ^
            mix64(static_cast<std::uint64_t>(candidate) ^
                  0x9e3779b97f4a7c15ULL) ^
            mix64(static_cast<std::uint64_t>(parameter) ^
                  0xd1b54a32d192ed03ULL) ^
            mix64(static_cast<std::uint64_t>(draw) ^
                  0x94d049bb133111ebULL);
        sum += uniform_unit(key);
    }
    return sum - 6.0;
}

using LatentParameters =
    std::array<double, kCemActiveParameterIndices.size()>;

[[nodiscard]] std::array<double, kPolicyFeatureCount>
physical_parameters(const LatentParameters& latent) noexcept {
    // Value-initialization is intentional: every inactive physical parameter
    // is the positive-zero bit pattern required by the registered search
    // space, rather than merely a numerically equivalent signed zero.
    std::array<double, kPolicyFeatureCount> result{};
    for (std::size_t latent_index = 0;
         latent_index < latent.size();
         ++latent_index) {
        result[kCemActiveParameterIndices[latent_index]] =
            kCemParameterScales[latent_index] * latent[latent_index];
    }
    return result;
}

[[nodiscard]] double worst_side_reward_rate_gain(
    const PolicyScore& score) noexcept {
    return std::min(
        score.negative_x_mean_reward_rate_gain_vs_zero,
        score.positive_x_mean_reward_rate_gain_vs_zero);
}

[[nodiscard]] double balanced_reward_rate_gain(
    const PolicyScore& score) noexcept {
    return 0.5 * score.negative_x_mean_reward_rate_gain_vs_zero +
        0.5 * score.positive_x_mean_reward_rate_gain_vs_zero;
}

void validate_score(const PolicyScore& score) {
    if (!std::isfinite(
            score.negative_x_mean_reward_rate_gain_vs_zero) ||
        !std::isfinite(
            score.positive_x_mean_reward_rate_gain_vs_zero) ||
        !std::isfinite(score.mean_return)) {
        throw std::runtime_error(
            "policy evaluator produced a non-finite side gain or mean return");
    }
}

[[nodiscard]] Json score_json(const PolicyScore& score) {
    return Json{
        {"invalid_samples", score.invalid_samples},
        {"terminated_episodes", score.terminated_episodes},
        {"safety_vetoes", score.safety_vetoes},
        {"negative_x_mean_reward_rate_gain_vs_zero",
         score.negative_x_mean_reward_rate_gain_vs_zero},
        {"positive_x_mean_reward_rate_gain_vs_zero",
         score.positive_x_mean_reward_rate_gain_vs_zero},
        {"worst_side_mean_reward_rate_gain_vs_zero",
         worst_side_reward_rate_gain(score)},
        {"balanced_mean_reward_rate_gain_vs_zero",
         balanced_reward_rate_gain(score)},
        {"mean_return", score.mean_return},
    };
}

[[nodiscard]] std::string parameter_sha256(
    std::span<const double> parameters) {
    Json encoded = Json::array();
    for (const double value : parameters) {
        encoded.push_back(encode_binary64_hex(value));
    }
    return sha256_hex(encoded.dump());
}

[[nodiscard]] std::vector<std::uint64_t> make_balanced_split(
    std::uint64_t base_seed,
    std::uint64_t tag,
    std::size_t pair_count,
    std::set<std::uint64_t>& used_seeds,
    std::set<std::uint32_t>& used_realizations) {
    if (pair_count == 0) {
        throw std::invalid_argument("mirrored seed split pair count must be positive");
    }
    std::vector<std::uint64_t> negative;
    std::vector<std::uint64_t> positive;
    std::uint64_t candidate_index = 0;
    while (negative.size() < pair_count || positive.size() < pair_count) {
        if (candidate_index == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("unable to derive balanced seed split");
        }
        const std::uint64_t seed = mix64(
            base_seed ^ tag ^
            mix64(candidate_index ^ 0x9e3779b97f4a7c15ULL));
        ++candidate_index;
        const std::uint32_t realization = light_search_realization_key(seed);
        if (used_seeds.contains(seed) ||
            used_realizations.contains(realization)) {
            continue;
        }
        const bool positive_side = light_search_seed_has_positive_side(seed);
        auto& destination = positive_side ? positive : negative;
        if (destination.size() >= pair_count) {
            continue;
        }
        used_seeds.emplace(seed);
        used_realizations.emplace(realization);
        destination.push_back(seed);
    }
    std::vector<std::uint64_t> result;
    result.reserve(pair_count * 2U);
    for (std::size_t index = 0; index < pair_count; ++index) {
        result.push_back(negative[index]);
        result.push_back(positive[index]);
    }
    return result;
}

[[nodiscard]] std::vector<std::vector<std::uint64_t>> make_train_batches(
    const std::vector<std::uint64_t>& train,
    std::size_t batch_size) {
    if (train.empty() || batch_size == 0 || batch_size > train.size() ||
        batch_size % 2U != 0U || train.size() % batch_size != 0U) {
        throw std::invalid_argument(
            "train batch size must be an even divisor of the balanced train split");
    }
    std::vector<std::vector<std::uint64_t>> result;
    for (std::size_t offset = 0; offset < train.size(); offset += batch_size) {
        result.emplace_back(
            train.begin() + static_cast<std::ptrdiff_t>(offset),
            train.begin() + static_cast<std::ptrdiff_t>(offset + batch_size));
    }
    return result;
}

void require_exact_seed_splits(const MirroredSeedSplits& supplied) {
    if (supplied.train.empty() || supplied.validation.empty() ||
        supplied.test.empty() || supplied.train.size() % 2U != 0U ||
        supplied.validation.size() % 2U != 0U ||
        supplied.test.size() % 2U != 0U) {
        throw std::invalid_argument(
            "train, validation, and test seed splits must contain nonzero pairs");
    }
    const MirroredSeedSplits expected = make_mirrored_seed_splits(
        supplied.base_seed,
        supplied.train.size() / 2U,
        supplied.validation.size() / 2U,
        supplied.test.size() / 2U);
    if (supplied.train != expected.train ||
        supplied.validation != expected.validation ||
        supplied.test != expected.test) {
        throw std::invalid_argument(
            "seed splits are not the exact disjoint balanced derived protocol");
    }
}

void validate_cem_options(const CemOptions& options) {
    if (options.generations == 0 || options.population_size < 2 ||
        options.elite_count == 0 ||
        options.elite_count > options.population_size ||
        !std::isfinite(options.initial_stddev) ||
        options.initial_stddev <= 0.0 ||
        !std::isfinite(options.minimum_stddev) ||
        options.minimum_stddev <= 0.0 ||
        options.minimum_stddev > options.initial_stddev ||
        !std::isfinite(options.update_rate) || options.update_rate <= 0.0 ||
        options.update_rate > 1.0) {
        throw std::invalid_argument(
            "invalid cem_masked_best_sample_v1 configuration");
    }
}

[[nodiscard]] PolicyArtifactData ephemeral_artifact(
    const PolicyTrainingRequest& request,
    std::span<const double> parameters,
    std::size_t selected_generation = 0) {
    if (parameters.size() != kPolicyFeatureCount) {
        throw std::invalid_argument("policy parameter vector has wrong dimension");
    }
    PolicyArtifactData artifact;
    artifact.compatibility = request.compatibility;
    std::copy(parameters.begin(), parameters.end(), artifact.parameters.begin());
    artifact.effort_limit = request.effort_limit;
    artifact.slew_limit = request.slew_limit;
    artifact.training.algorithm_id = std::string(kCemAlgorithmId);
    artifact.training.scenario_profile_id = request.episode_profile;
    artifact.training.seed_derivation_id =
        "balanced_light_side_splitmix64_v1";
    artifact.training.protocol_manifest_sha256 =
        request.protocol_manifest_sha256.empty()
        ? learning_experiment_manifest_sha256()
        : request.protocol_manifest_sha256;
    artifact.training.seed_split_base_seed = request.seed_splits.base_seed;
    artifact.training.optimizer_seed = request.optimizer.optimizer_seed;
    artifact.training.generations = request.optimizer.generations;
    artifact.training.population_size = request.optimizer.population_size;
    artifact.training.elite_count = request.optimizer.elite_count;
    artifact.training.train_batch_size = request.train_batch_size;
    artifact.training.max_control_steps = request.max_control_steps;
    artifact.training.initial_stddev = request.optimizer.initial_stddev;
    artifact.training.minimum_stddev = request.optimizer.minimum_stddev;
    artifact.training.update_rate = request.optimizer.update_rate;
    artifact.training.selected_generation = selected_generation;
    artifact.training.train_seed_count = request.seed_splits.train.size();
    artifact.training.validation_seed_count =
        request.seed_splits.validation.size();
    artifact.training.test_seed_count = request.seed_splits.test.size();
    artifact.training.train_seeds_sha256 =
        seed_list_sha256(request.seed_splits.train);
    artifact.training.validation_seeds_sha256 =
        seed_list_sha256(request.seed_splits.validation);
    artifact.training.test_seeds_sha256 =
        seed_list_sha256(request.seed_splits.test);
    return artifact;
}

using ZeroRewardRateBySeed = std::map<std::uint64_t, double>;

[[nodiscard]] PolicyScore score_from_evaluation(
    const Json& evaluation,
    const ZeroRewardRateBySeed& zero_reward_rates) {
    if (!evaluation.is_object() || !evaluation.contains("aggregate") ||
        !evaluation.at("aggregate").is_object() ||
        !evaluation.contains("episodes") ||
        !evaluation.at("episodes").is_array()) {
        throw std::runtime_error(
            "policy evaluation omitted aggregate or episode metrics");
    }
    const Json& aggregate = evaluation.at("aggregate");
    const Json& health = aggregate.at("health");
    const Json& safety = aggregate.at("safety");
    long double negative_gain_sum = 0.0L;
    long double positive_gain_sum = 0.0L;
    std::size_t negative_count = 0;
    std::size_t positive_count = 0;
    std::set<std::uint64_t> seen_seeds;
    for (const Json& episode : evaluation.at("episodes")) {
        if (!episode.is_object() ||
            !episode.contains("environment_seed") ||
            !episode.at("environment_seed").is_number_unsigned() ||
            !episode.contains("mean_reward_rate") ||
            !episode.at("mean_reward_rate").is_number()) {
            throw std::runtime_error(
                "policy evaluation episode omitted paired score fields");
        }
        const std::uint64_t seed =
            episode.at("environment_seed").get<std::uint64_t>();
        if (!seen_seeds.emplace(seed).second) {
            throw std::runtime_error(
                "policy evaluation repeated an environment seed");
        }
        const auto baseline = zero_reward_rates.find(seed);
        if (baseline == zero_reward_rates.end()) {
            throw std::runtime_error(
                "policy evaluation seed has no precomputed zero baseline");
        }
        const double learned_rate =
            episode.at("mean_reward_rate").get<double>();
        if (!std::isfinite(learned_rate) ||
            !std::isfinite(baseline->second)) {
            throw std::runtime_error(
                "policy or zero reward rate is non-finite");
        }
        const long double gain =
            static_cast<long double>(learned_rate) -
            static_cast<long double>(baseline->second);
        if (light_search_seed_has_positive_side(seed)) {
            positive_gain_sum += gain;
            ++positive_count;
        } else {
            negative_gain_sum += gain;
            ++negative_count;
        }
    }
    if (negative_count == 0 || negative_count != positive_count) {
        throw std::runtime_error(
            "policy score requires equal nonzero negative/positive sides");
    }
    PolicyScore score{
        .invalid_samples =
            health.at("invalid_reward_sensor_samples").get<std::uint64_t>() +
            health.at("invalid_actuator_feedback_samples").get<std::uint64_t>(),
        .terminated_episodes =
            aggregate.at("terminated_episodes").get<std::uint64_t>(),
        .safety_vetoes = safety.at("veto_count").get<std::uint64_t>(),
        .negative_x_mean_reward_rate_gain_vs_zero =
            static_cast<double>(
                negative_gain_sum /
                static_cast<long double>(negative_count)),
        .positive_x_mean_reward_rate_gain_vs_zero =
            static_cast<double>(
                positive_gain_sum /
                static_cast<long double>(positive_count)),
        .mean_return = aggregate.at("mean_return").get<double>(),
    };
    validate_score(score);
    return score;
}

[[nodiscard]] ZeroRewardRateBySeed zero_reward_rates_from_evaluation(
    const Json& evaluation,
    std::span<const std::uint64_t> expected_seeds) {
    if (!evaluation.is_object() || !evaluation.contains("episodes") ||
        !evaluation.at("episodes").is_array() ||
        evaluation.at("episodes").size() != expected_seeds.size()) {
        throw std::runtime_error(
            "zero evaluation does not match its declared seed scope");
    }
    ZeroRewardRateBySeed result;
    for (std::size_t index = 0; index < expected_seeds.size(); ++index) {
        const Json& episode = evaluation.at("episodes").at(index);
        if (!episode.is_object() ||
            !episode.contains("environment_seed") ||
            !episode.at("environment_seed").is_number_unsigned() ||
            !episode.contains("mean_reward_rate") ||
            !episode.at("mean_reward_rate").is_number()) {
            throw std::runtime_error(
                "zero evaluation episode omitted paired score fields");
        }
        const std::uint64_t seed =
            episode.at("environment_seed").get<std::uint64_t>();
        const double reward_rate =
            episode.at("mean_reward_rate").get<double>();
        if (seed != expected_seeds[index] || !std::isfinite(reward_rate) ||
            !result.emplace(seed, reward_rate).second) {
            throw std::runtime_error(
                "zero evaluation is not in exact unique seed order");
        }
    }
    return result;
}

[[nodiscard]] EpisodeOutcome episode_outcome(const Json& episode) {
    if (!episode.is_object()) {
        throw std::runtime_error("evaluation episode must be an object");
    }
    const Json& health = episode.at("health");
    EpisodeOutcome outcome{
        .environment_seed = episode.at("environment_seed").get<std::uint64_t>(),
        .episode_return = episode.at("return").get<double>(),
        .mean_reward_rate = episode.at("mean_reward_rate").get<double>(),
        .terminated = episode.at("environment_terminated").get<bool>(),
        .safety_vetoes =
            episode.at("safety").at("veto_count").get<std::uint64_t>(),
        .invalid_samples =
            health.at("invalid_reward_sensor_samples").get<std::uint64_t>() +
            health.at("invalid_actuator_feedback_samples").get<std::uint64_t>(),
    };
    if (!std::isfinite(outcome.episode_return) ||
        !std::isfinite(outcome.mean_reward_rate)) {
        throw std::runtime_error("evaluation episode metrics must be finite");
    }
    return outcome;
}

[[nodiscard]] double ordered_mean(std::span<const double> values) {
    if (values.empty()) {
        throw std::invalid_argument("mean requires at least one value");
    }
    long double sum = 0.0L;
    long double correction = 0.0L;
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("mean received a non-finite value");
        }
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

struct Interval {
    double lower{0.0};
    double upper{0.0};
};

[[nodiscard]] Interval side_stratified_paired_bootstrap_interval(
    std::span<const double> negative_side_differences,
    std::span<const double> positive_side_differences,
    const AcceptanceOptions& options,
    std::uint64_t comparison_tag) {
    if (negative_side_differences.empty() ||
        positive_side_differences.empty()) {
        throw std::invalid_argument(
            "side-stratified bootstrap requires both light-side strata");
    }
    std::vector<double> bootstrap_means;
    bootstrap_means.reserve(options.bootstrap_resamples);
    for (std::size_t sample = 0; sample < options.bootstrap_resamples; ++sample) {
        long double negative_sum = 0.0L;
        for (std::size_t draw = 0;
             draw < negative_side_differences.size();
             ++draw) {
            const std::uint64_t bits = mix64(
                options.bootstrap_seed ^ kBootstrapTag ^ comparison_tag ^
                mix64(static_cast<std::uint64_t>(sample)) ^
                mix64(static_cast<std::uint64_t>(draw) ^
                      0x9e3779b97f4a7c15ULL));
            const std::size_t index = static_cast<std::size_t>(
                bits % static_cast<std::uint64_t>(
                    negative_side_differences.size()));
            negative_sum += static_cast<long double>(
                negative_side_differences[index]);
        }
        long double positive_sum = 0.0L;
        for (std::size_t draw = 0;
             draw < positive_side_differences.size();
             ++draw) {
            const std::uint64_t bits = mix64(
                options.bootstrap_seed ^ kBootstrapTag ^ comparison_tag ^
                0x504f534954495645ULL ^
                mix64(static_cast<std::uint64_t>(sample)) ^
                mix64(static_cast<std::uint64_t>(draw) ^
                      0xd1b54a32d192ed03ULL));
            const std::size_t index = static_cast<std::size_t>(
                bits % static_cast<std::uint64_t>(
                    positive_side_differences.size()));
            positive_sum += static_cast<long double>(
                positive_side_differences[index]);
        }
        const long double negative_mean = negative_sum /
            static_cast<long double>(negative_side_differences.size());
        const long double positive_mean = positive_sum /
            static_cast<long double>(positive_side_differences.size());
        bootstrap_means.push_back(static_cast<double>(
            0.5L * negative_mean + 0.5L * positive_mean));
    }
    std::sort(bootstrap_means.begin(), bootstrap_means.end());
    const std::size_t lower_index =
        (bootstrap_means.size() * 25U) / 1000U;
    const std::size_t upper_index = std::min(
        bootstrap_means.size() - 1U,
        (bootstrap_means.size() * 975U) / 1000U);
    return {bootstrap_means[lower_index], bootstrap_means[upper_index]};
}

[[nodiscard]] Json episode_outcome_json(const EpisodeOutcome& outcome) {
    return Json{
        {"environment_seed", outcome.environment_seed},
        {"return", outcome.episode_return},
        {"mean_reward_rate", outcome.mean_reward_rate},
        {"terminated", outcome.terminated},
        {"safety_vetoes", outcome.safety_vetoes},
        {"invalid_samples", outcome.invalid_samples},
    };
}

void validate_episode_outcome(
    const EpisodeOutcome& outcome,
    std::uint64_t expected_seed,
    std::string_view context) {
    if (outcome.environment_seed != expected_seed) {
        throw std::invalid_argument(
            std::string(context) + " environment seed is not paired");
    }
    if (!std::isfinite(outcome.episode_return) ||
        !std::isfinite(outcome.mean_reward_rate)) {
        throw std::invalid_argument(
            std::string(context) + " contains non-finite reward metrics");
    }
}

[[nodiscard]] Json protocol_manifest_from_artifact(
    const PolicyArtifactData& artifact,
    const AcceptanceOptions& acceptance) {
    return Json{
        {"schema_version", kLearningExperimentSchemaVersion},
        {"scenario_profile_id", artifact.training.scenario_profile_id},
        {"assembly_id", artifact.compatibility.assembly_id},
        {"control_dt_s", artifact.compatibility.control_dt_s},
        {"max_control_steps", artifact.training.max_control_steps},
        {"seed_split",
         Json{
             {"derivation_id", artifact.training.seed_derivation_id},
             {"base_seed", artifact.training.seed_split_base_seed},
             {"train_pairs", artifact.training.train_seed_count / 2U},
             {"validation_pairs",
              artifact.training.validation_seed_count / 2U},
             {"test_pairs", artifact.training.test_seed_count / 2U},
         }},
        {"policy",
         Json{
             {"architecture_id", kPolicyArchitectureId},
             {"encoder_id", kPolicyEncoderId},
             {"activation_id", kPolicyActivationId},
             {"parameter_count", kPolicyFeatureCount},
             {"effort_limit", artifact.effort_limit},
             {"slew_limit_per_control_step", artifact.slew_limit},
             {"ambient_light_delta_scale", kPolicyAmbientDeltaScale},
             {"interaction_features",
              Json::array({
                  "scaled_ambient_delta_x_previous_local_effort",
                  "scaled_ambient_delta_x_local_velocity_fraction",
              })},
         }},
        {"optimizer",
         Json{
             {"algorithm_id", artifact.training.algorithm_id},
             {"generations", artifact.training.generations},
             {"population_size", artifact.training.population_size},
             {"elite_count", artifact.training.elite_count},
             {"train_batch_size", artifact.training.train_batch_size},
             {"search_space_id", kCemSearchSpaceId},
             {"active_parameter_indices", kCemActiveParameterIndices},
             {"parameter_scales", kCemParameterScales},
             {"inactive_parameters", kCemInactiveParameterPolicy},
             {"latent_initial_mean", 0.0},
             {"initial_stddev", artifact.training.initial_stddev},
             {"minimum_stddev", artifact.training.minimum_stddev},
             {"update_rate", artifact.training.update_rate},
             {"optimizer_seed", artifact.training.optimizer_seed},
             {"sampler_id", "splitmix64_irwin_hall_12_v1"},
             {"sampler_parameter_key", kCemSamplerParameterKey},
             {"population_sampling_rule", kCemPopulationSamplingRule},
             {"checkpoint_rule", kCemCheckpointRule},
             {"objective_id",
              "paired_zero_side_robust_reward_rate_v1"},
             {"zero_baseline_scope", "train_and_validation_only"},
             {"selection",
              "invalid_samples_then_terminations_then_vetoes_then_worst_side_"
              "paired_zero_mean_reward_rate_gain_then_balanced_paired_zero_"
              "mean_reward_rate_gain_then_mean_return"},
             {"validation_selection",
              "best_train_selected_candidate_validation_score_then_earliest_"
              "generation"},
         }},
        {"heldout_acceptance",
         Json{
             {"scenario_count", acceptance.expected_scenarios},
             {"random_streams_per_scenario",
              acceptance.random_streams_per_scenario},
             {"random_control_base_seed",
              acceptance.random_control_base_seed},
             {"random_effort_limit", acceptance.random_effort_limit},
             {"random_hold_steps", acceptance.random_hold_steps},
             {"random_slew_limit_per_control_step",
              acceptance.random_slew_limit_per_control_step},
             {"bootstrap_method",
              "paired_side_stratified_percentile_v1"},
             {"bootstrap_resamples", acceptance.bootstrap_resamples},
             {"bootstrap_seed", acceptance.bootstrap_seed},
             {"one_sided_lower_confidence", 0.975},
             {"multiplicity_control",
              "bonferroni_two_one_sided_0.025"},
             {"joint_familywise_confidence", 0.95},
             {"practical_reward_rate_gain",
              acceptance.practical_reward_rate_gain},
             {"minimum_wins", acceptance.minimum_wins},
             {"minimum_wins_per_side",
              acceptance.minimum_wins_per_side},
             {"requirements",
              Json::array({
                  "learned_has_zero_terminations",
                  "learned_has_zero_safety_vetoes",
                  "learned_has_zero_invalid_samples",
                  "paired_stratified_97_5pct_one_sided_lower_bound_above_zero_vs_zero",
                  "paired_stratified_97_5pct_one_sided_lower_bound_above_zero_vs_random_mean",
                  "mean_reward_rate_gain_at_least_practical_margin_vs_both",
                  "wins_at_least_minimum_vs_both",
                  "positive_mean_delta_in_each_light_side_vs_both",
                  "wins_at_least_minimum_per_light_side_vs_both",
              })},
         }},
    };
}

[[nodiscard]] MirroredSeedSplits require_artifact_seed_protocol(
    const PolicyArtifactData& artifact) {
    if (artifact.training.train_seed_count == 0 ||
        artifact.training.validation_seed_count == 0 ||
        artifact.training.test_seed_count == 0 ||
        artifact.training.train_seed_count % 2U != 0U ||
        artifact.training.validation_seed_count % 2U != 0U ||
        artifact.training.test_seed_count % 2U != 0U) {
        throw std::invalid_argument(
            "frozen artifact seed counts are not nonzero balanced pairs");
    }
    const MirroredSeedSplits expected = make_mirrored_seed_splits(
        artifact.training.seed_split_base_seed,
        artifact.training.train_seed_count / 2U,
        artifact.training.validation_seed_count / 2U,
        artifact.training.test_seed_count / 2U);
    if (seed_list_sha256(expected.train) !=
            artifact.training.train_seeds_sha256 ||
        seed_list_sha256(expected.validation) !=
            artifact.training.validation_seeds_sha256 ||
        seed_list_sha256(expected.test) !=
            artifact.training.test_seeds_sha256) {
        throw std::invalid_argument(
            "frozen artifact seed digests do not match its derived protocol");
    }
    return expected;
}

[[nodiscard]] PolicyArtifactData require_frozen_artifact(
    const PolicyArtifactData& supplied,
    std::string_view context) {
    if (supplied.self_sha256.empty()) {
        throw std::invalid_argument(
            std::string(context) +
            " requires a frozen artifact with an integrity checksum");
    }
    const Json canonical_document = policy_artifact_json(supplied);
    if (supplied.self_sha256 !=
        canonical_document.at("self_sha256").get<std::string>()) {
        throw std::invalid_argument(
            std::string(context) +
            " artifact integrity checksum does not match its contents");
    }
    return parse_policy_artifact(canonical_document);
}

void require_acceptance_protocol(
    const PolicyArtifactData& artifact,
    const AcceptanceOptions& options) {
    if (!policy_parameters_conform_to_cem_mask_v1(artifact.parameters)) {
        throw std::invalid_argument(
            "held-out policy parameters do not conform to the registered "
            "positive-zero CEM mask");
    }
    if (options.expected_scenarios == 0 ||
        options.expected_scenarios != artifact.training.test_seed_count ||
        options.expected_scenarios % 2U != 0U ||
        options.random_streams_per_scenario == 0 ||
        !std::isfinite(options.random_effort_limit) ||
        options.random_effort_limit <= 0.0 ||
        options.random_effort_limit > 1.0 ||
        options.random_hold_steps == 0 ||
        !std::isfinite(options.random_slew_limit_per_control_step) ||
        options.random_slew_limit_per_control_step <= 0.0 ||
        options.bootstrap_resamples < 100 || options.minimum_wins == 0 ||
        options.minimum_wins > options.expected_scenarios ||
        options.minimum_wins_per_side == 0 ||
        options.minimum_wins_per_side > options.expected_scenarios / 2U ||
        !std::isfinite(options.practical_reward_rate_gain) ||
        options.practical_reward_rate_gain < 0.0) {
        throw std::invalid_argument("invalid held-out acceptance configuration");
    }
    if (std::bit_cast<std::uint64_t>(options.random_effort_limit) !=
            std::bit_cast<std::uint64_t>(artifact.effort_limit) ||
        std::bit_cast<std::uint64_t>(
            options.random_slew_limit_per_control_step) !=
            std::bit_cast<std::uint64_t>(artifact.slew_limit)) {
        throw std::invalid_argument(
            "random control envelope does not match the learned policy");
    }
    const std::string reconstructed = sha256_hex(
        protocol_manifest_from_artifact(artifact, options).dump());
    if (reconstructed != artifact.training.protocol_manifest_sha256) {
        throw std::invalid_argument(
            "held-out acceptance configuration does not match the "
            "preregistered protocol manifest");
    }
}

void require_training_request(const PolicyTrainingRequest& request) {
    require_exact_seed_splits(request.seed_splits);
    validate_cem_options(request.optimizer);
    if (request.episode_profile != "light_search_1d_v1" ||
        request.max_control_steps == 0 ||
        request.candidate_evaluation_workers == 0) {
        throw std::invalid_argument(
            "native learner requires light_search_1d_v1, a positive horizon, "
            "and at least one candidate evaluator");
    }
    (void)make_train_batches(
        request.seed_splits.train, request.train_batch_size);
    const std::array<double, kPolicyFeatureCount> zero_parameters{};
    const PolicyArtifactData declared_protocol = ephemeral_artifact(
        request, zero_parameters);
    if (declared_protocol.training.protocol_manifest_sha256 ==
            learning_experiment_manifest_sha256() &&
        sha256_hex(protocol_manifest_from_artifact(
                       declared_protocol, AcceptanceOptions{})
                       .dump()) != learning_experiment_manifest_sha256()) {
        throw std::invalid_argument(
            "training request differs from the default protocol but claims "
            "its manifest digest");
    }
}

}  // namespace

bool light_search_seed_has_positive_side(std::uint64_t seed) noexcept {
    return light_search_1d_realization(seed).positive_side;
}

std::uint32_t light_search_realization_key(std::uint64_t seed) noexcept {
    return light_search_1d_realization_key(seed);
}

MirroredSeedSplits make_mirrored_seed_splits(
    std::uint64_t base_seed,
    std::size_t train_pairs,
    std::size_t validation_pairs,
    std::size_t test_pairs) {
    std::set<std::uint64_t> used_seeds;
    std::set<std::uint32_t> used_realizations;
    MirroredSeedSplits result;
    result.base_seed = base_seed;
    result.train = make_balanced_split(
        base_seed,
        kTrainSplitTag,
        train_pairs,
        used_seeds,
        used_realizations);
    result.validation = make_balanced_split(
        base_seed,
        kValidationSplitTag,
        validation_pairs,
        used_seeds,
        used_realizations);
    result.test = make_balanced_split(
        base_seed,
        kTestSplitTag,
        test_pairs,
        used_seeds,
        used_realizations);
    return result;
}

std::string seed_list_sha256(std::span<const std::uint64_t> seeds) {
    Json document = Json::array();
    for (const std::uint64_t seed : seeds) {
        document.push_back(seed);
    }
    return sha256_hex(document.dump());
}

bool policy_score_better(
    const PolicyScore& candidate,
    const PolicyScore& incumbent) noexcept {
    if (candidate.invalid_samples != incumbent.invalid_samples) {
        return candidate.invalid_samples < incumbent.invalid_samples;
    }
    if (candidate.terminated_episodes != incumbent.terminated_episodes) {
        return candidate.terminated_episodes < incumbent.terminated_episodes;
    }
    if (candidate.safety_vetoes != incumbent.safety_vetoes) {
        return candidate.safety_vetoes < incumbent.safety_vetoes;
    }
    const double candidate_worst = worst_side_reward_rate_gain(candidate);
    const double incumbent_worst = worst_side_reward_rate_gain(incumbent);
    if (candidate_worst != incumbent_worst) {
        return candidate_worst > incumbent_worst;
    }
    const double candidate_balanced = balanced_reward_rate_gain(candidate);
    const double incumbent_balanced = balanced_reward_rate_gain(incumbent);
    if (candidate_balanced != incumbent_balanced) {
        return candidate_balanced > incumbent_balanced;
    }
    return candidate.mean_return > incumbent.mean_return;
}

CemTrainingResult train_cem_diagonal(
    const CemOptions& options,
    const std::vector<std::vector<std::uint64_t>>& common_train_batches,
    std::span<const std::uint64_t> validation_seeds,
    const ParameterBatchEvaluator& evaluator,
    std::size_t candidate_evaluation_workers) {
    validate_cem_options(options);
    if (!evaluator || candidate_evaluation_workers == 0 ||
        common_train_batches.empty() || validation_seeds.empty() ||
        std::any_of(
            common_train_batches.begin(),
            common_train_batches.end(),
            [](const auto& batch) { return batch.empty(); })) {
        throw std::invalid_argument(
            "cem_masked_best_sample_v1 requires evaluator and non-empty seed batches");
    }

    LatentParameters mean{};
    LatentParameters standard_deviation{};
    standard_deviation.fill(options.initial_stddev);

    const std::array<double, kPolicyFeatureCount> zero_parameters{};
    PolicyScore best_validation = evaluator(zero_parameters, validation_seeds);
    validate_score(best_validation);
    std::array<double, kPolicyFeatureCount> best_parameters = zero_parameters;
    std::size_t selected_generation = 0;
    Json history = Json::array({
        Json{
            {"generation", 0},
            {"phase", "initial_validation"},
            {"validation", score_json(best_validation)},
            {"checkpoint_rule", kCemCheckpointRule},
            {"parameter_sha256", parameter_sha256(zero_parameters)},
            {"checkpoint_parameter_sha256",
             parameter_sha256(zero_parameters)},
        },
    });

    struct Candidate {
        std::size_t index{0};
        LatentParameters latent{};
        std::array<double, kPolicyFeatureCount> parameters{};
        PolicyScore score;
    };

    for (std::size_t generation = 1;
         generation <= options.generations;
         ++generation) {
        const auto& batch = common_train_batches[
            (generation - 1U) % common_train_batches.size()];
        std::vector<Candidate> candidates(options.population_size);
        for (std::size_t candidate_index = 0;
             candidate_index < candidates.size();
             ++candidate_index) {
            Candidate& candidate = candidates[candidate_index];
            candidate.index = candidate_index;
            for (std::size_t latent_index = 0;
                 latent_index < candidate.latent.size();
                 ++latent_index) {
                const std::size_t physical_index =
                    kCemActiveParameterIndices[latent_index];
                candidate.latent[latent_index] = mean[latent_index] +
                    standard_deviation[latent_index] *
                    irwin_hall_normal(
                        options.optimizer_seed,
                        generation,
                        candidate_index,
                        physical_index);
            }
            candidate.parameters = physical_parameters(candidate.latent);
        }

        // Parameter generation is complete before concurrency begins, so the
        // sampled population and candidate indices cannot depend on scheduling.
        // Each worker owns one indexed score slot; the ordered reduction below
        // remains identical to the serial path.
        std::vector<std::exception_ptr> evaluation_errors(candidates.size());
        const auto evaluate_candidate = [&](std::size_t candidate_index) {
            try {
                Candidate& candidate = candidates[candidate_index];
                candidate.score = evaluator(candidate.parameters, batch);
                validate_score(candidate.score);
            } catch (...) {
                evaluation_errors[candidate_index] =
                    std::current_exception();
            }
        };
        if (candidate_evaluation_workers == 1) {
            for (std::size_t candidate_index = 0;
                 candidate_index < candidates.size();
                 ++candidate_index) {
                evaluate_candidate(candidate_index);
            }
        } else {
            std::atomic<std::size_t> next_candidate{0};
            const std::size_t worker_count = std::min(
                candidate_evaluation_workers, candidates.size());
            std::vector<std::jthread> workers;
            workers.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                workers.emplace_back([&] {
                    while (true) {
                        const std::size_t candidate_index =
                            next_candidate.fetch_add(
                                1, std::memory_order_relaxed);
                        if (candidate_index >= candidates.size()) {
                            return;
                        }
                        evaluate_candidate(candidate_index);
                    }
                });
            }
            workers.clear();
        }
        for (const std::exception_ptr& error : evaluation_errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& first, const Candidate& second) {
                if (policy_score_better(first.score, second.score)) {
                    return true;
                }
                if (policy_score_better(second.score, first.score)) {
                    return false;
                }
                return first.index < second.index;
            });

        // Checkpoint the sample selected exclusively by the current training
        // batch. Validation can choose between such checkpoints, but its score
        // never feeds the population, distribution, or future samples.
        const std::array<double, kPolicyFeatureCount> checkpoint_parameters =
            candidates.front().parameters;
        const std::size_t checkpoint_candidate_index =
            candidates.front().index;
        const PolicyScore checkpoint_training_score =
            candidates.front().score;

        LatentParameters elite_mean{};
        for (std::size_t latent_index = 0;
             latent_index < elite_mean.size();
             ++latent_index) {
            long double sum = 0.0L;
            for (std::size_t elite = 0; elite < options.elite_count; ++elite) {
                sum += candidates[elite].latent[latent_index];
            }
            elite_mean[latent_index] = static_cast<double>(
                sum / static_cast<long double>(options.elite_count));
        }
        for (std::size_t latent_index = 0;
             latent_index < elite_mean.size();
             ++latent_index) {
            long double squared = 0.0L;
            for (std::size_t elite = 0; elite < options.elite_count; ++elite) {
                const long double difference =
                    candidates[elite].latent[latent_index] -
                    elite_mean[latent_index];
                squared += difference * difference;
            }
            const double elite_stddev = std::sqrt(static_cast<double>(
                squared / static_cast<long double>(options.elite_count)));
            mean[latent_index] =
                (1.0 - options.update_rate) * mean[latent_index] +
                options.update_rate * elite_mean[latent_index];
            standard_deviation[latent_index] = std::max(
                options.minimum_stddev,
                (1.0 - options.update_rate) *
                        standard_deviation[latent_index] +
                    options.update_rate * elite_stddev);
        }

        const PolicyScore validation = evaluator(
            checkpoint_parameters, validation_seeds);
        validate_score(validation);
        if (policy_score_better(validation, best_validation)) {
            best_validation = validation;
            best_parameters = checkpoint_parameters;
            selected_generation = generation;
        }
        const auto distribution_mean_parameters = physical_parameters(mean);
        history.push_back(Json{
            {"generation", generation},
            {"phase", "generation"},
            {"train_batch_sha256", seed_list_sha256(batch)},
            {"best_training_candidate_index", checkpoint_candidate_index},
            {"best_training", score_json(checkpoint_training_score)},
            {"validation", score_json(validation)},
            {"checkpoint_rule", kCemCheckpointRule},
            {"parameter_sha256", parameter_sha256(checkpoint_parameters)},
            {"checkpoint_parameter_sha256",
             parameter_sha256(checkpoint_parameters)},
            {"distribution_mean_parameter_sha256",
             parameter_sha256(distribution_mean_parameters)},
        });
    }

    return CemTrainingResult{
        .parameters = best_parameters,
        .validation_score = best_validation,
        .selected_generation = selected_generation,
        .deterministic_history = std::move(history),
    };
}

PolicyTrainingResult train_frozen_policy(
    const PolicyTrainingRequest& request,
    const ParameterBatchEvaluator& evaluator) {
    require_training_request(request);
    const auto batches = make_train_batches(
        request.seed_splits.train, request.train_batch_size);
    const CemTrainingResult trained = train_cem_diagonal(
        request.optimizer,
        batches,
        request.seed_splits.validation,
        evaluator,
        request.candidate_evaluation_workers);
    PolicyArtifactData artifact = ephemeral_artifact(
        request, trained.parameters, trained.selected_generation);
    // Round-trip through the strict schema so the returned object is exactly
    // the frozen, self-hashed representation that evaluation will reload.
    artifact = parse_policy_artifact(policy_artifact_json(artifact));
    return PolicyTrainingResult{
        .artifact = artifact,
        .deterministic_report = Json{
            {"schema_version", kLearningExperimentSchemaVersion},
            {"algorithm_id", kCemAlgorithmId},
            {"search_space",
             Json{
                 {"id", kCemSearchSpaceId},
                 {"active_parameter_indices", kCemActiveParameterIndices},
                 {"parameter_scales", kCemParameterScales},
                 {"inactive_parameters", kCemInactiveParameterPolicy},
                 {"latent_initial_mean", 0.0},
             }},
            {"sampler",
             Json{
                 {"id", "splitmix64_irwin_hall_12_v1"},
                 {"parameter_key", kCemSamplerParameterKey},
                 {"population_sampling_rule",
                  kCemPopulationSamplingRule},
             }},
            {"checkpoint_rule", kCemCheckpointRule},
            {"objective",
             Json{
                 {"id", "paired_zero_side_robust_reward_rate_v1"},
                 {"zero_baseline_scope", "train_and_validation_only"},
                 {"ordering",
                  Json::array({
                      "minimum_invalid_samples",
                      "minimum_terminated_episodes",
                      "minimum_safety_vetoes",
                      "maximum_worst_side_mean_reward_rate_gain_vs_zero",
                      "maximum_fixed_half_each_side_mean_reward_rate_gain_vs_zero",
                      "maximum_mean_return",
                      "stable_candidate_index_or_earliest_generation",
                  })},
             }},
            {"episode_profile", request.episode_profile},
            {"max_control_steps", request.max_control_steps},
            {"execution",
             Json{{"candidate_evaluation_workers",
                   request.candidate_evaluation_workers}}},
            {"artifact_sha256", artifact.self_sha256},
            {"artifact_sha256_role",
             "integrity_checksum_not_authentication"},
            {"protocol_manifest_sha256",
             artifact.training.protocol_manifest_sha256},
            {"selected_generation", trained.selected_generation},
            {"validation", score_json(trained.validation_score)},
            {"seed_splits",
             Json{
                 {"train_count", request.seed_splits.train.size()},
                 {"train_sha256", seed_list_sha256(request.seed_splits.train)},
                 {"validation_count", request.seed_splits.validation.size()},
                 {"validation_sha256",
                  seed_list_sha256(request.seed_splits.validation)},
                 {"test_count", request.seed_splits.test.size()},
                 {"test_sha256", seed_list_sha256(request.seed_splits.test)},
             }},
            {"history", trained.deterministic_history},
        },
    };
}

PolicyTrainingResult train_frozen_policy_in_environment(
    const std::filesystem::path& model_path,
    const PolicyTrainingRequest& request) {
    // Validate the entire declared protocol before opening an environment or
    // executing any baseline rollout.
    require_training_request(request);
    std::vector<std::uint64_t> zero_baseline_seeds;
    zero_baseline_seeds.reserve(
        request.seed_splits.train.size() +
        request.seed_splits.validation.size());
    zero_baseline_seeds.insert(
        zero_baseline_seeds.end(),
        request.seed_splits.train.begin(),
        request.seed_splits.train.end());
    zero_baseline_seeds.insert(
        zero_baseline_seeds.end(),
        request.seed_splits.validation.begin(),
        request.seed_splits.validation.end());

    PolicyEvaluationOptions zero_options;
    zero_options.assembly_id = request.compatibility.assembly_id;
    zero_options.episode_profile = request.episode_profile;
    zero_options.episodes = zero_baseline_seeds.size();
    zero_options.max_control_steps = request.max_control_steps;
    zero_options.base_seed = request.optimizer.optimizer_seed;
    zero_options.control_dt_s = request.compatibility.control_dt_s;
    zero_options.explicit_environment_seeds = zero_baseline_seeds;
    PolicyEvaluationDescriptor zero_descriptor;
    zero_descriptor.id = "zero_v0";
    zero_descriptor.uses_observation = false;
    zero_descriptor.uses_policy_seed = false;
    zero_descriptor.metadata = Json{
        {"role", "training_objective_baseline"},
        {"seed_scope", "train_and_validation_only"},
    };
    const Json zero_evaluation = evaluate_policy(
        model_path,
        zero_descriptor,
        [](const Json& unused_observation,
           const std::vector<std::string>& motor_ids,
           std::size_t unused_control_step,
           std::uint64_t unused_policy_seed) {
            (void)unused_observation;
            (void)unused_control_step;
            (void)unused_policy_seed;
            std::map<std::string, double> actions;
            for (const std::string& motor_id : motor_ids) {
                actions.emplace(motor_id, 0.0);
            }
            return actions;
        },
        zero_options);
    const ZeroRewardRateBySeed zero_reward_rates =
        zero_reward_rates_from_evaluation(
            zero_evaluation, zero_baseline_seeds);

    const ParameterBatchEvaluator evaluator =
        [model_path, request, zero_reward_rates](
            std::span<const double> parameters,
            std::span<const std::uint64_t> seeds) {
            PolicyArtifactData candidate = ephemeral_artifact(
                request, parameters);
            SharedLinearMemoryPolicy policy(candidate);
            PolicyEvaluationDescriptor descriptor;
            descriptor.id = std::string(kPolicyArchitectureId);
            descriptor.uses_observation = true;
            descriptor.uses_policy_seed = false;
            descriptor.metadata = Json{
                {"encoder_id", kPolicyEncoderId},
                {"training_candidate", true},
            };
            PolicyEvaluationOptions options;
            options.assembly_id = request.compatibility.assembly_id;
            options.episode_profile = request.episode_profile;
            options.episodes = seeds.size();
            options.max_control_steps = request.max_control_steps;
            options.base_seed = request.optimizer.optimizer_seed;
            options.control_dt_s = request.compatibility.control_dt_s;
            options.explicit_environment_seeds.assign(seeds.begin(), seeds.end());
            const Json evaluation = evaluate_policy(
                model_path,
                descriptor,
                [&policy](
                    const Json& observation,
                    const std::vector<std::string>& motor_ids,
                    std::size_t control_step,
                    std::uint64_t unused_policy_seed) {
                    (void)unused_policy_seed;
                    if (control_step == 0) {
                        policy.reset(motor_ids);
                    }
                    return policy.act(observation, motor_ids);
                },
                options);
            return score_from_evaluation(evaluation, zero_reward_rates);
        };
    PolicyTrainingResult result = train_frozen_policy(request, evaluator);
    result.deterministic_report["zero_baseline_precomputation"] = Json{
        {"policy_id", "zero_v0"},
        {"seed_scope", "train_and_validation_only"},
        {"train_seed_count", request.seed_splits.train.size()},
        {"train_seeds_sha256",
         seed_list_sha256(request.seed_splits.train)},
        {"validation_seed_count", request.seed_splits.validation.size()},
        {"validation_seeds_sha256",
         seed_list_sha256(request.seed_splits.validation)},
        {"total_seed_count", zero_baseline_seeds.size()},
        {"test_seed_count_evaluated", 0},
        {"aggregate", zero_evaluation.at("aggregate")},
    };
    return result;
}

Json evaluate_frozen_policy(
    const std::filesystem::path& model_path,
    const PolicyArtifactData& supplied_artifact,
    const PolicyCompatibility& actual_runtime,
    const PolicyEvaluationOptions& options) {
    const PolicyArtifactData artifact = require_frozen_artifact(
        supplied_artifact, "frozen policy evaluation");
    require_policy_compatible(artifact, actual_runtime);
    if (options.assembly_id != artifact.compatibility.assembly_id ||
        options.episode_profile != artifact.training.scenario_profile_id ||
        options.max_control_steps != artifact.training.max_control_steps ||
        std::bit_cast<std::uint64_t>(options.control_dt_s) !=
            std::bit_cast<std::uint64_t>(artifact.compatibility.control_dt_s)) {
        throw std::invalid_argument(
            "frozen evaluation options do not match policy artifact");
    }

    SharedLinearMemoryPolicy policy(artifact);
    PolicyEvaluationDescriptor descriptor;
    descriptor.id = std::string(kPolicyArchitectureId);
    descriptor.uses_observation = true;
    descriptor.uses_policy_seed = false;
    descriptor.metadata = Json{
        {"artifact_schema_version", kPolicyArtifactSchemaVersion},
        {"artifact_sha256", artifact.self_sha256},
        {"artifact_sha256_role", "integrity_checksum_not_authentication"},
        {"encoder_id", kPolicyEncoderId},
        {"activation_id", kPolicyActivationId},
        {"frozen", true},
    };
    return evaluate_policy(
        model_path,
        descriptor,
        [&policy](
            const Json& observation,
            const std::vector<std::string>& motor_ids,
            std::size_t control_step,
            std::uint64_t unused_policy_seed) {
            (void)unused_policy_seed;
            if (control_step == 0) {
                policy.reset(motor_ids);
            }
            return policy.act(observation, motor_ids);
        },
        options);
}

Json v3_heldout_report_governance() {
    return Json{
        {"intended_use", "one_preregistered_heldout_execution"},
        {"durable_exclusive_report_required", true},
        {"global_single_execution_enforced_by_software", false},
        {"limitation",
         "a new path or copied workspace can bypass local exclusivity"},
        {"current_protocol_feedback_rule",
         "do_not_use_this_v3_test_split_to_tune_or_select_v3_"
         "artifacts_checkpoints_or_optimizer_choices"},
        {"later_version_curriculum_rule",
         "after_the_v3_decision_is_fixed_this_split_may_be_declared_"
         "as_curriculum_for_a_later_protocol_version"},
        {"later_version_unseen_status",
         "if_promoted_it_is_not_unseen_heldout_data_for_that_later_version"},
    };
}

Json assess_heldout_policy(
    std::span<const HeldoutScenarioOutcome> supplied_scenarios,
    const PolicyArtifactData& supplied_artifact,
    const AcceptanceOptions& options) {
    const PolicyArtifactData artifact = require_frozen_artifact(
        supplied_artifact, "held-out assessment");
    const MirroredSeedSplits expected = require_artifact_seed_protocol(artifact);
    require_acceptance_protocol(artifact, options);
    if (supplied_scenarios.size() != expected.test.size()) {
        throw std::invalid_argument(
            "held-out scenarios do not match the frozen test split size");
    }

    std::map<std::uint64_t, const HeldoutScenarioOutcome*> supplied_by_seed;
    for (const HeldoutScenarioOutcome& scenario : supplied_scenarios) {
        if (!supplied_by_seed
                 .emplace(scenario.learned.environment_seed, &scenario)
                 .second) {
            throw std::invalid_argument(
                "held-out environment seeds must be unique");
        }
    }
    std::vector<HeldoutScenarioOutcome> scenarios;
    scenarios.reserve(expected.test.size());
    for (const std::uint64_t seed : expected.test) {
        const auto found = supplied_by_seed.find(seed);
        if (found == supplied_by_seed.end()) {
            throw std::invalid_argument(
                "held-out scenarios do not match the frozen test split");
        }
        scenarios.push_back(*found->second);
    }
    if (supplied_by_seed.size() != expected.test.size()) {
        throw std::invalid_argument(
            "held-out scenarios contain seeds outside the frozen test split");
    }

    std::set<std::uint32_t> unique_realizations;
    std::vector<double> learned_returns;
    std::vector<double> zero_returns;
    std::vector<double> random_returns;
    std::vector<double> learned_rates;
    std::vector<double> zero_rates;
    std::vector<double> random_rates;
    std::vector<double> rate_delta_zero;
    std::vector<double> rate_delta_random;
    std::vector<double> return_delta_zero;
    std::vector<double> return_delta_random;
    std::array<std::vector<double>, 2> side_rate_delta_zero;
    std::array<std::vector<double>, 2> side_rate_delta_random;
    std::array<std::size_t, 2> side_wins_zero{};
    std::array<std::size_t, 2> side_wins_random{};
    std::array<std::size_t, 2> side_ties_zero{};
    std::array<std::size_t, 2> side_ties_random{};
    std::uint64_t learned_terminations = 0;
    std::uint64_t learned_vetoes = 0;
    std::uint64_t learned_invalid = 0;
    std::size_t wins_zero = 0;
    std::size_t wins_random = 0;
    std::size_t ties_zero = 0;
    std::size_t ties_random = 0;
    Json scenario_documents = Json::array();

    for (const HeldoutScenarioOutcome& scenario : scenarios) {
        const std::uint64_t seed = scenario.learned.environment_seed;
        const LightSearch1dRealization realization =
            light_search_1d_realization(seed);
        if (!unique_realizations
                 .emplace(light_search_1d_realization_key(seed))
                 .second) {
            throw std::invalid_argument(
                "held-out environment realizations must be unique");
        }
        const std::size_t side_index = realization.positive_side ? 1U : 0U;
        validate_episode_outcome(scenario.learned, seed, "learned outcome");
        validate_episode_outcome(scenario.zero, seed, "zero outcome");
        if (scenario.random.size() != options.random_streams_per_scenario) {
            throw std::invalid_argument(
                "held-out scenario has wrong random-stream count");
        }
        std::vector<double> per_stream_returns;
        std::vector<double> per_stream_rates;
        Json random_documents = Json::array();
        for (const EpisodeOutcome& random : scenario.random) {
            validate_episode_outcome(random, seed, "random outcome");
            per_stream_returns.push_back(random.episode_return);
            per_stream_rates.push_back(random.mean_reward_rate);
            random_documents.push_back(episode_outcome_json(random));
        }
        const double random_return = ordered_mean(per_stream_returns);
        const double random_rate = ordered_mean(per_stream_rates);
        const double delta_zero_rate =
            scenario.learned.mean_reward_rate - scenario.zero.mean_reward_rate;
        const double delta_random_rate =
            scenario.learned.mean_reward_rate - random_rate;
        const double delta_zero_return =
            scenario.learned.episode_return - scenario.zero.episode_return;
        const double delta_random_return =
            scenario.learned.episode_return - random_return;
        learned_returns.push_back(scenario.learned.episode_return);
        zero_returns.push_back(scenario.zero.episode_return);
        random_returns.push_back(random_return);
        learned_rates.push_back(scenario.learned.mean_reward_rate);
        zero_rates.push_back(scenario.zero.mean_reward_rate);
        random_rates.push_back(random_rate);
        rate_delta_zero.push_back(delta_zero_rate);
        rate_delta_random.push_back(delta_random_rate);
        side_rate_delta_zero[side_index].push_back(delta_zero_rate);
        side_rate_delta_random[side_index].push_back(delta_random_rate);
        return_delta_zero.push_back(delta_zero_return);
        return_delta_random.push_back(delta_random_return);
        learned_terminations += scenario.learned.terminated ? 1U : 0U;
        learned_vetoes += scenario.learned.safety_vetoes;
        learned_invalid += scenario.learned.invalid_samples;
        if (delta_zero_rate > 0.0) {
            ++wins_zero;
            ++side_wins_zero[side_index];
        } else if (delta_zero_rate == 0.0) {
            ++ties_zero;
            ++side_ties_zero[side_index];
        }
        if (delta_random_rate > 0.0) {
            ++wins_random;
            ++side_wins_random[side_index];
        } else if (delta_random_rate == 0.0) {
            ++ties_random;
            ++side_ties_random[side_index];
        }
        scenario_documents.push_back(Json{
            {"environment_seed", seed},
            {"light_side",
             realization.positive_side ? "positive_x" : "negative_x"},
            {"realization_key", light_search_1d_realization_key(seed)},
            {"distance_bucket", realization.distance_bucket},
            {"source_x_m", realization.source_x_m},
            {"learned", episode_outcome_json(scenario.learned)},
            {"zero", episode_outcome_json(scenario.zero)},
            {"random_streams", std::move(random_documents)},
            {"random_mean_return", random_return},
            {"random_mean_reward_rate", random_rate},
            {"learned_minus_zero_return", delta_zero_return},
            {"learned_minus_random_return", delta_random_return},
            {"learned_minus_zero_reward_rate", delta_zero_rate},
            {"learned_minus_random_reward_rate", delta_random_rate},
        });
    }
    if (side_rate_delta_zero[0].size() != side_rate_delta_zero[1].size() ||
        side_rate_delta_zero[0].empty()) {
        throw std::invalid_argument(
            "held-out scenarios must be balanced across hidden light sides");
    }

    const Interval zero_interval = side_stratified_paired_bootstrap_interval(
        side_rate_delta_zero[0],
        side_rate_delta_zero[1],
        options,
        0x5a45524fULL);
    const Interval random_interval =
        side_stratified_paired_bootstrap_interval(
            side_rate_delta_random[0],
            side_rate_delta_random[1],
            options,
            0x52414e44ULL);
    const std::array<double, 2> side_mean_delta_zero{
        ordered_mean(side_rate_delta_zero[0]),
        ordered_mean(side_rate_delta_zero[1])};
    const std::array<double, 2> side_mean_delta_random{
        ordered_mean(side_rate_delta_random[0]),
        ordered_mean(side_rate_delta_random[1])};
    const double mean_rate_delta_zero = 0.5 *
        (side_mean_delta_zero[0] + side_mean_delta_zero[1]);
    const double mean_rate_delta_random = 0.5 *
        (side_mean_delta_random[0] + side_mean_delta_random[1]);
    const bool safe = learned_terminations == 0 && learned_vetoes == 0 &&
        learned_invalid == 0;
    const bool intervals_pass =
        zero_interval.lower > 0.0 && random_interval.lower > 0.0;
    const bool practical_pass =
        mean_rate_delta_zero >= options.practical_reward_rate_gain &&
        mean_rate_delta_random >= options.practical_reward_rate_gain;
    const bool wins_pass = wins_zero >= options.minimum_wins &&
        wins_random >= options.minimum_wins;
    const bool side_means_pass =
        side_mean_delta_zero[0] > 0.0 &&
        side_mean_delta_zero[1] > 0.0 &&
        side_mean_delta_random[0] > 0.0 &&
        side_mean_delta_random[1] > 0.0;
    const bool side_wins_pass =
        side_wins_zero[0] >= options.minimum_wins_per_side &&
        side_wins_zero[1] >= options.minimum_wins_per_side &&
        side_wins_random[0] >= options.minimum_wins_per_side &&
        side_wins_random[1] >= options.minimum_wins_per_side;

    const auto side_json = [&](std::size_t side) {
        const std::size_t count = side_rate_delta_zero[side].size();
        return Json{
            {"scenario_count", count},
            {"learned_minus_zero_mean_reward_rate",
             side_mean_delta_zero[side]},
            {"learned_minus_random_mean_reward_rate",
             side_mean_delta_random[side]},
            {"learned_wins_vs_zero", side_wins_zero[side]},
            {"learned_ties_vs_zero", side_ties_zero[side]},
            {"learned_losses_vs_zero",
             count - side_wins_zero[side] - side_ties_zero[side]},
            {"learned_wins_vs_random", side_wins_random[side]},
            {"learned_ties_vs_random", side_ties_random[side]},
            {"learned_losses_vs_random",
             count - side_wins_random[side] - side_ties_random[side]},
        };
    };

    return Json{
        {"schema_version", "droid-blocks.heldout-acceptance.v1"},
        {"artifact_sha256", artifact.self_sha256},
        {"artifact_sha256_role", "integrity_checksum_not_authentication"},
        {"protocol_manifest_sha256",
         artifact.training.protocol_manifest_sha256},
        {"configuration",
         Json{
             {"scenario_count", options.expected_scenarios},
             {"random_streams_per_scenario",
              options.random_streams_per_scenario},
             {"random_control_base_seed",
              options.random_control_base_seed},
             {"random_effort_limit", options.random_effort_limit},
             {"random_hold_steps", options.random_hold_steps},
             {"random_slew_limit_per_control_step",
              options.random_slew_limit_per_control_step},
             {"bootstrap_method",
              "paired_side_stratified_percentile_v1"},
             {"bootstrap_resamples", options.bootstrap_resamples},
             {"bootstrap_seed", options.bootstrap_seed},
             {"one_sided_lower_confidence", 0.975},
             {"multiplicity_control",
              "bonferroni_two_one_sided_0.025"},
             {"joint_familywise_confidence", 0.95},
             {"practical_reward_rate_gain",
              options.practical_reward_rate_gain},
             {"minimum_wins", options.minimum_wins},
             {"minimum_wins_per_side", options.minimum_wins_per_side},
             {"statistical_unit", "environment_scenario"},
             {"random_stream_reduction",
              "arithmetic_mean_within_scenario_before_comparison"},
             {"confidence_scope",
              "fixed_weight_half_negative_x_half_positive_x_scenarios"},
         }},
        {"aggregate",
         Json{
             {"learned_mean_return", ordered_mean(learned_returns)},
             {"zero_mean_return", ordered_mean(zero_returns)},
             {"random_mean_return", ordered_mean(random_returns)},
             {"learned_mean_reward_rate", ordered_mean(learned_rates)},
             {"zero_mean_reward_rate", ordered_mean(zero_rates)},
             {"random_mean_reward_rate", ordered_mean(random_rates)},
             {"learned_minus_zero_mean_return",
              ordered_mean(return_delta_zero)},
             {"learned_minus_random_mean_return",
              ordered_mean(return_delta_random)},
             {"learned_minus_zero_mean_reward_rate",
              mean_rate_delta_zero},
             {"learned_minus_random_mean_reward_rate",
              mean_rate_delta_random},
             {"learned_minus_zero_reward_rate_bootstrap_95pct",
              Json{{"lower", zero_interval.lower},
                   {"upper", zero_interval.upper}}},
             {"learned_minus_random_reward_rate_bootstrap_95pct",
              Json{{"lower", random_interval.lower},
                   {"upper", random_interval.upper}}},
             {"learned_wins_vs_zero", wins_zero},
             {"learned_ties_vs_zero", ties_zero},
             {"learned_losses_vs_zero",
              scenarios.size() - wins_zero - ties_zero},
             {"learned_wins_vs_random", wins_random},
             {"learned_ties_vs_random", ties_random},
             {"learned_losses_vs_random",
              scenarios.size() - wins_random - ties_random},
             {"learned_terminated_episodes", learned_terminations},
             {"learned_safety_vetoes", learned_vetoes},
             {"learned_invalid_samples", learned_invalid},
             {"by_light_side",
              Json{{"negative_x", side_json(0)},
                   {"positive_x", side_json(1)}}},
         }},
        {"criteria",
         Json{
             {"safe", safe},
             {"both_bootstrap_lower_bounds_above_zero", intervals_pass},
             {"both_practical_margins_met", practical_pass},
             {"both_minimum_win_counts_met", wins_pass},
             {"positive_mean_delta_in_each_light_side_vs_both",
              side_means_pass},
             {"minimum_win_counts_met_in_each_light_side_vs_both",
              side_wins_pass},
         }},
        {"accepted",
         safe && intervals_pass && practical_pass && wins_pass &&
             side_means_pass && side_wins_pass},
        {"scenarios", std::move(scenario_documents)},
    };
}

Json evaluate_frozen_heldout(
    const std::filesystem::path& model_path,
    const PolicyArtifactData& supplied_artifact,
    const PolicyCompatibility& actual_runtime,
    PolicyEvaluationOptions options,
    const AcceptanceOptions& acceptance) {
    const PolicyArtifactData artifact = require_frozen_artifact(
        supplied_artifact, "frozen held-out evaluation");
    const MirroredSeedSplits expected = require_artifact_seed_protocol(artifact);
    require_acceptance_protocol(artifact, acceptance);
    if (options.explicit_environment_seeds.size() != expected.test.size()) {
        throw std::invalid_argument(
            "held-out evaluation seed list does not match frozen test split");
    }
    const std::set<std::uint64_t> supplied_seeds(
        options.explicit_environment_seeds.begin(),
        options.explicit_environment_seeds.end());
    const std::set<std::uint64_t> expected_seeds(
        expected.test.begin(), expected.test.end());
    if (supplied_seeds.size() != expected.test.size() ||
        supplied_seeds != expected_seeds) {
        throw std::invalid_argument(
            "held-out evaluation seed list does not match frozen test split");
    }
    // Canonicalize rollout and reduction order after validating the exact set.
    options.explicit_environment_seeds = expected.test;
    options.episodes = expected.test.size();
    const Json learned = evaluate_frozen_policy(
        model_path, artifact, actual_runtime, options);

    PolicyEvaluationDescriptor zero_descriptor;
    zero_descriptor.id = "zero_v0";
    zero_descriptor.uses_observation = false;
    zero_descriptor.uses_policy_seed = false;
    const Json zero = evaluate_policy(
        model_path,
        zero_descriptor,
        [](const Json& unused_observation,
           const std::vector<std::string>& motor_ids,
           std::size_t unused_step,
           std::uint64_t unused_seed) {
            (void)unused_observation;
            (void)unused_step;
            (void)unused_seed;
            std::map<std::string, double> actions;
            for (const std::string& motor_id : motor_ids) {
                actions.emplace(motor_id, 0.0);
            }
            return actions;
        },
        options);

    std::vector<Json> random_evaluations;
    random_evaluations.reserve(acceptance.random_streams_per_scenario);
    BaselineEvaluationOptions random_options;
    random_options.assembly_id = options.assembly_id;
    random_options.episodes = options.episodes;
    random_options.max_control_steps = options.max_control_steps;
    random_options.control_dt_s = options.control_dt_s;
    random_options.random_effort_limit = acceptance.random_effort_limit;
    random_options.random_hold_steps = acceptance.random_hold_steps;
    for (std::size_t stream = 0;
         stream < acceptance.random_streams_per_scenario;
         ++stream) {
        PolicyEvaluationOptions random_run = options;
        random_run.base_seed = mix64(
            acceptance.random_control_base_seed ^ kRandomControlTag ^
            mix64(static_cast<std::uint64_t>(stream)));
        PolicyEvaluationDescriptor random_descriptor;
        random_descriptor.id = "seeded_random_sample_hold_slew_v1";
        random_descriptor.uses_observation = false;
        random_descriptor.uses_policy_seed = true;
        random_descriptor.metadata = Json{
            {"stream_index", stream},
            {"random_control_base_seed",
             acceptance.random_control_base_seed},
            {"effort_limit", acceptance.random_effort_limit},
            {"hold_steps", acceptance.random_hold_steps},
            {"slew_limit_per_control_step",
             acceptance.random_slew_limit_per_control_step},
        };
        random_evaluations.push_back(evaluate_policy(
            model_path,
            random_descriptor,
            [random_options,
             slew_limit = acceptance.random_slew_limit_per_control_step,
             previous = std::map<std::string, double>{}](
                const Json& unused_observation,
                const std::vector<std::string>& motor_ids,
                std::size_t control_step,
                std::uint64_t policy_seed) mutable {
                (void)unused_observation;
                if (control_step == 0) {
                    previous.clear();
                    for (const std::string& motor_id : motor_ids) {
                        previous.emplace(motor_id, 0.0);
                    }
                }
                const std::map<std::string, double> targets =
                    make_baseline_actions(
                    BaselinePolicyKind::seeded_random,
                    motor_ids,
                    control_step,
                    policy_seed,
                    random_options);
                std::map<std::string, double> actions;
                for (const std::string& motor_id : motor_ids) {
                    const auto found = previous.find(motor_id);
                    if (found == previous.end()) {
                        throw std::runtime_error(
                            "random control motor topology changed mid-episode");
                    }
                    actions.emplace(
                        motor_id,
                        std::clamp(
                            targets.at(motor_id),
                            found->second - slew_limit,
                            found->second + slew_limit));
                }
                previous = actions;
                return actions;
            },
            random_run));
    }

    const Json& learned_episodes = learned.at("episodes");
    const Json& zero_episodes = zero.at("episodes");
    std::vector<HeldoutScenarioOutcome> scenarios;
    scenarios.reserve(options.episodes);
    for (std::size_t index = 0; index < options.episodes; ++index) {
        HeldoutScenarioOutcome scenario;
        scenario.learned = episode_outcome(learned_episodes.at(index));
        scenario.zero = episode_outcome(zero_episodes.at(index));
        for (const Json& random : random_evaluations) {
            scenario.random.push_back(
                episode_outcome(random.at("episodes").at(index)));
        }
        scenarios.push_back(std::move(scenario));
    }
    Json random_documents = Json::array();
    for (Json& random : random_evaluations) {
        random_documents.push_back(std::move(random));
    }
    return Json{
        {"schema_version", "droid-blocks.frozen-heldout-evaluation.v1"},
        {"learned", learned},
        {"zero", zero},
        {"random_streams", std::move(random_documents)},
        {"acceptance", assess_heldout_policy(scenarios, artifact, acceptance)},
    };
}

Json default_learning_experiment_manifest() {
    PolicyArtifactData artifact;
    artifact.compatibility.assembly_id = "demo_rover_v0";
    artifact.compatibility.control_dt_s = 0.02;
    artifact.training.algorithm_id = std::string(kCemAlgorithmId);
    artifact.training.scenario_profile_id = "light_search_1d_v1";
    artifact.training.seed_derivation_id =
        "balanced_light_side_splitmix64_v1";
    artifact.training.seed_split_base_seed = 0;
    artifact.training.optimizer_seed = 0;
    artifact.training.generations = 40;
    artifact.training.population_size = 32;
    artifact.training.elite_count = 8;
    artifact.training.train_batch_size = 8;
    artifact.training.max_control_steps = 1'000;
    artifact.training.initial_stddev = 1.0;
    artifact.training.minimum_stddev = 0.1;
    artifact.training.update_rate = 0.25;
    artifact.training.train_seed_count = 32;
    artifact.training.validation_seed_count = 16;
    artifact.training.test_seed_count = 64;
    artifact.effort_limit = 0.6;
    artifact.slew_limit = 0.1;
    return protocol_manifest_from_artifact(artifact, AcceptanceOptions{});
}

std::string learning_experiment_manifest_sha256() {
    static const std::string digest =
        sha256_hex(default_learning_experiment_manifest().dump());
    return digest;
}

}  // namespace droid
