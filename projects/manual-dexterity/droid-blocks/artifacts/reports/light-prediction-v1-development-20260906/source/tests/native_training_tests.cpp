#include "droid/learner.hpp"
#include "droid/environment.hpp"
#include "droid/module_catalog.hpp"
#include "droid/simulation.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
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

template <typename Callable>
void require_throws_containing(
    Callable&& callable,
    std::string_view expected,
    std::string_view message) {
    try {
        std::invoke(std::forward<Callable>(callable));
    } catch (const std::exception& error) {
        if (std::string_view(error.what()).find(expected) !=
            std::string_view::npos) {
            return;
        }
        throw std::runtime_error(
            std::string(message) + ": wrong error: " + error.what());
    }
    throw std::runtime_error(std::string(message));
}

[[nodiscard]] std::string digest(char digit) {
    return std::string(64, digit);
}

[[nodiscard]] droid::PolicyCompatibility compatibility() {
    return droid::PolicyCompatibility{
        .environment_api_version = "agent_environment_v1",
        .assembly_id = "demo_rover_v0",
        .motor_ids = {
            "motor-0001", "motor-0002", "motor-0003", "motor-0004"},
        .control_dt_s = 0.02,
        .model_sha256 = digest('1'),
        .module_catalog_sha256 = digest('2'),
        .environment_spec_sha256 = digest('3'),
        .feature_schema_sha256 = droid::policy_feature_schema_sha256(),
    };
}

[[nodiscard]] droid::PolicyTrainingRequest small_request() {
    droid::PolicyTrainingRequest request;
    request.compatibility = compatibility();
    request.optimizer.generations = 12;
    request.optimizer.population_size = 20;
    request.optimizer.elite_count = 5;
    request.optimizer.initial_stddev = 0.8;
    request.optimizer.minimum_stddev = 0.01;
    request.optimizer.update_rate = 0.65;
    request.optimizer.optimizer_seed = 0x123456789abcdef0ULL;
    request.seed_splits = droid::make_mirrored_seed_splits(91, 4, 2, 4);
    request.protocol_manifest_sha256 = digest('a');
    request.train_batch_size = 4;
    return request;
}

[[nodiscard]] droid::PolicyScore synthetic_score(
    std::span<const double> parameters,
    std::span<const std::uint64_t> unused_seeds) {
    (void)unused_seeds;
    const double error = parameters[29] - 1.75;
    return droid::PolicyScore{.mean_return = -(error * error)};
}

[[nodiscard]] droid::PolicyScore scalar_score(double value) {
    return droid::PolicyScore{
        .negative_x_mean_reward_rate_gain_vs_zero = value,
        .positive_x_mean_reward_rate_gain_vs_zero = value,
        .mean_return = value,
    };
}

void require_masked_parameters(
    std::span<const double> parameters,
    std::string_view context) {
    require(parameters.size() == droid::kPolicyFeatureCount,
            "masked parameter vector has wrong size");
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const bool active = std::find(
            droid::kCemActiveParameterIndices.begin(),
            droid::kCemActiveParameterIndices.end(),
            index) != droid::kCemActiveParameterIndices.end();
        if (!active) {
            require(
                std::bit_cast<std::uint64_t>(parameters[index]) == 0U,
                std::string(context) +
                    " contains a non-positive-zero inactive parameter");
        }
    }
}

void test_masked_search_space_and_sample_checkpoint() {
    droid::CemOptions options;
    options.generations = 1;
    options.population_size = 2;
    options.elite_count = 1;
    options.initial_stddev = 1.0;
    options.minimum_stddev = 0.1;
    options.update_rate = 1.0;
    options.optimizer_seed = 0;
    const std::vector<std::vector<std::uint64_t>> batches{{1, 2}};
    const std::vector<std::uint64_t> validation{9, 10};
    std::vector<std::array<double, droid::kPolicyFeatureCount>> calls;
    const auto evaluator = [&calls](
                               std::span<const double> parameters,
                               std::span<const std::uint64_t> unused_seeds) {
        (void)unused_seeds;
        std::array<double, droid::kPolicyFeatureCount> copy{};
        std::copy(parameters.begin(), parameters.end(), copy.begin());
        require_masked_parameters(copy, "CEM callback");
        calls.push_back(copy);
        return scalar_score(
            std::abs(copy[29]) + std::abs(copy[30]) +
            std::abs(copy[32]));
    };
    const droid::CemTrainingResult trained = droid::train_cem_diagonal(
        options, batches, validation, evaluator);

    require(calls.size() == 4,
            "one-generation two-candidate CEM made unexpected calls");
    require(calls[0] == std::array<double, droid::kPolicyFeatureCount>{},
            "CEM omitted its separate zero validation anchor");
    require(calls[1] != std::array<double, droid::kPolicyFeatureCount>{} &&
                calls[2] !=
                    std::array<double, droid::kPolicyFeatureCount>{},
            "a population slot was replaced by the distribution mean");
    require(
        droid::encode_binary64_hex(calls[1][29]) ==
                "0x3ffb6de99dd81610" &&
            droid::encode_binary64_hex(calls[1][30]) ==
                "0x3f91cbfb84405580" &&
            droid::encode_binary64_hex(calls[1][32]) ==
                "0xc04689b1097d8a28",
        "candidate-zero stochastic sampler golden changed");
    const auto& sampled = calls[2];
    require(
        droid::encode_binary64_hex(sampled[29]) ==
                "0x3ffb14e3d3c635d0" &&
            droid::encode_binary64_hex(sampled[30]) ==
                "0xbfcec566bb5ba208" &&
            droid::encode_binary64_hex(sampled[32]) ==
                "0x4048a058bd5b4d08",
        "physical-index-keyed masked sampler golden changed");
    require(calls[3] == sampled && trained.parameters == sampled,
            "validation/freeze checkpoint was not the train-best sample");
    require(trained.selected_generation == 1,
            "superior sampled checkpoint did not replace the zero anchor");
}

void test_bimodal_sample_checkpoint_escapes_zero() {
    droid::CemOptions options;
    options.generations = 2;
    options.population_size = 32;
    options.elite_count = 8;
    options.optimizer_seed = 0;
    std::vector<std::array<double, droid::kPolicyFeatureCount>> train_samples;
    const auto evaluator = [&train_samples](
                               std::span<const double> parameters,
                               std::span<const std::uint64_t> seeds) {
        std::array<double, droid::kPolicyFeatureCount> copy{};
        std::copy(parameters.begin(), parameters.end(), copy.begin());
        require_masked_parameters(copy, "bimodal callback");
        if (seeds.front() == 1U) {
            train_samples.push_back(copy);
        }
        // Either probe sign is equally useful; an inactive mean is not.
        const bool active_mode = copy[30] != 0.0 && copy[32] > 0.0;
        return scalar_score(active_mode ? 1.0 : 0.0);
    };
    const auto trained = droid::train_cem_diagonal(
        options, {{1, 2}}, std::array<std::uint64_t, 2>{9, 10}, evaluator);
    require(trained.selected_generation > 0 &&
                trained.parameters[30] != 0.0 &&
                trained.parameters[32] > 0.0,
            "best-sample checkpoint collapsed between symmetric probe modes");
    require(std::find(
                train_samples.begin(), train_samples.end(),
                trained.parameters) != train_samples.end(),
            "frozen bimodal solution was not an evaluated training sample");
}

void test_validation_cannot_change_future_samples() {
    const auto run = [](bool prefer_positive_recurrence) {
        droid::CemOptions options;
        options.generations = 3;
        options.population_size = 8;
        options.elite_count = 2;
        options.optimizer_seed = 123;
        const std::vector<std::vector<std::uint64_t>> batches{{1, 2}};
        const std::vector<std::uint64_t> validation{9, 10};
        std::vector<std::string> train_hashes;
        std::vector<std::pair<
            std::array<double, droid::kPolicyFeatureCount>,
            droid::PolicyScore>> generation;
        bool initial_validation = true;
        const auto evaluator = [&, prefer_positive_recurrence](
                                   std::span<const double> parameters,
                                   std::span<const std::uint64_t> seeds) {
            std::array<double, droid::kPolicyFeatureCount> copy{};
            std::copy(parameters.begin(), parameters.end(), copy.begin());
            require_masked_parameters(copy, "isolation callback");
            if (seeds.front() == 1U) {
                const double value =
                    -std::abs(copy[29] - 1.0) -
                    0.01 * std::abs(copy[32] - 32.0);
                const droid::PolicyScore score = scalar_score(value);
                train_hashes.push_back(droid::sha256_hex(
                    Json(copy).dump()));
                generation.emplace_back(copy, score);
                return score;
            }
            if (initial_validation) {
                initial_validation = false;
                require(copy ==
                            std::array<double, droid::kPolicyFeatureCount>{},
                        "initial validation did not receive zero");
            } else {
                require(generation.size() == options.population_size,
                        "validation ran before a complete train population");
                std::size_t winner = 0;
                for (std::size_t index = 1; index < generation.size(); ++index) {
                    if (droid::policy_score_better(
                            generation[index].second,
                            generation[winner].second)) {
                        winner = index;
                    }
                }
                require(copy == generation[winner].first,
                        "validation did not receive the preceding train winner");
                generation.clear();
            }
            return scalar_score(
                (prefer_positive_recurrence ? 1.0 : -1.0) * copy[29]);
        };
        (void)droid::train_cem_diagonal(
            options, batches, validation, evaluator);
        require(generation.empty(),
                "final training population was not checkpointed");
        return train_hashes;
    };

    require(run(true) == run(false),
            "validation scores changed later training samples");
}

void test_real_train_pair_recurrence_ablation() {
    const std::filesystem::path model{"models/droid.xml"};
    const droid::MirroredSeedSplits splits =
        droid::make_mirrored_seed_splits();
    const std::vector<std::uint64_t> train_pair{
        splits.train.at(0), splits.train.at(1)};
    require(
        train_pair == std::vector<std::uint64_t>{
            6106049681611768608ULL, 6990021049862345368ULL},
        "registered train-only recurrence pair changed");
    for (const std::uint64_t seed : train_pair) {
        require(std::find(splits.validation.begin(), splits.validation.end(), seed) ==
                        splits.validation.end() &&
                    std::find(splits.test.begin(), splits.test.end(), seed) ==
                        splits.test.end(),
                "recurrence ablation escaped the training split");
    }

    droid::PolicyArtifactData template_artifact =
        droid::train_frozen_policy(small_request(), synthetic_score).artifact;
    const auto evaluate_parameters = [&, template_artifact](
                                         std::array<double,
                                             droid::kPolicyFeatureCount>
                                             parameters,
                                         std::string id) {
        droid::PolicyArtifactData artifact = template_artifact;
        artifact.parameters = parameters;
        artifact.self_sha256.clear();
        droid::PolicyEvaluationDescriptor descriptor;
        descriptor.id = std::move(id);
        descriptor.uses_observation = true;
        descriptor.uses_policy_seed = false;
        descriptor.metadata = Json::object();
        droid::PolicyEvaluationOptions options;
        options.assembly_id = "demo_rover_v0";
        options.episode_profile = "light_search_1d_v1";
        options.episodes = train_pair.size();
        options.max_control_steps = 1'000;
        options.control_dt_s = 0.02;
        options.explicit_environment_seeds = train_pair;
        return droid::evaluate_policy(
            model,
            descriptor,
            [policy = droid::SharedLinearMemoryPolicy(std::move(artifact))](
                const Json& observation,
                const std::vector<std::string>& motor_ids,
                std::size_t control_step,
                std::uint64_t unused_policy_seed) mutable {
                (void)unused_policy_seed;
                if (control_step == 0) {
                    policy.reset(motor_ids);
                }
                return policy.act(observation, motor_ids);
            },
            options);
    };

    std::array<double, droid::kPolicyFeatureCount> recurrent{};
    recurrent[29] = 2.0;
    recurrent[30] = 0.03;
    recurrent[32] = 64.0;
    std::array<double, droid::kPolicyFeatureCount> feedforward = recurrent;
    feedforward[29] = 0.0;
    const std::array<double, droid::kPolicyFeatureCount> zero{};
    const Json zero_evaluation = evaluate_parameters(zero, "zero_train_pair_v0");
    const Json recurrent_evaluation =
        evaluate_parameters(recurrent, "recurrent_train_pair_v0");
    const Json feedforward_evaluation =
        evaluate_parameters(feedforward, "feedforward_train_pair_v0");

    const auto gains = [&zero_evaluation](const Json& evaluation) {
        std::array<double, 2> result{};
        const Json& zero_episodes = zero_evaluation.at("episodes");
        const Json& episodes = evaluation.at("episodes");
        require(episodes.size() == 2 && zero_episodes.size() == 2,
                "train-only ablation evaluated an unexpected episode count");
        for (std::size_t index = 0; index < 2; ++index) {
            require(episodes.at(index).at("environment_seed") ==
                        zero_episodes.at(index).at("environment_seed"),
                    "train-only ablation is not paired with zero control");
            result[index] =
                episodes.at(index).at("mean_reward_rate").get<double>() -
                zero_episodes.at(index)
                    .at("mean_reward_rate")
                    .get<double>();
        }
        const Json& aggregate = evaluation.at("aggregate");
        require(aggregate.at("terminated_episodes") == 0 &&
                    aggregate.at("safety").at("veto_count") == 0 &&
                    aggregate.at("health")
                            .at("invalid_reward_sensor_samples") == 0 &&
                    aggregate.at("health")
                            .at("invalid_actuator_feedback_samples") == 0,
                "train-only ablation was not safe and valid");
        return result;
    };
    const auto recurrent_gains = gains(recurrent_evaluation);
    const auto feedforward_gains = gains(feedforward_evaluation);
    require(!droid::light_search_seed_has_positive_side(train_pair[0]) &&
                droid::light_search_seed_has_positive_side(train_pair[1]),
            "recurrence ablation pair lost negative/positive ordering");
    require(recurrent_gains[0] > 0.07 && recurrent_gains[1] > 0.08,
            "registered recurrent controller did not improve both train sides");
    require(feedforward_gains[0] < 0.0 && feedforward_gains[1] > 0.07,
            "removing previous-effort recurrence no longer causes the train-side failure");
    require(recurrent_gains[0] > feedforward_gains[0] + 0.075,
            "previous-effort recurrence did not materially repair the failed train side");
}

void test_light_seed_mapping_and_splits() {
    require(droid::light_search_seed_has_positive_side(0),
            "golden seed 0 did not select positive light side");
    require(!droid::light_search_seed_has_positive_side(1),
            "golden seed 1 did not select negative light side");
    require(droid::light_search_realization_key(0) == 0x01e742acU &&
                droid::light_search_realization_key(1) == 0x00f9da24U &&
                droid::light_search_realization_key(
                    std::numeric_limits<std::uint64_t>::max()) ==
                    0x01a4e32eU,
            "light-search realization key golden vectors changed");
    require(
        droid::light_search_1d_realization(0).source_x_m ==
                19.2268886566162109375 &&
            droid::light_search_1d_realization(1).source_x_m ==
                -19.8078784942626953125 &&
            droid::light_search_1d_realization(
                std::numeric_limits<std::uint64_t>::max()).source_x_m ==
                17.15273189544677734375,
        "light-search distance golden vectors changed");
    const auto first = droid::make_mirrored_seed_splits();
    const auto second = droid::make_mirrored_seed_splits();
    require(first.train == second.train &&
                first.validation == second.validation && first.test == second.test,
            "seed split derivation was not exact");
    require(first.train.size() == 32 && first.validation.size() == 16 &&
                first.test.size() == 64,
            "default seed split sizes are wrong");
    std::set<std::uint64_t> all;
    std::set<std::uint32_t> realizations;
    const auto inspect = [&all, &realizations](
                             const std::vector<std::uint64_t>& split) {
        std::size_t positive = 0;
        for (const auto seed : split) {
            require(all.emplace(seed).second, "seed splits overlap");
            require(
                realizations.emplace(
                    droid::light_search_realization_key(seed)).second,
                "seed splits repeat a world realization");
            positive += droid::light_search_seed_has_positive_side(seed) ? 1U : 0U;
        }
        require(positive * 2U == split.size(), "seed split is not balanced");
        for (std::size_t index = 0; index < split.size(); index += 2U) {
            require(!droid::light_search_seed_has_positive_side(split[index]) &&
                        droid::light_search_seed_has_positive_side(
                            split[index + 1U]),
                    "seed split pair is not negative/positive mirrored");
        }
    };
    inspect(first.train);
    inspect(first.validation);
    inspect(first.test);
    const auto different = droid::make_mirrored_seed_splits(1);
    require(first.train != different.train,
            "base seed did not change derived split");
}

void test_safety_first_score_order() {
    const droid::PolicyScore safe{.mean_return = -1000.0};
    const droid::PolicyScore invalid{
        .invalid_samples = 1,
        .negative_x_mean_reward_rate_gain_vs_zero = 1.0e9,
        .positive_x_mean_reward_rate_gain_vs_zero = 1.0e9,
        .mean_return = 1.0e9};
    const droid::PolicyScore terminated{
        .terminated_episodes = 1,
        .negative_x_mean_reward_rate_gain_vs_zero = 1.0e9,
        .positive_x_mean_reward_rate_gain_vs_zero = 1.0e9,
        .mean_return = 1.0e9};
    const droid::PolicyScore vetoed{
        .safety_vetoes = 1,
        .negative_x_mean_reward_rate_gain_vs_zero = 1.0e9,
        .positive_x_mean_reward_rate_gain_vs_zero = 1.0e9,
        .mean_return = 1.0e9};
    require(droid::policy_score_better(safe, invalid),
            "reward compensated for invalid samples");
    require(droid::policy_score_better(safe, terminated),
            "reward compensated for termination");
    require(droid::policy_score_better(safe, vetoed),
            "reward compensated for safety veto");
    require(droid::policy_score_better(
                droid::PolicyScore{.mean_return = 2.0}, safe),
            "return did not rank candidates after equal safety");

    const droid::PolicyScore symmetric{
        .negative_x_mean_reward_rate_gain_vs_zero = 0.02,
        .positive_x_mean_reward_rate_gain_vs_zero = 0.01,
        .mean_return = -1.0e6};
    const droid::PolicyScore one_sided{
        .negative_x_mean_reward_rate_gain_vs_zero = 0.20,
        .positive_x_mean_reward_rate_gain_vs_zero = -0.10,
        .mean_return = 1.0e6};
    require(droid::policy_score_better(symmetric, one_sided),
            "strong performance on one side subsidized failure on the other");

    const droid::PolicyScore larger_balanced_gain{
        .negative_x_mean_reward_rate_gain_vs_zero = 0.01,
        .positive_x_mean_reward_rate_gain_vs_zero = 0.05,
        .mean_return = -1.0e6};
    const droid::PolicyScore smaller_balanced_gain{
        .negative_x_mean_reward_rate_gain_vs_zero = 0.01,
        .positive_x_mean_reward_rate_gain_vs_zero = 0.03,
        .mean_return = 1.0e6};
    require(
        droid::policy_score_better(
            larger_balanced_gain, smaller_balanced_gain),
        "balanced side gain did not break an equal worst-side tie");

    const droid::PolicyScore lower_return{
        .negative_x_mean_reward_rate_gain_vs_zero = 0.01,
        .positive_x_mean_reward_rate_gain_vs_zero = 0.05,
        .mean_return = 4.0};
    const droid::PolicyScore higher_return{
        .negative_x_mean_reward_rate_gain_vs_zero = 0.05,
        .positive_x_mean_reward_rate_gain_vs_zero = 0.01,
        .mean_return = 5.0};
    require(droid::policy_score_better(higher_return, lower_return),
            "mean return did not break an equal side-objective tie");
}

void test_common_batches_and_deterministic_cem() {
    droid::CemOptions options;
    options.generations = 3;
    options.population_size = 6;
    options.elite_count = 2;
    options.initial_stddev = 0.7;
    options.minimum_stddev = 0.02;
    options.update_rate = 0.7;
    options.optimizer_seed = 77;
    const std::vector<std::vector<std::uint64_t>> batches{{1, 2}, {3, 4}};
    const std::vector<std::uint64_t> validation{9, 10};
    std::vector<std::string> calls;
    const auto evaluator = [&calls](
                               std::span<const double> parameters,
                               std::span<const std::uint64_t> seeds) {
        calls.push_back(droid::seed_list_sha256(seeds));
        return synthetic_score(parameters, seeds);
    };
    const auto first = droid::train_cem_diagonal(
        options, batches, validation, evaluator);
    const std::string validation_hash = droid::seed_list_sha256(validation);
    require(calls.front() == validation_hash,
            "CEM did not validate the zero initialization first");
    std::size_t cursor = 1;
    for (std::size_t generation = 0; generation < options.generations;
         ++generation) {
        const std::string expected_batch =
            droid::seed_list_sha256(batches[generation % batches.size()]);
        for (std::size_t candidate = 0;
             candidate < options.population_size;
             ++candidate) {
            require(calls.at(cursor++) == expected_batch,
                    "candidates in one generation saw different train batches");
        }
        require(calls.at(cursor++) == validation_hash,
                "generation checkpoint used training rather than validation seeds");
    }
    require(cursor == calls.size(), "unexpected CEM evaluator calls");

    const auto repeated = droid::train_cem_diagonal(
        options,
        batches,
        validation,
        synthetic_score,
        4);
    require(first.parameters == repeated.parameters &&
                first.selected_generation == repeated.selected_generation &&
                first.deterministic_history.dump() ==
                    repeated.deterministic_history.dump(),
            "one-worker and four-worker CEM histories were not byte identical");
    const Json& initial_validation =
        first.deterministic_history.at(0).at("validation");
    require(
        initial_validation.contains(
            "negative_x_mean_reward_rate_gain_vs_zero") &&
            initial_validation.contains(
                "positive_x_mean_reward_rate_gain_vs_zero") &&
            initial_validation.contains(
                "worst_side_mean_reward_rate_gain_vs_zero") &&
            initial_validation.contains(
                "balanced_mean_reward_rate_gain_vs_zero"),
        "CEM history omitted the explicit side-robust score fields");
    require(first.validation_score.mean_return > -(1.75 * 1.75),
            "synthetic CEM objective did not improve over zero initialization");
}

void test_frozen_training_artifact() {
    const droid::PolicyTrainingRequest request = small_request();
    const auto first = droid::train_frozen_policy(request, synthetic_score);
    const auto second = droid::train_frozen_policy(request, synthetic_score);
    droid::PolicyTrainingRequest parallel_request = request;
    parallel_request.candidate_evaluation_workers = 4;
    const auto parallel = droid::train_frozen_policy(
        parallel_request, synthetic_score);
    const Json first_document = droid::policy_artifact_json(first.artifact);
    const Json second_document = droid::policy_artifact_json(second.artifact);
    require(first_document.dump() == second_document.dump(),
            "repeated training produced different frozen artifacts");
    require(first.deterministic_report.dump() == second.deterministic_report.dump(),
            "repeated training produced different deterministic reports");
    require(first_document.dump() ==
                droid::policy_artifact_json(parallel.artifact).dump(),
            "execution worker count changed frozen artifact identity");
    require(
        parallel.deterministic_report.at("execution")
                .at("candidate_evaluation_workers") == 4,
        "parallel worker count was not reported as execution metadata");
    require(
        first.deterministic_report.at("schema_version") ==
                droid::kLearningExperimentSchemaVersion &&
            first.deterministic_report.at("objective").at("id") ==
                "paired_zero_side_robust_reward_rate_v1" &&
            first.deterministic_report.at("objective")
                    .at("zero_baseline_scope") ==
                "train_and_validation_only",
        "training report omitted the side-robust paired-zero objective");
    require(
        first.deterministic_report.at("validation").contains(
            "worst_side_mean_reward_rate_gain_vs_zero"),
        "training report validation score omitted its worst-side objective");
    require(first.artifact.training.test_seed_count == 8,
            "artifact omitted untouched test split provenance");
    require(first.artifact.training.test_seeds_sha256 ==
                droid::seed_list_sha256(request.seed_splits.test),
            "artifact test split digest is wrong");
    require(first.artifact.training.selected_generation <=
                request.optimizer.generations,
            "artifact selected an impossible validation checkpoint");
    require(first.artifact.training.max_control_steps ==
                request.max_control_steps,
            "artifact omitted training horizon provenance");
}

[[nodiscard]] droid::PolicyArtifactData acceptance_artifact() {
    droid::PolicyTrainingRequest request;
    request.compatibility = compatibility();
    request.seed_splits = droid::make_mirrored_seed_splits();
    const auto trained = droid::train_frozen_policy(request, synthetic_score);
    return trained.artifact;
}

[[nodiscard]] std::vector<droid::HeldoutScenarioOutcome> passing_scenarios(
    const droid::PolicyArtifactData& artifact) {
    const auto splits = droid::make_mirrored_seed_splits(
        artifact.training.seed_split_base_seed,
        artifact.training.train_seed_count / 2U,
        artifact.training.validation_seed_count / 2U,
        artifact.training.test_seed_count / 2U);
    std::vector<droid::HeldoutScenarioOutcome> scenarios;
    for (std::size_t index = 0; index < splits.test.size(); ++index) {
        const std::uint64_t seed = splits.test[index];
        const double jitter = static_cast<double>(index) * 0.001;
        droid::HeldoutScenarioOutcome scenario;
        scenario.learned = droid::EpisodeOutcome{
            .environment_seed = seed,
            .episode_return = 14.0 + jitter,
            .mean_reward_rate = 0.70 + jitter,
        };
        scenario.zero = droid::EpisodeOutcome{
            .environment_seed = seed,
            .episode_return = 10.0,
            .mean_reward_rate = 0.50,
        };
        for (std::size_t stream = 0; stream < 4; ++stream) {
            scenario.random.push_back(droid::EpisodeOutcome{
                .environment_seed = seed,
                .episode_return = 9.0 + static_cast<double>(stream),
                .mean_reward_rate = 0.44 + 0.02 * static_cast<double>(stream),
            });
        }
        scenarios.push_back(std::move(scenario));
    }
    return scenarios;
}

enum class AcceptanceComparison {
    zero,
    random_mean,
};

void set_acceptance_comparison_deltas(
    std::vector<droid::HeldoutScenarioOutcome>& scenarios,
    AcceptanceComparison comparison,
    const std::array<std::vector<double>, 2>& side_deltas) {
    std::array<std::size_t, 2> side_offsets{};
    for (auto& scenario : scenarios) {
        const std::size_t side = droid::light_search_seed_has_positive_side(
                                     scenario.learned.environment_seed)
            ? 1U
            : 0U;
        require(side_offsets[side] < side_deltas[side].size(),
                "acceptance delta fixture is too short for its side");
        const double delta = side_deltas[side][side_offsets[side]++];
        const double baseline_rate =
            scenario.learned.mean_reward_rate - delta;
        const double baseline_return = scenario.learned.episode_return -
            20.0 * delta;
        if (comparison == AcceptanceComparison::zero) {
            scenario.zero.mean_reward_rate = baseline_rate;
            scenario.zero.episode_return = baseline_return;
        } else {
            for (auto& random : scenario.random) {
                random.mean_reward_rate = baseline_rate;
                random.episode_return = baseline_return;
            }
        }
    }
    require(side_offsets[0] == side_deltas[0].size() &&
                side_offsets[1] == side_deltas[1].size(),
            "acceptance delta fixture has unused side entries");
}

void require_only_acceptance_criterion_failed(
    const Json& assessment,
    std::string_view expected_failure) {
    require(!assessment.at("accepted").get<bool>(),
            "negative acceptance fixture unexpectedly passed");
    constexpr std::array<const char*, 6> criteria{
        "safe",
        "both_bootstrap_lower_bounds_above_zero",
        "both_practical_margins_met",
        "both_minimum_win_counts_met",
        "positive_mean_delta_in_each_light_side_vs_both",
        "minimum_win_counts_met_in_each_light_side_vs_both",
    };
    for (const char* criterion : criteria) {
        const bool expected = criterion != expected_failure;
        require(
            assessment.at("criteria").at(criterion).get<bool>() == expected,
            std::string("acceptance fixture did not isolate criterion: ") +
                std::string(expected_failure));
    }
}

[[nodiscard]] droid::PolicyArtifactData artifact_with_minimum_wins(
    droid::PolicyArtifactData artifact,
    std::size_t minimum_wins) {
    Json manifest = droid::default_learning_experiment_manifest();
    manifest["heldout_acceptance"]["minimum_wins"] = minimum_wins;
    artifact.training.protocol_manifest_sha256 =
        droid::sha256_hex(manifest.dump());
    artifact.self_sha256.clear();
    return droid::parse_policy_artifact(
        droid::policy_artifact_json(artifact));
}

void test_heldout_acceptance() {
    const droid::AcceptanceOptions options;
    const auto artifact = acceptance_artifact();
    auto scenarios = passing_scenarios(artifact);
    const Json accepted = droid::assess_heldout_policy(
        scenarios, artifact, options);
    require(accepted.at("accepted").get<bool>(),
            "clearly superior safe held-out policy was rejected");
    require(
        accepted.at("aggregate")
                .at("learned_minus_random_reward_rate_bootstrap_95pct")
                .at("lower")
                .get<double>() > 0.0,
        "paired random bootstrap lower bound was not positive");
    require(
        accepted.at("configuration").at("multiplicity_control") ==
            "bonferroni_two_one_sided_0.025",
        "acceptance report omitted the two-comparison correction");

    std::reverse(scenarios.begin(), scenarios.end());
    const Json reordered = droid::assess_heldout_policy(
        scenarios, artifact, options);
    require(accepted.dump() == reordered.dump(),
            "held-out assessment depended on input scenario order");

    scenarios.front().learned.terminated = true;
    const Json unsafe = droid::assess_heldout_policy(
        scenarios, artifact, options);
    require(!unsafe.at("accepted").get<bool>() &&
                !unsafe.at("criteria").at("safe").get<bool>(),
            "unsafe learned policy passed held-out acceptance");

    scenarios = passing_scenarios(artifact);
    for (auto& scenario : scenarios) {
        if (!droid::light_search_seed_has_positive_side(
                scenario.learned.environment_seed)) {
            scenario.learned.mean_reward_rate = 0.40;
        }
    }
    const Json one_sided_failure = droid::assess_heldout_policy(
        scenarios, artifact, options);
    require(
        !one_sided_failure.at("accepted").get<bool>() &&
            !one_sided_failure.at("criteria")
                 .at("positive_mean_delta_in_each_light_side_vs_both")
                 .get<bool>(),
        "strong performance on one light side subsidized failure on the other");
}

void test_heldout_acceptance_negative_gates() {
    const droid::AcceptanceOptions options;
    const droid::PolicyArtifactData artifact = acceptance_artifact();

    auto invalid = passing_scenarios(artifact);
    invalid.front().learned.invalid_samples = 1;
    require_only_acceptance_criterion_failed(
        droid::assess_heldout_policy(invalid, artifact, options), "safe");

    auto vetoed = passing_scenarios(artifact);
    vetoed.front().learned.safety_vetoes = 1;
    require_only_acceptance_criterion_failed(
        droid::assess_heldout_policy(vetoed, artifact, options), "safe");

    const auto noisy_positive_mean = [] {
        std::vector<double> values(32, -0.25);
        std::fill(values.begin(), values.begin() + 24, 0.10);
        return values;
    };
    for (const AcceptanceComparison comparison : {
             AcceptanceComparison::zero,
             AcceptanceComparison::random_mean}) {
        auto uncertain = passing_scenarios(artifact);
        set_acceptance_comparison_deltas(
            uncertain,
            comparison,
            {noisy_positive_mean(), noisy_positive_mean()});
        const Json assessment = droid::assess_heldout_policy(
            uncertain, artifact, options);
        const char* failed_interval =
            comparison == AcceptanceComparison::zero
            ? "learned_minus_zero_reward_rate_bootstrap_95pct"
            : "learned_minus_random_reward_rate_bootstrap_95pct";
        const char* passing_interval =
            comparison == AcceptanceComparison::zero
            ? "learned_minus_random_reward_rate_bootstrap_95pct"
            : "learned_minus_zero_reward_rate_bootstrap_95pct";
        require(
            assessment.at("aggregate")
                    .at(failed_interval)
                    .at("lower")
                    .get<double>() <= 0.0 &&
                assessment.at("aggregate")
                        .at(passing_interval)
                        .at("lower")
                        .get<double>() > 0.0,
            "bootstrap fixture did not isolate its requested comparison");
        require_only_acceptance_criterion_failed(
            assessment, "both_bootstrap_lower_bounds_above_zero");
    }

    auto impractical = passing_scenarios(artifact);
    set_acceptance_comparison_deltas(
        impractical,
        AcceptanceComparison::zero,
        {std::vector<double>(32, 0.005),
         std::vector<double>(32, 0.005)});
    require_only_acceptance_criterion_failed(
        droid::assess_heldout_policy(impractical, artifact, options),
        "both_practical_margins_met");

    droid::AcceptanceOptions stricter_overall = options;
    stricter_overall.minimum_wins = 49;
    const droid::PolicyArtifactData stricter_artifact =
        artifact_with_minimum_wins(artifact, stricter_overall.minimum_wins);
    auto insufficient_overall = passing_scenarios(stricter_artifact);
    std::vector<double> twenty_four_wins(32, 0.0);
    std::fill(
        twenty_four_wins.begin(), twenty_four_wins.begin() + 24, 0.10);
    set_acceptance_comparison_deltas(
        insufficient_overall,
        AcceptanceComparison::zero,
        {twenty_four_wins, twenty_four_wins});
    require_only_acceptance_criterion_failed(
        droid::assess_heldout_policy(
            insufficient_overall, stricter_artifact, stricter_overall),
        "both_minimum_win_counts_met");

    auto insufficient_one_side = passing_scenarios(artifact);
    std::vector<double> twenty_three_wins(32, 0.0);
    std::fill(
        twenty_three_wins.begin(), twenty_three_wins.begin() + 23, 0.10);
    set_acceptance_comparison_deltas(
        insufficient_one_side,
        AcceptanceComparison::zero,
        {twenty_three_wins, std::vector<double>(32, 0.10)});
    require_only_acceptance_criterion_failed(
        droid::assess_heldout_policy(
            insufficient_one_side, artifact, options),
        "minimum_win_counts_met_in_each_light_side_vs_both");
}

void test_v3_heldout_report_governance() {
    const Json governance = droid::v3_heldout_report_governance();
    require(
        governance.at("current_protocol_feedback_rule") ==
            "do_not_use_this_v3_test_split_to_tune_or_select_v3_"
            "artifacts_checkpoints_or_optimizer_choices",
        "held-out report did not prohibit feedback into v3 selection");
    require(
        governance.at("later_version_curriculum_rule") ==
            "after_the_v3_decision_is_fixed_this_split_may_be_declared_"
            "as_curriculum_for_a_later_protocol_version",
        "held-out report prohibited declared later-version curriculum reuse");
    require(
        governance.at("later_version_unseen_status") ==
            "if_promoted_it_is_not_unseen_heldout_data_for_that_later_version",
        "held-out report failed to revoke unseen status after promotion");
    require(
        governance.at("global_single_execution_enforced_by_software") ==
            false,
        "held-out report overstated technical one-shot enforcement");
}

void test_heldout_rejects_wrong_frozen_split_before_rollout() {
    const droid::PolicyArtifactData artifact = acceptance_artifact();
    const auto splits = droid::make_mirrored_seed_splits(
        artifact.training.seed_split_base_seed,
        artifact.training.train_seed_count / 2U,
        artifact.training.validation_seed_count / 2U,
        artifact.training.test_seed_count / 2U);
    droid::PolicyEvaluationOptions options;
    options.assembly_id = artifact.compatibility.assembly_id;
    options.episode_profile = artifact.training.scenario_profile_id;
    options.episodes = splits.test.size();
    options.max_control_steps = artifact.training.max_control_steps;
    options.control_dt_s = artifact.compatibility.control_dt_s;
    options.explicit_environment_seeds = splits.test;
    options.explicit_environment_seeds.front() =
        std::numeric_limits<std::uint64_t>::max();
    require_throws_containing(
        [&] {
            (void)droid::evaluate_frozen_heldout(
                "this-model-must-never-be-opened.xml",
                artifact,
                artifact.compatibility,
                options,
                {});
        },
        "seed list does not match frozen test split",
        "wrong held-out seed set reached environment rollout");

    auto scenarios = passing_scenarios(artifact);
    const std::uint64_t unrelated_seed =
        std::numeric_limits<std::uint64_t>::max();
    scenarios.front().learned.environment_seed = unrelated_seed;
    scenarios.front().zero.environment_seed = unrelated_seed;
    for (auto& random : scenarios.front().random) {
        random.environment_seed = unrelated_seed;
    }
    require_throws_containing(
        [&] { (void)droid::assess_heldout_policy(scenarios, artifact); },
        "frozen test split",
        "direct assessor accepted outcomes from an unrelated scenario set");
}

void test_heldout_rejects_off_mask_artifact_before_model_open() {
    droid::PolicyArtifactData artifact = acceptance_artifact();
    artifact.parameters[0] = 1.0;
    artifact.self_sha256.clear();
    artifact.self_sha256 = droid::policy_artifact_json(artifact)
                               .at("self_sha256")
                               .get<std::string>();
    const auto splits = droid::make_mirrored_seed_splits(
        artifact.training.seed_split_base_seed,
        artifact.training.train_seed_count / 2U,
        artifact.training.validation_seed_count / 2U,
        artifact.training.test_seed_count / 2U);
    droid::PolicyEvaluationOptions options;
    options.assembly_id = artifact.compatibility.assembly_id;
    options.episode_profile = artifact.training.scenario_profile_id;
    options.episodes = splits.test.size();
    options.max_control_steps = artifact.training.max_control_steps;
    options.control_dt_s = artifact.compatibility.control_dt_s;
    options.explicit_environment_seeds = splits.test;
    require_throws_containing(
        [&] {
            (void)droid::evaluate_frozen_heldout(
                "this-model-must-never-be-opened.xml",
                artifact,
                artifact.compatibility,
                options,
                {});
        },
        "positive-zero CEM mask",
        "off-mask v3 artifact reached the held-out model boundary");
}

void test_manifest_matches_checked_in_protocol() {
    std::ifstream input("config/learning_experiment_v3.json");
    require(static_cast<bool>(input), "checked-in learning manifest is missing");
    Json checked_in;
    input >> checked_in;
    require(checked_in == droid::default_learning_experiment_manifest(),
            "checked-in and executable experiment manifests diverged");
    require(
        droid::learning_experiment_manifest_sha256() ==
            droid::sha256_hex(checked_in.dump()),
        "executable protocol digest does not identify the checked-in manifest");
}

void test_split_and_frozen_identity_rejections() {
    droid::PolicyTrainingRequest reordered = small_request();
    std::swap(reordered.seed_splits.train[0], reordered.seed_splits.train[1]);
    require_throws_containing(
        [&] { (void)droid::train_frozen_policy(reordered, synthetic_score); },
        "exact disjoint balanced derived protocol",
        "manually reordered training split was accepted");

    droid::PolicyTrainingRequest overlapping = small_request();
    overlapping.seed_splits.validation[0] = overlapping.seed_splits.train[0];
    require_throws_containing(
        [&] { (void)droid::train_frozen_policy(overlapping, synthetic_score); },
        "exact disjoint balanced derived protocol",
        "overlapping seed splits were accepted");

    droid::PolicyTrainingRequest false_default_claim = small_request();
    false_default_claim.protocol_manifest_sha256 =
        droid::learning_experiment_manifest_sha256();
    require_throws_containing(
        [&] {
            (void)droid::train_frozen_policy(
                false_default_claim, synthetic_score);
        },
        "differs from the default protocol",
        "non-default training request claimed the default manifest digest");

    const droid::PolicyTrainingRequest request = small_request();
    droid::PolicyArtifactData unfrozen =
        droid::train_frozen_policy(request, synthetic_score).artifact;
    unfrozen.self_sha256.clear();
    droid::PolicyEvaluationOptions options;
    options.assembly_id = request.compatibility.assembly_id;
    options.episode_profile = request.episode_profile;
    options.episodes = request.seed_splits.test.size();
    options.max_control_steps = request.max_control_steps;
    options.control_dt_s = request.compatibility.control_dt_s;
    options.explicit_environment_seeds = request.seed_splits.test;
    require_throws_containing(
        [&] {
            (void)droid::evaluate_frozen_policy(
                "this-model-must-never-be-opened.xml",
                unfrozen,
                request.compatibility,
                options);
        },
        "integrity checksum",
        "frozen evaluation silently minted a missing artifact identity");
}

void test_tiny_real_environment_training_and_evaluation() {
    const std::filesystem::path model{"models/droid.xml"};
    droid::DroidEnvironment environment(model, false);
    const Json spec = environment.spec();

    droid::PolicyTrainingRequest request;
    request.compatibility.environment_api_version =
        spec.at("api_version").get<std::string>();
    request.compatibility.assembly_id =
        spec.at("assembly_id").get<std::string>();
    request.compatibility.motor_ids =
        spec.at("action").at("motor_ids").get<std::vector<std::string>>();
    request.compatibility.control_dt_s = 0.02;
    request.compatibility.model_sha256 = droid::sha256_file(model);
    request.compatibility.module_catalog_sha256 =
        droid::sha256_hex(droid::catalog_json().dump());
    request.compatibility.environment_spec_sha256 =
        droid::sha256_hex(spec.dump());
    request.compatibility.feature_schema_sha256 =
        droid::policy_feature_schema_sha256();
    request.optimizer.generations = 1;
    request.optimizer.population_size = 2;
    request.optimizer.elite_count = 1;
    request.optimizer.initial_stddev = 0.1;
    request.optimizer.minimum_stddev = 0.01;
    request.optimizer.update_rate = 1.0;
    request.optimizer.optimizer_seed = 17;
    request.seed_splits = droid::make_mirrored_seed_splits(73, 1, 1, 1);
    request.protocol_manifest_sha256 = digest('b');
    request.train_batch_size = 2;
    request.max_control_steps = 2;
    request.candidate_evaluation_workers = 1;

    const auto trained =
        droid::train_frozen_policy_in_environment(model, request);
    droid::PolicyTrainingRequest parallel_request = request;
    parallel_request.candidate_evaluation_workers = 4;
    const auto parallel = droid::train_frozen_policy_in_environment(
        model, parallel_request);
    require(!trained.artifact.self_sha256.empty(),
            "real environment training did not freeze an artifact");
    require(
        droid::policy_artifact_json(trained.artifact).dump() ==
                droid::policy_artifact_json(parallel.artifact).dump() &&
            trained.deterministic_report.at("history").dump() ==
                parallel.deterministic_report.at("history").dump() &&
            trained.deterministic_report
                    .at("zero_baseline_precomputation")
                    .dump() ==
                parallel.deterministic_report
                    .at("zero_baseline_precomputation")
                    .dump(),
        "native side-robust training changed between one and four workers");
    const Json& zero_baseline =
        trained.deterministic_report.at("zero_baseline_precomputation");
    require(
        zero_baseline.at("seed_scope") ==
                "train_and_validation_only" &&
            zero_baseline.at("train_seed_count") ==
                request.seed_splits.train.size() &&
            zero_baseline.at("validation_seed_count") ==
                request.seed_splits.validation.size() &&
            zero_baseline.at("total_seed_count") ==
                request.seed_splits.train.size() +
                    request.seed_splits.validation.size() &&
            zero_baseline.at("test_seed_count_evaluated") == 0,
        "zero baseline precomputation escaped the train/validation scope");
    require(
        trained.deterministic_report.at("validation")
                .contains("negative_x_mean_reward_rate_gain_vs_zero") &&
            trained.deterministic_report.at("validation")
                .contains("positive_x_mean_reward_rate_gain_vs_zero"),
        "native score report omitted paired per-side gains");

    droid::PolicyEvaluationOptions options;
    options.assembly_id = request.compatibility.assembly_id;
    options.episode_profile = request.episode_profile;
    options.episodes = request.seed_splits.validation.size();
    options.max_control_steps = request.max_control_steps;
    options.base_seed = 101;
    options.control_dt_s = request.compatibility.control_dt_s;
    options.explicit_environment_seeds = request.seed_splits.validation;
    const Json evaluation = droid::evaluate_frozen_policy(
        model, trained.artifact, request.compatibility, options);
    require(
        evaluation.at("episodes").size() ==
            request.seed_splits.validation.size(),
        "real frozen-policy smoke evaluated the wrong validation count");
}

void test_invalid_configuration() {
    droid::CemOptions invalid;
    invalid.elite_count = invalid.population_size + 1U;
    require_throws(
        [&] {
            (void)droid::train_cem_diagonal(
                invalid, {{1, 2}}, std::array<std::uint64_t, 1>{3}, synthetic_score);
        },
        "invalid elite count was accepted");
    require_throws(
        [] { (void)droid::make_mirrored_seed_splits(0, 0, 1, 1); },
        "zero-sized seed split was accepted");
    require_throws(
        [] {
            (void)droid::train_cem_diagonal(
                droid::CemOptions{},
                {{1, 2}},
                std::array<std::uint64_t, 1>{3},
                synthetic_score,
                0);
        },
        "zero candidate-evaluation workers were accepted");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string_view, std::function<void()>>> tests{
        {"light seed mapping and disjoint balanced splits",
         test_light_seed_mapping_and_splits},
        {"safety-first score order", test_safety_first_score_order},
        {"masked search space and sampled checkpoint",
         test_masked_search_space_and_sample_checkpoint},
        {"bimodal best-sample escape",
         test_bimodal_sample_checkpoint_escapes_zero},
        {"validation isolation from future samples",
         test_validation_cannot_change_future_samples},
        {"real train-pair recurrence ablation",
         test_real_train_pair_recurrence_ablation},
        {"common batches and deterministic CEM",
         test_common_batches_and_deterministic_cem},
        {"frozen training artifact", test_frozen_training_artifact},
        {"held-out acceptance", test_heldout_acceptance},
        {"held-out acceptance negative gates",
         test_heldout_acceptance_negative_gates},
        {"v3 held-out report governance",
         test_v3_heldout_report_governance},
        {"held-out frozen split rejection",
         test_heldout_rejects_wrong_frozen_split_before_rollout},
        {"held-out off-mask rejection before model open",
         test_heldout_rejects_off_mask_artifact_before_model_open},
        {"checked-in experiment manifest",
         test_manifest_matches_checked_in_protocol},
        {"split and frozen identity rejections",
         test_split_and_frozen_identity_rejections},
        {"tiny real environment training and evaluation",
         test_tiny_real_environment_training_and_evaluation},
        {"invalid configuration", test_invalid_configuration},
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
    std::cout << passed << " native training tests passed\n";
    return 0;
}
