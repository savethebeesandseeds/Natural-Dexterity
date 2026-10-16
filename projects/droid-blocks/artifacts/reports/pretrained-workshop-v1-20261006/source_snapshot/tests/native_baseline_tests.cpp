#include "droid/baseline.hpp"
#include "droid/environment.hpp"
#include "droid/evaluation.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

using ExpectedPolicyActionCallback =
    std::function<std::map<std::string, double>(
        const Json&,
        const std::vector<std::string>&,
        std::size_t,
        std::uint64_t)>;
static_assert(std::is_same_v<
              droid::PolicyActionCallback,
              ExpectedPolicyActionCallback>);

[[nodiscard]] std::filesystem::path default_model_path() {
#if defined(DROID_TEST_MODEL_PATH)
    return DROID_TEST_MODEL_PATH;
#elif defined(DROID_MODEL_PATH)
    return DROID_MODEL_PATH;
#else
    return "models/droid.xml";
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

[[nodiscard]] std::vector<std::string> motor_ids() {
    return {
        "motor-0001", "motor-0002", "motor-0003", "motor-0004"};
}

[[nodiscard]] bool contains_key_recursive(
    const Json& value,
    const std::set<std::string>& keys) {
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            if (keys.contains(key) || contains_key_recursive(child, keys)) {
                return true;
            }
        }
    } else if (value.is_array()) {
        for (const Json& child : value) {
            if (contains_key_recursive(child, keys)) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] double object_number_sum(const Json& object) {
    require(object.is_object(), "expected a JSON object of numeric values");
    long double sum = 0.0L;
    for (const auto& [unused, value] : object.items()) {
        (void)unused;
        require(value.is_number(), "metric map contains a non-number");
        const double number = value.get<double>();
        require(std::isfinite(number), "metric map contains a non-finite number");
        sum += static_cast<long double>(number);
    }
    return static_cast<double>(sum);
}

void test_policy_ids_and_seed_streams() {
    require(
        droid::baseline_policy_id(droid::BaselinePolicyKind::zero) ==
            "zero_v0",
        "zero policy ID changed");
    require(
        droid::baseline_policy_id(droid::BaselinePolicyKind::seeded_random) ==
            "seeded_random_sample_hold_v0",
        "random policy ID changed");
    require(
        droid::parse_baseline_policy("zero_v0") ==
            droid::BaselinePolicyKind::zero,
        "canonical zero policy did not parse");
    require(
        droid::parse_baseline_policy("random") ==
            droid::BaselinePolicyKind::seeded_random,
        "random shorthand did not parse");
    require_throws(
        [] { (void)droid::parse_baseline_policy("oracle"); },
        "unknown policy ID must fail");

    const std::uint64_t base = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t environment_0 = droid::baseline_environment_seed(base, 0);
    const std::uint64_t environment_1 = droid::baseline_environment_seed(base, 1);
    const std::uint64_t policy_0 = droid::baseline_policy_seed(base, 0);
    require(
        environment_0 ==
                droid::policy_evaluation_environment_seed(base, 0) &&
            policy_0 == droid::policy_evaluation_policy_seed(base, 0),
        "baseline adapters changed the generic evaluation seed contract");
    require(
        environment_0 == droid::baseline_environment_seed(base, 0),
        "environment seed stream is not deterministic");
    require(environment_0 != environment_1, "episode seeds did not advance");
    require(
        environment_0 != policy_0,
        "environment and policy seed streams are not domain separated");
}

void test_zero_policy_actions() {
    droid::BaselineEvaluationOptions options;
    const auto actions = droid::make_baseline_actions(
        droid::BaselinePolicyKind::zero,
        motor_ids(),
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        options);
    require(actions.size() == 4, "zero policy omitted a motor");
    for (const auto& [id, effort] : actions) {
        require(!id.empty(), "zero policy emitted an empty motor ID");
        require(effort == 0.0, "zero policy emitted non-zero effort");
    }
}

void test_random_policy_is_stable_order_independent_and_held() {
    droid::BaselineEvaluationOptions options;
    options.random_effort_limit = 0.6;
    options.random_hold_steps = 5;
    const std::uint64_t seed = 0xfedcba9876543210ULL;
    std::vector<std::string> forward = motor_ids();
    std::vector<std::string> reverse = forward;
    std::reverse(reverse.begin(), reverse.end());

    const auto first = droid::make_baseline_actions(
        droid::BaselinePolicyKind::seeded_random,
        forward,
        0,
        seed,
        options);
    const auto repeated = droid::make_baseline_actions(
        droid::BaselinePolicyKind::seeded_random,
        reverse,
        0,
        seed,
        options);
    require(first == repeated, "random actions depend on motor input order");
    for (std::size_t step = 1; step < options.random_hold_steps; ++step) {
        require(
            first == droid::make_baseline_actions(
                         droid::BaselinePolicyKind::seeded_random,
                         forward,
                         step,
                         seed,
                         options),
            "sample-and-hold action changed inside its hold block");
    }
    const auto next_block = droid::make_baseline_actions(
        droid::BaselinePolicyKind::seeded_random,
        forward,
        options.random_hold_steps,
        seed,
        options);
    require(first != next_block, "random action did not change at hold boundary");
    const auto distinct_seed = droid::make_baseline_actions(
        droid::BaselinePolicyKind::seeded_random,
        forward,
        0,
        seed + 1U,
        options);
    require(first != distinct_seed, "distinct policy seeds produced same action map");

    for (std::size_t step = 0; step < 100; ++step) {
        const auto actions = droid::make_baseline_actions(
            droid::BaselinePolicyKind::seeded_random,
            forward,
            step,
            std::numeric_limits<std::uint64_t>::max(),
            options);
        for (const auto& [unused, effort] : actions) {
            (void)unused;
            require(std::isfinite(effort), "random policy emitted non-finite effort");
            require(
                std::abs(effort) <= options.random_effort_limit,
                "random policy exceeded its effort limit");
        }
    }
}

[[nodiscard]] std::map<std::string, double> handoff_test_actions(
    const std::vector<std::string>& ids,
    std::size_t control_step) {
    std::map<std::string, double> actions;
    const double effort = 0.1 + 0.05 * static_cast<double>(control_step);
    for (const std::string& id : ids) {
        actions.emplace(id, effort);
    }
    return actions;
}

void test_generic_policy_observation_handoff_and_profile_provenance(
    const std::filesystem::path& model_path) {
    droid::PolicyEvaluationOptions options;
    options.assembly_id = "demo_rover_v0";
    options.episode_profile = "light_search_1d_v1";
    options.episodes = 1;
    options.max_control_steps = 3;
    options.base_seed = 0xA11CE;
    options.control_dt_s = 0.02;
    options.explicit_environment_seeds = {1};

    droid::PolicyEvaluationDescriptor descriptor;
    descriptor.schema_version = "test.policy.v1";
    descriptor.id = "observation_handoff_test_v0";
    descriptor.deterministic = true;
    descriptor.uses_observation = true;
    descriptor.uses_policy_seed = true;
    descriptor.metadata = Json{{"artifact_id", "test-artifact-0001"}};

    const std::set<std::string> forbidden_callback_keys{
        "info",
        "reward",
        "reward_components",
        "safety",
        "terminated",
        "truncated",
        "robot",
        "world",
        "controller",
        "light_source_position",
        "target_velocity_rad_s",
        "direction",
    };

    const auto run = [&](std::vector<Json>& observations) {
        std::vector<std::size_t> steps;
        std::vector<std::uint64_t> policy_seeds;
        const Json result = droid::evaluate_policy(
            model_path,
            descriptor,
            [&](
                const Json& observation,
                const std::vector<std::string>& ids,
                std::size_t control_step,
                std::uint64_t policy_seed) {
                require(
                    observation.is_object() && observation.size() == 2 &&
                        observation.contains("reward_sensors") &&
                        observation.contains("actuator_feedback"),
                    "generic policy did not receive the policy-safe observation");
                require(
                    !contains_key_recursive(observation, forbidden_callback_keys),
                    "generic policy callback received privileged or evaluator data");
                require(
                    ids == motor_ids(),
                    "generic policy callback received wrong opaque motor IDs");
                observations.push_back(observation);
                steps.push_back(control_step);
                policy_seeds.push_back(policy_seed);
                return handoff_test_actions(ids, control_step);
            },
            options);

        require(
            steps == std::vector<std::size_t>({0, 1, 2}),
            "generic evaluator did not issue exactly one ordered callback per step");
        require(
            policy_seeds == std::vector<std::uint64_t>(
                3,
                droid::policy_evaluation_policy_seed(options.base_seed, 0)),
            "generic evaluator exposed the wrong policy-only seed");
        return result;
    };

    std::vector<Json> observed;
    const Json first = run(observed);
    std::vector<Json> repeated_observed;
    const Json repeated = run(repeated_observed);
    require(
        first.dump() == repeated.dump(),
        "generic policy evaluation was not exactly repeatable");
    require(
        observed == repeated_observed,
        "generic policy observation handoff was not repeatable");

    droid::DroidEnvironment reference(model_path, false);
    const Json reset = reference.reset(
        options.assembly_id,
        options.explicit_environment_seeds.at(0),
        false,
        options.episode_profile);
    Json expected_observation = reset.at("observation");
    for (std::size_t step = 0; step < options.max_control_steps; ++step) {
        require(
            observed.at(step).dump() == expected_observation.dump(),
            "callback did not receive the current environment observation");
        const Json transition = reference.step(
            handoff_test_actions(motor_ids(), step), options.control_dt_s);
        expected_observation = transition.at("observation");
    }

    require(
        first.at("schema_version") == droid::kPolicyEvaluationSchemaVersion,
        "generic evaluation schema version is missing");
    require(
        first.at("policy").at("schema_version") == descriptor.schema_version &&
            first.at("policy").at("id") == descriptor.id &&
            first.at("policy").at("uses_observation") == true &&
            first.at("policy").at("uses_policy_seed") == true &&
            first.at("policy").at("artifact_id") == "test-artifact-0001",
        "generic policy descriptor provenance was not retained");
    require(
        first.at("environment").at("episode_profile") ==
                options.episode_profile &&
            first.at("configuration").at("episode_profile") ==
                options.episode_profile &&
            first.at("episodes").at(0).at("episode_profile") ==
                options.episode_profile,
        "evaluation omitted episode-profile provenance");
    require(
        first.at("configuration").at("environment_seed_source") ==
                "explicit" &&
            first.at("episodes").at(0).at("environment_seed") == 1,
        "generic evaluator did not honor the explicit environment seed");
    require(
        first.at("episodes").at(0).at("policy_seed").get<std::uint64_t>() ==
            droid::policy_evaluation_policy_seed(options.base_seed, 0),
        "generic evaluation omitted its independently derived policy seed");
}

void test_invalid_configuration_is_rejected(
    const std::filesystem::path& model_path) {
    droid::BaselineEvaluationOptions options;
    options.episodes = 0;
    require_throws(
        [&] {
            (void)droid::evaluate_baseline(
                model_path, droid::BaselinePolicyKind::zero, options);
        },
        "zero episode evaluation must fail");
    options.episodes = 1;
    options.max_control_steps = 0;
    require_throws(
        [&] {
            (void)droid::evaluate_baseline(
                model_path, droid::BaselinePolicyKind::zero, options);
        },
        "zero horizon evaluation must fail");
    options.max_control_steps = 1;
    options.control_dt_s = std::numeric_limits<double>::quiet_NaN();
    require_throws(
        [&] {
            (void)droid::evaluate_baseline(
                model_path, droid::BaselinePolicyKind::zero, options);
        },
        "non-finite control interval must fail");
    options.control_dt_s = 0.02;
    options.random_effort_limit = 1.01;
    require_throws(
        [&] {
            (void)droid::make_baseline_actions(
                droid::BaselinePolicyKind::seeded_random,
                motor_ids(),
                0,
                0,
                options);
        },
        "out-of-contract random effort limit must fail");
    options.random_effort_limit = 0.6;
    options.random_hold_steps = 0;
    require_throws(
        [&] {
            (void)droid::make_baseline_actions(
                droid::BaselinePolicyKind::seeded_random,
                motor_ids(),
                0,
                0,
                options);
        },
        "zero sample-and-hold interval must fail");
    options.random_hold_steps = 5;
    require_throws(
        [&] {
            (void)droid::make_baseline_actions(
                droid::BaselinePolicyKind::zero, {}, 0, 0, options);
        },
        "empty motor list must fail");
    require_throws(
        [&] {
            auto duplicate = motor_ids();
            duplicate.push_back(duplicate.front());
            (void)droid::make_baseline_actions(
                droid::BaselinePolicyKind::zero,
                duplicate,
                0,
                0,
                options);
        },
        "duplicate motor IDs must fail");

    droid::PolicyEvaluationOptions generic_options;
    generic_options.episodes = 2;
    generic_options.max_control_steps = 1;
    generic_options.explicit_environment_seeds = {7};
    droid::PolicyEvaluationDescriptor descriptor;
    descriptor.id = "validation_test_v0";
    const droid::PolicyActionCallback zero_callback = [](
        const Json&,
        const std::vector<std::string>& ids,
        std::size_t,
        std::uint64_t) {
        return handoff_test_actions(ids, 0);
    };
    require_throws(
        [&] {
            (void)droid::evaluate_policy(
                model_path, descriptor, zero_callback, generic_options);
        },
        "mismatched explicit environment seed count must fail");

    generic_options.episodes = 1;
    descriptor.metadata = Json{{"id", "metadata-collision"}};
    require_throws(
        [&] {
            (void)droid::evaluate_policy(
                model_path, descriptor, zero_callback, generic_options);
        },
        "reserved policy metadata must fail");

    descriptor.metadata = Json::object();
    require_throws(
        [&] {
            (void)droid::evaluate_policy(
                model_path,
                descriptor,
                droid::PolicyActionCallback{},
                generic_options);
        },
        "empty policy callback must fail");

    descriptor.metadata = Json{{"score", std::numeric_limits<double>::infinity()}};
    require_throws(
        [&] {
            (void)droid::evaluate_policy(
                model_path, descriptor, zero_callback, generic_options);
        },
        "non-finite policy metadata must fail");
}

[[nodiscard]] droid::BaselineEvaluationOptions short_evaluation_options() {
    droid::BaselineEvaluationOptions options;
    options.episodes = 2;
    options.max_control_steps = 12;
    options.base_seed = std::numeric_limits<std::uint64_t>::max();
    options.random_effort_limit = 0.6;
    options.random_hold_steps = 3;
    return options;
}

void test_evaluation_is_exact_and_policy_safe(
    const std::filesystem::path& model_path) {
    const auto options = short_evaluation_options();
    const Json first = droid::evaluate_baseline(
        model_path, droid::BaselinePolicyKind::seeded_random, options);
    const Json second = droid::evaluate_baseline(
        model_path, droid::BaselinePolicyKind::seeded_random, options);
    require(
        first.dump() == second.dump(),
        "repeated full-width-seed evaluations are not byte deterministic");
    require(
        first.at("schema_version") ==
            droid::kBaselineEvaluationSchemaVersion,
        "evaluation schema version is missing");
    require(
        first.at("configuration").at("base_seed").get<std::uint64_t>() ==
            std::numeric_limits<std::uint64_t>::max(),
        "full-width base seed was narrowed");
    require(
        !first.at("policy").at("uses_observation").get<bool>(),
        "observation-free baseline claims to read observations");
    require(
        first.at("configuration").at("recording") == false,
        "baseline should not pay trace-recording cost");

    const std::set<std::string> forbidden{
        "robot",
        "world",
        "controller",
        "position",
        "position_rad",
        "quaternion",
        "yaw",
        "speed",
        "distance",
        "contact_count",
        "light_source_position",
        "target_velocity_rad_s",
        "direction",
        "wall_time_s",
        "timestamp",
        "peak_rss_bytes",
    };
    require(
        !contains_key_recursive(first, forbidden),
        "evaluation document contains privileged or nondeterministic data");
}

void test_metrics_reconcile_and_outcomes_are_honest(
    const std::filesystem::path& model_path) {
    const auto options = short_evaluation_options();
    const Json result = droid::evaluate_baseline(
        model_path, droid::BaselinePolicyKind::zero, options);
    const Json& episodes = result.at("episodes");
    require(episodes.size() == options.episodes, "evaluation lost an episode");
    long double total_return = 0.0L;
    std::uint64_t total_steps = 0;
    std::uint64_t total_physics_steps = 0;
    double total_time = 0.0;
    for (const Json& episode : episodes) {
        const double episode_return = episode.at("return").get<double>();
        const double sensor_sum = object_number_sum(
            episode.at("reward").at("by_sensor"));
        require(
            std::abs(episode_return - sensor_sum) <= 1e-12,
            "episode sensor rewards do not reconcile to return");
        require(
            std::abs(
                episode.at("reward").at("reconciliation_error").get<double>()) <=
                1e-12,
            "episode reports a material reward reconciliation error");
        const double simulated_time =
            episode.at("simulated_time_s").get<double>();
        require(simulated_time > 0.0, "episode executed no simulated time");
        require(
            std::abs(
                episode.at("mean_reward_rate").get<double>() -
                episode_return / simulated_time) <= 1e-12,
            "episode mean reward rate is inconsistent");
        const bool stopped =
            episode.at("environment_terminated").get<bool>() ||
            episode.at("environment_truncated").get<bool>();
        require(
            episode.at("horizon_reached").get<bool>() == !stopped,
            "external horizon was conflated with environment truncation");
        require(
            !episode.at("environment_truncated").get<bool>(),
            "v0 environment unexpectedly reported truncation");
        total_return += episode_return;
        total_steps += episode.at("control_steps").get<std::uint64_t>();
        total_physics_steps += episode.at("physics_steps").get<std::uint64_t>();
        total_time += simulated_time;
    }

    const Json& aggregate = result.at("aggregate");
    require(
        aggregate.at("total_control_steps") == total_steps,
        "aggregate control-step count is inconsistent");
    require(
        aggregate.at("total_physics_steps") == total_physics_steps,
        "aggregate physics-step count is inconsistent");
    require(
        std::abs(
            aggregate.at("total_return").get<double>() -
            static_cast<double>(total_return)) <= 1e-12,
        "aggregate total return is inconsistent");
    require(
        std::abs(
            aggregate.at("exposure_weighted_reward_rate").get<double>() -
            static_cast<double>(total_return) / total_time) <= 1e-12,
        "exposure-weighted reward rate is inconsistent");
    require(
        aggregate.contains("population_stddev") &&
            aggregate.at("population_stddev").get<double>() >= 0.0,
        "return dispersion is missing or ambiguously named");
    require(
        aggregate.at("reward").at("by_sensor").size() == 3,
        "aggregate did not preserve three reward-sensor identities");
    require(
        aggregate.at("reward").at("by_family").size() == 2,
        "aggregate did not preserve sensor-family provenance");
    require(
        aggregate.at("health").at("invalid_reward_sensor_samples") == 0 &&
            aggregate.at("health").at("invalid_actuator_feedback_samples") == 0,
        "valid reference rollout reported invalid module samples");
    require(
        aggregate.at("actuators").at("by_module").size() == 4,
        "actuator health metrics lost an opaque motor instance");
}

void test_zero_and_random_rollouts_are_meaningful_and_paired(
    const std::filesystem::path& model_path) {
    droid::BaselineEvaluationOptions options;
    options.episode_profile = "light_search_1d_v1";
    options.episodes = 2;
    options.max_control_steps = 40;
    options.base_seed = 23;
    options.random_effort_limit = 0.6;
    options.random_hold_steps = 5;
    const Json zero = droid::evaluate_baseline(
        model_path, droid::BaselinePolicyKind::zero, options);
    const Json random = droid::evaluate_baseline(
        model_path, droid::BaselinePolicyKind::seeded_random, options);
    require(
        zero.at("environment").at("episode_profile") ==
                options.episode_profile &&
            random.at("environment").at("episode_profile") ==
                options.episode_profile,
        "baseline evaluations omitted paired episode-profile provenance");
    for (std::size_t index = 0; index < options.episodes; ++index) {
        require(
            zero.at("episodes").at(index).at("environment_seed") ==
                random.at("episodes").at(index).at("environment_seed"),
            "policies were not evaluated on paired environment seeds");
        require(
            zero.at("episodes").at(index).at("policy_seed").is_null(),
            "zero policy falsely reports using an RNG seed");
        require(
            random.at("episodes").at(index).at("policy_seed")
                .is_number_unsigned(),
            "seeded-random policy omitted its RNG seed");
        require(
            random.at("episodes").at(index).at("policy_seed")
                    .get<std::uint64_t>() ==
                droid::baseline_policy_seed(options.base_seed, index),
            "seeded-random policy reported the wrong RNG stream seed");
    }
    require(
        zero.at("aggregate").at("actions").at("mean_abs_effort") == 0.0 &&
            zero.at("aggregate").at("actions").at("max_abs_effort") == 0.0,
        "zero rollout action metrics are not zero");
    require(
        random.at("aggregate").at("actions").at("mean_abs_effort").get<double>() >
            0.0,
        "random rollout produced no non-zero actions");
    require(
        random.at("aggregate").at("actions").at("max_abs_effort").get<double>() <=
            options.random_effort_limit,
        "random rollout exceeded configured effort limit");
    require(
        random.at("aggregate").at("actuators").at("peak_abs_current_a")
                .get<double>() >
            zero.at("aggregate").at("actuators").at("peak_abs_current_a")
                .get<double>(),
        "random effort did not produce meaningful actuator feedback");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path model_path =
            argc > 1 ? std::filesystem::path(argv[1]) : default_model_path();
        if (argc > 2) {
            throw std::invalid_argument(
                "usage: droid-native-baseline-tests [model.xml]");
        }

        const std::vector<std::pair<std::string, std::function<void()>>> tests{
            {"policy IDs and seed streams", test_policy_ids_and_seed_streams},
            {"zero policy actions", test_zero_policy_actions},
            {"stable held bounded random actions",
             test_random_policy_is_stable_order_independent_and_held},
            {"generic observation handoff and profile provenance",
             [&] {
                 test_generic_policy_observation_handoff_and_profile_provenance(
                     model_path);
             }},
            {"invalid configuration",
             [&] { test_invalid_configuration_is_rejected(model_path); }},
            {"exact policy-safe evaluation",
             [&] { test_evaluation_is_exact_and_policy_safe(model_path); }},
            {"metric reconciliation and outcomes",
             [&] { test_metrics_reconcile_and_outcomes_are_honest(model_path); }},
            {"meaningful paired rollouts",
             [&] {
                 test_zero_and_random_rollouts_are_meaningful_and_paired(
                     model_path);
             }},
        };

        for (const auto& [name, test] : tests) {
            test();
            std::cout << "[PASS] " << name << '\n';
        }
        std::cout << tests.size() << " native baseline tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] native baseline: " << error.what() << '\n';
        return 1;
    }
}
