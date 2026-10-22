#include "droid/baseline.hpp"

#include "droid/evaluation.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace droid {
namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kActionStreamTag{0x414354494f4e5630ULL};
constexpr std::uint64_t kFnvOffsetBasis{14'695'981'039'346'656'037ULL};
constexpr std::uint64_t kFnvPrime{1'099'511'628'211ULL};
constexpr double kTwoToMinus53{0x1.0p-53};

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

[[nodiscard]] std::uint64_t hash_module_id(std::string_view module_id) noexcept {
    std::uint64_t value = kFnvOffsetBasis;
    for (const char character : module_id) {
        value ^= static_cast<unsigned char>(character);
        value *= kFnvPrime;
    }
    return value;
}

void validate_options(
    BaselinePolicyKind policy,
    const BaselineEvaluationOptions& options) {
    if (baseline_policy_id(policy) == "unknown") {
        throw std::invalid_argument("unknown baseline policy kind");
    }
    if (options.assembly_id.empty()) {
        throw std::invalid_argument("baseline assembly_id must not be empty");
    }
    if (options.episode_profile.empty()) {
        throw std::invalid_argument("baseline episode_profile must not be empty");
    }
    if (options.episodes == 0) {
        throw std::invalid_argument("baseline episodes must be greater than zero");
    }
    if (options.max_control_steps == 0) {
        throw std::invalid_argument(
            "baseline max_control_steps must be greater than zero");
    }
    if (!std::isfinite(options.control_dt_s) ||
        options.control_dt_s <= 0.0) {
        throw std::invalid_argument(
            "baseline control_dt_s must be finite and greater than zero");
    }
    if (!std::isfinite(options.random_effort_limit) ||
        options.random_effort_limit < 0.0 ||
        options.random_effort_limit > 1.0) {
        throw std::invalid_argument(
            "baseline random_effort_limit must be finite and inside [0, 1]");
    }
    if (options.random_hold_steps == 0) {
        throw std::invalid_argument(
            "baseline random_hold_steps must be greater than zero");
    }
}

[[nodiscard]] Json policy_metadata(
    BaselinePolicyKind policy,
    const BaselineEvaluationOptions& options) {
    Json metadata;
    switch (policy) {
        case BaselinePolicyKind::zero:
            metadata = Json{
                {"description", "constant zero effort for every opaque motor ID"},
                {"effort_limit", 0.0},
                {"sample_and_hold_steps", 1},
            };
            break;
        case BaselinePolicyKind::seeded_random:
            metadata = Json{
                {"description",
                 "independent seeded sample-and-hold effort per opaque motor ID"},
                {"effort_limit", options.random_effort_limit},
                {"sample_and_hold_steps", options.random_hold_steps},
                {"generator",
                 Json{
                     {"module_id_hash", "fnv1a_64"},
                     {"integer_mixer", "splitmix64_finalizer"},
                     {"unit_conversion", "high_53_bits_times_2^-53"},
                     {"range", "[-effort_limit, effort_limit)"},
                 }},
            };
            break;
    }
    return metadata;
}

}  // namespace

std::string_view baseline_policy_id(BaselinePolicyKind policy) noexcept {
    switch (policy) {
        case BaselinePolicyKind::zero:
            return "zero_v0";
        case BaselinePolicyKind::seeded_random:
            return "seeded_random_sample_hold_v0";
    }
    return "unknown";
}

BaselinePolicyKind parse_baseline_policy(std::string_view policy_id) {
    if (policy_id == "zero_v0" || policy_id == "zero") {
        return BaselinePolicyKind::zero;
    }
    if (policy_id == "seeded_random_sample_hold_v0" ||
        policy_id == "seeded_random" || policy_id == "random") {
        return BaselinePolicyKind::seeded_random;
    }
    throw std::invalid_argument(
        "baseline policy must be zero_v0 or seeded_random_sample_hold_v0");
}

std::uint64_t baseline_environment_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept {
    return policy_evaluation_environment_seed(base_seed, episode_index);
}

std::uint64_t baseline_policy_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept {
    return policy_evaluation_policy_seed(base_seed, episode_index);
}

std::map<std::string, double> make_baseline_actions(
    BaselinePolicyKind policy,
    const std::vector<std::string>& motor_ids,
    std::size_t control_step,
    std::uint64_t policy_seed,
    const BaselineEvaluationOptions& options) {
    validate_options(policy, options);
    if (motor_ids.empty()) {
        throw std::invalid_argument("baseline requires at least one motor ID");
    }

    std::map<std::string, double> actions;
    const std::uint64_t hold_block = static_cast<std::uint64_t>(
        control_step / options.random_hold_steps);
    for (const std::string& motor_id : motor_ids) {
        if (motor_id.empty()) {
            throw std::invalid_argument("baseline motor IDs must not be empty");
        }
        double effort = 0.0;
        switch (policy) {
            case BaselinePolicyKind::zero:
                break;
            case BaselinePolicyKind::seeded_random: {
                if (options.random_effort_limit == 0.0) {
                    effort = 0.0;
                    break;
                }
                const std::uint64_t bits = mix64(
                    policy_seed ^ hash_module_id(motor_id) ^
                    mix64(hold_block ^ kActionStreamTag));
                const std::uint64_t high_53_bits = bits >> 11U;
                const double unit =
                    static_cast<double>(high_53_bits) * kTwoToMinus53;
                effort = options.random_effort_limit * (2.0 * unit - 1.0);
                break;
            }
        }
        if (!actions.emplace(motor_id, effort).second) {
            throw std::invalid_argument("baseline motor IDs must be unique");
        }
    }
    return actions;
}

Json evaluate_baseline(
    const std::filesystem::path& model_path,
    BaselinePolicyKind policy,
    const BaselineEvaluationOptions& options) {
    validate_options(policy, options);

    PolicyEvaluationOptions evaluation_options;
    evaluation_options.assembly_id = options.assembly_id;
    evaluation_options.episode_profile = options.episode_profile;
    evaluation_options.episodes = options.episodes;
    evaluation_options.max_control_steps = options.max_control_steps;
    evaluation_options.base_seed = options.base_seed;
    evaluation_options.control_dt_s = options.control_dt_s;

    PolicyEvaluationDescriptor descriptor;
    descriptor.schema_version = "droid-blocks.baseline-policy.v1";
    descriptor.id = std::string(baseline_policy_id(policy));
    descriptor.deterministic = true;
    descriptor.uses_observation = false;
    descriptor.uses_policy_seed =
        policy == BaselinePolicyKind::seeded_random;
    descriptor.metadata = policy_metadata(policy, options);

    Json result = evaluate_policy(
        model_path,
        descriptor,
        [policy, options](
            const Json&,
            const std::vector<std::string>& motor_ids,
            std::size_t control_step,
            std::uint64_t policy_seed) {
            return make_baseline_actions(
                policy, motor_ids, control_step, policy_seed, options);
        },
        evaluation_options);
    result["schema_version"] = kBaselineEvaluationSchemaVersion;
    result["configuration"]["random_effort_limit"] =
        options.random_effort_limit;
    result["configuration"]["random_hold_steps"] =
        options.random_hold_steps;
    return result;
}

}  // namespace droid
