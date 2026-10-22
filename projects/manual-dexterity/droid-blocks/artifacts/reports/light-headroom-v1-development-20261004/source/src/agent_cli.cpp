#include "droid/baseline.hpp"
#include "droid/environment.hpp"
#include "droid/learner.hpp"
#include "droid/module_catalog.hpp"
#include "droid/policy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kAssemblyId{"demo_rover_v0"};
constexpr double kDefaultControlDtS = 0.02;
constexpr std::size_t kDefaultSteps = 10'000;
constexpr std::size_t kDefaultEvaluationSteps = 1'000;
constexpr std::size_t kDefaultEvaluationEpisodes = 8;
constexpr double kDefaultRandomEffort = 0.6;
constexpr std::size_t kDefaultHoldSteps = 5;
constexpr std::string_view kBalancedLightSideSeedDerivationId{
    "balanced_light_side_splitmix64_v1"};
enum class Command {
    benchmark,
    record,
    replay,
    evaluate,
    train,
    validate_policy,
    evaluate_policy,
};

enum class EvaluationPolicy {
    zero,
    random,
    both,
};

struct Options {
    Command command{Command::benchmark};
    std::filesystem::path model{"models/droid.xml"};
    std::size_t steps{kDefaultSteps};
    std::uint64_t seed{0};
    double control_dt_s{kDefaultControlDtS};
    EvaluationPolicy policy{EvaluationPolicy::both};
    std::size_t episodes{kDefaultEvaluationEpisodes};
    double random_effort{kDefaultRandomEffort};
    std::size_t hold_steps{kDefaultHoldSteps};
    std::optional<std::filesystem::path> output;
    std::optional<std::filesystem::path> input;
    std::optional<std::filesystem::path> output_policy;
    std::optional<std::filesystem::path> input_policy;
    std::optional<std::filesystem::path> output_report;
    std::optional<std::size_t> generations;
    std::optional<std::size_t> population_size;
    std::optional<std::size_t> elite_count;
    std::optional<std::size_t> train_batch_size;
    std::optional<double> initial_stddev;
    std::optional<double> minimum_stddev;
    std::optional<double> update_rate;
    std::optional<std::uint64_t> optimizer_seed;
    std::optional<std::size_t> workers;
    bool help{false};
};

[[nodiscard]] std::string require_value(
    int& index,
    int argc,
    char** argv,
    std::string_view option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return argv[++index];
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(
    std::string_view text,
    std::string_view option) {
    Integer value{};
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument(std::string(option) + " must be an integer");
    }
    return value;
}

[[nodiscard]] double parse_double(
    std::string_view text,
    std::string_view option) {
    double value{};
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end || !std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(option) + " must be a finite number");
    }
    return value;
}

[[nodiscard]] Command parse_command(std::string_view text) {
    if (text == "benchmark") {
        return Command::benchmark;
    }
    if (text == "record") {
        return Command::record;
    }
    if (text == "replay") {
        return Command::replay;
    }
    if (text == "evaluate") {
        return Command::evaluate;
    }
    if (text == "train") {
        return Command::train;
    }
    if (text == "validate-policy") {
        return Command::validate_policy;
    }
    if (text == "evaluate-policy") {
        return Command::evaluate_policy;
    }
    throw std::invalid_argument(
        "command must be benchmark, record, replay, evaluate, train, or "
        "validate-policy or evaluate-policy");
}

[[nodiscard]] EvaluationPolicy parse_evaluation_policy(
    std::string_view text) {
    if (text == "zero") {
        return EvaluationPolicy::zero;
    }
    if (text == "random") {
        return EvaluationPolicy::random;
    }
    if (text == "both") {
        return EvaluationPolicy::both;
    }
    throw std::invalid_argument("--policy must be zero, random, or both");
}

[[nodiscard]] std::string_view evaluation_policy_name(
    EvaluationPolicy policy) noexcept {
    switch (policy) {
        case EvaluationPolicy::zero:
            return "zero";
        case EvaluationPolicy::random:
            return "random";
        case EvaluationPolicy::both:
            return "both";
    }
    return "unknown";
}

[[nodiscard]] std::vector<droid::BaselinePolicyKind> evaluation_policies(
    EvaluationPolicy policy) {
    switch (policy) {
        case EvaluationPolicy::zero:
            return {droid::BaselinePolicyKind::zero};
        case EvaluationPolicy::random:
            return {droid::BaselinePolicyKind::seeded_random};
        case EvaluationPolicy::both:
            // This order is part of the CLI contract and deliberately does
            // not depend on enum values or caller input ordering.
            return {
                droid::BaselinePolicyKind::zero,
                droid::BaselinePolicyKind::seeded_random,
            };
    }
    throw std::logic_error("unknown evaluation policy selection");
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument("a command is required");
    }

    Options options;
    const std::string_view first(argv[1]);
    if (first == "-h" || first == "--help") {
        options.help = true;
        return options;
    }
    options.command = parse_command(first);
    if (options.command == Command::evaluate) {
        options.steps = kDefaultEvaluationSteps;
    }

    bool policy_supplied = false;
    bool episodes_supplied = false;
    bool random_effort_supplied = false;
    bool hold_steps_supplied = false;
    bool steps_supplied = false;
    bool seed_supplied = false;
    bool control_dt_supplied = false;
    bool training_option_supplied = false;
    std::set<std::string> seen_options;

    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument.starts_with("--") && argument != "--help" &&
            !seen_options.emplace(argument).second) {
            throw std::invalid_argument(
                "option supplied more than once: " + std::string(argument));
        }
        if (argument == "-h" || argument == "--help") {
            options.help = true;
        } else if (argument == "--model") {
            options.model = require_value(index, argc, argv, argument);
        } else if (argument == "--steps") {
            options.steps = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (options.steps == 0) {
                throw std::invalid_argument("--steps must be greater than zero");
            }
            steps_supplied = true;
        } else if (argument == "--seed") {
            options.seed = parse_integer<std::uint64_t>(
                require_value(index, argc, argv, argument), argument);
            seed_supplied = true;
        } else if (argument == "--control-dt") {
            options.control_dt_s = parse_double(
                require_value(index, argc, argv, argument), argument);
            if (options.control_dt_s <= 0.0) {
                throw std::invalid_argument(
                    "--control-dt must be greater than zero");
            }
            control_dt_supplied = true;
        } else if (argument == "--policy") {
            options.policy = parse_evaluation_policy(
                require_value(index, argc, argv, argument));
            policy_supplied = true;
        } else if (argument == "--episodes") {
            options.episodes = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (options.episodes == 0) {
                throw std::invalid_argument(
                    "--episodes must be greater than zero");
            }
            episodes_supplied = true;
        } else if (argument == "--random-effort") {
            options.random_effort = parse_double(
                require_value(index, argc, argv, argument), argument);
            if (options.random_effort < 0.0 ||
                options.random_effort > 1.0) {
                throw std::invalid_argument(
                    "--random-effort must be between 0 and 1");
            }
            random_effort_supplied = true;
        } else if (argument == "--hold-steps") {
            options.hold_steps = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (options.hold_steps == 0) {
                throw std::invalid_argument(
                    "--hold-steps must be greater than zero");
            }
            hold_steps_supplied = true;
        } else if (argument == "--output") {
            options.output = require_value(index, argc, argv, argument);
        } else if (argument == "--input") {
            options.input = require_value(index, argc, argv, argument);
        } else if (argument == "--output-policy") {
            options.output_policy = require_value(index, argc, argv, argument);
        } else if (argument == "--input-policy") {
            options.input_policy = require_value(index, argc, argv, argument);
        } else if (argument == "--output-report") {
            options.output_report = require_value(index, argc, argv, argument);
        } else if (argument == "--generations") {
            options.generations = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (*options.generations == 0) {
                throw std::invalid_argument(
                    "--generations must be greater than zero");
            }
            training_option_supplied = true;
        } else if (argument == "--population-size") {
            options.population_size = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (*options.population_size < 2) {
                throw std::invalid_argument(
                    "--population-size must be at least two");
            }
            training_option_supplied = true;
        } else if (argument == "--elite-count") {
            options.elite_count = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (*options.elite_count == 0) {
                throw std::invalid_argument(
                    "--elite-count must be greater than zero");
            }
            training_option_supplied = true;
        } else if (argument == "--train-batch-size") {
            options.train_batch_size = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (*options.train_batch_size == 0) {
                throw std::invalid_argument(
                    "--train-batch-size must be greater than zero");
            }
            training_option_supplied = true;
        } else if (argument == "--initial-stddev") {
            options.initial_stddev = parse_double(
                require_value(index, argc, argv, argument), argument);
            if (*options.initial_stddev <= 0.0) {
                throw std::invalid_argument(
                    "--initial-stddev must be greater than zero");
            }
            training_option_supplied = true;
        } else if (argument == "--minimum-stddev") {
            options.minimum_stddev = parse_double(
                require_value(index, argc, argv, argument), argument);
            if (*options.minimum_stddev <= 0.0) {
                throw std::invalid_argument(
                    "--minimum-stddev must be greater than zero");
            }
            training_option_supplied = true;
        } else if (argument == "--update-rate") {
            options.update_rate = parse_double(
                require_value(index, argc, argv, argument), argument);
            if (*options.update_rate <= 0.0 || *options.update_rate > 1.0) {
                throw std::invalid_argument(
                    "--update-rate must be in (0, 1]");
            }
            training_option_supplied = true;
        } else if (argument == "--optimizer-seed") {
            options.optimizer_seed = parse_integer<std::uint64_t>(
                require_value(index, argc, argv, argument), argument);
            training_option_supplied = true;
        } else if (argument == "--workers") {
            options.workers = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
            if (*options.workers == 0) {
                throw std::invalid_argument(
                    "--workers must be greater than zero");
            }
            training_option_supplied = true;
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument));
        }
    }

    if (!options.help && options.command == Command::record && !options.output) {
        throw std::invalid_argument("record requires --output PATH");
    }
    if (!options.help && options.command == Command::replay && !options.input) {
        throw std::invalid_argument("replay requires --input PATH");
    }
    if (!options.help && options.command == Command::train &&
        !options.output_policy) {
        throw std::invalid_argument("train requires --output-policy PATH");
    }
    if (!options.help &&
        (options.command == Command::validate_policy ||
         options.command == Command::evaluate_policy) &&
        !options.input_policy) {
        throw std::invalid_argument(
            std::string(
                options.command == Command::validate_policy
                    ? "validate-policy"
                    : "evaluate-policy") +
            " requires --input-policy PATH");
    }
    if (!options.help && options.command == Command::evaluate_policy &&
        !options.output_report) {
        throw std::invalid_argument(
            "evaluate-policy requires --output-report PATH as its durable "
            "report destination");
    }
    if (options.command != Command::record && options.output) {
        throw std::invalid_argument("--output is only valid with record");
    }
    if (options.command != Command::replay && options.input) {
        throw std::invalid_argument("--input is only valid with replay");
    }
    if (options.command != Command::train && options.output_policy) {
        throw std::invalid_argument(
            "--output-policy is only valid with train");
    }
    if (options.command != Command::validate_policy &&
        options.command != Command::evaluate_policy &&
        options.input_policy) {
        throw std::invalid_argument(
            "--input-policy is only valid with validate-policy or "
            "evaluate-policy");
    }
    if (options.command != Command::evaluate_policy && options.output_report) {
        throw std::invalid_argument(
            "--output-report is only valid with evaluate-policy");
    }
    if (options.command != Command::evaluate &&
        (policy_supplied || episodes_supplied || random_effort_supplied ||
         hold_steps_supplied)) {
        throw std::invalid_argument(
            "--policy, --episodes, --random-effort, and --hold-steps are "
            "only valid with evaluate");
    }
    if (options.command != Command::train && training_option_supplied) {
        throw std::invalid_argument(
            "optimizer options are only valid with train");
    }
    if ((options.command == Command::train ||
         options.command == Command::validate_policy ||
         options.command == Command::evaluate_policy) &&
        (steps_supplied || seed_supplied || control_dt_supplied)) {
        throw std::invalid_argument(
            "--steps, --seed, and --control-dt are fixed by the learning "
            "protocol and are not valid with train, validate-policy, or "
            "evaluate-policy");
    }
    return options;
}

void print_usage(std::ostream& output, std::string_view executable) {
    output
        << "Usage:\n"
        << "  " << executable << " benchmark [options]\n"
        << "  " << executable << " record --output PATH [options]\n"
        << "  " << executable << " replay --input PATH [options]\n"
        << "  " << executable << " evaluate [options]\n"
        << "  " << executable << " train --output-policy PATH [options]\n"
        << "  " << executable
        << " validate-policy --input-policy PATH [--model PATH]\n"
        << "  " << executable
        << " evaluate-policy --input-policy PATH --output-report PATH\n"
        << "\n"
        << "Native synchronous tools for the Droid Blocks agent boundary.\n"
        << "\n"
        << "Options:\n"
        << "  --model PATH         MuJoCo XML (default: models/droid.xml)\n"
        << "  --steps N            Control transitions (benchmark/record: 10000;\n"
        << "                       evaluate, per episode: 1000)\n"
        << "  --seed N             Deterministic episode seed (default: 0)\n"
        << "  --control-dt SECONDS Control interval (default: 0.02)\n"
        << "  --policy NAME        zero, random, or both (evaluate; default: both)\n"
        << "  --episodes N         Paired episodes (evaluate; default: 8)\n"
        << "  --random-effort X    Random effort magnitude in [0, 1]\n"
        << "                       (evaluate; default: 0.6)\n"
        << "  --hold-steps N       Steps per random action (evaluate; default: 5)\n"
        << "  --output PATH        Trace destination for record\n"
        << "  --input PATH         Trace source for replay\n"
        << "  --output-policy PATH New frozen policy destination (train; no overwrite)\n"
        << "  --input-policy PATH  Frozen policy source (validate-policy or\n"
        << "                       evaluate-policy)\n"
        << "  --output-report PATH Required held-out report destination (no overwrite)\n"
        << "\n"
        << "Training optimizer options (train only; compiled protocol defaults):\n"
        << "  --generations N      CEM generations (default: 40)\n"
        << "  --population-size N  Candidates per generation (default: 32)\n"
        << "  --elite-count N      Elite candidates per generation (default: 8)\n"
        << "  --train-batch-size N Balanced train scenarios per batch (default: 8)\n"
        << "  --initial-stddev X   Initial latent CEM deviation (default: 1.0)\n"
        << "  --minimum-stddev X   Minimum latent CEM deviation (default: 0.1)\n"
        << "  --update-rate X      CEM update rate in (0, 1] (default: 0.25)\n"
        << "  --optimizer-seed N   Deterministic optimizer seed (default: 0)\n"
        << "  --workers N          Parallel candidate evaluators (default: 1;\n"
        << "                       execution-only, not protocol identity)\n"
        << "\n"
        << "train uses the compiled light_search_1d_v1 train/validation/test\n"
        << "split. validate-policy reruns only the exact validation split stored\n"
        << "in an artifact and is diagnostic, never official. evaluate-policy\n"
        << "alone runs the preregistered held-out acceptance protocol.\n"
        << "  -h, --help           Show this help\n";
}

[[nodiscard]] std::map<std::string, double> deterministic_actions(
    const std::vector<std::string>& motor_ids,
    std::size_t step,
    std::uint64_t seed) {
    // Integer-only phase generation keeps the action sequence independent of
    // libm implementations. It depends only on public actuator IDs, step, and
    // the caller-owned seed; it never reads privileged simulator state.
    constexpr std::uint64_t period = 400;
    const std::uint64_t phase =
        (static_cast<std::uint64_t>(step % period) + seed % period) % period;
    const std::uint64_t triangle = phase <= period / 2U
        ? phase
        : period - phase;
    const double effort =
        (static_cast<double>(triangle) - 100.0) / 250.0;

    std::map<std::string, double> actions;
    for (const std::string& motor_id : motor_ids) {
        actions.emplace(motor_id, effort);
    }
    return actions;
}

[[nodiscard]] std::vector<std::string> action_motor_ids(const Json& spec) {
    if (!spec.contains("action") || !spec.at("action").is_object() ||
        !spec.at("action").contains("motor_ids") ||
        !spec.at("action").at("motor_ids").is_array()) {
        throw std::runtime_error(
            "environment specification has no action.motor_ids array");
    }

    std::vector<std::string> motor_ids;
    for (const Json& value : spec.at("action").at("motor_ids")) {
        if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
            throw std::runtime_error(
                "environment action motor IDs must be non-empty strings");
        }
        const std::string motor_id = value.get<std::string>();
        if (std::find(motor_ids.begin(), motor_ids.end(), motor_id) !=
            motor_ids.end()) {
            throw std::runtime_error(
                "environment action motor IDs must be unique");
        }
        motor_ids.push_back(motor_id);
    }
    if (motor_ids.empty()) {
        throw std::runtime_error(
            "environment action motor ID list must not be empty");
    }
    return motor_ids;
}

struct LearningProtocol {
    Json manifest;
    std::string manifest_sha256;
    std::string assembly_id;
    std::string episode_profile;
    std::string seed_derivation_id;
    double control_dt_s{0.0};
    std::size_t max_control_steps{0};
    std::uint64_t split_base_seed{0};
    std::size_t train_pairs{0};
    std::size_t validation_pairs{0};
    std::size_t test_pairs{0};
    droid::CemOptions optimizer;
    std::size_t train_batch_size{0};
    double effort_limit{0.0};
    double slew_limit{0.0};
    droid::AcceptanceOptions acceptance;
};

[[nodiscard]] bool same_binary64(double left, double right) noexcept {
    return std::bit_cast<std::uint64_t>(left) ==
        std::bit_cast<std::uint64_t>(right);
}

[[nodiscard]] LearningProtocol learning_protocol() {
    LearningProtocol protocol;
    protocol.manifest = droid::default_learning_experiment_manifest();
    protocol.manifest_sha256 =
        droid::learning_experiment_manifest_sha256();
    if (!protocol.manifest.is_object() ||
        protocol.manifest.value("schema_version", "") !=
            droid::kLearningExperimentSchemaVersion) {
        throw std::logic_error(
            "compiled learning experiment manifest has an invalid schema");
    }
    if (protocol.manifest_sha256 !=
        droid::sha256_hex(protocol.manifest.dump())) {
        throw std::logic_error(
            "compiled learning experiment manifest hash is not canonical");
    }

    protocol.assembly_id =
        protocol.manifest.at("assembly_id").get<std::string>();
    protocol.episode_profile =
        protocol.manifest.at("scenario_profile_id").get<std::string>();
    protocol.control_dt_s =
        protocol.manifest.at("control_dt_s").get<double>();
    protocol.max_control_steps =
        protocol.manifest.at("max_control_steps").get<std::size_t>();

    const Json& split = protocol.manifest.at("seed_split");
    protocol.seed_derivation_id =
        split.at("derivation_id").get<std::string>();
    protocol.split_base_seed =
        split.at("base_seed").get<std::uint64_t>();
    protocol.train_pairs = split.at("train_pairs").get<std::size_t>();
    protocol.validation_pairs =
        split.at("validation_pairs").get<std::size_t>();
    protocol.test_pairs = split.at("test_pairs").get<std::size_t>();

    const Json& optimizer = protocol.manifest.at("optimizer");
    if (optimizer.at("algorithm_id").get<std::string>() !=
        droid::kCemAlgorithmId) {
        throw std::logic_error(
            "compiled learning experiment has an unsupported optimizer");
    }
    protocol.optimizer.generations =
        optimizer.at("generations").get<std::size_t>();
    protocol.optimizer.population_size =
        optimizer.at("population_size").get<std::size_t>();
    protocol.optimizer.elite_count =
        optimizer.at("elite_count").get<std::size_t>();
    protocol.train_batch_size =
        optimizer.at("train_batch_size").get<std::size_t>();
    protocol.optimizer.initial_stddev =
        optimizer.at("initial_stddev").get<double>();
    protocol.optimizer.minimum_stddev =
        optimizer.at("minimum_stddev").get<double>();
    protocol.optimizer.update_rate =
        optimizer.at("update_rate").get<double>();
    protocol.optimizer.optimizer_seed =
        optimizer.at("optimizer_seed").get<std::uint64_t>();

    const Json& policy = protocol.manifest.at("policy");
    protocol.effort_limit = policy.at("effort_limit").get<double>();
    protocol.slew_limit =
        policy.at("slew_limit_per_control_step").get<double>();

    const Json& acceptance = protocol.manifest.at("heldout_acceptance");
    protocol.acceptance.expected_scenarios =
        acceptance.at("scenario_count").get<std::size_t>();
    protocol.acceptance.random_streams_per_scenario =
        acceptance.at("random_streams_per_scenario").get<std::size_t>();
    protocol.acceptance.random_control_base_seed =
        acceptance.at("random_control_base_seed").get<std::uint64_t>();
    protocol.acceptance.random_effort_limit =
        acceptance.at("random_effort_limit").get<double>();
    protocol.acceptance.random_hold_steps =
        acceptance.at("random_hold_steps").get<std::size_t>();
    protocol.acceptance.random_slew_limit_per_control_step =
        acceptance.at("random_slew_limit_per_control_step").get<double>();
    protocol.acceptance.bootstrap_resamples =
        acceptance.at("bootstrap_resamples").get<std::size_t>();
    protocol.acceptance.bootstrap_seed =
        acceptance.at("bootstrap_seed").get<std::uint64_t>();
    protocol.acceptance.practical_reward_rate_gain =
        acceptance.at("practical_reward_rate_gain").get<double>();
    protocol.acceptance.minimum_wins =
        acceptance.at("minimum_wins").get<std::size_t>();
    protocol.acceptance.minimum_wins_per_side =
        acceptance.at("minimum_wins_per_side").get<std::size_t>();

    if (protocol.assembly_id != kAssemblyId ||
        protocol.episode_profile != droid::kLightSearch1dEpisodeProfile ||
        !std::isfinite(protocol.control_dt_s) ||
        protocol.control_dt_s <= 0.0 || protocol.max_control_steps == 0 ||
        protocol.train_pairs == 0 || protocol.validation_pairs == 0 ||
        protocol.test_pairs == 0) {
        throw std::logic_error(
            "compiled learning experiment has invalid fixed protocol fields");
    }
    return protocol;
}

[[nodiscard]] droid::MirroredSeedSplits protocol_seed_splits(
    const LearningProtocol& protocol) {
    droid::MirroredSeedSplits splits = droid::make_mirrored_seed_splits(
        protocol.split_base_seed,
        protocol.train_pairs,
        protocol.validation_pairs,
        protocol.test_pairs);
    if (splits.test.size() != protocol.acceptance.expected_scenarios) {
        throw std::logic_error(
            "compiled held-out scenario count does not match the test split");
    }
    return splits;
}

[[nodiscard]] droid::PolicyCompatibility runtime_compatibility(
    const std::filesystem::path& model,
    std::string_view assembly_id,
    double control_dt_s) {
    if (assembly_id.empty() || !std::isfinite(control_dt_s) ||
        control_dt_s <= 0.0) {
        throw std::invalid_argument(
            "policy runtime identity has an invalid assembly or control "
            "interval");
    }
    droid::DroidEnvironment environment(model, false);
    const Json spec = environment.spec();
    if (!spec.contains("api_version") ||
        !spec.at("api_version").is_string() ||
        spec.at("api_version").get_ref<const std::string&>().empty() ||
        spec.value("assembly_id", "") != assembly_id) {
        throw std::runtime_error(
            "environment specification is incompatible with the learning "
            "protocol");
    }
    return droid::PolicyCompatibility{
        .environment_api_version =
            spec.at("api_version").get<std::string>(),
        .assembly_id = std::string(assembly_id),
        .motor_ids = action_motor_ids(spec),
        .control_dt_s = control_dt_s,
        .model_sha256 = droid::sha256_file(model),
        .module_catalog_sha256 =
            droid::sha256_hex(droid::catalog_json().dump()),
        .environment_spec_sha256 = droid::sha256_hex(spec.dump()),
        .feature_schema_sha256 = droid::policy_feature_schema_sha256(),
    };
}

[[nodiscard]] droid::PolicyCompatibility runtime_compatibility(
    const std::filesystem::path& model,
    const LearningProtocol& protocol) {
    return runtime_compatibility(
        model, protocol.assembly_id, protocol.control_dt_s);
}

[[nodiscard]] Json compatibility_json(
    const droid::PolicyCompatibility& compatibility) {
    return Json{
        {"environment_api_version", compatibility.environment_api_version},
        {"assembly_id", compatibility.assembly_id},
        {"motor_ids", compatibility.motor_ids},
        {"control_dt_s", compatibility.control_dt_s},
        {"control_dt_binary64",
         droid::encode_binary64_hex(compatibility.control_dt_s)},
        {"model_sha256", compatibility.model_sha256},
        {"module_catalog_sha256", compatibility.module_catalog_sha256},
        {"environment_spec_sha256",
         compatibility.environment_spec_sha256},
        {"feature_schema_sha256", compatibility.feature_schema_sha256},
    };
}

[[nodiscard]] Json optimizer_json(
    const droid::CemOptions& optimizer,
    std::size_t train_batch_size) {
    return Json{
        {"algorithm_id", droid::kCemAlgorithmId},
        {"generations", optimizer.generations},
        {"population_size", optimizer.population_size},
        {"elite_count", optimizer.elite_count},
        {"train_batch_size", train_batch_size},
        {"initial_stddev", optimizer.initial_stddev},
        {"minimum_stddev", optimizer.minimum_stddev},
        {"update_rate", optimizer.update_rate},
        {"optimizer_seed", optimizer.optimizer_seed},
    };
}

[[nodiscard]] Json seed_splits_json(
    const droid::MirroredSeedSplits& splits) {
    return Json{
        {"train_count", splits.train.size()},
        {"train_sha256", droid::seed_list_sha256(splits.train)},
        {"validation_count", splits.validation.size()},
        {"validation_sha256", droid::seed_list_sha256(splits.validation)},
        {"test_count", splits.test.size()},
        {"test_sha256", droid::seed_list_sha256(splits.test)},
    };
}

[[nodiscard]] droid::MirroredSeedSplits artifact_seed_splits(
    const droid::PolicyArtifactData& artifact) {
    if (artifact.training.scenario_profile_id !=
        droid::kLightSearch1dEpisodeProfile) {
        throw std::invalid_argument(
            "validate-policy requires a light_search_1d_v1 artifact");
    }
    if (artifact.training.seed_derivation_id !=
        kBalancedLightSideSeedDerivationId) {
        throw std::invalid_argument(
            "validate-policy does not recognize the artifact seed "
            "derivation");
    }
    if (artifact.training.train_seed_count == 0 ||
        artifact.training.validation_seed_count == 0 ||
        artifact.training.test_seed_count == 0 ||
        artifact.training.train_seed_count % 2U != 0U ||
        artifact.training.validation_seed_count % 2U != 0U ||
        artifact.training.test_seed_count % 2U != 0U) {
        throw std::invalid_argument(
            "artifact seed counts must be nonzero balanced pairs");
    }

    droid::MirroredSeedSplits splits = droid::make_mirrored_seed_splits(
        artifact.training.seed_split_base_seed,
        artifact.training.train_seed_count / 2U,
        artifact.training.validation_seed_count / 2U,
        artifact.training.test_seed_count / 2U);
    if (droid::seed_list_sha256(splits.train) !=
            artifact.training.train_seeds_sha256 ||
        droid::seed_list_sha256(splits.validation) !=
            artifact.training.validation_seeds_sha256 ||
        droid::seed_list_sha256(splits.test) !=
            artifact.training.test_seeds_sha256) {
        throw std::invalid_argument(
            "artifact seed digests do not match its exact derived split");
    }
    return splits;
}

[[nodiscard]] Json effective_training_manifest(
    const LearningProtocol& protocol,
    const droid::CemOptions& optimizer,
    std::size_t train_batch_size) {
    Json effective = protocol.manifest;
    Json& configuration = effective.at("optimizer");
    configuration["generations"] = optimizer.generations;
    configuration["population_size"] = optimizer.population_size;
    configuration["elite_count"] = optimizer.elite_count;
    configuration["train_batch_size"] = train_batch_size;
    configuration["initial_stddev"] = optimizer.initial_stddev;
    configuration["minimum_stddev"] = optimizer.minimum_stddev;
    configuration["update_rate"] = optimizer.update_rate;
    configuration["optimizer_seed"] = optimizer.optimizer_seed;
    return effective;
}

[[nodiscard]] droid::PolicyTrainingRequest training_request(
    const Options& options,
    const LearningProtocol& protocol,
    droid::PolicyCompatibility compatibility) {
    droid::PolicyTrainingRequest request;
    request.compatibility = std::move(compatibility);
    request.optimizer = protocol.optimizer;
    request.seed_splits = protocol_seed_splits(protocol);
    request.episode_profile = protocol.episode_profile;
    request.train_batch_size = protocol.train_batch_size;
    request.max_control_steps = protocol.max_control_steps;
    request.effort_limit = protocol.effort_limit;
    request.slew_limit = protocol.slew_limit;

    if (options.generations) {
        request.optimizer.generations = *options.generations;
    }
    if (options.population_size) {
        request.optimizer.population_size = *options.population_size;
    }
    if (options.elite_count) {
        request.optimizer.elite_count = *options.elite_count;
    }
    if (options.train_batch_size) {
        request.train_batch_size = *options.train_batch_size;
    }
    if (options.initial_stddev) {
        request.optimizer.initial_stddev = *options.initial_stddev;
    }
    if (options.minimum_stddev) {
        request.optimizer.minimum_stddev = *options.minimum_stddev;
    }
    if (options.update_rate) {
        request.optimizer.update_rate = *options.update_rate;
    }
    if (options.optimizer_seed) {
        request.optimizer.optimizer_seed = *options.optimizer_seed;
    }
    if (options.workers) {
        request.candidate_evaluation_workers = *options.workers;
    }

    if (request.optimizer.elite_count > request.optimizer.population_size) {
        throw std::invalid_argument(
            "--elite-count must not exceed --population-size");
    }
    if (request.optimizer.minimum_stddev >
        request.optimizer.initial_stddev) {
        throw std::invalid_argument(
            "--minimum-stddev must not exceed --initial-stddev");
    }
    if (request.train_batch_size > request.seed_splits.train.size() ||
        request.train_batch_size % 2U != 0U ||
        request.seed_splits.train.size() % request.train_batch_size != 0U) {
        throw std::invalid_argument(
            "--train-batch-size must be an even divisor of the fixed train "
            "split");
    }
    request.protocol_manifest_sha256 = droid::sha256_hex(
        effective_training_manifest(
            protocol, request.optimizer, request.train_batch_size)
            .dump());
    return request;
}

void validate_exclusive_destination(
    const std::filesystem::path& path,
    std::string_view description) {
    if (path.empty() || path.filename().empty()) {
        throw std::invalid_argument(
            std::string(description) + " destination must name a file");
    }
    const std::filesystem::path parent = path.has_parent_path()
        ? path.parent_path()
        : std::filesystem::current_path();
    if (!std::filesystem::is_directory(parent)) {
        throw std::runtime_error(
            std::string(description) + " destination directory does not "
            "exist: " + parent.string());
    }
    if (std::filesystem::exists(path)) {
        throw std::runtime_error(
            "refusing to overwrite existing " + std::string(description) +
            ": " + path.string());
    }
}

[[nodiscard]] std::filesystem::path unique_report_sibling(
    const std::filesystem::path& destination) {
    static std::atomic<std::uint64_t> sequence{0};
    const auto stamp = static_cast<std::uint64_t>(
        Clock::now().time_since_epoch().count());
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
    throw std::runtime_error(
        "unable to reserve temporary held-out report path");
}

void save_json_exclusive(
    const std::filesystem::path& path,
    const Json& document) {
    validate_exclusive_destination(path, "held-out report");
    const std::filesystem::path temporary = unique_report_sibling(path);
    try {
        {
            std::ofstream output(
                temporary, std::ios::binary | std::ios::out | std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "unable to open temporary held-out report: " +
                    temporary.string());
            }
            output << document.dump(2) << '\n';
            output.flush();
            if (!output) {
                throw std::runtime_error(
                    "unable to write temporary held-out report: " +
                    temporary.string());
            }
        }
        std::error_code error;
        std::filesystem::create_hard_link(temporary, path, error);
        if (error) {
            if (std::filesystem::exists(path)) {
                throw std::runtime_error(
                    "refusing to overwrite existing held-out report: " +
                    path.string());
            }
            throw std::runtime_error(
                "unable to install held-out report: " + error.message());
        }
        std::filesystem::remove(temporary, error);
        if (error) {
            throw std::runtime_error(
                "held-out report saved but temporary link removal failed: " +
                error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

[[nodiscard]] Json policy_artifact_summary(
    const droid::PolicyArtifactData& artifact) {
    const Json document = droid::policy_artifact_json(artifact);
    return Json{
        {"schema_version", document.at("schema_version")},
        {"self_sha256", document.at("self_sha256")},
        {"architecture", document.at("architecture")},
        {"compatibility", document.at("compatibility")},
        {"training", document.at("training")},
    };
}

void require_official_heldout_artifact(
    const droid::PolicyArtifactData& artifact,
    const LearningProtocol& protocol,
    const droid::MirroredSeedSplits& splits) {
    const auto mismatch = [](std::string_view field) {
        throw std::invalid_argument(
            "policy artifact is not eligible for the official held-out "
            "protocol: " + std::string(field) + " mismatch");
    };
    if (artifact.training.algorithm_id != droid::kCemAlgorithmId) {
        mismatch("training.algorithm_id");
    }
    if (!droid::policy_parameters_conform_to_cem_mask_v1(
            artifact.parameters)) {
        mismatch("parameters.cem_mask_v1");
    }
    if (artifact.training.protocol_manifest_sha256 !=
        protocol.manifest_sha256) {
        mismatch("training.protocol_manifest_sha256");
    }
    if (artifact.training.scenario_profile_id != protocol.episode_profile) {
        mismatch("training.scenario_profile_id");
    }
    if (artifact.training.seed_derivation_id != protocol.seed_derivation_id) {
        mismatch("training.seed_derivation_id");
    }
    if (artifact.training.max_control_steps != protocol.max_control_steps) {
        mismatch("training.max_control_steps");
    }
    if (artifact.training.seed_split_base_seed !=
        protocol.split_base_seed) {
        mismatch("training.seed_split_base_seed");
    }
    if (artifact.training.generations != protocol.optimizer.generations) {
        mismatch("training.generations");
    }
    if (artifact.training.population_size !=
        protocol.optimizer.population_size) {
        mismatch("training.population_size");
    }
    if (artifact.training.elite_count != protocol.optimizer.elite_count) {
        mismatch("training.elite_count");
    }
    if (artifact.training.train_batch_size != protocol.train_batch_size) {
        mismatch("training.train_batch_size");
    }
    if (artifact.training.optimizer_seed !=
        protocol.optimizer.optimizer_seed) {
        mismatch("training.optimizer_seed");
    }
    if (!same_binary64(
            artifact.training.initial_stddev,
            protocol.optimizer.initial_stddev)) {
        mismatch("training.initial_stddev");
    }
    if (!same_binary64(
            artifact.training.minimum_stddev,
            protocol.optimizer.minimum_stddev)) {
        mismatch("training.minimum_stddev");
    }
    if (!same_binary64(
            artifact.training.update_rate, protocol.optimizer.update_rate)) {
        mismatch("training.update_rate");
    }
    if (artifact.training.train_seed_count != splits.train.size() ||
        artifact.training.train_seeds_sha256 !=
            droid::seed_list_sha256(splits.train)) {
        mismatch("training train split");
    }
    if (artifact.training.validation_seed_count != splits.validation.size() ||
        artifact.training.validation_seeds_sha256 !=
            droid::seed_list_sha256(splits.validation)) {
        mismatch("training validation split");
    }
    if (artifact.training.test_seed_count != splits.test.size() ||
        artifact.training.test_seeds_sha256 !=
            droid::seed_list_sha256(splits.test)) {
        mismatch("training test split");
    }
    if (artifact.compatibility.assembly_id != protocol.assembly_id) {
        mismatch("compatibility.assembly_id");
    }
    if (!same_binary64(
            artifact.compatibility.control_dt_s, protocol.control_dt_s)) {
        mismatch("compatibility.control_dt_s");
    }
    if (!same_binary64(artifact.effort_limit, protocol.effort_limit)) {
        mismatch("architecture.effort_limit");
    }
    if (!same_binary64(artifact.slew_limit, protocol.slew_limit)) {
        mismatch("architecture.slew_limit");
    }
}

struct ValidationPairAccumulator {
    std::size_t count{0};
    std::size_t learned_wins{0};
    std::size_t zero_wins{0};
    std::size_t ties{0};
    long double learned_sum{0.0L};
    long double zero_sum{0.0L};
    long double delta_sum{0.0L};

    void add(double learned, double zero) {
        ++count;
        learned_sum += learned;
        zero_sum += zero;
        delta_sum +=
            static_cast<long double>(learned) -
            static_cast<long double>(zero);
        if (learned > zero) {
            ++learned_wins;
        } else if (zero > learned) {
            ++zero_wins;
        } else {
            ++ties;
        }
    }

    [[nodiscard]] Json document() const {
        if (count == 0) {
            throw std::logic_error(
                "validation comparison stratum must not be empty");
        }
        const long double divisor = static_cast<long double>(count);
        const double learned_mean = static_cast<double>(learned_sum / divisor);
        const double zero_mean = static_cast<double>(zero_sum / divisor);
        const double delta_mean = static_cast<double>(delta_sum / divisor);
        if (!std::isfinite(learned_mean) || !std::isfinite(zero_mean) ||
            !std::isfinite(delta_mean)) {
            throw std::runtime_error(
                "validation comparison produced a non-finite mean");
        }
        return Json{
            {"scenario_count", count},
            {"learned_mean_reward_rate", learned_mean},
            {"zero_mean_reward_rate", zero_mean},
            {"learned_minus_zero_mean_reward_rate",
             delta_mean},
            {"learned_win_count", learned_wins},
            {"zero_win_count", zero_wins},
            {"tie_count", ties},
        };
    }
};

[[nodiscard]] double episode_reward_rate(
    const Json& episode,
    std::string_view policy_name) {
    if (!episode.is_object() || !episode.contains("mean_reward_rate") ||
        !episode.at("mean_reward_rate").is_number()) {
        throw std::runtime_error(
            std::string(policy_name) +
            " validation episode omitted mean_reward_rate");
    }
    const double value = episode.at("mean_reward_rate").get<double>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string(policy_name) +
            " validation episode has non-finite mean_reward_rate");
    }
    return value;
}

[[nodiscard]] Json paired_validation_comparison(
    const Json& learned,
    const Json& zero,
    const std::vector<std::uint64_t>& validation_seeds) {
    if (!learned.contains("episodes") ||
        !learned.at("episodes").is_array() ||
        !zero.contains("episodes") || !zero.at("episodes").is_array() ||
        learned.at("episodes").size() != validation_seeds.size() ||
        zero.at("episodes").size() != validation_seeds.size()) {
        throw std::runtime_error(
            "validation evaluations do not match the exact validation split");
    }

    ValidationPairAccumulator overall;
    ValidationPairAccumulator negative;
    ValidationPairAccumulator positive;
    Json episodes = Json::array();
    for (std::size_t index = 0; index < validation_seeds.size(); ++index) {
        const Json& learned_episode = learned.at("episodes").at(index);
        const Json& zero_episode = zero.at("episodes").at(index);
        if (!learned_episode.contains("environment_seed") ||
            !learned_episode.at("environment_seed").is_number_unsigned() ||
            !zero_episode.contains("environment_seed") ||
            !zero_episode.at("environment_seed").is_number_unsigned()) {
            throw std::runtime_error(
                "validation episode omitted its unsigned environment seed");
        }
        const std::uint64_t expected_seed = validation_seeds.at(index);
        if (learned_episode.at("environment_seed").get<std::uint64_t>() !=
                expected_seed ||
            zero_episode.at("environment_seed").get<std::uint64_t>() !=
                expected_seed) {
            throw std::runtime_error(
                "validation evaluations are not paired in derived seed order");
        }

        const double learned_rate =
            episode_reward_rate(learned_episode, "learned");
        const double zero_rate = episode_reward_rate(zero_episode, "zero");
        const bool positive_side =
            droid::light_search_seed_has_positive_side(expected_seed);
        overall.add(learned_rate, zero_rate);
        (positive_side ? positive : negative).add(learned_rate, zero_rate);

        episodes.push_back(Json{
            {"index", index},
            {"environment_seed", expected_seed},
            {"light_side", positive_side ? "positive_x" : "negative_x"},
            {"learned_mean_reward_rate", learned_rate},
            {"zero_mean_reward_rate", zero_rate},
            {"learned_minus_zero_mean_reward_rate",
             learned_rate - zero_rate},
            {"winner",
             learned_rate > zero_rate
                 ? "learned"
                 : (zero_rate > learned_rate ? "zero" : "tie")},
        });
    }
    if (negative.count != positive.count) {
        throw std::runtime_error(
            "derived validation split is not balanced by light side");
    }
    return Json{
        {"metric", "mean_reward_rate"},
        {"pairing", "exact_environment_seed_and_order"},
        {"win_rule", "strict_greater_mean_reward_rate"},
        {"overall", overall.document()},
        {"by_light_side",
         Json{
             {"negative_x", negative.document()},
             {"positive_x", positive.document()},
         }},
        {"episodes", std::move(episodes)},
    };
}

[[nodiscard]] std::uint64_t peak_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
#endif
#else
    return 0;
#endif
}

[[nodiscard]] double physics_timestep(const Json& spec) {
    double value = 0.0;
    if (spec.contains("physics_timestep_s") &&
        spec.at("physics_timestep_s").is_number()) {
        value = spec.at("physics_timestep_s").get<double>();
    } else if (spec.contains("timing") && spec.at("timing").is_object() &&
        spec.at("timing").contains("physics_timestep_s") &&
        spec.at("timing").at("physics_timestep_s").is_number()) {
        value = spec.at("timing").at("physics_timestep_s").get<double>();
    }
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(
            "environment specification has no positive finite physics timestep");
    }
    return value;
}

[[nodiscard]] double transition_reward(const Json& transition) {
    if (!transition.contains("reward") ||
        !transition.at("reward").is_number()) {
        throw std::runtime_error(
            "environment transition did not contain a numeric reward");
    }
    return transition.at("reward").get<double>();
}

[[nodiscard]] Json run_episode(
    droid::DroidEnvironment& environment,
    const std::vector<std::string>& motor_ids,
    const Options& options,
    bool record) {
    const Json reset = environment.reset(kAssemblyId, options.seed, record);
    double reward_checksum = 0.0;
    std::size_t executed_steps = 0;
    std::uint64_t executed_physics_steps = 0;
    double simulated_time_s = 0.0;
    for (std::size_t step = 0; step < options.steps; ++step) {
        const Json transition = environment.step(
            deterministic_actions(motor_ids, step, options.seed),
            options.control_dt_s);
        reward_checksum +=
            transition_reward(transition) * static_cast<double>(step + 1U);
        ++executed_steps;
        const Json& info = transition.at("info");
        executed_physics_steps +=
            info.at("physics_steps_executed").get<std::uint64_t>();
        simulated_time_s += info.at("elapsed_control_dt_s").get<double>();
        if (transition.value("terminated", false) ||
            transition.value("truncated", false)) {
            break;
        }
    }
    return Json{
        {"reset", reset},
        {"executed_steps", executed_steps},
        {"executed_physics_steps", executed_physics_steps},
        {"simulated_time_s", simulated_time_s},
        {"reward_checksum", reward_checksum},
    };
}

[[nodiscard]] int benchmark(const Options& options) {
    droid::DroidEnvironment environment(options.model, false);
    const Json spec = environment.spec();
    const std::vector<std::string> motor_ids = action_motor_ids(spec);
    const double physics_dt_s = physics_timestep(spec);

    const auto started = Clock::now();
    const Json result = run_episode(environment, motor_ids, options, false);
    const double wall_time_s = std::chrono::duration<double>(
        Clock::now() - started).count();
    const double safe_wall_time = std::max(
        wall_time_s, std::numeric_limits<double>::min());
    const std::size_t environment_steps =
        result.at("executed_steps").get<std::size_t>();
    const std::uint64_t physics_steps =
        result.at("executed_physics_steps").get<std::uint64_t>();
    const double simulated_time_s =
        result.at("simulated_time_s").get<double>();

    std::cout << Json{
        {"command", "benchmark"},
        {"assembly_id", kAssemblyId},
        {"seed", options.seed},
        {"recording", false},
        {"requested_environment_steps", options.steps},
        {"environment_steps", environment_steps},
        {"physics_steps", physics_steps},
        {"control_dt_s", options.control_dt_s},
        {"physics_timestep_s", physics_dt_s},
        {"wall_time_s", wall_time_s},
        {"environment_steps_per_s",
         static_cast<double>(environment_steps) / safe_wall_time},
        {"physics_steps_per_s",
         static_cast<double>(physics_steps) / safe_wall_time},
        {"simulated_seconds_per_wall_second",
         simulated_time_s / safe_wall_time},
        {"peak_rss_bytes", peak_rss_bytes()},
        {"reward_checksum", result.at("reward_checksum")},
    }.dump() << '\n';
    return 0;
}

[[nodiscard]] int record(const Options& options) {
    droid::DroidEnvironment environment(options.model, false);
    const std::vector<std::string> motor_ids =
        action_motor_ids(environment.spec());
    const Json result = run_episode(environment, motor_ids, options, true);
    environment.save_trace(*options.output);
    const Json trace = environment.trace();
    std::cout << Json{
        {"command", "record"},
        {"assembly_id", kAssemblyId},
        {"seed", options.seed},
        {"path", options.output->string()},
        {"recorded_steps", trace.at("steps").size()},
        {"reward_checksum", result.at("reward_checksum")},
    }.dump() << '\n';
    return 0;
}

[[nodiscard]] int replay(const Options& options) {
    droid::DroidEnvironment environment(options.model, false);
    Json report = environment.replay_and_verify_file(*options.input);
    report["command"] = "replay";
    report["path"] = options.input->string();
    std::cout << report.dump() << '\n';
    return report.value("ok", false) ? 0 : 1;
}

[[nodiscard]] const Json& evaluation_aggregate(const Json& evaluation) {
    if (!evaluation.contains("aggregate") ||
        !evaluation.at("aggregate").is_object()) {
        throw std::runtime_error(
            "baseline evaluation did not contain an aggregate object");
    }
    return evaluation.at("aggregate");
}

template <typename Value>
[[nodiscard]] Value aggregate_value(
    const Json& evaluation,
    std::string_view key) {
    const Json& aggregate = evaluation_aggregate(evaluation);
    const std::string key_string(key);
    if (!aggregate.contains(key_string) ||
        !aggregate.at(key_string).is_number()) {
        throw std::runtime_error(
            "baseline aggregate did not contain numeric " + key_string);
    }
    return aggregate.at(key_string).get<Value>();
}

[[nodiscard]] int evaluate(const Options& options) {
    droid::BaselineEvaluationOptions evaluation_options;
    evaluation_options.assembly_id = std::string(kAssemblyId);
    evaluation_options.episodes = options.episodes;
    evaluation_options.max_control_steps = options.steps;
    evaluation_options.base_seed = options.seed;
    evaluation_options.control_dt_s = options.control_dt_s;
    evaluation_options.random_effort_limit = options.random_effort;
    evaluation_options.random_hold_steps = options.hold_steps;

    const auto started = Clock::now();
    Json evaluations = Json::array();
    std::uint64_t total_environment_steps = 0;
    std::uint64_t total_physics_steps = 0;
    double total_simulated_time_s = 0.0;

    for (const droid::BaselinePolicyKind policy :
         evaluation_policies(options.policy)) {
        Json result = droid::evaluate_baseline(
            options.model, policy, evaluation_options);
        total_environment_steps +=
            aggregate_value<std::uint64_t>(result, "total_control_steps");
        total_physics_steps +=
            aggregate_value<std::uint64_t>(result, "total_physics_steps");
        total_simulated_time_s +=
            aggregate_value<double>(result, "total_simulated_time_s");
        evaluations.push_back(std::move(result));
    }

    const double wall_time_s = std::chrono::duration<double>(
        Clock::now() - started).count();
    const double safe_wall_time = std::max(
        wall_time_s, std::numeric_limits<double>::min());

    Json output{
        {"command", "evaluate"},
        {"assembly_id", kAssemblyId},
        {"policy", evaluation_policy_name(options.policy)},
        {"seed", options.seed},
        {"evaluations", std::move(evaluations)},
        {"performance",
         {
             {"wall_time_s", wall_time_s},
             {"environment_steps", total_environment_steps},
             {"physics_steps", total_physics_steps},
             {"environment_steps_per_s",
              static_cast<double>(total_environment_steps) / safe_wall_time},
             {"physics_steps_per_s",
              static_cast<double>(total_physics_steps) / safe_wall_time},
             {"simulated_seconds_per_wall_second",
              total_simulated_time_s / safe_wall_time},
             {"peak_rss_bytes", peak_rss_bytes()},
         }},
    };

    if (options.policy == EvaluationPolicy::both) {
        const Json& zero = output.at("evaluations").at(0);
        const Json& random = output.at("evaluations").at(1);
        const double return_delta =
            aggregate_value<double>(random, "mean_return") -
            aggregate_value<double>(zero, "mean_return");
        const auto random_terminated = aggregate_value<std::int64_t>(
            random, "terminated_episodes");
        const auto zero_terminated = aggregate_value<std::int64_t>(
            zero, "terminated_episodes");
        output["comparison"] = Json{
            {"random_minus_zero_mean_return", return_delta},
            {"random_minus_zero_termination_count",
             random_terminated - zero_terminated},
        };
    }

    std::cout << output.dump() << '\n';
    return 0;
}

[[nodiscard]] int train(const Options& options) {
    const LearningProtocol protocol = learning_protocol();
    validate_exclusive_destination(*options.output_policy, "policy artifact");

    const auto started = Clock::now();
    droid::PolicyTrainingRequest request = training_request(
        options,
        protocol,
        runtime_compatibility(options.model, protocol));
    const droid::PolicyTrainingResult result =
        droid::train_frozen_policy_in_environment(options.model, request);
    droid::save_policy_artifact_exclusive(
        *options.output_policy, result.artifact);
    const double wall_time_s = std::chrono::duration<double>(
        Clock::now() - started).count();

    const Json artifact = policy_artifact_summary(result.artifact);
    const Json effective_manifest = effective_training_manifest(
        protocol, request.optimizer, request.train_batch_size);
    std::cout << Json{
        {"schema_version", "droid-blocks.agent-training-command.v1"},
        {"command", "train"},
        {"official_protocol_manifest", protocol.manifest},
        {"official_protocol_manifest_sha256", protocol.manifest_sha256},
        {"effective_training_manifest", effective_manifest},
        {"effective_training_manifest_sha256",
         request.protocol_manifest_sha256},
        {"effective_training_configuration",
         Json{
             {"episode_profile", request.episode_profile},
             {"max_control_steps", request.max_control_steps},
             {"control_dt_s", request.compatibility.control_dt_s},
             {"optimizer",
              optimizer_json(request.optimizer, request.train_batch_size)},
             {"effort_limit", request.effort_limit},
             {"slew_limit_per_control_step", request.slew_limit},
             {"seed_splits", seed_splits_json(request.seed_splits)},
         }},
        {"runtime_compatibility",
         compatibility_json(request.compatibility)},
        {"execution",
         Json{{"candidate_evaluation_workers",
               request.candidate_evaluation_workers}}},
        {"policy_artifact",
         Json{
             {"path", options.output_policy->string()},
             {"schema_version", artifact.at("schema_version")},
             {"self_sha256", artifact.at("self_sha256")},
             {"architecture", artifact.at("architecture")},
             {"compatibility", artifact.at("compatibility")},
             {"training", artifact.at("training")},
         }},
        {"training_report", result.deterministic_report},
        {"performance",
         Json{
             {"wall_time_s", wall_time_s},
             {"peak_rss_bytes", peak_rss_bytes()},
         }},
    }.dump() << '\n';
    return 0;
}

[[nodiscard]] int validate_frozen_command(const Options& options) {
    const auto started = Clock::now();
    const droid::PolicyArtifactData artifact =
        droid::load_policy_artifact(*options.input_policy);
    const droid::MirroredSeedSplits splits = artifact_seed_splits(artifact);
    const droid::PolicyCompatibility compatibility = runtime_compatibility(
        options.model,
        artifact.compatibility.assembly_id,
        artifact.compatibility.control_dt_s);
    droid::require_policy_compatible(artifact, compatibility);

    droid::PolicyEvaluationOptions evaluation_options;
    evaluation_options.assembly_id = artifact.compatibility.assembly_id;
    evaluation_options.episode_profile =
        artifact.training.scenario_profile_id;
    evaluation_options.episodes = splits.validation.size();
    evaluation_options.max_control_steps =
        artifact.training.max_control_steps;
    evaluation_options.base_seed = artifact.training.optimizer_seed;
    evaluation_options.control_dt_s =
        artifact.compatibility.control_dt_s;
    evaluation_options.explicit_environment_seeds = splits.validation;

    const Json learned = droid::evaluate_frozen_policy(
        options.model, artifact, compatibility, evaluation_options);
    droid::PolicyEvaluationDescriptor zero_descriptor;
    zero_descriptor.id = "zero_v0";
    zero_descriptor.uses_observation = false;
    zero_descriptor.uses_policy_seed = false;
    zero_descriptor.metadata = Json{
        {"role", "validation_control"},
        {"split", "validation"},
    };
    const Json zero = droid::evaluate_policy(
        options.model,
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
        evaluation_options);
    const Json comparison = paired_validation_comparison(
        learned, zero, splits.validation);
    const double wall_time_s = std::chrono::duration<double>(
        Clock::now() - started).count();

    std::cout << Json{
        {"schema_version", "droid-blocks.agent-policy-validation-command.v1"},
        {"command", "validate-policy"},
        {"classification", "diagnostic_non_official"},
        {"split_evaluated", "validation"},
        {"test_rollouts_executed", false},
        {"heldout_decision_emitted", false},
        {"input_policy", options.input_policy->string()},
        {"policy_artifact_sha256", artifact.self_sha256},
        {"seed_split_provenance",
         Json{
             {"derivation_id", artifact.training.seed_derivation_id},
             {"base_seed", artifact.training.seed_split_base_seed},
             {"digests", seed_splits_json(splits)},
         }},
        {"runtime_compatibility", compatibility_json(compatibility)},
        {"policy_artifact", policy_artifact_summary(artifact)},
        {"validation_evaluation",
         Json{
             {"schema_version",
              "droid-blocks.frozen-validation-evaluation.v1"},
             {"split", "validation"},
             {"learned", learned},
             {"zero", zero},
             {"paired_comparison", comparison},
         }},
        {"performance",
         Json{
             {"wall_time_s", wall_time_s},
             {"peak_rss_bytes", peak_rss_bytes()},
         }},
    }.dump() << '\n';
    return 0;
}

[[nodiscard]] int evaluate_frozen_command(const Options& options) {
    const LearningProtocol protocol = learning_protocol();
    const droid::MirroredSeedSplits splits = protocol_seed_splits(protocol);
    validate_exclusive_destination(*options.output_report, "held-out report");

    const auto started = Clock::now();
    const droid::PolicyArtifactData artifact =
        droid::load_policy_artifact(*options.input_policy);
    require_official_heldout_artifact(artifact, protocol, splits);
    const droid::PolicyCompatibility compatibility =
        runtime_compatibility(options.model, protocol);
    droid::require_policy_compatible(artifact, compatibility);

    droid::PolicyEvaluationOptions evaluation_options;
    evaluation_options.assembly_id = protocol.assembly_id;
    evaluation_options.episode_profile = protocol.episode_profile;
    evaluation_options.episodes = splits.test.size();
    evaluation_options.max_control_steps = protocol.max_control_steps;
    evaluation_options.base_seed = protocol.split_base_seed;
    evaluation_options.control_dt_s = protocol.control_dt_s;
    evaluation_options.explicit_environment_seeds = splits.test;
    const Json heldout = droid::evaluate_frozen_heldout(
        options.model,
        artifact,
        compatibility,
        std::move(evaluation_options),
        protocol.acceptance);

    const Json report{
        {"schema_version", "droid-blocks.official-heldout-report.v1"},
        {"protocol_manifest", protocol.manifest},
        {"protocol_manifest_sha256", protocol.manifest_sha256},
        {"seed_split_provenance", seed_splits_json(splits)},
        {"runtime_compatibility", compatibility_json(compatibility)},
        {"policy_artifact", policy_artifact_summary(artifact)},
        {"one_shot_governance", droid::v3_heldout_report_governance()},
        {"heldout_evaluation", heldout},
    };
    const std::string report_sha256 = droid::sha256_hex(report.dump());
    save_json_exclusive(*options.output_report, report);
    const double wall_time_s = std::chrono::duration<double>(
        Clock::now() - started).count();
    const bool accepted = heldout.at("acceptance").at("accepted").get<bool>();

    Json output{
        {"schema_version", "droid-blocks.agent-policy-evaluation-command.v1"},
        {"command", "evaluate-policy"},
        {"accepted", accepted},
        {"input_policy", options.input_policy->string()},
        {"policy_artifact_sha256", artifact.self_sha256},
        {"report_canonical_json_sha256", report_sha256},
        {"report", report},
        {"performance",
         Json{
             {"wall_time_s", wall_time_s},
             {"peak_rss_bytes", peak_rss_bytes()},
         }},
    };
    output["output_report"] = options.output_report->string();
    std::cout << output.dump() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_usage(std::cout, argc > 0 ? argv[0] : "droid-agent");
            return 0;
        }
        switch (options.command) {
            case Command::benchmark:
                return benchmark(options);
            case Command::record:
                return record(options);
            case Command::replay:
                return replay(options);
            case Command::evaluate:
                return evaluate(options);
            case Command::train:
                return train(options);
            case Command::validate_policy:
                return validate_frozen_command(options);
            case Command::evaluate_policy:
                return evaluate_frozen_command(options);
        }
        throw std::logic_error("unreachable command");
    } catch (const std::invalid_argument& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << "Try --help for usage.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
