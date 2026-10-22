#include "droid/evaluation.hpp"

#include "droid/environment.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace droid {
namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kEnvironmentStreamTag{0x44524f4944454e56ULL};
constexpr std::uint64_t kPolicyStreamTag{0x44524f4944504f4cULL};

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    // The SplitMix64 finalizer is written out here so the evaluation streams are
    // independent of any C++ standard-library random implementation.
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

[[nodiscard]] std::uint64_t stream_seed(
    std::uint64_t base_seed,
    std::size_t episode_index,
    std::uint64_t tag) noexcept {
    const auto episode = static_cast<std::uint64_t>(episode_index);
    return mix64(base_seed ^ tag ^ mix64(episode ^ 0x9e3779b97f4a7c15ULL));
}

class CompensatedSum final {
public:
    void add(long double value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("metric accumulation received a non-finite value");
        }
        const long double tentative = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            correction_ += (sum_ - tentative) + value;
        } else {
            correction_ += (value - tentative) + sum_;
        }
        sum_ = tentative;
    }

    [[nodiscard]] long double value() const noexcept {
        return sum_ + correction_;
    }

private:
    long double sum_{0.0L};
    long double correction_{0.0L};
};

[[nodiscard]] double finite_double(long double value, std::string_view name) {
    if (!std::isfinite(value) ||
        std::abs(value) >
            static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::runtime_error(
            std::string(name) + " cannot be represented as a finite double");
    }
    const double converted = static_cast<double>(value);
    if (!std::isfinite(converted)) {
        throw std::runtime_error(std::string(name) + " is non-finite");
    }
    return converted;
}

[[nodiscard]] double finite_number(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.is_object() || !object.contains(key) ||
        !object.at(key).is_number()) {
        throw std::runtime_error(
            std::string(context) + "." + key + " must be numeric");
    }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string(context) + "." + key + " must be finite");
    }
    return value;
}

[[nodiscard]] std::string nonempty_string(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.is_object() || !object.contains(key) ||
        !object.at(key).is_string() ||
        object.at(key).get_ref<const std::string&>().empty()) {
        throw std::runtime_error(
            std::string(context) + "." + key +
            " must be a non-empty string");
    }
    return object.at(key).get<std::string>();
}

[[nodiscard]] bool boolean_member(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.is_object() || !object.contains(key) ||
        !object.at(key).is_boolean()) {
        throw std::runtime_error(
            std::string(context) + "." + key + " must be boolean");
    }
    return object.at(key).get<bool>();
}

[[nodiscard]] const Json& array_member(
    const Json& object,
    std::string_view member,
    std::string_view context) {
    const std::string key(member);
    if (!object.is_object() || !object.contains(key) ||
        !object.at(key).is_array()) {
        throw std::runtime_error(
            std::string(context) + "." + key + " must be an array");
    }
    return object.at(key);
}

void checked_increment(std::uint64_t& destination, std::uint64_t amount = 1) {
    if (amount > std::numeric_limits<std::uint64_t>::max() - destination) {
        throw std::overflow_error("baseline metric counter overflow");
    }
    destination += amount;
}

void merge_counter(
    std::map<std::string, std::uint64_t>& destination,
    const std::map<std::string, std::uint64_t>& source) {
    for (const auto& [key, count] : source) {
        checked_increment(destination[key], count);
    }
}

[[nodiscard]] Json counter_json(
    const std::map<std::string, std::uint64_t>& counts) {
    Json result = Json::object();
    for (const auto& [key, count] : counts) {
        result[key] = count;
    }
    return result;
}

[[nodiscard]] Json sum_json(
    const std::map<std::string, CompensatedSum>& sums,
    std::string_view context) {
    Json result = Json::object();
    for (const auto& [key, sum] : sums) {
        result[key] = finite_double(sum.value(), context);
    }
    return result;
}

struct ActuatorMetrics {
    std::uint64_t valid_sample_count{0};
    bool has_valid_sample{false};
    double peak_abs_current_a{0.0};
    double peak_temperature_c{0.0};
    double peak_stuck_score{0.0};
    std::uint64_t stuck_sample_count{0};
    std::map<std::string, std::uint64_t> fault_flag_counts;
};

struct EpisodeMetrics {
    std::size_t index{0};
    std::string episode_profile;
    std::uint64_t environment_seed{0};
    std::uint64_t policy_seed{0};
    bool policy_seed_used{false};
    CompensatedSum episode_return;
    CompensatedSum component_total;
    std::map<std::string, CompensatedSum> reward_by_sensor;
    std::map<std::string, CompensatedSum> reward_by_family;
    std::map<std::string, std::string> reward_sensor_families;
    std::uint64_t control_steps{0};
    std::uint64_t physics_steps{0};
    CompensatedSum simulated_time_s;
    bool environment_terminated{false};
    bool environment_truncated{false};
    bool horizon_reached{false};
    std::uint64_t safety_veto_count{0};
    std::map<std::string, std::uint64_t> safety_flag_counts;
    std::map<std::string, std::map<std::string, std::uint64_t>>
        actuator_safety_flag_counts;
    std::uint64_t observation_snapshots{0};
    std::uint64_t reward_sensor_samples{0};
    std::uint64_t actuator_feedback_samples{0};
    std::uint64_t invalid_reward_sensor_samples{0};
    std::uint64_t invalid_actuator_feedback_samples{0};
    std::map<std::string, ActuatorMetrics> actuators;
    std::uint64_t action_value_count{0};
    CompensatedSum absolute_action_effort;
    double maximum_absolute_action_effort{0.0};
};

void validate_string_array_and_count(
    const Json& values,
    std::map<std::string, std::uint64_t>& counts,
    std::string_view context) {
    if (!values.is_array()) {
        throw std::runtime_error(std::string(context) + " must be an array");
    }
    for (const Json& value : values) {
        if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
            throw std::runtime_error(
                std::string(context) + " entries must be non-empty strings");
        }
        checked_increment(counts[value.get<std::string>()]);
    }
}

void inspect_policy_observation(
    const Json& observation,
    EpisodeMetrics& metrics) {
    if (!observation.is_object() || observation.size() != 2 ||
        !observation.contains("reward_sensors") ||
        !observation.at("reward_sensors").is_array() ||
        !observation.contains("actuator_feedback") ||
        !observation.at("actuator_feedback").is_array()) {
        throw std::runtime_error(
            "policy observation must contain exactly reward_sensors and "
            "actuator_feedback arrays");
    }
    checked_increment(metrics.observation_snapshots);

    std::set<std::string> sensor_ids;
    for (const Json& sample : observation.at("reward_sensors")) {
        const std::string module_id = nonempty_string(
            sample, "module_id", "reward sensor sample");
        (void)nonempty_string(sample, "family_id", "reward sensor sample");
        if (!sensor_ids.insert(module_id).second) {
            throw std::runtime_error(
                "policy observation contains a duplicate reward sensor ID");
        }
        const bool valid = boolean_member(
            sample, "valid", "reward sensor sample");
        checked_increment(metrics.reward_sensor_samples);
        if (!valid) {
            checked_increment(metrics.invalid_reward_sensor_samples);
        } else if (!sample.contains("observations") ||
                   !sample.at("observations").is_object()) {
            throw std::runtime_error(
                "valid reward sensor sample must contain observations object");
        }
    }

    std::set<std::string> actuator_ids;
    for (const Json& sample : observation.at("actuator_feedback")) {
        const std::string module_id = nonempty_string(
            sample, "module_id", "actuator feedback sample");
        if (!actuator_ids.insert(module_id).second) {
            throw std::runtime_error(
                "policy observation contains a duplicate actuator ID");
        }
        const bool valid = boolean_member(
            sample, "valid", "actuator feedback sample");
        checked_increment(metrics.actuator_feedback_samples);
        ActuatorMetrics& actuator = metrics.actuators[module_id];
        if (!valid) {
            checked_increment(metrics.invalid_actuator_feedback_samples);
            continue;
        }
        if (!sample.contains("feedback") || !sample.at("feedback").is_object()) {
            throw std::runtime_error(
                "valid actuator feedback sample must contain feedback object");
        }
        const Json& feedback = sample.at("feedback");
        const double current_a = finite_number(
            feedback, "current_a", "actuator feedback");
        const double temperature_c = finite_number(
            feedback, "temperature_c", "actuator feedback");
        const double stuck_score = finite_number(
            feedback, "stuck_score", "actuator feedback");
        if (stuck_score < 0.0 || stuck_score > 1.0) {
            throw std::runtime_error(
                "actuator feedback.stuck_score must be inside [0, 1]");
        }
        const bool stuck = boolean_member(
            feedback, "stuck", "actuator feedback");
        const Json& fault_flags = array_member(
            feedback, "fault_flags", "actuator feedback");

        checked_increment(actuator.valid_sample_count);
        actuator.peak_abs_current_a = std::max(
            actuator.peak_abs_current_a, std::abs(current_a));
        if (!actuator.has_valid_sample) {
            actuator.peak_temperature_c = temperature_c;
            actuator.has_valid_sample = true;
        } else {
            actuator.peak_temperature_c = std::max(
                actuator.peak_temperature_c, temperature_c);
        }
        actuator.peak_stuck_score = std::max(
            actuator.peak_stuck_score, stuck_score);
        if (stuck) {
            checked_increment(actuator.stuck_sample_count);
        }
        validate_string_array_and_count(
            fault_flags,
            actuator.fault_flag_counts,
            "actuator feedback.fault_flags");
    }
}

void inspect_actions(
    const std::map<std::string, double>& actions,
    EpisodeMetrics& metrics) {
    for (const auto& [module_id, effort] : actions) {
        if (module_id.empty() || !std::isfinite(effort)) {
            throw std::runtime_error("baseline generated an invalid action");
        }
        const double magnitude = std::abs(effort);
        metrics.absolute_action_effort.add(magnitude);
        metrics.maximum_absolute_action_effort = std::max(
            metrics.maximum_absolute_action_effort, magnitude);
        checked_increment(metrics.action_value_count);
    }
}

void inspect_reward(const Json& transition, EpisodeMetrics& metrics) {
    const double reward = finite_number(transition, "reward", "transition");
    metrics.episode_return.add(reward);
    const Json& components = array_member(
        transition, "reward_components", "transition");
    std::set<std::string> seen_module_ids;
    for (const Json& component : components) {
        const std::string module_id = nonempty_string(
            component, "module_id", "reward component");
        const std::string family_id = nonempty_string(
            component, "family_id", "reward component");
        if (!seen_module_ids.insert(module_id).second) {
            throw std::runtime_error(
                "transition contains a duplicate reward component module ID");
        }
        const auto [family, inserted] =
            metrics.reward_sensor_families.emplace(module_id, family_id);
        if (!inserted && family->second != family_id) {
            throw std::runtime_error(
                "reward component changed family for module ID " + module_id);
        }
        const double component_reward = finite_number(
            component, "transition_reward", "reward component");
        metrics.component_total.add(component_reward);
        metrics.reward_by_sensor[module_id].add(component_reward);
        metrics.reward_by_family[family_id].add(component_reward);
    }
}

void inspect_safety(const Json& transition, EpisodeMetrics& metrics) {
    if (!transition.contains("safety") ||
        !transition.at("safety").is_object()) {
        throw std::runtime_error("transition.safety must be an object");
    }
    const Json& safety = transition.at("safety");
    if (boolean_member(safety, "veto", "transition.safety")) {
        checked_increment(metrics.safety_veto_count);
    }
    validate_string_array_and_count(
        array_member(safety, "flags", "transition.safety"),
        metrics.safety_flag_counts,
        "transition.safety.flags");

    const Json& actuator_flags = array_member(
        safety, "actuator_flags", "transition.safety");
    std::set<std::string> seen_ids;
    for (const Json& entry : actuator_flags) {
        const std::string module_id = nonempty_string(
            entry, "module_id", "actuator safety entry");
        if (!seen_ids.insert(module_id).second) {
            throw std::runtime_error(
                "transition safety contains a duplicate actuator ID");
        }
        validate_string_array_and_count(
            array_member(entry, "flags", "actuator safety entry"),
            metrics.actuator_safety_flag_counts[module_id],
            "actuator safety flags");
    }
}

void inspect_timing_and_outcome(
    const Json& transition,
    EpisodeMetrics& metrics) {
    if (!transition.contains("info") || !transition.at("info").is_object()) {
        throw std::runtime_error("transition.info must be an object");
    }
    const Json& info = transition.at("info");
    if (!info.contains("physics_steps_executed") ||
        !info.at("physics_steps_executed").is_number_unsigned()) {
        throw std::runtime_error(
            "transition.info.physics_steps_executed must be unsigned");
    }
    const std::uint64_t physics_steps =
        info.at("physics_steps_executed").get<std::uint64_t>();
    if (physics_steps == 0) {
        throw std::runtime_error(
            "transition executed no physics steps");
    }
    checked_increment(metrics.physics_steps, physics_steps);
    metrics.simulated_time_s.add(finite_number(
        info, "elapsed_control_dt_s", "transition.info"));
    metrics.environment_terminated = boolean_member(
        transition, "terminated", "transition");
    metrics.environment_truncated = boolean_member(
        transition, "truncated", "transition");
}

[[nodiscard]] Json nested_counter_json(
    const std::map<
        std::string,
        std::map<std::string, std::uint64_t>>& counts) {
    Json result = Json::object();
    for (const auto& [module_id, module_counts] : counts) {
        result[module_id] = counter_json(module_counts);
    }
    return result;
}

[[nodiscard]] Json actuator_metrics_json(
    const std::map<std::string, ActuatorMetrics>& actuators) {
    Json by_module = Json::object();
    double overall_current = 0.0;
    double overall_temperature = 0.0;
    double overall_stuck_score = 0.0;
    bool has_valid_sample = false;
    std::uint64_t total_stuck_samples = 0;
    std::map<std::string, std::uint64_t> total_fault_counts;

    for (const auto& [module_id, actuator] : actuators) {
        Json entry{
            {"valid_sample_count", actuator.valid_sample_count},
            {"stuck_sample_count", actuator.stuck_sample_count},
            {"fault_flag_counts", counter_json(actuator.fault_flag_counts)},
        };
        if (actuator.has_valid_sample) {
            entry["peak_abs_current_a"] = actuator.peak_abs_current_a;
            entry["peak_temperature_c"] = actuator.peak_temperature_c;
            entry["peak_stuck_score"] = actuator.peak_stuck_score;
            overall_current = std::max(
                overall_current, actuator.peak_abs_current_a);
            if (!has_valid_sample) {
                overall_temperature = actuator.peak_temperature_c;
                has_valid_sample = true;
            } else {
                overall_temperature = std::max(
                    overall_temperature, actuator.peak_temperature_c);
            }
            overall_stuck_score = std::max(
                overall_stuck_score, actuator.peak_stuck_score);
        } else {
            entry["peak_abs_current_a"] = nullptr;
            entry["peak_temperature_c"] = nullptr;
            entry["peak_stuck_score"] = nullptr;
        }
        checked_increment(total_stuck_samples, actuator.stuck_sample_count);
        merge_counter(total_fault_counts, actuator.fault_flag_counts);
        by_module[module_id] = std::move(entry);
    }

    return Json{
        {"by_module", std::move(by_module)},
        {"peak_abs_current_a", has_valid_sample ? Json(overall_current) : Json(nullptr)},
        {"peak_temperature_c", has_valid_sample ? Json(overall_temperature) : Json(nullptr)},
        {"peak_stuck_score", has_valid_sample ? Json(overall_stuck_score) : Json(nullptr)},
        {"stuck_sample_count", total_stuck_samples},
        {"fault_flag_counts", counter_json(total_fault_counts)},
    };
}

[[nodiscard]] Json episode_json(const EpisodeMetrics& metrics) {
    const long double elapsed = metrics.simulated_time_s.value();
    const long double episode_return = metrics.episode_return.value();
    const double mean_reward_rate = elapsed > 0.0L
        ? finite_double(episode_return / elapsed, "episode mean reward rate")
        : 0.0;
    const long double reconciliation =
        episode_return - metrics.component_total.value();
    const double mean_abs_effort = metrics.action_value_count > 0
        ? finite_double(
              metrics.absolute_action_effort.value() /
                  static_cast<long double>(metrics.action_value_count),
              "episode mean absolute action effort")
        : 0.0;

    return Json{
        {"index", metrics.index},
        {"episode_profile", metrics.episode_profile},
        {"environment_seed", metrics.environment_seed},
        {"policy_seed",
         metrics.policy_seed_used ? Json(metrics.policy_seed) : Json(nullptr)},
        {"return", finite_double(episode_return, "episode return")},
        {"mean_reward_rate", mean_reward_rate},
        {"control_steps", metrics.control_steps},
        {"physics_steps", metrics.physics_steps},
        {"simulated_time_s", finite_double(elapsed, "episode simulated time")},
        {"environment_terminated", metrics.environment_terminated},
        {"environment_truncated", metrics.environment_truncated},
        {"horizon_reached", metrics.horizon_reached},
        {"reward",
         Json{
             {"by_sensor", sum_json(metrics.reward_by_sensor, "sensor reward")},
             {"by_family", sum_json(metrics.reward_by_family, "family reward")},
             {"reconciliation_error",
              finite_double(reconciliation, "episode reward reconciliation")},
         }},
        {"safety",
         Json{
             {"veto_count", metrics.safety_veto_count},
             {"flag_counts", counter_json(metrics.safety_flag_counts)},
             {"actuator_flag_counts",
              nested_counter_json(metrics.actuator_safety_flag_counts)},
         }},
        {"health",
         Json{
             {"observation_snapshots", metrics.observation_snapshots},
             {"reward_sensor_samples", metrics.reward_sensor_samples},
             {"actuator_feedback_samples", metrics.actuator_feedback_samples},
             {"invalid_reward_sensor_samples",
              metrics.invalid_reward_sensor_samples},
             {"invalid_actuator_feedback_samples",
              metrics.invalid_actuator_feedback_samples},
         }},
        {"actuators", actuator_metrics_json(metrics.actuators)},
        {"actions",
         Json{
             {"value_count", metrics.action_value_count},
             {"mean_abs_effort", mean_abs_effort},
             {"max_abs_effort", metrics.maximum_absolute_action_effort},
         }},
    };
}

struct AggregateMetrics {
    std::vector<long double> returns;
    CompensatedSum total_return;
    CompensatedSum total_episode_reward_rate;
    CompensatedSum total_simulated_time_s;
    std::uint64_t total_control_steps{0};
    std::uint64_t total_physics_steps{0};
    std::uint64_t terminated_episodes{0};
    std::uint64_t truncated_episodes{0};
    std::uint64_t horizon_reached_episodes{0};
    std::map<std::string, CompensatedSum> reward_by_sensor;
    std::map<std::string, CompensatedSum> reward_by_family;
    CompensatedSum component_total;
    std::uint64_t safety_veto_count{0};
    std::uint64_t episodes_with_safety_veto{0};
    std::map<std::string, std::uint64_t> safety_flag_counts;
    std::map<std::string, std::map<std::string, std::uint64_t>>
        actuator_safety_flag_counts;
    std::uint64_t observation_snapshots{0};
    std::uint64_t reward_sensor_samples{0};
    std::uint64_t actuator_feedback_samples{0};
    std::uint64_t invalid_reward_sensor_samples{0};
    std::uint64_t invalid_actuator_feedback_samples{0};
    std::uint64_t episodes_with_invalid_samples{0};
    std::map<std::string, ActuatorMetrics> actuators;
    std::uint64_t action_value_count{0};
    CompensatedSum absolute_action_effort;
    double maximum_absolute_action_effort{0.0};
};

void merge_episode(
    const EpisodeMetrics& episode,
    AggregateMetrics& aggregate) {
    const long double episode_return = episode.episode_return.value();
    const long double elapsed = episode.simulated_time_s.value();
    aggregate.returns.push_back(episode_return);
    aggregate.total_return.add(episode_return);
    aggregate.total_simulated_time_s.add(elapsed);
    aggregate.total_episode_reward_rate.add(
        elapsed > 0.0L ? episode_return / elapsed : 0.0L);
    checked_increment(aggregate.total_control_steps, episode.control_steps);
    checked_increment(aggregate.total_physics_steps, episode.physics_steps);
    if (episode.environment_terminated) {
        checked_increment(aggregate.terminated_episodes);
    }
    if (episode.environment_truncated) {
        checked_increment(aggregate.truncated_episodes);
    }
    if (episode.horizon_reached) {
        checked_increment(aggregate.horizon_reached_episodes);
    }
    for (const auto& [id, value] : episode.reward_by_sensor) {
        aggregate.reward_by_sensor[id].add(value.value());
    }
    for (const auto& [id, value] : episode.reward_by_family) {
        aggregate.reward_by_family[id].add(value.value());
    }
    aggregate.component_total.add(episode.component_total.value());
    checked_increment(aggregate.safety_veto_count, episode.safety_veto_count);
    if (episode.safety_veto_count > 0) {
        checked_increment(aggregate.episodes_with_safety_veto);
    }
    merge_counter(aggregate.safety_flag_counts, episode.safety_flag_counts);
    for (const auto& [module_id, counts] :
         episode.actuator_safety_flag_counts) {
        merge_counter(aggregate.actuator_safety_flag_counts[module_id], counts);
    }
    checked_increment(
        aggregate.observation_snapshots, episode.observation_snapshots);
    checked_increment(
        aggregate.reward_sensor_samples, episode.reward_sensor_samples);
    checked_increment(
        aggregate.actuator_feedback_samples, episode.actuator_feedback_samples);
    checked_increment(
        aggregate.invalid_reward_sensor_samples,
        episode.invalid_reward_sensor_samples);
    checked_increment(
        aggregate.invalid_actuator_feedback_samples,
        episode.invalid_actuator_feedback_samples);
    if (episode.invalid_reward_sensor_samples > 0 ||
        episode.invalid_actuator_feedback_samples > 0) {
        checked_increment(aggregate.episodes_with_invalid_samples);
    }
    for (const auto& [module_id, source] : episode.actuators) {
        ActuatorMetrics& destination = aggregate.actuators[module_id];
        checked_increment(
            destination.valid_sample_count, source.valid_sample_count);
        checked_increment(
            destination.stuck_sample_count, source.stuck_sample_count);
        if (source.has_valid_sample) {
            destination.peak_abs_current_a = std::max(
                destination.peak_abs_current_a, source.peak_abs_current_a);
            if (!destination.has_valid_sample) {
                destination.peak_temperature_c = source.peak_temperature_c;
                destination.has_valid_sample = true;
            } else {
                destination.peak_temperature_c = std::max(
                    destination.peak_temperature_c,
                    source.peak_temperature_c);
            }
            destination.peak_stuck_score = std::max(
                destination.peak_stuck_score, source.peak_stuck_score);
        }
        merge_counter(
            destination.fault_flag_counts, source.fault_flag_counts);
    }
    checked_increment(
        aggregate.action_value_count, episode.action_value_count);
    aggregate.absolute_action_effort.add(
        episode.absolute_action_effort.value());
    aggregate.maximum_absolute_action_effort = std::max(
        aggregate.maximum_absolute_action_effort,
        episode.maximum_absolute_action_effort);
}

[[nodiscard]] Json aggregate_json(const AggregateMetrics& aggregate) {
    if (aggregate.returns.empty()) {
        throw std::logic_error("cannot summarize zero baseline episodes");
    }
    const long double count =
        static_cast<long double>(aggregate.returns.size());
    const long double total_return = aggregate.total_return.value();
    const long double mean_return = total_return / count;
    const auto [minimum, maximum] = std::minmax_element(
        aggregate.returns.begin(), aggregate.returns.end());
    CompensatedSum squared_deviations;
    for (const long double value : aggregate.returns) {
        const long double deviation = value - mean_return;
        squared_deviations.add(deviation * deviation);
    }
    const long double population_stddev = std::sqrt(
        std::max(0.0L, squared_deviations.value() / count));
    const long double total_time = aggregate.total_simulated_time_s.value();
    const double exposure_weighted_rate = total_time > 0.0L
        ? finite_double(
              total_return / total_time,
              "aggregate exposure weighted reward rate")
        : 0.0;
    const double mean_episode_rate = finite_double(
        aggregate.total_episode_reward_rate.value() / count,
        "aggregate mean episode reward rate");
    const double mean_abs_effort = aggregate.action_value_count > 0
        ? finite_double(
              aggregate.absolute_action_effort.value() /
                  static_cast<long double>(aggregate.action_value_count),
              "aggregate mean absolute action effort")
        : 0.0;

    return Json{
        {"episode_count", aggregate.returns.size()},
        {"total_return", finite_double(total_return, "aggregate total return")},
        {"mean_return", finite_double(mean_return, "aggregate mean return")},
        {"min_return", finite_double(*minimum, "aggregate minimum return")},
        {"max_return", finite_double(*maximum, "aggregate maximum return")},
        {"population_stddev",
         finite_double(population_stddev, "aggregate population stddev")},
        {"total_control_steps", aggregate.total_control_steps},
        {"total_physics_steps", aggregate.total_physics_steps},
        {"total_simulated_time_s",
         finite_double(total_time, "aggregate simulated time")},
        {"exposure_weighted_reward_rate", exposure_weighted_rate},
        {"mean_episode_reward_rate", mean_episode_rate},
        {"terminated_episodes", aggregate.terminated_episodes},
        {"truncated_episodes", aggregate.truncated_episodes},
        {"horizon_reached_episodes", aggregate.horizon_reached_episodes},
        {"reward",
         Json{
             {"by_sensor", sum_json(aggregate.reward_by_sensor, "sensor reward")},
             {"by_family", sum_json(aggregate.reward_by_family, "family reward")},
             {"reconciliation_error",
              finite_double(
                  total_return - aggregate.component_total.value(),
                  "aggregate reward reconciliation")},
         }},
        {"safety",
         Json{
             {"veto_count", aggregate.safety_veto_count},
             {"episodes_with_veto", aggregate.episodes_with_safety_veto},
             {"flag_counts", counter_json(aggregate.safety_flag_counts)},
             {"actuator_flag_counts",
              nested_counter_json(
                  aggregate.actuator_safety_flag_counts)},
         }},
        {"health",
         Json{
             {"observation_snapshots", aggregate.observation_snapshots},
             {"reward_sensor_samples", aggregate.reward_sensor_samples},
             {"actuator_feedback_samples",
              aggregate.actuator_feedback_samples},
             {"invalid_reward_sensor_samples",
              aggregate.invalid_reward_sensor_samples},
             {"invalid_actuator_feedback_samples",
              aggregate.invalid_actuator_feedback_samples},
             {"episodes_with_invalid_samples",
              aggregate.episodes_with_invalid_samples},
         }},
        {"actuators", actuator_metrics_json(aggregate.actuators)},
        {"actions",
         Json{
             {"value_count", aggregate.action_value_count},
             {"mean_abs_effort", mean_abs_effort},
             {"max_abs_effort", aggregate.maximum_absolute_action_effort},
         }},
    };
}

[[nodiscard]] std::vector<std::string> motor_ids_from_spec(const Json& spec) {
    if (!spec.is_object() || !spec.contains("action") ||
        !spec.at("action").is_object()) {
        throw std::runtime_error("environment spec.action must be an object");
    }
    const Json& ids = array_member(
        spec.at("action"), "motor_ids", "environment spec.action");
    std::vector<std::string> result;
    std::set<std::string> unique;
    for (const Json& id : ids) {
        if (!id.is_string() || id.get_ref<const std::string&>().empty()) {
            throw std::runtime_error(
                "environment action IDs must be non-empty strings");
        }
        const std::string value = id.get<std::string>();
        if (!unique.insert(value).second) {
            throw std::runtime_error("environment action IDs must be unique");
        }
        result.push_back(value);
    }
    if (result.empty()) {
        throw std::runtime_error("environment action ID list must not be empty");
    }
    return result;
}

void validate_options(const PolicyEvaluationOptions& options) {
    if (options.assembly_id.empty()) {
        throw std::invalid_argument(
            "policy evaluation assembly_id must not be empty");
    }
    if (options.episode_profile.empty()) {
        throw std::invalid_argument(
            "policy evaluation episode_profile must not be empty");
    }
    if (options.episodes == 0) {
        throw std::invalid_argument(
            "policy evaluation episodes must be greater than zero");
    }
    if (options.max_control_steps == 0) {
        throw std::invalid_argument(
            "policy evaluation max_control_steps must be greater than zero");
    }
    if (!std::isfinite(options.control_dt_s) ||
        options.control_dt_s <= 0.0) {
        throw std::invalid_argument(
            "policy evaluation control_dt_s must be finite and greater than zero");
    }
    if (!options.explicit_environment_seeds.empty() &&
        options.explicit_environment_seeds.size() != options.episodes) {
        throw std::invalid_argument(
            "explicit_environment_seeds must contain exactly one seed per episode");
    }
}

void validate_metadata_value(const Json& value, std::string_view context) {
    if (value.is_number_float() && !std::isfinite(value.get<double>())) {
        throw std::invalid_argument(
            std::string(context) + " contains a non-finite number");
    }
    if (value.is_binary() || value.is_discarded()) {
        throw std::invalid_argument(
            std::string(context) + " contains a non-JSON value");
    }
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            validate_metadata_value(
                child, std::string(context) + "." + key);
        }
    } else if (value.is_array()) {
        for (std::size_t index = 0; index < value.size(); ++index) {
            validate_metadata_value(
                value.at(index),
                std::string(context) + "[" + std::to_string(index) + "]");
        }
    }
}

void validate_policy(const PolicyEvaluationDescriptor& policy) {
    if (policy.schema_version.empty()) {
        throw std::invalid_argument(
            "policy descriptor schema_version must not be empty");
    }
    if (policy.id.empty()) {
        throw std::invalid_argument("policy descriptor id must not be empty");
    }
    if (!policy.metadata.is_object()) {
        throw std::invalid_argument("policy descriptor metadata must be an object");
    }
    validate_metadata_value(policy.metadata, "policy descriptor metadata");
    constexpr std::array<std::string_view, 5> reserved{
        "schema_version",
        "id",
        "deterministic",
        "uses_observation",
        "uses_policy_seed",
    };
    for (const std::string_view key : reserved) {
        if (policy.metadata.contains(std::string(key))) {
            throw std::invalid_argument(
                "policy descriptor metadata must not redefine " +
                std::string(key));
        }
    }
}

[[nodiscard]] Json policy_json(const PolicyEvaluationDescriptor& policy) {
    Json result = policy.metadata;
    result["schema_version"] = policy.schema_version;
    result["id"] = policy.id;
    result["deterministic"] = policy.deterministic;
    result["uses_observation"] = policy.uses_observation;
    result["uses_policy_seed"] = policy.uses_policy_seed;
    return result;
}

void require_profile_provenance(
    const Json& result,
    std::string_view expected_profile,
    std::string_view context) {
    if (!result.is_object() || !result.contains("info") ||
        !result.at("info").is_object() ||
        !result.at("info").contains("episode_profile") ||
        !result.at("info").at("episode_profile").is_string()) {
        throw std::runtime_error(
            std::string(context) +
            ".info.episode_profile must be a string");
    }
    if (result.at("info").at("episode_profile").get<std::string>() !=
        expected_profile) {
        throw std::runtime_error(
            std::string(context) +
            ".info.episode_profile did not match the requested profile");
    }
}

}  // namespace

std::uint64_t policy_evaluation_environment_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept {
    return stream_seed(base_seed, episode_index, kEnvironmentStreamTag);
}

std::uint64_t policy_evaluation_policy_seed(
    std::uint64_t base_seed,
    std::size_t episode_index) noexcept {
    return stream_seed(base_seed, episode_index, kPolicyStreamTag);
}

Json evaluate_policy(
    const std::filesystem::path& model_path,
    const PolicyEvaluationDescriptor& policy,
    const PolicyActionCallback& action_callback,
    const PolicyEvaluationOptions& options) {
    validate_options(options);
    validate_policy(policy);
    if (!action_callback) {
        throw std::invalid_argument(
            "policy evaluation action callback must not be empty");
    }
    DroidEnvironment environment(model_path, false);
    const Json spec = environment.spec();
    const std::vector<std::string> motor_ids = motor_ids_from_spec(spec);
    const double physics_timestep_s = finite_number(
        spec, "physics_timestep_s", "environment spec");
    if (physics_timestep_s <= 0.0) {
        throw std::runtime_error(
            "environment physics_timestep_s must be positive");
    }
    if (!spec.contains("api_version") ||
        !spec.at("api_version").is_string()) {
        throw std::runtime_error(
            "environment spec.api_version must be a string");
    }

    Json episode_documents = Json::array();
    AggregateMetrics aggregate;
    for (std::size_t episode_index = 0;
         episode_index < options.episodes;
         ++episode_index) {
        EpisodeMetrics episode;
        episode.index = episode_index;
        episode.episode_profile = options.episode_profile;
        episode.environment_seed = options.explicit_environment_seeds.empty()
            ? policy_evaluation_environment_seed(
                  options.base_seed, episode_index)
            : options.explicit_environment_seeds.at(episode_index);
        episode.policy_seed = policy_evaluation_policy_seed(
            options.base_seed, episode_index);
        episode.policy_seed_used = policy.uses_policy_seed;
        for (const std::string& motor_id : motor_ids) {
            episode.actuators.try_emplace(motor_id);
            episode.actuator_safety_flag_counts.try_emplace(motor_id);
        }

        const Json reset = environment.reset(
            options.assembly_id,
            episode.environment_seed,
            false,
            options.episode_profile);
        require_profile_provenance(
            reset, options.episode_profile, "environment reset");
        if (!reset.is_object() || !reset.contains("observation")) {
            throw std::runtime_error(
                "environment reset omitted the policy observation");
        }
        inspect_policy_observation(reset.at("observation"), episode);
        Json current_observation = reset.at("observation");

        for (std::size_t control_step = 0;
             control_step < options.max_control_steps;
             ++control_step) {
            const std::map<std::string, double> actions =
                action_callback(
                    current_observation,
                    motor_ids,
                    control_step,
                    episode.policy_seed);
            inspect_actions(actions, episode);
            const Json transition = environment.step(
                actions, options.control_dt_s);
            require_profile_provenance(
                transition, options.episode_profile, "environment transition");
            checked_increment(episode.control_steps);
            inspect_reward(transition, episode);
            inspect_safety(transition, episode);
            inspect_timing_and_outcome(transition, episode);
            if (!transition.contains("observation")) {
                throw std::runtime_error(
                    "environment transition omitted the policy observation");
            }
            inspect_policy_observation(
                transition.at("observation"), episode);
            current_observation = transition.at("observation");
            if (episode.environment_terminated ||
                episode.environment_truncated) {
                break;
            }
        }
        episode.horizon_reached =
            episode.control_steps == options.max_control_steps &&
            !episode.environment_terminated &&
            !episode.environment_truncated;
        episode_documents.push_back(episode_json(episode));
        merge_episode(episode, aggregate);
    }

    return Json{
        {"schema_version", kPolicyEvaluationSchemaVersion},
        {"environment",
         Json{
              {"api_version", spec.at("api_version")},
              {"assembly_id", options.assembly_id},
              {"episode_profile", options.episode_profile},
              {"motor_ids", motor_ids},
             {"policy_safe_boundary", true},
         }},
        {"configuration",
         Json{
              {"episodes", options.episodes},
              {"max_control_steps", options.max_control_steps},
              {"base_seed", options.base_seed},
              {"episode_profile", options.episode_profile},
              {"environment_seed_source",
               options.explicit_environment_seeds.empty()
                   ? "derived_from_base_seed"
                   : "explicit"},
              {"explicit_environment_seeds",
               options.explicit_environment_seeds},
              {"control_dt_s", options.control_dt_s},
              {"physics_timestep_s", physics_timestep_s},
              {"recording", false},
          }},
        {"policy", policy_json(policy)},
        {"episodes", std::move(episode_documents)},
        {"aggregate", aggregate_json(aggregate)},
    };
}

}  // namespace droid
