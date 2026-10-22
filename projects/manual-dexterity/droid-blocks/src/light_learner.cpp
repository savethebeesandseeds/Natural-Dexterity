#include "droid/light_learner.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr std::string_view kId{"history_expected_sarsa_v2"};
constexpr double kDt{0.02};
constexpr std::size_t kQuietSteps{5};
constexpr std::size_t kOptionSteps{20};
constexpr std::size_t kHistoryCapacity{64};
constexpr std::size_t kReplayCapacity{32};
constexpr std::size_t kFeatureCount{56};
constexpr std::size_t kActionCount{3};
constexpr std::size_t kCheckpointIdentityLimit{128};
constexpr std::string_view kCheckpointArchitecture{"history_expected_sarsa_v2;value-checkpoint-v1;3x56-binary64"};
constexpr std::size_t kReplayUpdates{4};
constexpr double kOptionDiscount{0.92};
constexpr double kLambda{0.7};
constexpr double kOnlineAlpha{0.12};
constexpr double kReplayAlpha{0.08};
constexpr double kTdLimit{2.0};
constexpr double kWeightLimit{20.0};
constexpr double kEffortLimit{0.35};
constexpr double kSlew{0.1};
constexpr double kShaftTarget{3.0};
constexpr double kSpeedGain{0.05};
constexpr double kTolerance{0.000002};
constexpr std::array<double, kActionCount> kRequests{-0.15, 0.0, 0.15};
constexpr std::array<std::string_view, kActionCount> kActionIds{"negative", "coast", "positive"};
using Features = std::array<double, kFeatureCount>;
using Parameters = std::array<Features, kActionCount>;

[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument("light learner: " + std::string(message));
}
void keys(const Json& value, std::initializer_list<std::string_view> expected) {
    if (!value.is_object() || value.size() != expected.size()) invalid("unexpected fields at physical/learning boundary");
    for (const auto key : expected) if (!value.contains(key)) invalid("missing physical/learning field");
}
std::string text_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_string() || field.get_ref<const std::string&>().empty()) invalid("expected nonempty string");
    return field.get<std::string>();
}
double number(const Json& value, std::string_view key) {
    if (!value.at(key).is_number()) invalid("expected physical numeric field");
    const double result = value.at(key).get<double>();
    if (!std::isfinite(result)) invalid("nonfinite physical/learning field");
    return result;
}
bool boolean(const Json& value, std::string_view key) {
    if (!value.at(key).is_boolean()) invalid("expected boolean field");
    return value.at(key).get<bool>();
}
std::array<double, 3> vector(const Json& object, std::string_view key) {
    const Json& value = object.at(key);
    if (!value.is_array() || value.size() != 3) invalid("physical vector must have three components");
    std::array<double, 3> result{};
    for (std::size_t index = 0; index < 3; ++index) {
        if (!value[index].is_number()) invalid("nonnumeric vector component");
        result[index] = value[index].get<double>();
        if (!std::isfinite(result[index])) invalid("nonfinite vector component");
    }
    if (!std::isfinite(std::hypot(result[0], result[1], result[2]))) invalid("nonfinite vector magnitude");
    return result;
}

struct Sample {
    std::string id;
    bool valid{false};
    std::uint64_t sequence{0};
    double sampled{0.0}, delivered{0.0}, age{0.0};
    Json channels;
};
struct Physical {
    Sample imu, light;
    std::string motor_id;
    double time{0.0}, position{0.0}, velocity{0.0}, current{0.0}, lux{0.0};
    std::array<double, 3> force{}, gyro{};
};

Sample sample(const Json& object, double period, double latency, double stale) {
    keys(object, {"module_id", "family_id", "valid", "sequence", "sample_time_s", "delivered_time_s",
                   "age_s", "sample_period_s", "latency_s", "observations"});
    Sample result;
    result.id = text_field(object, "module_id");
    result.valid = boolean(object, "valid");
    const Json& sequence = object.at("sequence");
    if (!sequence.is_number_unsigned() && !(sequence.is_number_integer() && sequence.get<std::int64_t>() >= 0)) invalid("invalid sensor sequence");
    result.sequence = sequence.get<std::uint64_t>();
    if (std::abs(number(object, "sample_period_s") - period) > kTolerance ||
        std::abs(number(object, "latency_s") - latency) > kTolerance) invalid("sensor clock differs from declared version");
    result.channels = object.at("observations");
    if (!result.valid) {
        if (result.sequence != 0 || !object.at("sample_time_s").is_null() ||
            !object.at("delivered_time_s").is_null() || !object.at("age_s").is_null()) invalid("malformed initial undelivered sample");
    } else {
        result.sampled = number(object, "sample_time_s");
        result.delivered = number(object, "delivered_time_s");
        result.age = number(object, "age_s");
        if (result.sequence == 0 || result.sampled < 0.0 || result.age < 0.0 || result.age > stale + kTolerance ||
            std::abs(result.delivered - result.sampled - latency) > kTolerance ||
            result.age + kTolerance < latency) invalid("stale, future or inconsistently timed sample");
    }
    return result;
}

Physical physical(const Json& observation, std::optional<double> expected_time = std::nullopt) {
    keys(observation, {"schema_version", "reward_sensors", "actuator_feedback"});
    if (text_field(observation, "schema_version") != "construction_observation_v2") invalid("unsupported physical observation schema");
    const Json& sensors = observation.at("reward_sensors");
    const Json& motors = observation.at("actuator_feedback");
    if (!sensors.is_array() || sensors.size() != 2 || !motors.is_array() || motors.size() != 1) invalid("exactly one IMU, light sensor and motor are required");
    Physical result;
    bool have_imu = false, have_light = false;
    for (const Json& sensor : sensors) {
        const std::string family = text_field(sensor, "family_id");
        if (family == "imu_6axis_v0" && !have_imu) {
            have_imu = true;
            result.imu = sample(sensor, 0.01, 0.006, 0.05);
            keys(result.imu.channels, {"specific_force_m_s2", "angular_velocity_rad_s"});
            result.force = vector(result.imu.channels, "specific_force_m_s2");
            result.gyro = vector(result.imu.channels, "angular_velocity_rad_s");
            if (!result.imu.valid && (result.force != std::array<double, 3>{} || result.gyro != std::array<double, 3>{})) invalid("initial IMU contains undeclared readings");
        } else if (family == "ambient_light_v0" && !have_light) {
            have_light = true;
            result.light = sample(sensor, 0.1, 0.02, 0.5);
            keys(result.light.channels, {"illuminance_lux", "saturated"});
            result.lux = number(result.light.channels, "illuminance_lux");
            const bool saturated = boolean(result.light.channels, "saturated");
            if (result.lux < 0.0 || (!result.light.valid && (result.lux != 0.0 || saturated))) invalid("invalid illuminance reading");
        } else invalid("unknown or duplicate sensor family");
    }
    if (!have_imu || !have_light || result.imu.id == result.light.id) invalid("missing sensor or duplicate module identity");
    result.time = expected_time.value_or(result.imu.valid ? result.imu.sampled + result.imu.age :
        result.light.valid ? result.light.sampled + result.light.age : 0.0);
    for (const auto& [entry, latency] : {std::pair{&result.imu, 0.006}, std::pair{&result.light, 0.02}}) {
        if (entry->valid) {
            if (std::abs(entry->sampled + entry->age - result.time) > kTolerance ||
                entry->delivered > result.time + kTolerance) invalid("observation clock does not match control cadence");
        } else if (result.time + kTolerance >= latency) invalid("sensor invalid after initial delivery window");
    }
    const Json& motor = motors.at(0);
    keys(motor, {"module_id", "sku_id", "valid", "feedback"});
    result.motor_id = text_field(motor, "module_id");
    if (result.motor_id == result.imu.id || result.motor_id == result.light.id) invalid("duplicate module identity");
    if (text_field(motor, "sku_id") != "rotary_dc_gearmotor_v0" || !boolean(motor, "valid")) invalid("unsupported or invalid motor feedback");
    const Json& feedback = motor.at("feedback");
    keys(feedback, {"position_rad", "velocity_rad_s", "current_a", "bus_voltage_v", "temperature_c",
                     "output_torque_nm", "load_impedance_nm_s_per_rad", "stuck_score", "stuck", "fault_flags"});
    for (const auto key : {"position_rad", "velocity_rad_s", "current_a", "bus_voltage_v", "temperature_c",
                           "output_torque_nm", "load_impedance_nm_s_per_rad", "stuck_score"}) (void)number(feedback, key);
    (void)boolean(feedback, "stuck");
    result.position = number(feedback, "position_rad");
    result.velocity = number(feedback, "velocity_rad_s");
    result.current = number(feedback, "current_a");
    if (!feedback.at("fault_flags").is_array() || !feedback.at("fault_flags").empty()) invalid("motor " + result.motor_id + " fault_flags=" + feedback.at("fault_flags").dump());
    return result;
}

double envelope(double requested, double previous, const Physical& observation) {
    if (!std::isfinite(requested) || !std::isfinite(previous) || std::abs(previous) > kEffortLimit + 1e-12) invalid("invalid effort history");
    double target = std::clamp(requested, -kEffortLimit, kEffortLimit);
    if (!observation.imu.valid || !observation.light.valid) return 0.0;
    if (target != 0.0) {
        const double sign = target > 0.0 ? 1.0 : -1.0;
        target = sign * std::min(std::abs(target), kSpeedGain * std::max(0.0, kShaftTarget - sign * observation.velocity));
    }
    return std::clamp(target, previous - kSlew, previous + kSlew);
}

double validate_reward(const Json& transition, const Physical& observation) {
    keys(transition, {"observation", "reward", "reward_components", "safety", "terminated", "truncated", "info"});
    keys(transition.at("safety"), {"veto", "flags"});
    if (!transition.at("safety").at("flags").is_array()) invalid("malformed safety flags");
    if (boolean(transition.at("safety"), "veto") || boolean(transition, "terminated") ||
        boolean(transition, "truncated") || !transition.at("safety").at("flags").empty()) invalid("native world stopped or flagged transition");
    keys(transition.at("info"), {"elapsed_control_dt_s", "physics_steps_executed"});
    if (std::abs(number(transition.at("info"), "elapsed_control_dt_s") - kDt) > kTolerance ||
        transition.at("info").at("physics_steps_executed") != 10) invalid("transition duration does not match control cadence");
    const Json& components = transition.at("reward_components");
    if (!components.is_array() || components.size() != 2) invalid("reward needs exactly the two attached sensor components");
    std::set<std::string> seen;
    double sum = 0.0;
    for (const Json& component : components) {
        keys(component, {"module_id", "family_id", "transition_reward", "reward_rate", "valid"});
        const std::string id = text_field(component, "module_id");
        const std::string family = text_field(component, "family_id");
        if (!seen.insert(id).second || !((id == observation.imu.id && family == "imu_6axis_v0") ||
            (id == observation.light.id && family == "ambient_light_v0"))) invalid("reward component is not an attached sensor");
        const bool valid = boolean(component, "valid");
        if (valid != (id == observation.imu.id ? observation.imu.valid : observation.light.valid)) invalid("reward component validity differs from its sensor");
        const double reward = number(component, "transition_reward");
        const double rate = number(component, "reward_rate");
        if (std::abs(reward) > kDt + 1e-7 || std::abs(rate) > 1.0 + 1e-7) invalid("sensor reward exceeds declared bounds");
        sum += reward;
    }
    const double reward = number(transition, "reward");
    if (std::abs(reward - sum) > 1e-7) invalid("transition reward is not the sum of attached sensor rewards");
    return reward;
}

struct Random {
    std::uint64_t state{1};
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double unit() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    std::size_t index(std::size_t count) { return static_cast<std::size_t>(next() % count); }
};
double dot(const Features& a, const Features& b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}
double clip(double value) { return std::clamp(value, -1.0, 1.0); }
double light_level(const Physical& physical) { return std::clamp(std::log1p(physical.lux) / std::log1p(1000.0), 0.0, 1.0); }

// Fixed-order, length-delimited encoding. Doubles retain all IEEE-754 bits,
// including signed zero; integers are unsigned 64-bit big endian. Physical
// channel JSON is canonical ordered-key, round-trip numeric JSON. This emits
// fingerprints only, not a restorable or policy-visible state representation.
struct Fingerprint {
    std::string bytes;
    void integer(std::uint64_t value) {
        for (int shift = 56; shift >= 0; shift -= 8)
            bytes.push_back(static_cast<char>((value >> shift) & 0xff));
    }
    void real(double value) { integer(std::bit_cast<std::uint64_t>(value)); }
    void flag(bool value) { bytes.push_back(value ? '\1' : '\0'); }
    void text(std::string_view value) { integer(value.size()); bytes.append(value); }
    void feature(const Features& value) { for (const double item : value) real(item); }
    void parameters(const Parameters& value) { for (const auto& row : value) feature(row); }
    void sample(const Sample& value) {
        text(value.id); flag(value.valid); integer(value.sequence);
        real(value.sampled); real(value.delivered); real(value.age);
        text(value.channels.dump());
    }
    void physical(const Physical& value) {
        sample(value.imu); sample(value.light); text(value.motor_id);
        for (const double number : {value.time, value.position, value.velocity, value.current, value.lux}) real(number);
        for (const double number : value.force) real(number);
        for (const double number : value.gyro) real(number);
    }
    std::string hash() const { return sha256_hex(bytes); }
};
} // namespace

struct LightLearner::ValueCheckpoint::Data {
    Parameters weights{};
    std::string architecture;
    std::string motor_id, imu_id, light_id;
};

LightLearner::ValueCheckpoint::ValueCheckpoint(std::shared_ptr<const Data> data)
    : data_(std::move(data)) {}

struct LightLearner::Impl {
    struct Experience { Physical observation; double effort; double reward; };
    struct Option { Features start; std::size_t action; double reward; double discount; Features next; };
    Parameters weights{}, traces{};
    std::deque<Experience> history;
    std::deque<Option> replay;
    Random action_random, replay_random;
    std::array<std::size_t, kActionCount> visits{};
    std::array<std::size_t, kActionCount> initial_order{0, 1, 2};
    std::size_t steps{0}, action{1}, remaining{0}, decisions{0}, updates{0}, replay_updates{0}, exploratory{0};
    std::size_t unique_light{0}, unique_imu{0};
    bool pending{false}, bound{false};
    std::string motor_id, imu_id, light_id;
    Sample seen_light, seen_imu;
    Physical latest;
    double effort{0.0}, requested{0.0}, target{0.0};
    double option_return{0.0}, option_discount{1.0};
    Features option_start{};
    double last_reward{0.0}, cumulative_reward{0.0}, last_td{0.0}, parameter_change{0.0};
    double initial_reward_sum{0.0}, initial_value{0.0};
    std::size_t initial_reward_samples{0};
    bool values_initialized{false};
    bool learning_frozen{false};
    std::size_t suppressed_updates{0}, suppressed_replay_updates{0};

    explicit Impl(std::uint64_t seed = 1) : action_random{seed}, replay_random{seed ^ 0xd1b54a32d192ed03ULL} {
        for (std::size_t i = initial_order.size(); i > 1; --i) std::swap(initial_order[i - 1], initial_order[action_random.index(i)]);
    }
    double epsilon() const { return std::max(0.12, 0.4 * std::exp(-static_cast<double>(decisions) / 150.0)); }

    void check_sample(const Sample& now, Sample& previous, std::size_t& count) {
        if (!now.valid) return;
        if (previous.valid) {
            if (now.sequence < previous.sequence || now.sampled + kTolerance < previous.sampled) invalid("sensor sample went backwards");
            if (now.sequence == previous.sequence) {
                if (now.sampled != previous.sampled || now.delivered != previous.delivered || now.channels != previous.channels) invalid("held sensor reading changed without new sequence");
                return;
            }
            if (now.sampled <= previous.sampled + kTolerance) invalid("new sequence did not advance acquisition time");
        }
        previous = now;
        ++count;
    }
    void accept(const Physical& observation) {
        if (!bound) {
            bound = true; motor_id = observation.motor_id; imu_id = observation.imu.id; light_id = observation.light.id;
        } else if (motor_id != observation.motor_id || imu_id != observation.imu.id || light_id != observation.light.id) invalid("physical module identity changed without reset");
        check_sample(observation.imu, seen_imu, unique_imu);
        check_sample(observation.light, seen_light, unique_light);
        latest = observation;
    }

    Features features(const Physical& observation) const {
        Features f{};
        f[0] = 1.0;
        f[1] = std::sin(observation.position); f[2] = std::cos(observation.position);
        f[3] = clip(observation.velocity / 6.0); f[4] = std::abs(f[3]);
        f[5] = clip(observation.current / 2.5);
        for (std::size_t i = 0; i < 3; ++i) { f[6 + i] = clip(observation.force[i] / 39.24); f[9 + i] = clip(observation.gyro[i] / 10.0); }
        f[12] = light_level(observation);
        f[13] = observation.imu.valid && observation.light.valid ? 1.0 : 0.0;
        const Experience* older = nullptr;
        const Experience* older_light = nullptr;
        double mean_effort = 0, mean_velocity = 0, mean_reward = 0;
        std::size_t count = 0;
        for (auto entry = history.rbegin(); entry != history.rend(); ++entry) {
            if (!older_light && entry->observation.light.valid &&
                entry->observation.light.sampled + kTolerance < observation.light.sampled) older_light = &*entry;
            if (!older && entry->observation.time <= observation.time - 0.2 + kTolerance) older = &*entry;
            if (entry->observation.time >= observation.time - 0.4 - kTolerance) {
                mean_effort += entry->effort; mean_velocity += entry->observation.velocity;
                mean_reward += entry->reward / kDt; ++count;
            }
        }
        if (older_light) {
            const double dt = observation.light.sampled - older_light->observation.light.sampled;
            f[14] = clip(4.0 * (f[12] - light_level(older_light->observation)) / dt);
        }
        if (older) {
            const double dt = observation.time - older->observation.time;
            for (std::size_t i = 0; i < 3; ++i) {
                f[15 + i] = clip((observation.gyro[i] - older->observation.gyro[i]) / (40.0 * dt));
                f[18 + i] = clip((observation.force[i] - older->observation.force[i]) / (100.0 * dt));
            }
            f[23] = clip((observation.velocity - older->observation.velocity) / (40.0 * dt));
        }
        f[21] = clip(effort / 0.15);
        f[22] = clip(std::hypot(observation.gyro[0], observation.gyro[1], observation.gyro[2]) / 10.0);
        f[24] = std::clamp(observation.light.age / 0.5, 0.0, 1.0);
        if (count) { f[25] = clip(mean_reward / (2.0 * count)); f[26] = clip(mean_effort / (0.15 * count)); f[27] = clip(mean_velocity / (6.0 * count)); }
        f[28] = f[1] * f[3]; f[29] = f[2] * f[3]; f[30] = f[12] * f[3]; f[31] = f[14] * f[3];
        for (std::size_t angle = 0; angle < 8; ++angle) {
            const double center = 2.0 * std::numbers::pi * static_cast<double>(angle) / 8.0;
            const double angular = std::exp(3.0 * (std::cos(observation.position - center) - 1.0));
            for (std::size_t speed = 0; speed < 3; ++speed) {
                const double offset = clip(observation.velocity / 6.0) - (static_cast<double>(speed) - 1.0);
                f[32 + angle * 3 + speed] = angular * std::exp(-2.0 * offset * offset);
            }
        }
        return f;
    }
    std::array<double, kActionCount> values(const Features& f) const {
        return {dot(weights[0], f), dot(weights[1], f), dot(weights[2], f)};
    }
    double expected(const Features& f) const {
        const auto q = values(f);
        if (decisions < kActionCount) return q[initial_order[decisions]];
        const double best = *std::max_element(q.begin(), q.end());
        return (1.0 - epsilon()) * best + epsilon() * (q[0] + q[1] + q[2]) / 3.0;
    }
    void choose(const Features& f) {
        bool exploring = false;
        if (decisions < kActionCount) { action = initial_order[decisions]; exploring = true; }
        else if (action_random.unit() < epsilon()) { action = action_random.index(kActionCount); exploring = true; }
        else {
            const auto q = values(f);
            const double best = *std::max_element(q.begin(), q.end());
            std::array<std::size_t, kActionCount> ties{};
            std::size_t count = 0;
            for (std::size_t i = 0; i < kActionCount; ++i) if (std::abs(q[i] - best) <= 1e-12) ties[count++] = i;
            action = ties[action_random.index(count)];
        }
        ++visits[action]; ++decisions; exploratory += exploring ? 1 : 0;
        remaining = kOptionSteps; option_return = 0.0; option_discount = 1.0; option_start = f;
    }
    void change(double& parameter, double delta) {
        if (learning_frozen) return;
        const double before = parameter;
        parameter = std::clamp(parameter + delta, -kWeightLimit, kWeightLimit);
        if (!std::isfinite(parameter)) invalid("nonfinite learned value parameter");
        parameter_change += std::abs(parameter - before);
    }
    void learn(const Features& next) {
        const Option transition{option_start, action, option_return, option_discount, next};
        last_td = transition.reward + transition.discount * expected(next) - dot(weights[action], option_start);
        const double delta = std::clamp(last_td, -kTdLimit, kTdLimit);
        const double alpha = kOnlineAlpha / (1.0 + dot(option_start, option_start));
        for (std::size_t a = 0; a < kActionCount; ++a) {
            for (std::size_t feature = 0; feature < kFeatureCount; ++feature) {
                traces[a][feature] *= transition.discount * kLambda;
                if (a == action) traces[a][feature] += option_start[feature];
                traces[a][feature] = std::clamp(traces[a][feature], -5.0, 5.0);
                change(weights[a][feature], alpha * delta * traces[a][feature]);
            }
        }
        ++updates;
        if (learning_frozen) ++suppressed_updates;
        replay.push_back(transition);
        if (replay.size() > kReplayCapacity) replay.pop_front();
        for (std::size_t update = 0; update < kReplayUpdates; ++update) {
            const Option& entry = replay[replay_random.index(replay.size())];
            const double error = std::clamp(entry.reward + entry.discount * expected(entry.next) -
                dot(weights[entry.action], entry.start), -kTdLimit, kTdLimit);
            const double rate = kReplayAlpha / (1.0 + dot(entry.start, entry.start));
            for (std::size_t feature = 0; feature < kFeatureCount; ++feature) change(weights[entry.action][feature], rate * error * entry.start[feature]);
            ++replay_updates;
            if (learning_frozen) ++suppressed_replay_updates;
        }
    }
    void freeze_learning() {
        if (pending) invalid("finish the pending act/observe pair before freezing learning");
        if (!values_initialized) invalid("finish quiet value initialization before freezing learning");
        learning_frozen = true;
    }
    double act(const Json& observation) {
        if (pending) invalid("observe the pending transition before another act");
        const Physical current = physical(observation, static_cast<double>(steps) * kDt);
        accept(current);
        if (steps < kQuietSteps) { action = 1; requested = 0.0; }
        else {
            if (remaining == 0) choose(features(current));
            requested = kRequests[action];
        }
        target = requested == 0.0 ? 0.0 : std::copysign(std::min(std::abs(requested),
            kSpeedGain * std::max(0.0, kShaftTarget - std::copysign(1.0, requested) * current.velocity)), requested);
        if (!current.imu.valid || !current.light.valid) target = 0.0;
        effort = envelope(requested, effort, current);
        pending = true;
        return effort;
    }
    void observe(const Json& transition) {
        if (!pending) invalid("observe requires one act followed by one physics transition");
        keys(transition, {"observation", "reward", "reward_components", "safety", "terminated", "truncated", "info"});
        const Physical current = physical(transition.at("observation"), static_cast<double>(steps + 1) * kDt);
        const double reward = validate_reward(transition, current);
        // A poststep valid bit alone does not establish a fully delivered
        // interval. The first interval includes both initial sensor deliveries.
        const bool quiet_interval_delivered = steps < kQuietSteps && latest.imu.valid &&
            latest.light.valid && current.imu.valid && current.light.valid;
        accept(current);
        last_reward = reward; cumulative_reward += reward;
        history.push_back({current, effort, reward});
        if (history.size() > kHistoryCapacity) history.pop_front();
        if (quiet_interval_delivered) {
            initial_reward_sum += reward;
            ++initial_reward_samples;
        }
        if (steps + 1 == kQuietSteps) {
            if (initial_reward_samples != kQuietSteps - 1) invalid("quiet initialization lacks fully delivered intervals");
            const double mean_reward = initial_reward_sum / static_cast<double>(initial_reward_samples);
            const double discount_step = std::pow(kOptionDiscount, 1.0 / static_cast<double>(kOptionSteps));
            initial_value = mean_reward / (1.0 - discount_step);
            for (auto& action_weights : weights) action_weights[0] = initial_value;
            values_initialized = true;
        }
        if (steps >= kQuietSteps) {
            option_return += option_discount * reward;
            option_discount *= std::pow(kOptionDiscount, 1.0 / static_cast<double>(kOptionSteps));
            --remaining;
            if (remaining == 0) learn(features(current));
        }
        ++steps; pending = false;
    }
    std::string parameter_fingerprint() const {
        Fingerprint result;
        result.text("light-learner-parameters-binary64-v1");
        result.parameters(weights);
        return result.hash();
    }
    void append_nonparameter_state(Fingerprint& result) const {
        result.parameters(traces);
        result.integer(history.size());
        for (const auto& entry : history) {
            result.physical(entry.observation); result.real(entry.effort); result.real(entry.reward);
        }
        result.integer(replay.size());
        for (const auto& entry : replay) {
            result.feature(entry.start); result.integer(entry.action);
            result.real(entry.reward); result.real(entry.discount); result.feature(entry.next);
        }
        result.integer(action_random.state); result.integer(replay_random.state);
        for (const auto value : visits) result.integer(value);
        for (const auto value : initial_order) result.integer(value);
        for (const auto value : {steps, action, remaining, decisions, updates, replay_updates,
                                exploratory, unique_light, unique_imu}) result.integer(value);
        result.flag(pending); result.flag(bound);
        result.text(motor_id); result.text(imu_id); result.text(light_id);
        result.sample(seen_light); result.sample(seen_imu); result.physical(latest);
        for (const double value : {effort, requested, target, option_return, option_discount}) result.real(value);
        result.feature(option_start);
        for (const double value : {last_reward, cumulative_reward, last_td, parameter_change,
                                   initial_reward_sum, initial_value}) result.real(value);
        result.integer(initial_reward_samples); result.flag(values_initialized);
        result.flag(learning_frozen); result.integer(suppressed_updates); result.integer(suppressed_replay_updates);
    }
    std::string state_fingerprint() const {
        Fingerprint result;
        result.text("light-learner-complete-state-v1");
        result.text(kId);
        result.parameters(weights);
        append_nonparameter_state(result);
        return result.hash();
    }
    std::string nonparameter_state_fingerprint() const {
        Fingerprint result;
        result.text("light-learner-nonparameter-state-v1");
        result.text(kId);
        append_nonparameter_state(result);
        return result.hash();
    }
    void check_checkpoint_boundary() const {
        if (pending) invalid("finish the pending act/observe pair before accessing a value checkpoint");
        if (!values_initialized) invalid("finish quiet value initialization before accessing a value checkpoint");
    }
    Json diagnostics() const {
        const auto q = values(features(latest));
        Json options = Json::array();
        double l1 = 0.0;
        for (std::size_t a = 0; a < kActionCount; ++a) {
            options.push_back(Json{{"action_id", kActionIds[a]}, {"requested_effort", kRequests[a]}, {"value", q[a]}, {"visits", visits[a]}});
            for (const double parameter : weights[a]) l1 += std::abs(parameter);
        }
        Json recent = Json::array();
        const std::size_t first = history.size() > 8 ? history.size() - 8 : 0;
        for (std::size_t i = first; i < history.size(); ++i) {
            const auto& entry = history[i];
            recent.push_back(Json{{"time_s", entry.observation.time}, {"effort", entry.effort}, {"reward", entry.reward},
                {"light_sequence", entry.observation.light.sequence}, {"imu_sequence", entry.observation.imu.sequence},
                {"light_sample_time_s", entry.observation.light.sampled}, {"imu_sample_time_s", entry.observation.imu.sampled},
                {"illuminance_lux", entry.observation.lux}, {"shaft_velocity_rad_s", entry.observation.velocity},
                {"angular_velocity_rad_s", entry.observation.gyro}, {"specific_force_m_s2", entry.observation.force}});
        }
        return Json{{"learner_id", kId}, {"learning_enabled", !learning_frozen}, {"learning_frozen", learning_frozen},
            {"parameter_fingerprint_sha256", parameter_fingerprint()}, {"state_fingerprint_sha256", state_fingerprint()},
            {"nonparameter_state_fingerprint_sha256", nonparameter_state_fingerprint()},
            {"elapsed_s", static_cast<double>(steps) * kDt},
            {"step", steps}, {"phase", steps < kQuietSteps ? "warming" : "acting"}, {"action_id", kActionIds[action]},
            {"effort", effort}, {"requested_effort", requested}, {"target_effort", target},
            {"speed_limited", std::abs(target) < std::abs(requested)}, {"awaiting_observation", pending},
            {"option_remaining_s", static_cast<double>(remaining) * kDt}, {"decision_count", decisions},
            {"update_count", updates}, {"replay_update_count", replay_updates}, {"exploratory_decisions", exploratory},
            {"suppressed_update_count", suppressed_updates}, {"suppressed_replay_update_count", suppressed_replay_updates},
            {"exploration_rate", epsilon()}, {"history_size", history.size()}, {"replay_size", replay.size()},
            {"unique_light_samples", unique_light}, {"unique_imu_samples", unique_imu},
            {"last_reward", last_reward}, {"cumulative_reward", cumulative_reward}, {"last_td_error", last_td},
            {"values_initialized", values_initialized}, {"initial_value", initial_value},
            {"initial_reward_samples", initial_reward_samples},
            {"initial_reward_rate", initial_reward_samples ? initial_reward_sum / (kDt * static_cast<double>(initial_reward_samples)) : 0.0},
            {"parameter_l1", l1}, {"parameter_change_l1", parameter_change}, {"option_values", options}, {"recent_experience", recent}};
    }
};

LightLearner::LightLearner(std::uint64_t seed) : impl_(std::make_unique<Impl>(seed)) {}
LightLearner::~LightLearner() = default;
void LightLearner::reset(std::uint64_t seed) { impl_ = std::make_unique<Impl>(seed); }
double LightLearner::act(const Json& observation) {
    Impl next = *impl_; const double effort = next.act(observation); *impl_ = std::move(next); return effort;
}
void LightLearner::observe(const Json& transition) { Impl next = *impl_; next.observe(transition); *impl_ = std::move(next); }
void LightLearner::freeze_learning() { impl_->freeze_learning(); }
LightLearner::ValueCheckpoint LightLearner::capture_values() const {
    impl_->check_checkpoint_boundary();
    for (const auto& id : {impl_->motor_id, impl_->imu_id, impl_->light_id})
        if (id.empty() || id.size() > kCheckpointIdentityLimit) invalid("checkpoint module identity exceeds its bounded format");
    auto data = std::make_shared<ValueCheckpoint::Data>();
    data->weights = impl_->weights;
    data->architecture = kCheckpointArchitecture;
    data->motor_id = impl_->motor_id;
    data->imu_id = impl_->imu_id;
    data->light_id = impl_->light_id;
    return ValueCheckpoint(std::move(data));
}
void LightLearner::restore_values(const ValueCheckpoint& checkpoint) {
    impl_->check_checkpoint_boundary();
    if (!checkpoint.data_ || checkpoint.data_->architecture != kCheckpointArchitecture ||
        checkpoint.data_->motor_id != impl_->motor_id || checkpoint.data_->imu_id != impl_->imu_id ||
        checkpoint.data_->light_id != impl_->light_id) invalid("incompatible value checkpoint architecture or module identities");
    for (const auto& row : checkpoint.data_->weights)
        for (const double weight : row)
            if (!std::isfinite(weight) || std::abs(weight) > kWeightLimit) invalid("value checkpoint weight is outside finite bounds");
    // Assignment is the sole mutation. In particular, parameter_change counts
    // learning writes and is not repurposed as checkpoint-import accounting.
    impl_->weights = checkpoint.data_->weights;
}
Json LightLearner::diagnostics() const { return impl_->diagnostics(); }
double LightLearner::bounded_effort(double requested, double previous, const Json& observation) {
    return envelope(requested, previous, physical(observation));
}

Json LightLearner::specification() {
    return Json{{"learner_id", kId}, {"schema_version", "droid-blocks.online-light-learner.v2"},
        {"algorithm", "linear Expected SARSA with accumulating eligibility traces, fixed-duration options and bounded recent one-step replay"},
        {"control_dt_s", kDt}, {"quiet_start_s", kQuietSteps * kDt}, {"option_duration_s", kOptionSteps * kDt},
        {"requests", kRequests}, {"action_ids", kActionIds}, {"feature_count", kFeatureCount},
        {"feature_encoder", Json{{"0", "bias"}, {"1-2", "sin/cos of local motor encoder"},
            {"3-5", "signed/absolute shaft speed divided by6, current divided by2.5"},
            {"6-11", "three local specific-force axes divided by39.24 and gyro axes divided by10"},
            {"12-14", "log1p(lux)/log1p(1000), sensor-valid mask, fresh-light slope times4"},
            {"15-20", "gyro and specific-force changes over at least.2s divided by40/s and100/s"},
            {"21-24", "previous effort/.15, gyro magnitude/10, shaft acceleration/40, light age/.5"},
            {"25-27", ".4s means of total reward rate/2, effort/.15 and shaft velocity/6"},
            {"28-31", "encoder sin/cos, light level and light slope multiplied by shaft-speed feature"},
            {"32-55", "eight circular encoder radial bases times three coarse local shaft-speed bases"},
            {"normalization", "signed features clipped to[-1,1]; light level and age clipped to[0,1]; IDs never become features"}}},
        {"objective", "discounted time-integrated SUM of exactly the attached sensor reward components; no lux-delta surrogate, target angle, pose or hidden motion score"},
        {"discount_per_option", kOptionDiscount}, {"discount_within_option", "0.92^(elapsed/.4s)"},
        {"value_initialization", Json{
            {"method", "equal bias for all actions from measured quiet reward; all other parameters and traces start zero"},
            {"measurement_interval_start_s", kDt}, {"measurement_interval_end_s", kQuietSteps * kDt},
            {"fully_delivered_intervals", kQuietSteps - 1},
            {"formula", "mean actual SUM transition reward / (1 - 0.92^(1/20))"},
            {"excluded_interval", "[0,.02]s contains initial sensor deliveries"},
            {"purpose", "give all actions the same measured background value before evidence distinguishes them"},
            {"limitations", "a finite initial value estimate, not a reward target, policy guarantee or privileged body/light cue"}}},
        {"lambda", kLambda}, {"online_step_size", kOnlineAlpha}, {"replay_step_size", kReplayAlpha},
        {"step_size_normalization", "divide by1+squared feature norm"}, {"td_update_clip", kTdLimit},
        {"parameter_clip", kWeightLimit}, {"trace_clip", 5.0},
        {"exploration", "seeded shuffled first visit to each option, then epsilon-greedy max(.12,.4*exp(-decision_count/150)) with seeded tie handling"},
        {"random_generator", "splitmix64; separate deterministic action and replay streams"},
        {"history_capacity_transitions", kHistoryCapacity}, {"replay_capacity_options", kReplayCapacity},
        {"replay_updates_per_option", kReplayUpdates}, {"replay_sampling", "uniform over the most recent32 completed options"},
        {"sensor_period_s", Json{{"imu_6axis_v0", 0.01}, {"ambient_light_v0", 0.1}}},
        {"sensor_latency_s", Json{{"imu_6axis_v0", 0.006}, {"ambient_light_v0", 0.02}}},
        {"effort_limit", kEffortLimit}, {"slew_limit", kSlew}, {"shaft_target_rad_s", kShaftTarget},
        {"speed_gain", kSpeedGain}, {"governor", "same prospective local-shaft envelope as bounded rhythm probes; soft target is not a guaranteed physical speed limit"},
        {"memory", "reset clears values/traces/history/replay; ordinary sensory changes retain and update them without a privileged change cue"},
        {"freeze_intervention", Json{{"id", "one_way_value_freeze_v1"},
            {"precondition", "values initialized and no pending act/observe pair"},
            {"effect", "suppress online and replay parameter writes only; histories, options, traces, replay sampling, RNG draws and exploration schedule continue"},
            {"reset", "clears the freeze; no unfreeze operation during a run"},
            {"counters", "update_count and replay_update_count count scheduled processing; suppressed counters count processing with parameter writes disabled"},
            {"fingerprints", "SHA-256 of explicit complete state and exact binary64 parameters; diagnostic only; not state export or restore"}}},
        {"value_checkpoint", Json{{"id", "typed_value_checkpoint_v1"}, {"parameter_count", kActionCount * kFeatureCount},
            {"architecture", kCheckpointArchitecture}, {"module_identity_max_bytes", kCheckpointIdentityLimit},
            {"precondition", "initialized values and no pending act/observe pair"},
            {"payload", "immutable exact 168 binary64 weights plus architecture and bounded module identity compatibility; no world state, history, optimizer, RNG, time or source metadata"},
            {"restore_effect", "assign weights only, preserving every nonparameter state including the learning-freeze flag"},
            {"scope", "diagnostic intervention only; identical module IDs do not prove identical physical bodies; callers must enforce the same-body protocol"},
            {"fingerprints", "nonparameter_state_fingerprint_sha256 covers every stored controller member except weights; complete-state encoding remains unchanged"}}},
        {"failure_policy", "strict act/observe cadence; reject stale/invalid/malformed observations, all motor/native flags and inconsistent reward sums without partial learner mutation"},
        {"scope", "small exposed-development learner, not a convergence, optimality, topology-transfer or hardware claim"}};
}

} // namespace droid
