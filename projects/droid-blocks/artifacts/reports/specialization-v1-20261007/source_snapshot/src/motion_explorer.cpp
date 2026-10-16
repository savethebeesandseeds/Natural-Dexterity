#include "droid/motion_explorer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr std::size_t kWarmupSteps{5};
constexpr std::size_t kCoastSteps{15};
constexpr double kEffortCeiling{0.35};
constexpr double kSlew{0.1};
constexpr double kShaftTargetRadS{3.0};
constexpr double kShaftGain{0.05};
constexpr double kSamplePeriodS{0.01};
constexpr double kLatencyS{0.006};
constexpr double kStaleAfterS{0.05};
constexpr double kTimeToleranceS{0.000002};

struct Pattern {
    std::string_view id;
    std::string_view name;
    std::string_view description;
    std::string_view waveform;
    double amplitude;
    double frequency_hz;
};

constexpr std::array<Pattern, 5> kPatterns{{
    {"sway_slow", "Slow sway", "A gentle back-and-forth rhythm, once every four seconds.", "sine", 0.2, 0.25},
    {"sway_medium", "Steady sway", "A back-and-forth rhythm, once every two seconds.", "sine", 0.2, 0.5},
    {"sway_quick", "Quick sway", "A quicker back-and-forth rhythm, once each second.", "sine", 0.2, 1.0},
    {"paired_taps", "Paired taps", "A short push each way, with a rest between pushes.", "alternating_pulses", 0.2, 0.5},
    {"push_and_coast", "Push and drift", "One short push, a long rest, then one push back.", "two_separated_pulses", 0.2, 0.0},
}};

[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument("motion explorer: " + std::string(message));
}

void exact_keys(const Json& value, std::initializer_list<std::string_view> keys) {
    if (!value.is_object() || value.size() != keys.size()) invalid("unexpected observation fields");
    for (const auto key : keys) if (!value.contains(key)) invalid("missing observation field");
}

[[nodiscard]] std::string string_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_string() || field.get_ref<const std::string&>().empty()) invalid("expected non-empty string");
    return field.get<std::string>();
}

[[nodiscard]] double number_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_number()) invalid("expected numeric physical field");
    const double result = field.get<double>();
    if (!std::isfinite(result)) invalid("non-finite physical field");
    return result;
}

[[nodiscard]] bool bool_field(const Json& value, std::string_view key) {
    if (!value.at(key).is_boolean()) invalid("expected boolean physical field");
    return value.at(key).get<bool>();
}

[[nodiscard]] std::array<double, 3> vector_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_array() || field.size() != 3) invalid("IMU vectors must have exactly three components");
    std::array<double, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (!field[index].is_number()) invalid("IMU vector component must be numeric");
        result[index] = field[index].get<double>();
        if (!std::isfinite(result[index])) invalid("non-finite IMU vector component");
    }
    if (!std::isfinite(std::hypot(result[0], result[1], result[2]))) invalid("IMU vector magnitude is non-finite");
    return result;
}

struct PhysicalObservation {
    std::string motor_id;
    std::string sensor_id;
    double shaft_velocity{0.0};
    bool valid{false};
    std::uint64_t sequence{0};
    double sampled_s{0.0};
    double delivered_s{0.0};
    std::array<double, 3> force{};
    std::array<double, 3> gyro{};
};

[[nodiscard]] PhysicalObservation validate(const Json& observation, double now_s) {
    exact_keys(observation, {"schema_version", "reward_sensors", "actuator_feedback"});
    if (string_field(observation, "schema_version") != "construction_observation_v1") invalid("unsupported observation schema");
    const Json& sensors = observation.at("reward_sensors");
    const Json& motors = observation.at("actuator_feedback");
    if (!sensors.is_array() || sensors.size() != 1 || !motors.is_array() || motors.size() != 1) {
        invalid("exactly one physical IMU and one motor feedback entry are required");
    }
    PhysicalObservation result;
    const Json& sensor = sensors.at(0);
    exact_keys(sensor, {"module_id", "family_id", "valid", "sequence", "sample_time_s",
                        "delivered_time_s", "age_s", "sample_period_s", "latency_s", "observations"});
    result.sensor_id = string_field(sensor, "module_id");
    if (string_field(sensor, "family_id") != "imu_6axis_v0") invalid("unsupported sensor family");
    result.valid = bool_field(sensor, "valid");
    exact_keys(sensor.at("observations"), {"specific_force_m_s2", "angular_velocity_rad_s"});
    result.force = vector_field(sensor.at("observations"), "specific_force_m_s2");
    result.gyro = vector_field(sensor.at("observations"), "angular_velocity_rad_s");
    if (std::abs(number_field(sensor, "sample_period_s") - kSamplePeriodS) > kTimeToleranceS ||
        std::abs(number_field(sensor, "latency_s") - kLatencyS) > kTimeToleranceS) invalid("IMU timing differs from this explorer version");
    const Json& sequence = sensor.at("sequence");
    if (!sequence.is_number_unsigned() &&
        !(sequence.is_number_integer() && sequence.get<std::int64_t>() >= 0)) invalid("IMU sequence must be a nonnegative integer");
    result.sequence = sequence.get<std::uint64_t>();
    if (!result.valid) {
        if (now_s + kTimeToleranceS >= kLatencyS || result.sequence != 0 ||
            !sensor.at("sample_time_s").is_null() || !sensor.at("delivered_time_s").is_null() ||
            !sensor.at("age_s").is_null() || result.force != std::array<double, 3>{} ||
            result.gyro != std::array<double, 3>{}) invalid("IMU is invalid outside its initial delivery window");
    } else {
        result.sampled_s = number_field(sensor, "sample_time_s");
        result.delivered_s = number_field(sensor, "delivered_time_s");
        const double age = number_field(sensor, "age_s");
        if (result.sequence == 0 || result.sampled_s < 0.0 ||
            result.sampled_s > now_s + kTimeToleranceS || result.delivered_s > now_s + kTimeToleranceS ||
            std::abs(result.delivered_s - result.sampled_s - kLatencyS) > kTimeToleranceS ||
            age < 0.0 || age > kStaleAfterS + kTimeToleranceS ||
            std::abs(age - (now_s - result.sampled_s)) > kTimeToleranceS) invalid("IMU is stale or has inconsistent timing");
    }
    const Json& motor = motors.at(0);
    exact_keys(motor, {"module_id", "sku_id", "valid", "feedback"});
    result.motor_id = string_field(motor, "module_id");
    if (string_field(motor, "sku_id") != "rotary_dc_gearmotor_v0" || !bool_field(motor, "valid")) {
        invalid("unsupported or invalid motor feedback");
    }
    const Json& feedback = motor.at("feedback");
    exact_keys(feedback, {"position_rad", "velocity_rad_s", "current_a", "bus_voltage_v", "temperature_c",
                          "output_torque_nm", "load_impedance_nm_s_per_rad", "stuck_score", "stuck", "fault_flags"});
    for (const auto key : {"position_rad", "velocity_rad_s", "current_a", "bus_voltage_v", "temperature_c",
                           "output_torque_nm", "load_impedance_nm_s_per_rad", "stuck_score"}) (void)number_field(feedback, key);
    (void)bool_field(feedback, "stuck");
    result.shaft_velocity = number_field(feedback, "velocity_rad_s");
    if (!feedback.at("fault_flags").is_array() || !feedback.at("fault_flags").empty()) {
        invalid("motor " + result.motor_id + " feedback reports fault_flags=" + feedback.at("fault_flags").dump());
    }
    return result;
}

[[nodiscard]] Json pattern_json(const Pattern& pattern) {
    return Json{{"id", pattern.id}, {"name", pattern.name},
        {"pattern_id", pattern.id}, {"label", pattern.name}, {"description", pattern.description},
        {"duration_s", kMotionExplorerDurationS}, {"control_dt_s", kMotionExplorerControlDtS},
        {"max_steps", kMotionExplorerMaxSteps}, {"waveform", pattern.waveform},
        {"amplitude", pattern.amplitude}, {"frequency_hz", pattern.frequency_hz},
        {"initial_quiet_s", kWarmupSteps * kMotionExplorerControlDtS},
        {"final_coast_s", kCoastSteps * kMotionExplorerControlDtS}};
}

[[nodiscard]] double requested_effort(std::size_t pattern_index, std::size_t step) {
    if (step < kWarmupSteps || step >= kMotionExplorerMaxSteps - kCoastSteps) return 0.0;
    const std::size_t active_step = step - kWarmupSteps;
    const Pattern& pattern = kPatterns.at(pattern_index);
    if (pattern.waveform == "sine") {
        const double time_s = static_cast<double>(active_step) * kMotionExplorerControlDtS;
        return pattern.amplitude * std::sin(2.0 * std::numbers::pi * pattern.frequency_hz * time_s);
    }
    if (pattern.waveform == "alternating_pulses") {
        const std::size_t phase = active_step % 100;
        if (phase < 12) return pattern.amplitude;
        if (phase >= 50 && phase < 62) return -pattern.amplitude;
        return 0.0;
    }
    if (active_step < 12) return pattern.amplitude;
    if (active_step >= 140 && active_step < 152) return -pattern.amplitude;
    return 0.0;
}
}  // namespace

struct MotionExplorer::Impl {
    bool initialized{false};
    std::size_t pattern_index{0};
    std::size_t steps{0};
    bool pending{false};
    double effort{0.0};
    double requested{0.0};
    double target{0.0};
    bool speed_limited{false};
    std::vector<double> efforts;
    std::string motor_id;
    std::string sensor_id;
    bool has_seen_sample{false};
    PhysicalObservation seen_sample;
    std::uint64_t last_recorded_sequence{0};
    std::size_t sample_count{0};
    double first_sample_s{0.0};
    double last_sample_s{0.0};
    long double gyro_square_sum{0.0L};
    long double force_square_sum{0.0L};
    long double force_mean{0.0L};
    long double force_m2{0.0L};
    double gyro_peak{0.0};
    double force_peak{0.0};

    void check_identity_and_sample(const PhysicalObservation& observation) {
        if (motor_id.empty()) {
            motor_id = observation.motor_id;
            sensor_id = observation.sensor_id;
        } else if (motor_id != observation.motor_id || sensor_id != observation.sensor_id) {
            invalid("physical module identity changed without reset");
        }
        if (!observation.valid) return;
        if (has_seen_sample) {
            if (observation.sequence < seen_sample.sequence ||
                observation.sampled_s + kTimeToleranceS < seen_sample.sampled_s) invalid("IMU sample went backwards");
            if (observation.sequence == seen_sample.sequence) {
                if (observation.sampled_s != seen_sample.sampled_s || observation.delivered_s != seen_sample.delivered_s ||
                    observation.force != seen_sample.force || observation.gyro != seen_sample.gyro) invalid("held IMU sample changed without a new sequence");
                return;
            }
            if (observation.sampled_s <= seen_sample.sampled_s + kTimeToleranceS) invalid("new IMU sequence did not advance sample time");
        }
        seen_sample = observation;
        has_seen_sample = true;
    }

    [[nodiscard]] double act(const Json& observation) {
        if (!initialized) invalid("reset is required before a trial");
        if (pending) invalid("observe the pending physics transition before another act");
        if (steps >= kMotionExplorerMaxSteps) invalid("trial is complete; reset before another act");
        const PhysicalObservation physical = validate(observation, static_cast<double>(steps) * kMotionExplorerControlDtS);
        check_identity_and_sample(physical);
        requested = requested_effort(pattern_index, steps);
        target = requested;
        if (target != 0.0) {
            const double direction = target > 0.0 ? 1.0 : -1.0;
            target = direction * std::min({std::abs(target), kEffortCeiling,
                kShaftGain * std::max(0.0, kShaftTargetRadS - direction * physical.shaft_velocity)});
        }
        if (!physical.valid) target = 0.0;
        speed_limited = std::abs(target) < std::abs(requested);
        effort = std::clamp(target, effort - kSlew, effort + kSlew);
        effort = std::clamp(effort, -kEffortCeiling, kEffortCeiling);
        pending = true;
        return effort;
    }

    void observe(const Json& observation) {
        if (!initialized || !pending) invalid("observe requires one pending act and physics transition");
        const PhysicalObservation physical = validate(observation, static_cast<double>(steps + 1) * kMotionExplorerControlDtS);
        check_identity_and_sample(physical);
        if (physical.valid && physical.sequence != last_recorded_sequence) {
            const double gyro = std::hypot(physical.gyro[0], physical.gyro[1], physical.gyro[2]);
            const double force = std::hypot(physical.force[0], physical.force[1], physical.force[2]);
            if (sample_count == 0) first_sample_s = physical.sampled_s;
            last_sample_s = physical.sampled_s;
            ++sample_count;
            const long double force_long = force;
            const long double delta = force_long - force_mean;
            force_mean += delta / static_cast<long double>(sample_count);
            force_m2 += delta * (force_long - force_mean);
            gyro_square_sum += static_cast<long double>(gyro) * gyro;
            force_square_sum += force_long * force_long;
            gyro_peak = std::max(gyro_peak, gyro);
            force_peak = std::max(force_peak, force);
            last_recorded_sequence = physical.sequence;
        }
        efforts.push_back(effort);
        ++steps;
        pending = false;
    }

    [[nodiscard]] std::string_view status() const {
        return steps >= kMotionExplorerMaxSteps ? "completed" : steps == 0 && !pending ? "ready" : "running";
    }

    [[nodiscard]] Json result() const {
        const Pattern& pattern = kPatterns.at(pattern_index);
        const long double count = static_cast<long double>(std::max<std::size_t>(sample_count, 1));
        return Json{{"explorer_id", kMotionExplorerId},
            {"pattern_id", initialized ? Json(pattern.id) : Json(nullptr)},
            {"pattern_name", initialized ? Json(pattern.name) : Json(nullptr)},
            {"label", initialized ? Json(pattern.name) : Json(nullptr)}, {"status", status()},
            {"steps", steps}, {"duration_s", kMotionExplorerDurationS},
            {"elapsed_s", static_cast<double>(steps) * kMotionExplorerControlDtS},
            {"unique_sample_count", sample_count},
            {"observed_sample_span_s", sample_count > 1 ? last_sample_s - first_sample_s : 0.0},
            {"first_sample_time_s", sample_count > 0 ? Json(first_sample_s) : Json(nullptr)},
            {"last_sample_time_s", sample_count > 0 ? Json(last_sample_s) : Json(nullptr)},
            {"gyro_rms_rad_s", static_cast<double>(std::sqrt(gyro_square_sum / count))},
            {"gyro_peak_rad_s", gyro_peak},
            {"specific_force_rms_m_s2", static_cast<double>(std::sqrt(force_square_sum / count))},
            {"specific_force_peak_m_s2", force_peak},
            {"specific_force_variation_m_s2", static_cast<double>(std::sqrt(std::max(0.0L, force_m2 / count)))},
            {"specific_force_includes_gravity", true}, {"efforts", efforts}};
    }

    [[nodiscard]] Json diagnostics() const {
        const Pattern& pattern = kPatterns.at(pattern_index);
        const std::string_view phase = steps >= kMotionExplorerMaxSteps ? "completed" :
            steps < kWarmupSteps ? "warmup" : steps >= kMotionExplorerMaxSteps - kCoastSteps ? "coasting" : "exploring";
        return Json{{"explorer_id", kMotionExplorerId},
            {"pattern_id", initialized ? Json(pattern.id) : Json(nullptr)},
            {"pattern_name", initialized ? Json(pattern.name) : Json(nullptr)},
            {"label", initialized ? Json(pattern.name) : Json(nullptr)}, {"status", status()},
            {"phase", phase}, {"steps", steps}, {"max_steps", kMotionExplorerMaxSteps},
            {"duration_s", kMotionExplorerDurationS},
            {"elapsed_s", static_cast<double>(steps) * kMotionExplorerControlDtS},
            {"remaining_s", static_cast<double>(kMotionExplorerMaxSteps - steps) * kMotionExplorerControlDtS},
            {"requested_effort", requested}, {"target_effort", target}, {"effort", effort},
            {"speed_limited", speed_limited}, {"awaiting_observation", pending},
            {"unique_sample_count", sample_count}};
    }
};

MotionExplorer::MotionExplorer() : impl_(std::make_unique<Impl>()) {}
MotionExplorer::~MotionExplorer() = default;

Json MotionExplorer::patterns() {
    Json result = Json::array();
    for (const auto& pattern : kPatterns) result.push_back(pattern_json(pattern));
    return result;
}

void MotionExplorer::reset(std::string pattern_id) {
    const auto found = std::find_if(kPatterns.begin(), kPatterns.end(), [&](const auto& pattern) { return pattern.id == pattern_id; });
    if (found == kPatterns.end()) invalid("unknown pattern ID");
    Impl next;
    next.initialized = true;
    next.pattern_index = static_cast<std::size_t>(found - kPatterns.begin());
    next.efforts.reserve(kMotionExplorerMaxSteps);
    *impl_ = std::move(next);
}

double MotionExplorer::act(const Json& observation) {
    Impl next = *impl_;
    const double effort = next.act(observation);
    *impl_ = std::move(next);
    return effort;
}

void MotionExplorer::observe(const Json& observation) {
    Impl next = *impl_;
    next.observe(observation);
    *impl_ = std::move(next);
}

bool MotionExplorer::complete() const { return impl_->initialized && impl_->steps >= kMotionExplorerMaxSteps; }
Json MotionExplorer::result() const { return impl_->result(); }
Json MotionExplorer::diagnostics() const { return impl_->diagnostics(); }

Json motion_explorer_spec() {
    return Json{{"schema_version", "droid-blocks.motion-explorer.v1"}, {"explorer_id", kMotionExplorerId},
        {"patterns", MotionExplorer::patterns()}, {"control_dt_s", kMotionExplorerControlDtS},
        {"max_control_steps", kMotionExplorerMaxSteps}, {"duration_s", kMotionExplorerDurationS},
        {"initial_quiet_s", kWarmupSteps * kMotionExplorerControlDtS},
        {"final_coast_s", kCoastSteps * kMotionExplorerControlDtS},
        {"pattern_clock", "active time starts after initial quiet; final coast stops waveform requests for the last 15 steps"},
        {"paired_taps", "12 positive steps then coast through step49, 12 negative steps then coast through step99, repeated on the active clock"},
        {"push_and_coast", "12 positive active steps, coast, 12 negative active steps starting at step140, then coast"},
        {"effort_ceiling", kEffortCeiling}, {"slew_limit_per_step", kSlew},
        {"shaft_governor", Json{{"target_rad_s", kShaftTargetRadS}, {"effort_per_rad_s", kShaftGain},
            {"rule", "requested sign * min(requested magnitude, 0.35, 0.05 * max(0, 3 - requested sign * local shaft velocity)), then slew0.1"},
            {"limitation", "soft sampled-feedback envelope; not a guaranteed speed bound or hardware safety qualification"}}},
        {"observation_schema", "construction_observation_v1"},
        {"sensor_sample_period_s", kSamplePeriodS}, {"sensor_latency_s", kLatencyS}, {"sensor_stale_after_s", kStaleAfterS},
        {"sampling", "unique delivered IMU samples seen after each20ms control transition; intermediate10ms samples are not reconstructed or double counted"},
        {"gyro_descriptors", "RMS and peak of the local angular-velocity vector magnitude, in rad/s"},
        {"force_descriptors", "RMS, peak and population standard deviation of the local specific-force vector magnitude, in m/s^2; includes gravity"},
        {"comparison_scope", "all patterns have the same6s physical-time budget and include initial quiet/final coast; reset the same assembly before each trial"},
        {"memory_scope", "one trial; reset forgets all samples, identity and effort history"},
        {"failure_policy", "malformed, missing, nonfinite, stale or invalid post-warmup readings and every nonempty motor fault_flags stop inference without mutating explorer state; native world remains authoritative for physical safety"},
        {"selection_or_ranking", "none; fixed patterns describe observed responses without a reward score or optimality claim"},
        {"trained_parameters", false}};
}

}  // namespace droid
