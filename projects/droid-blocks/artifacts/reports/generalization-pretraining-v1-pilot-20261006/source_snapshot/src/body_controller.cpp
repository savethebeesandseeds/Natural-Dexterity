#include "droid/body_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr std::size_t kInitialSteps{15};
constexpr std::size_t kProbeSteps{50};
constexpr std::size_t kWashSteps{30};
constexpr double kProbeEvidenceWarmupS{0.4};
constexpr double kSensorPeriodS{0.1};
constexpr double kSensorLatencyS{0.04};
constexpr double kStaleAfterS{0.5};
constexpr double kTimeToleranceS{0.000002};
constexpr double kMinimumSlopeLuxPerS{0.02};
constexpr double kRelativeSlopePerS{0.001};

[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument("body probe controller: " + std::string(message));
}

void exact_keys(const Json& value, std::initializer_list<std::string_view> keys) {
    if (!value.is_object() || value.size() != keys.size()) {
        invalid("unexpected observation fields");
    }
    for (const auto key : keys) {
        if (!value.contains(key)) invalid("missing observation field");
    }
}

[[nodiscard]] std::string string_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_string() || field.get_ref<const std::string&>().empty()) {
        invalid("expected a non-empty string field");
    }
    return field.get<std::string>();
}

[[nodiscard]] double number_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_number()) invalid("expected a physical numeric field");
    const double result = field.get<double>();
    if (!std::isfinite(result)) invalid("non-finite observation");
    return result;
}

[[nodiscard]] bool bool_field(const Json& value, std::string_view key) {
    const Json& field = value.at(key);
    if (!field.is_boolean()) invalid("expected a boolean field");
    return field.get<bool>();
}

[[nodiscard]] std::set<std::string> identity_set(
    std::span<const std::string> ids) {
    std::set<std::string> result;
    for (const std::string& id : ids) {
        if (id.empty() || !result.insert(id).second) {
            invalid("motor IDs must be non-empty and unique");
        }
    }
    if (result.empty()) invalid("at least one motor is required");
    return result;
}

struct LightSample {
    bool valid{false};
    std::uint64_t sequence{0};
    double sampled_s{0.0};
    double delivered_s{0.0};
    double lux{0.0};
    bool saturated{false};
};

[[nodiscard]] LightSample validate_observation(
    const Json& observation, const std::set<std::string>& ids, double now_s,
    std::map<std::string, double>& shaft_velocities) {
    exact_keys(observation, {"reward_sensors", "actuator_feedback"});
    const Json& sensors = observation.at("reward_sensors");
    const Json& actuators = observation.at("actuator_feedback");
    if (!sensors.is_array() || !actuators.is_array()) {
        invalid("observation collections must be arrays");
    }
    std::set<std::string> sensor_ids;
    bool has_light = false;
    LightSample result;
    for (const Json& sensor : sensors) {
        if (!sensor.is_object() || !sensor.contains("family_id")) {
            invalid("malformed sensor entry");
        }
        const std::string family = string_field(sensor, "family_id");
        if (family == "ambient_light_v0") {
            exact_keys(sensor, {"module_id", "family_id", "valid", "observations",
                               "sequence", "sample_time_s", "delivered_time_s",
                               "age_s", "sample_period_s", "latency_s"});
            if (has_light) invalid("exactly one ambient light sensor is supported");
            has_light = true;
            const Json& channels = sensor.at("observations");
            exact_keys(channels, {"illuminance_lux", "saturated"});
            result.lux = number_field(channels, "illuminance_lux");
            result.saturated = bool_field(channels, "saturated");
            result.valid = bool_field(sensor, "valid");
            if (result.lux < 0.0) invalid("negative illuminance");
            if (std::abs(number_field(sensor, "sample_period_s") - kSensorPeriodS) >
                    kTimeToleranceS ||
                std::abs(number_field(sensor, "latency_s") - kSensorLatencyS) >
                    kTimeToleranceS) {
                invalid("sensor timing does not match this controller version");
            }
            if (!sensor.at("sequence").is_number_unsigned() &&
                !(sensor.at("sequence").is_number_integer() &&
                  sensor.at("sequence").get<std::int64_t>() >= 0)) {
                invalid("sensor sequence must be a nonnegative integer");
            }
            result.sequence = sensor.at("sequence").get<std::uint64_t>();
            if (!result.valid) {
                if (now_s + kTimeToleranceS >= kSensorLatencyS ||
                    result.sequence != 0 || !sensor.at("sample_time_s").is_null() ||
                    !sensor.at("delivered_time_s").is_null() ||
                    !sensor.at("age_s").is_null()) {
                    invalid("light sensor is invalid after its initial delivery window");
                }
            } else {
                result.sampled_s = number_field(sensor, "sample_time_s");
                result.delivered_s = number_field(sensor, "delivered_time_s");
                const double age = number_field(sensor, "age_s");
                if (result.sequence == 0 || result.sampled_s < 0.0 ||
                    result.sampled_s > now_s + kTimeToleranceS ||
                    result.delivered_s > now_s + kTimeToleranceS ||
                    std::abs(result.delivered_s - result.sampled_s - kSensorLatencyS) >
                        kTimeToleranceS ||
                    age < 0.0 || age > kStaleAfterS ||
                    std::abs(age - (now_s - result.sampled_s)) > kTimeToleranceS) {
                    invalid("light sensor is stale or has inconsistent timing");
                }
            }
        } else if (family == "touch_force_v0") {
            exact_keys(sensor, {"module_id", "family_id", "valid", "observations"});
            if (!bool_field(sensor, "valid")) invalid("touch sensor is invalid");
            exact_keys(sensor.at("observations"), {"contact", "normal_force_n"});
            (void)bool_field(sensor.at("observations"), "contact");
            if (number_field(sensor.at("observations"), "normal_force_n") < 0.0) {
                invalid("negative touch force");
            }
        } else {
            invalid("unsupported sensor family");
        }
        if (!sensor_ids.insert(string_field(sensor, "module_id")).second) {
            invalid("duplicate sensor ID");
        }
    }
    if (!has_light) invalid("missing ambient light sensor");

    std::set<std::string> observed_ids;
    for (const Json& actuator : actuators) {
        exact_keys(actuator, {"module_id", "sku_id", "valid", "feedback"});
        const std::string id = string_field(actuator, "module_id");
        if (!ids.contains(id) || !observed_ids.insert(id).second) {
            invalid("unknown or duplicate motor feedback");
        }
        if (string_field(actuator, "sku_id") != "rotary_dc_gearmotor_v0" ||
            !bool_field(actuator, "valid")) {
            invalid("unsupported or invalid motor feedback");
        }
        const Json& feedback = actuator.at("feedback");
        exact_keys(feedback, {"position_rad", "velocity_rad_s", "current_a",
                              "bus_voltage_v", "temperature_c", "output_torque_nm",
                              "load_impedance_nm_s_per_rad", "stuck_score", "stuck",
                              "fault_flags"});
        for (const auto key : {"position_rad", "velocity_rad_s", "current_a",
                               "bus_voltage_v", "temperature_c", "output_torque_nm",
                               "load_impedance_nm_s_per_rad", "stuck_score"}) {
            (void)number_field(feedback, key);
        }
        (void)bool_field(feedback, "stuck");
        shaft_velocities.emplace(id, number_field(feedback, "velocity_rad_s"));
        if (!feedback.at("fault_flags").is_array() ||
            !feedback.at("fault_flags").empty()) {
            invalid("motor " + id + " feedback reports fault_flags=" +
                    feedback.at("fault_flags").dump());
        }
    }
    if (observed_ids != ids) invalid("missing motor feedback");
    return result;
}

struct EvidencePoint { double time_s; double lux; };
struct Slope { double lux_per_s{0.0}; double mean_lux{0.0}; std::size_t count{0}; };

[[nodiscard]] Slope slope(const std::vector<EvidencePoint>& samples) {
    Slope result;
    result.count = samples.size();
    if (samples.empty()) return result;
    double mean_t = 0.0;
    for (const auto& point : samples) {
        mean_t += point.time_s;
        result.mean_lux += point.lux;
    }
    mean_t /= static_cast<double>(samples.size());
    result.mean_lux /= static_cast<double>(samples.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (const auto& point : samples) {
        const double dt = point.time_s - mean_t;
        numerator += dt * (point.lux - result.mean_lux);
        denominator += dt * dt;
    }
    if (denominator > 0.0) result.lux_per_s = numerator / denominator;
    return result;
}
}  // namespace

struct BodyProbeController::Impl {
    enum class Stage { initial, positive, positive_wash, negative, negative_wash, seeking };
    struct Motor {
        std::string id;
        int preference{0};
        double evidence{0.0};
        std::string confidence{"untried"};
        double effort{0.0};
        double target_effort{0.0};
        bool speed_limited{false};
        std::size_t positive_samples{0};
        std::size_t negative_samples{0};
    };
    std::vector<Motor> motors;
    std::set<std::string> identities;
    std::string strategy{"discover"};
    Stage stage{Stage::initial};
    std::size_t step{0};
    std::size_t stage_start{0};
    std::size_t probe_start{0};
    std::size_t active{0};
    std::vector<EvidencePoint> positive_samples;
    std::vector<EvidencePoint> negative_samples;
    LightSample last_sample;
    bool has_sample{false};

    [[nodiscard]] std::size_t duration() const {
        switch (stage) {
            case Stage::initial: return kInitialSteps;
            case Stage::positive:
            case Stage::negative: return kProbeSteps;
            case Stage::positive_wash:
            case Stage::negative_wash: return kWashSteps;
            case Stage::seeking: return 0;
        }
        return 0;
    }

    void finish_pair() {
        const Slope plus = slope(positive_samples);
        const Slope minus = slope(negative_samples);
        const double tolerance = std::max(kMinimumSlopeLuxPerS,
            kRelativeSlopePerS * std::max(plus.mean_lux, minus.mean_lux));
        int preference = 0;
        if (plus.count >= 3 && minus.count >= 3) {
            if (plus.lux_per_s > tolerance && minus.lux_per_s < -tolerance) preference = 1;
            if (plus.lux_per_s < -tolerance && minus.lux_per_s > tolerance) preference = -1;
        }
        const auto remember = [&](Motor& motor) {
            motor.preference = preference;
            motor.evidence = (plus.lux_per_s - minus.lux_per_s) / 2.0;
            motor.confidence = preference == 0 ? "uncertain" : "measured";
            motor.positive_samples = plus.count;
            motor.negative_samples = minus.count;
        };
        if (strategy == "together") {
            for (Motor& motor : motors) remember(motor);
        } else {
            remember(motors.at(active));
        }
    }

    void advance_stage() {
        if (stage == Stage::seeking || step - stage_start < duration()) return;
        switch (stage) {
            case Stage::initial:
                stage = Stage::positive;
                probe_start = step;
                break;
            case Stage::positive: stage = Stage::positive_wash; break;
            case Stage::positive_wash:
                stage = Stage::negative;
                probe_start = step;
                break;
            case Stage::negative: stage = Stage::negative_wash; break;
            case Stage::negative_wash:
                finish_pair();
                ++active;
                if (strategy == "together" || active == motors.size()) {
                    stage = Stage::seeking;
                } else {
                    stage = Stage::positive;
                    probe_start = step;
                    positive_samples.clear();
                    negative_samples.clear();
                }
                break;
            case Stage::seeking: break;
        }
        stage_start = step;
    }

    void observe(const LightSample& sample) {
        if (!sample.valid) return;
        if (has_sample) {
            if (sample.sequence < last_sample.sequence ||
                sample.sampled_s + kTimeToleranceS < last_sample.sampled_s) {
                invalid("light sample went backwards");
            }
            if (sample.sequence == last_sample.sequence) {
                if (sample.sampled_s != last_sample.sampled_s ||
                    sample.delivered_s != last_sample.delivered_s ||
                    sample.lux != last_sample.lux ||
                    sample.saturated != last_sample.saturated) {
                    invalid("held light sample changed without a new sequence");
                }
                return;
            }
            if (sample.sampled_s <= last_sample.sampled_s + kTimeToleranceS) {
                invalid("new sequence did not advance physical sample time");
            }
        }
        last_sample = sample;
        has_sample = true;
        if (sample.saturated) return;
        if (stage != Stage::positive && stage != Stage::positive_wash &&
            stage != Stage::negative && stage != Stage::negative_wash) return;
        const double started = static_cast<double>(probe_start) * kBodyControllerControlDtS;
        const double ended = started + static_cast<double>(kProbeSteps) * kBodyControllerControlDtS;
        if (sample.sampled_s + kTimeToleranceS < started + kProbeEvidenceWarmupS ||
            sample.sampled_s > ended + kTimeToleranceS) return;
        auto& samples = (stage == Stage::positive || stage == Stage::positive_wash)
            ? positive_samples : negative_samples;
        samples.push_back({sample.sampled_s - started, sample.lux});
    }

    [[nodiscard]] std::map<std::string, double> act(
        const Json& observation, std::span<const std::string> motor_ids) {
        if (motors.empty()) invalid("reset is required before inference");
        if (identity_set(motor_ids) != identities) invalid("motor topology changed without reset");
        const double now_s = static_cast<double>(step) * kBodyControllerControlDtS;
        std::map<std::string, double> shaft_velocities;
        const LightSample sample = validate_observation(
            observation, identities, now_s, shaft_velocities);
        advance_stage();
        observe(sample);
        std::map<std::string, double> actions;
        for (std::size_t index = 0; index < motors.size(); ++index) {
            Motor& motor = motors[index];
            double target = 0.0;
            if (stage == Stage::seeking) {
                target = static_cast<double>(motor.preference) * kBodyControllerEffortLimit;
            } else if ((stage == Stage::positive || stage == Stage::negative) &&
                       (strategy == "together" || index == active)) {
                target = stage == Stage::positive
                    ? kBodyControllerEffortLimit : -kBodyControllerEffortLimit;
            }
            if (!sample.valid) target = 0.0;
            const double requested_target = target;
            if (target != 0.0) {
                const double direction = target > 0.0 ? 1.0 : -1.0;
                const double headroom = std::max(0.0,
                    kBodyControllerShaftTargetRadS - direction * shaft_velocities.at(motor.id));
                target = direction * std::min(std::abs(target),
                    kBodyControllerSpeedGain * headroom);
            }
            motor.target_effort = target;
            motor.speed_limited = std::abs(target) < std::abs(requested_target);
            motor.effort = std::clamp(target,
                motor.effort - kBodyControllerSlewLimit,
                motor.effort + kBodyControllerSlewLimit);
            actions.emplace(motor.id, motor.effort);
        }
        ++step;
        return actions;
    }

    [[nodiscard]] Json diagnostics() const {
        Json entries = Json::array();
        for (const Motor& motor : motors) {
            entries.push_back(Json{{"module_id", motor.id},
                {"preference", motor.preference}, {"evidence_lux", motor.evidence},
                {"confidence", motor.confidence}, {"effort", motor.effort},
                {"target_effort", motor.target_effort}, {"speed_limited", motor.speed_limited},
                {"positive_samples", motor.positive_samples},
                {"negative_samples", motor.negative_samples}});
        }
        const bool probing = stage == Stage::positive || stage == Stage::negative;
        const bool measuring = probing || stage == Stage::positive_wash || stage == Stage::negative_wash;
        return Json{{"controller_id", kBodyControllerId}, {"strategy", strategy},
            {"phase", stage == Stage::seeking ? "seeking" : probing ? "probing" : "settling"},
            {"active_motor_id", measuring && strategy == "discover" && active < motors.size()
                ? Json(motors[active].id) : Json(nullptr)},
            {"probe_polarity", stage == Stage::positive ? 1 : stage == Stage::negative ? -1 : 0},
            {"elapsed_s", static_cast<double>(step) * kBodyControllerControlDtS},
            {"phase_elapsed_s", static_cast<double>(step - stage_start) * kBodyControllerControlDtS},
            {"phase_duration_s", static_cast<double>(duration()) * kBodyControllerControlDtS},
            {"evidence_units", "lx/s"},
            {"evidence_definition", "(positive-probe lux slope minus negative-probe lux slope) / 2"},
            {"minimum_slope_lux_per_s", kMinimumSlopeLuxPerS},
            {"relative_slope_threshold_per_s", kRelativeSlopePerS},
            {"memory_scope", "this run; reset forgets all measured preferences"},
            {"motors", std::move(entries)}};
    }
};

BodyProbeController::BodyProbeController() : impl_(std::make_unique<Impl>()) {}
BodyProbeController::~BodyProbeController() = default;

void BodyProbeController::reset(std::span<const std::string> motor_ids,
                                std::string_view strategy) {
    const auto identities = identity_set(motor_ids);
    if (strategy != "discover" && strategy != "together") invalid("unknown probe strategy");
    Impl next;
    next.identities = identities;
    next.strategy = strategy;
    for (const std::string& id : motor_ids) next.motors.push_back(Impl::Motor{.id = id});
    *impl_ = std::move(next);
}

std::map<std::string, double> BodyProbeController::act(
    const Json& observation, std::span<const std::string> motor_ids) {
    Impl next = *impl_;
    auto actions = next.act(observation, motor_ids);
    *impl_ = std::move(next);
    return actions;
}

Json BodyProbeController::diagnostics() const { return impl_->diagnostics(); }

Json body_controller_spec() {
    return Json{{"controller_id", kBodyControllerId},
        {"schema_version", "droid-blocks.body-probe-controller.v2"},
        {"strategies", Json::array({"discover", "together"})},
        {"control_dt_s", kBodyControllerControlDtS},
        {"effort_limit", kBodyControllerEffortLimit},
        {"slew_limit_per_control_step", kBodyControllerSlewLimit},
        {"shaft_governor", Json{
            {"target_rad_s", kBodyControllerShaftTargetRadS},
            {"effort_per_rad_s", kBodyControllerSpeedGain},
            {"magnitude_rule", "min(requested magnitude, 0.05 * max(0, 3 - requested sign * local shaft velocity)) before the unchanged 0.1 slew limit"},
            {"scope", "every motor and both strategies, during probing and seeking"},
            {"purpose", "prospectively reduce drive demand as a lightly loaded shaft accelerates; all nonempty fault_flags still stop inference"},
            {"limitation", "sampled-feedback soft envelope, not a guaranteed speed bound or a hardware safety qualification"}}},
        {"initial_settle_s", static_cast<double>(kInitialSteps) * kBodyControllerControlDtS},
        {"probe_s", static_cast<double>(kProbeSteps) * kBodyControllerControlDtS},
        {"washout_s", static_cast<double>(kWashSteps) * kBodyControllerControlDtS},
        {"evidence_warmup_s", kProbeEvidenceWarmupS},
        {"minimum_fresh_samples_per_polarity", 3},
        {"minimum_slope_lux_per_s", kMinimumSlopeLuxPerS},
        {"relative_slope_threshold_per_s", kRelativeSlopePerS},
        {"sensor_sample_period_s", kSensorPeriodS},
        {"sensor_latency_s", kSensorLatencyS},
        {"sensor_stale_after_s", kStaleAfterS},
        {"evidence_units", "lx/s"},
        {"evidence_definition", "(positive-probe lux slope minus negative-probe lux slope) / 2"},
        {"evidence_estimator", "least-squares slope of unique unsaturated delivered light samples captured from probe+0.4 s through probe+1.0 s; late arrivals retained during washout"},
        {"preference_rule", "paired slopes must both exceed max(0.02 lx/s, 0.001/s * maximum paired mean lux) in opposite directions; otherwise preference zero"},
        {"probe_order", "reset enumeration, retained under ID relabel; act collection ordering does not affect results"},
        {"memory_scope", "one run; reset clears all state; pause must not call act"},
        {"seeking_rule", "retain each measured local sign, with the same shaft-speed effort envelope; no further learning or moving-source tracking"},
        {"trained_parameters", false}};
}

}  // namespace droid
