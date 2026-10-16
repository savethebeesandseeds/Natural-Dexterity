#include "droid/light_context.hpp"
#include "droid/light_learner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr std::size_t kHistoryFrames = 11;
constexpr std::size_t kNeighbors = 5;
[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument("light context: " + std::string(message));
}
void finite_features(std::span<const double> features, std::size_t expected) {
    if (features.size() != expected) invalid("feature dimension differs from declared mode");
    for (const double value : features)
        if (!std::isfinite(value)) invalid("nonfinite feature");
}
double squared_mean_distance(std::span<const double> a, std::span<const double> b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        sum += difference * difference;
        if (!std::isfinite(sum)) invalid("squared distance is not finite");
    }
    const double distance = sum / static_cast<double>(a.size());
    if (!std::isfinite(distance)) invalid("mean squared distance is not finite");
    return distance;
}
} // namespace

std::string_view light_context_mode_id(LightContextMode mode) {
    switch (mode) {
        case LightContextMode::instantaneous: return "instantaneous";
        case LightContextMode::history: return "history";
        case LightContextMode::light_only: return "light_only";
        case LightContextMode::no_light_history: return "no_light_history";
    }
    invalid("unknown decoder mode");
}
std::size_t light_context_dimension(LightContextMode mode) {
    switch (mode) {
        case LightContextMode::instantaneous: return 12;
        case LightContextMode::history: return 132;
        case LightContextMode::light_only: return 1;
        case LightContextMode::no_light_history: return 121;
    }
    invalid("unknown decoder mode");
}
LightContextFrame project_light_context_frame(const Json& observation, double executed_effort) {
    // This existing parser checks the entire observation shape, local feedback,
    // finite channels, declared sample timing and empty motor fault flags. Its
    // return value is ignored: projection neither chooses nor changes effort.
    (void)LightLearner::bounded_effort(0.0, executed_effort, observation);
    const Json* imu = nullptr;
    const Json* light = nullptr;
    for (const Json& sensor : observation.at("reward_sensors")) {
        if (sensor.at("valid") != true) invalid("projection requires delivered valid sensors");
        if (sensor.at("family_id") == "imu_6axis_v0") imu = &sensor.at("observations");
        else if (sensor.at("family_id") == "ambient_light_v0") light = &sensor.at("observations");
    }
    if (!imu || !light) invalid("missing required local sensor");
    const Json& motor = observation.at("actuator_feedback").at(0).at("feedback");
    const double position = motor.at("position_rad").get<double>();
    LightContextFrame frame{
        executed_effort / .15,
        std::sin(position), std::cos(position),
        motor.at("velocity_rad_s").get<double>() / 10.0,
        motor.at("current_a").get<double>() / .5,
        imu->at("specific_force_m_s2").at(0).get<double>() / 50.0,
        imu->at("specific_force_m_s2").at(1).get<double>() / 50.0,
        imu->at("specific_force_m_s2").at(2).get<double>() / 50.0,
        imu->at("angular_velocity_rad_s").at(0).get<double>() / 10.0,
        imu->at("angular_velocity_rad_s").at(1).get<double>() / 10.0,
        imu->at("angular_velocity_rad_s").at(2).get<double>() / 10.0,
        light->at("illuminance_lux").get<double>() / 1000.0};
    finite_features(frame, 12);
    return frame;
}
std::vector<double> light_context_features(LightContextMode mode, std::span<const LightContextFrame> frames) {
    const std::size_t dimension = light_context_dimension(mode);
    if (frames.empty() || frames.size() > kHistoryFrames) invalid("feature window requires one through eleven frames");
    if ((mode == LightContextMode::history || mode == LightContextMode::no_light_history) &&
        frames.size() != kHistoryFrames) invalid("history requires exactly eleven causal frames");
    for (const auto& frame : frames) finite_features(frame, 12);
    std::vector<double> result;
    result.reserve(dimension);
    if (mode == LightContextMode::instantaneous) result.assign(frames.back().begin(), frames.back().end());
    else if (mode == LightContextMode::light_only) result.push_back(frames.back().back());
    else {
        const std::size_t width = mode == LightContextMode::no_light_history ? 11 : 12;
        for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
            result.insert(result.end(), frame->begin(), frame->begin() + width);
    }
    finite_features(result, dimension);
    return result;
}
LightContextDecoder::LightContextDecoder(LightContextMode mode, std::span<const LightContextExample> training)
    : mode_(mode) {
    const std::size_t dimension = light_context_dimension(mode_);
    if (training.size() < kNeighbors) invalid("five-neighbor decoder requires at least five training examples");
    for (const auto& example : training) finite_features(example.features, dimension);
    training_.assign(training.begin(), training.end());
}
LightContextPrediction LightContextDecoder::predict(std::span<const double> features) const {
    finite_features(features, light_context_dimension(mode_));
    std::vector<std::pair<double, bool>> distances;
    distances.reserve(training_.size());
    for (const auto& example : training_)
        distances.emplace_back(squared_mean_distance(features, example.features), example.is_a);
    std::nth_element(distances.begin(), distances.begin() + (kNeighbors - 1), distances.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    const double cutoff = distances[kNeighbors - 1].first;
    std::size_t count = 0, a_votes = 0;
    for (const auto& [distance, is_a] : distances) {
        if (distance <= cutoff) { ++count; a_votes += is_a ? 1 : 0; }
    }
    const double probability = (static_cast<double>(a_votes) + 1.0) / (static_cast<double>(count) + 2.0);
    return {probability, probability >= .5, count, cutoff};
}
Json LightContextDecoder::serialize() const {
    Json training = Json::array();
    for (const auto& example : training_)
        training.push_back(Json{{"features", example.features}, {"is_a", example.is_a}});
    return Json{{"schema", "light-context-decoder-v1"}, {"mode", light_context_mode_id(mode_)},
        {"dimension", light_context_dimension(mode_)}, {"k", kNeighbors},
        {"distance", "mean squared distance over the fixed feature coordinates"},
        {"ties", "include every example with distance at or below the exact fifth-distance cutoff"},
        {"probability_a", "(A_votes+1)/(neighbor_count+2)"}, {"predict_a", "probability_a >= .5"},
        {"training", std::move(training)}};
}
Json light_context_specification() {
    return Json{{"schema", "light-context-features-v1"}, {"frame_dimension", 12},
        {"frame_order", Json::array({"executed_effort/.15", "sin(local_motor_position)", "cos(local_motor_position)",
            "local_motor_velocity/10", "local_motor_current/.5", "local_specific_force_x/50",
            "local_specific_force_y/50", "local_specific_force_z/50", "local_gyro_x/10",
            "local_gyro_y/10", "local_gyro_z/10", "illuminance_lux/1000"})},
        {"normalization", "fixed declared divisors; no clipping or fitted transformation"},
        {"validation", "strict construction_observation_v2 local schema; delivered valid IMU/light; finite channels and effort; existing motor envelope validation"},
        {"effort_domain", Json::array({-.35, .35})},
        {"metadata", "timestamps, sequence IDs and opaque module IDs validate transport only; no source, body, rewards, labels or metadata enter feature vectors"},
        {"history", Json{{"frames", kHistoryFrames}, {"spacing_s", .1}, {"order", "latest to oldest"},
            {"causality", "collector supplies only completed frames up to the query endpoint; collector owns timing validation"}}},
        {"dimensions", Json{{"instantaneous", 12}, {"history", 132}, {"light_only", 1}, {"no_light_history", 121}}},
        {"decoder", Json{{"k", kNeighbors}, {"distance", "mean squared Euclidean coordinate difference"},
            {"cutoff_ties", "all exact ties included"}, {"probability_a", "(A_votes+1)/(neighbors+2)"},
            {"predict_a", "probability_a >= .5"}, {"fit", "copy and freeze numeric examples and boolean training labels"},
            {"prediction_inputs", "numeric feature vector only"}, {"online_updates", false}}}};
}
} // namespace droid