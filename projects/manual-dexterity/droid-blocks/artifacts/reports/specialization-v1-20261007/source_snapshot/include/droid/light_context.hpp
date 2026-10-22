#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>

namespace droid {
using LightContextFrame = std::array<double, 12>;
enum class LightContextMode { instantaneous, history, light_only, no_light_history };

// Projection accepts only the declared local observation and the effort that
// produced it. Timing/identity metadata is validated, never emitted as features.
[[nodiscard]] LightContextFrame project_light_context_frame(
    const nlohmann::json& observation, double executed_effort);
// Caller supplies causal frames oldest to newest, sampled .1s apart. History
// modes require exactly 11; instantaneous/light-only accept 1 through 11.
[[nodiscard]] std::vector<double> light_context_features(
    LightContextMode mode, std::span<const LightContextFrame> oldest_to_newest);
[[nodiscard]] std::string_view light_context_mode_id(LightContextMode mode);
[[nodiscard]] std::size_t light_context_dimension(LightContextMode mode);
[[nodiscard]] nlohmann::json light_context_specification();

struct LightContextExample {
    std::vector<double> features;
    bool is_a{};
};
struct LightContextPrediction {
    double probability_a{};
    bool predicts_a{};
    std::size_t neighbor_count{};
    double squared_mean_distance_cutoff{};
};

// Constructor fits by copying the numeric training set. There is no mutable
// prediction state, online update API, metadata input or model import parser.
class LightContextDecoder final {
public:
    LightContextDecoder(LightContextMode mode, std::span<const LightContextExample> training);
    [[nodiscard]] LightContextPrediction predict(std::span<const double> features) const;
    [[nodiscard]] nlohmann::json serialize() const;
private:
    LightContextMode mode_;
    std::vector<LightContextExample> training_;
};
} // namespace droid