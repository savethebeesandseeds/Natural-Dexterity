#pragma once

#include "droid/light_context.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>

namespace droid {
enum class LightPredictionMode { current_input, history_input, history_no_input };

[[nodiscard]] std::string_view light_prediction_mode_id(LightPredictionMode mode);
[[nodiscard]] std::size_t light_prediction_taps(LightPredictionMode mode);
[[nodiscard]] bool light_prediction_uses_input(LightPredictionMode mode);
[[nodiscard]] std::size_t light_prediction_state_dimension(LightPredictionMode mode);
[[nodiscard]] nlohmann::json light_prediction_specification();

// Exactly p local output frames, oldest first. For an input model, the p-1
// requests span the completed intervals between these frames, also oldest first.
// All requests are normalized by .15 by the caller. No-input models require an
// empty request history. Timing, source identity and physical state are absent.
[[nodiscard]] std::vector<double> light_prediction_state(
    LightPredictionMode mode,
    std::span<const LightContextFrame> outputs_oldest_to_newest,
    std::span<const double> previous_requests_oldest_to_newest);

struct LightPredictionExample {
    std::vector<double> state;
    double requested_input{};
    LightContextFrame next_output{};
};

// Affine ridge ARX model. Fitting copies numeric evidence and freezes the model.
// Prediction accepts only the declared delay state and next known request.
// A no-input model validates, then ignores, the request scalar.
class LightPredictionModel final {
public:
    LightPredictionModel(LightPredictionMode mode,
                         std::span<const LightPredictionExample> training);
    [[nodiscard]] LightContextFrame predict(
        std::span<const double> state, double requested_input) const;
    [[nodiscard]] std::vector<double> advance(
        std::span<const double> state, double requested_input) const;
    // Every state after the initial one contains predicted outputs only. No
    // overload accepts actual future outputs or a teacher-forcing callback.
    [[nodiscard]] std::vector<LightContextFrame> forecast(
        std::span<const double> initial_state,
        std::span<const double> future_requests) const;
    [[nodiscard]] nlohmann::json serialize() const;
private:
    LightPredictionMode mode_;
    std::vector<LightPredictionExample> training_;
    std::vector<std::vector<double>> coefficients_;
    double normal_equation_residual_max_abs_{};
    double normal_equation_residual_relative_{};
};
} // namespace droid
