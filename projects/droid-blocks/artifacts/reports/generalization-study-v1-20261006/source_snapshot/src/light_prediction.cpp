#include "droid/light_prediction.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace droid {
namespace {
using Json = nlohmann::json;
using Matrix = std::vector<std::vector<double>>;
constexpr std::size_t kOutputDimension = 12;
constexpr double kRidge = .001;

[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument("light prediction: " + std::string(message));
}
double finite(double value, std::string_view message) {
    if (!std::isfinite(value)) invalid(message);
    return value;
}
void finite_vector(std::span<const double> values, std::size_t dimension) {
    if (values.size() != dimension) invalid("dimension differs from declared model");
    for (double value : values) (void)finite(value, "nonfinite numeric input");
}
std::vector<double> design(LightPredictionMode mode,
                           std::span<const double> state, double request) {
    finite_vector(state, light_prediction_state_dimension(mode));
    (void)finite(request, "nonfinite request");
    std::vector<double> result;
    result.reserve(state.size() + 2);
    result.push_back(1.0);
    result.insert(result.end(), state.begin(), state.end());
    if (light_prediction_uses_input(mode)) result.push_back(request);
    return result;
}
double checked_dot(std::span<const double> a, std::span<const double> b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        sum = finite(sum + finite(a[i] * b[i], "numeric product overflow"),
                     "numeric accumulation overflow");
    return sum;
}
} // namespace

std::string_view light_prediction_mode_id(LightPredictionMode mode) {
    switch (mode) {
        case LightPredictionMode::current_input: return "current_input";
        case LightPredictionMode::history_input: return "history_input";
        case LightPredictionMode::history_no_input: return "history_no_input";
    }
    invalid("unknown model mode");
}
std::size_t light_prediction_taps(LightPredictionMode mode) {
    switch (mode) {
        case LightPredictionMode::current_input: return 1;
        case LightPredictionMode::history_input:
        case LightPredictionMode::history_no_input: return 11;
    }
    invalid("unknown model mode");
}
bool light_prediction_uses_input(LightPredictionMode mode) {
    (void)light_prediction_mode_id(mode);
    return mode != LightPredictionMode::history_no_input;
}
std::size_t light_prediction_state_dimension(LightPredictionMode mode) {
    const std::size_t taps = light_prediction_taps(mode);
    return kOutputDimension * taps + (light_prediction_uses_input(mode) ? taps - 1 : 0);
}
Json light_prediction_specification() {
    return Json{{"schema_version", "light_prediction_specification_v1"},
        {"sample_interval_s", .1}, {"output_dimension", kOutputDimension},
        {"output_projection", "project_light_context_frame"},
        {"output_channels", Json::array({"executed_effort/.15", "sin(motor_position)",
            "cos(motor_position)", "motor_velocity/10", "motor_current/.5",
            "imu_specific_force_x/50", "imu_specific_force_y/50", "imu_specific_force_z/50",
            "imu_gyro_x/10", "imu_gyro_y/10", "imu_gyro_z/10", "illuminance/1000"})},
        {"request_normalization", .15},
        {"state_order", "[y_n,...,y_(n-p+1),r_(n-1),...,r_(n-p+1)]; omit r history for no-input"},
        {"regression", "y_next = coefficients * [1,state,current_request_if_enabled]"},
        {"loss", "mean squared residual per output + lambda * squared nonbias coefficients"},
        {"lambda", kRidge}, {"bias_regularized", false}, {"fitted_scaling", false},
        {"clipping", false}, {"unit_circle_projection", false},
        {"solver", "double-precision Cholesky of mean Gram plus ridge"},
        {"residual", "maximum absolute element of regularized normal-equation residual"},
        {"relative_residual", "max_abs_residual / max(1,max_abs_normal_equation_rhs)"},
        {"forecast", "recursive predicted state; known future requests; no future measured outputs"},
        {"modes", Json::array({
            Json{{"id", "current_input"}, {"taps", 1}, {"uses_input", true}, {"state_dimension", 12}},
            Json{{"id", "history_input"}, {"taps", 11}, {"uses_input", true}, {"state_dimension", 142}},
            Json{{"id", "history_no_input"}, {"taps", 11}, {"uses_input", false}, {"state_dimension", 132}}
        })}};
}
std::vector<double> light_prediction_state(
    LightPredictionMode mode, std::span<const LightContextFrame> outputs,
    std::span<const double> previous_requests) {
    const std::size_t taps = light_prediction_taps(mode);
    const bool input = light_prediction_uses_input(mode);
    if (outputs.size() != taps) invalid("state requires exactly the declared causal output taps");
    finite_vector(previous_requests, input ? taps - 1 : 0);
    std::vector<double> state;
    state.reserve(light_prediction_state_dimension(mode));
    for (auto output = outputs.rbegin(); output != outputs.rend(); ++output) {
        finite_vector(*output, kOutputDimension);
        state.insert(state.end(), output->begin(), output->end());
    }
    for (auto request = previous_requests.rbegin(); request != previous_requests.rend(); ++request)
        state.push_back(*request);
    return state;
}

LightPredictionModel::LightPredictionModel(
    LightPredictionMode mode, std::span<const LightPredictionExample> training)
    : mode_(mode), training_(training.begin(), training.end()) {
    const std::size_t dimension = light_prediction_state_dimension(mode) + 1 +
        (light_prediction_uses_input(mode) ? 1 : 0);
    if (training_.empty()) invalid("training evidence is empty");
    Matrix gram(dimension, std::vector<double>(dimension, 0.0));
    Matrix rhs(kOutputDimension, std::vector<double>(dimension, 0.0));
    for (const auto& row : training_) {
        const auto phi = design(mode_, row.state, row.requested_input);
        finite_vector(row.next_output, kOutputDimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            for (std::size_t j = 0; j <= i; ++j)
                gram[i][j] = finite(gram[i][j] + finite(phi[i] * phi[j], "Gram product overflow"),
                                    "Gram accumulation overflow");
            for (std::size_t channel = 0; channel < kOutputDimension; ++channel)
                rhs[channel][i] = finite(rhs[channel][i] +
                    finite(phi[i] * row.next_output[channel], "RHS product overflow"),
                    "RHS accumulation overflow");
        }
    }
    const double count = static_cast<double>(training_.size());
    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            gram[i][j] /= count;
            gram[j][i] = gram[i][j];
        }
        if (i != 0) gram[i][i] = finite(gram[i][i] + kRidge, "ridge diagonal overflow");
        for (std::size_t channel = 0; channel < kOutputDimension; ++channel)
            rhs[channel][i] /= count;
    }
    Matrix lower(dimension, std::vector<double>(dimension, 0.0));
    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = gram[i][j];
            for (std::size_t k = 0; k < j; ++k)
                value = finite(value - finite(lower[i][k] * lower[j][k], "Cholesky product overflow"),
                               "Cholesky accumulation overflow");
            if (i == j) {
                if (!(value > 0.0)) invalid("regularized Gram is not numerically positive definite");
                lower[i][j] = finite(std::sqrt(value), "invalid Cholesky diagonal");
            } else {
                lower[i][j] = finite(value / lower[j][j], "Cholesky division overflow");
            }
        }
    }
    coefficients_.assign(kOutputDimension, std::vector<double>(dimension, 0.0));
    double rhs_max_abs = 0.0;
    for (std::size_t channel = 0; channel < kOutputDimension; ++channel) {
        std::vector<double> intermediate(dimension, 0.0);
        for (std::size_t i = 0; i < dimension; ++i) {
            double value = rhs[channel][i];
            for (std::size_t j = 0; j < i; ++j)
                value = finite(value - finite(lower[i][j] * intermediate[j], "forward solve product overflow"),
                               "forward solve accumulation overflow");
            intermediate[i] = finite(value / lower[i][i], "forward solve division overflow");
        }
        for (std::size_t reverse = 0; reverse < dimension; ++reverse) {
            const std::size_t i = dimension - 1 - reverse;
            double value = intermediate[i];
            for (std::size_t j = i + 1; j < dimension; ++j)
                value = finite(value - finite(lower[j][i] * coefficients_[channel][j], "back solve product overflow"),
                               "back solve accumulation overflow");
            coefficients_[channel][i] = finite(value / lower[i][i], "back solve division overflow");
        }
        for (std::size_t i = 0; i < dimension; ++i) {
            const double residual = finite(checked_dot(gram[i], coefficients_[channel]) - rhs[channel][i],
                                           "normal equation residual overflow");
            normal_equation_residual_max_abs_ = std::max(normal_equation_residual_max_abs_, std::abs(residual));
            rhs_max_abs = std::max(rhs_max_abs, std::abs(rhs[channel][i]));
        }
    }
    normal_equation_residual_relative_ = normal_equation_residual_max_abs_ / std::max(1.0, rhs_max_abs);
}

LightContextFrame LightPredictionModel::predict(std::span<const double> state, double request) const {
    const auto phi = design(mode_, state, request);
    LightContextFrame result{};
    for (std::size_t channel = 0; channel < kOutputDimension; ++channel)
        result[channel] = checked_dot(coefficients_[channel], phi);
    return result;
}
std::vector<double> LightPredictionModel::advance(std::span<const double> state, double request) const {
    const auto next = predict(state, request);
    const std::size_t taps = light_prediction_taps(mode_);
    std::vector<double> result;
    result.reserve(state.size());
    result.insert(result.end(), next.begin(), next.end());
    result.insert(result.end(), state.begin(), state.begin() + kOutputDimension * (taps - 1));
    if (light_prediction_uses_input(mode_) && taps > 1) {
        result.push_back(request);
        result.insert(result.end(), state.begin() + kOutputDimension * taps,
                      state.begin() + kOutputDimension * taps + taps - 2);
    }
    return result;
}
std::vector<LightContextFrame> LightPredictionModel::forecast(
    std::span<const double> initial, std::span<const double> requests) const {
    finite_vector(initial, light_prediction_state_dimension(mode_));
    finite_vector(requests, requests.size());
    std::vector<double> state(initial.begin(), initial.end());
    std::vector<LightContextFrame> result;
    result.reserve(requests.size());
    for (double request : requests) {
        state = advance(state, request);
        LightContextFrame output{};
        std::copy_n(state.begin(), kOutputDimension, output.begin());
        result.push_back(output);
    }
    return result;
}
Json LightPredictionModel::serialize() const {
    const std::size_t dimension = light_prediction_state_dimension(mode_);
    const std::size_t taps = light_prediction_taps(mode_);
    const bool input = light_prediction_uses_input(mode_);
    Matrix a(dimension, std::vector<double>(dimension, 0.0));
    Matrix b(dimension, std::vector<double>(1, 0.0));
    Matrix c(kOutputDimension, std::vector<double>(dimension, 0.0));
    Matrix d(kOutputDimension, std::vector<double>(1, 0.0));
    std::vector<double> bias(dimension, 0.0);
    for (std::size_t channel = 0; channel < kOutputDimension; ++channel) {
        bias[channel] = coefficients_[channel][0];
        std::copy_n(coefficients_[channel].begin() + 1, dimension, a[channel].begin());
        if (input) b[channel][0] = coefficients_[channel].back();
        c[channel][channel] = 1.0;
    }
    for (std::size_t row = kOutputDimension; row < kOutputDimension * taps; ++row)
        a[row][row - kOutputDimension] = 1.0;
    if (input && taps > 1) {
        b[kOutputDimension * taps][0] = 1.0;
        for (std::size_t tap = 1; tap < taps - 1; ++tap)
            a[kOutputDimension * taps + tap][kOutputDimension * taps + tap - 1] = 1.0;
    }
    return Json{{"schema_version", "light_prediction_model_v1"},
        {"mode", light_prediction_mode_id(mode_)}, {"taps", taps}, {"uses_input", input},
        {"state_dimension", dimension}, {"output_dimension", kOutputDimension},
        {"coefficient_order", "[bias,state...,current_request_if_enabled]"},
        {"coefficients", coefficients_},
        {"state_space", Json{{"A", a}, {"B", b}, {"C", c}, {"D", d}, {"c", bias},
            {"equations", "z_next=A*z+B*r+c; y=C*z+D*r"}}},
        {"fit", Json{{"training_rows", training_.size()}, {"lambda", kRidge},
            {"normal_equation_residual_max_abs", normal_equation_residual_max_abs_},
            {"normal_equation_residual_relative", normal_equation_residual_relative_}}}};
}
} // namespace droid
