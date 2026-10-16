#include "droid/light_prediction.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;
using Mode = droid::LightPredictionMode;
using Frame = droid::LightContextFrame;
using Example = droid::LightPredictionExample;
using Model = droid::LightPredictionModel;
void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void close(double actual, double expected, std::string_view message, double tolerance = 1e-11) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}
template<class Function> void rejects(Function&& function, std::string_view message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}
// Balanced, mutually orthogonal deterministic regressors give an independent
// analytic ridge solution: every nonbias coefficient is divided by 1+lambda.
std::vector<Example> synthetic(Mode mode) {
    const std::size_t dimension = droid::light_prediction_state_dimension(mode);
    std::size_t count = 1;
    while (count <= dimension + 1) count *= 2;
    std::vector<Example> rows;
    for (std::size_t row = 0; row < count; ++row) {
        Example example;
        for (std::size_t col = 0; col < dimension; ++col)
            example.state.push_back(std::popcount(row & (col + 1)) % 2 == 0 ? 1.0 : -1.0);
        example.requested_input = std::popcount(row & (dimension + 1)) % 2 == 0 ? 1.0 : -1.0;
        for (std::size_t channel = 0; channel < 12; ++channel) {
            example.next_output[channel] = .1 * (channel + 1) +
                (.2 + .01 * channel) * example.state[channel] +
                .1 * example.state[(channel + 1) % 12];
            if (dimension > 12) example.next_output[channel] += .05 * example.state[12 + channel];
            if (droid::light_prediction_uses_input(mode))
                example.next_output[channel] += .3 * example.requested_input;
        }
        rows.push_back(example);
    }
    return rows;
}
void analytic_mimo_ridge_recovery() {
    auto rows = synthetic(Mode::current_input);
    Model model(Mode::current_input, rows);
    const auto saved = model.serialize();
    require(saved.at("fit").at("training_rows") == rows.size(), "wrong evidence row count");
    close(saved.at("fit").at("lambda"), .001, "ridge differs from fixed declared value");
    require(saved.at("fit").at("normal_equation_residual_max_abs").get<double>() < 1e-13,
            "normal equation residual is too large");
    for (std::size_t channel = 0; channel < 12; ++channel) {
        const auto coefficients = saved.at("coefficients").at(channel).get<std::vector<double>>();
        close(coefficients[0], .1 * (channel + 1), "intercept was regularized");
        for (std::size_t col = 0; col < 12; ++col) {
            const double actual_coefficient = (col == channel ? .2 + .01 * channel : 0.0) +
                (col == (channel + 1) % 12 ? .1 : 0.0);
            close(coefficients[col + 1], actual_coefficient / 1.001, "MIMO state coefficient is incorrect");
        }
        close(coefficients.back(), .3 / 1.001, "known command coefficient is incorrect");
    }
    std::vector<double> held_out(12);
    for (std::size_t i = 0; i < held_out.size(); ++i) held_out[i] = .17 * i - .6;
    const auto prediction = model.predict(held_out, -.7);
    for (std::size_t channel = 0; channel < 12; ++channel) {
        const double expected = .1 * (channel + 1) + ((.2 + .01 * channel) * held_out[channel] +
            .1 * held_out[(channel + 1) % 12] + .3 * -.7) / 1.001;
        close(prediction[channel], expected, "held-out affine prediction differs from analytic solution");
    }
}
void causal_state_and_companion_matrices() {
    std::array<Frame, 11> frames{};
    std::array<double, 10> requests{};
    for (std::size_t t = 0; t < frames.size(); ++t)
        for (std::size_t c = 0; c < 12; ++c) frames[t][c] = .01 * (100 * t + c);
    for (std::size_t t = 0; t < requests.size(); ++t) requests[t] = .1 * t;
    auto state = droid::light_prediction_state(Mode::history_input, frames, requests);
    require(state.size() == 142, "history state dimension");
    for (std::size_t tap = 0; tap < 11; ++tap)
        for (std::size_t c = 0; c < 12; ++c)
            close(state[12 * tap + c], frames[10 - tap][c], "output delay order is not causal latest first");
    for (std::size_t tap = 0; tap < 10; ++tap)
        close(state[132 + tap], requests[9 - tap], "request delay order is incorrect");
    Model model(Mode::history_input, synthetic(Mode::history_input));
    const auto saved = model.serialize();
    const auto& matrices = saved.at("state_space");
    const auto next = model.advance(state, -.4);
    const auto output = model.predict(state, -.4);
    for (std::size_t row = 0; row < state.size(); ++row) {
        double expected = matrices.at("c").at(row).get<double>() +
            matrices.at("B").at(row).at(0).get<double>() * -.4;
        for (std::size_t col = 0; col < state.size(); ++col)
            expected += matrices.at("A").at(row).at(col).get<double>() * state[col];
        close(next[row], expected, "explicit A/B/c and direct advancement disagree");
        if (row < 12) close(next[row], output[row], "predicted output not inserted at state head");
        else if (row < 132) close(next[row], state[row - 12], "output shift is incorrect");
        else if (row == 132) close(next[row], -.4, "current request not inserted into history");
        else close(next[row], state[row - 1], "request shift is incorrect");
    }
    for (std::size_t row = 0; row < 12; ++row) {
        double measured = matrices.at("D").at(row).at(0).get<double>() * -.4;
        for (std::size_t col = 0; col < state.size(); ++col)
            measured += matrices.at("C").at(row).at(col).get<double>() * state[col];
        close(measured, frames.back()[row], "C/D does not select the current measured output");
        close(matrices.at("D").at(row).at(0), 0.0, "current-output feedthrough should be zero");
    }
    require(droid::light_prediction_state(Mode::current_input,
            std::span<const Frame>(frames).last(1), {}).size() == 12, "current state dimension");
}
void input_ablation_and_freezing() {
    auto rows = synthetic(Mode::history_no_input);
    Model model(Mode::history_no_input, rows);
    const auto before = model.serialize();
    const auto state = rows[0].state;
    const auto prediction = model.predict(state, -1.0);
    require(prediction == model.predict(state, 1.0), "no-input prediction depends on current command");
    require(model.forecast(state, std::array<double, 3>{-1, 0, 1}) ==
            model.forecast(state, std::array<double, 3>{1, 1, -1}),
            "no-input recursive dynamics depend on commands");
    for (auto& row : rows) {
        row.state.assign(1, 999);
        row.requested_input = 999;
        row.next_output.fill(999);
    }
    require(model.serialize() == before, "caller mutation changed fitted evidence or model");
    require(model.predict(state, 0.0) == prediction, "prediction or forecast changed model weights");
    for (const auto& row : before.at("state_space").at("B"))
        close(row.at(0), 0.0, "no-input B matrix is not zero");
    std::array<Frame, 11> frames{};
    require(droid::light_prediction_state(Mode::history_no_input, frames, {}).size() == 132,
            "input ablation retained request history");
}
void recursive_forecast_without_teacher_forcing() {
    Model model(Mode::current_input, synthetic(Mode::current_input));
    const std::vector<double> initial(12, 0.2);
    const std::array<double, 4> commands{1, -.5, 0, 1};
    const auto recursive = model.forecast(initial, commands);
    require(recursive.size() == commands.size(), "forecast horizon is incorrect");
    auto state = initial;
    for (std::size_t step = 0; step < commands.size(); ++step) {
        const auto predicted = model.predict(state, commands[step]);
        require(recursive[step] == predicted, "forecast consumed something beyond its predicted state");
        state.assign(predicted.begin(), predicted.end());
    }
    // A hypothetical next measurement is intentionally inconsistent with the
    // first prediction. Teacher forcing must differ from the recursive result.
    auto disturbed_measurement = std::vector<double>(recursive[0].begin(), recursive[0].end());
    disturbed_measurement[0] += 3.0;
    const auto teacher_forced_second = model.predict(disturbed_measurement, commands[1]);
    require(std::abs(teacher_forced_second[0] - recursive[1][0]) > .5,
            "recursive rollout was silently equivalent to teacher forcing");
    require(initial == std::vector<double>(12, .2), "forecast mutated its initial state");
    require(model.forecast(initial, {}).empty(), "empty horizon should return no predicted samples");
}
void unpenalized_bias_and_degenerate_evidence() {
    Example row;
    row.state.assign(12, 0.0);
    row.next_output.fill(2.75);
    Model model(Mode::current_input, std::span<const Example>(&row, 1));
    const auto output = model.predict(row.state, 0.0);
    for (double value : output) close(value, 2.75, "unpenalized bias was shrunk");
    const auto saved = model.serialize();
    for (const auto& coefficients : saved.at("coefficients"))
        for (std::size_t i = 1; i < coefficients.size(); ++i)
            close(coefficients.at(i), 0, "ridge failed with entirely unexcited regressors");
    // Duplicating every row leaves the mean-loss regularized fit unchanged.
    auto once = synthetic(Mode::current_input);
    auto twice = once;
    twice.insert(twice.end(), once.begin(), once.end());
    Model first(Mode::current_input, once), second(Mode::current_input, twice);
    const auto a = first.predict(once[0].state, .3), b = second.predict(once[0].state, .3);
    for (std::size_t c = 0; c < 12; ++c) close(a[c], b[c], "ridge was applied to sum loss instead of mean loss");
}
void malformed_and_overflow_rejection() {
    auto rows = synthetic(Mode::current_input);
    Model model(Mode::current_input, rows);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double maximum = std::numeric_limits<double>::max();
    rejects([&] { (void)Model(Mode::current_input, {}); }, "empty evidence accepted");
    rejects([&] { (void)droid::light_prediction_taps(static_cast<Mode>(99)); }, "unknown model accepted");
    rejects([&] { (void)model.predict(std::vector<double>(11), 0); }, "wrong prediction dimension accepted");
    rejects([&] { (void)model.predict(rows[0].state, nan); }, "nonfinite request accepted");
    auto malformed = rows;
    malformed[0].state[0] = inf;
    rejects([&] { (void)Model(Mode::current_input, malformed); }, "nonfinite training feature accepted");
    malformed = rows;
    malformed[0].next_output[0] = nan;
    rejects([&] { (void)Model(Mode::current_input, malformed); }, "nonfinite target accepted");
    malformed = rows;
    malformed[0].state.pop_back();
    rejects([&] { (void)Model(Mode::current_input, malformed); }, "malformed training dimension accepted");
    malformed = rows;
    malformed[0].state[0] = maximum;
    rejects([&] { (void)Model(Mode::current_input, malformed); }, "Gram overflow accepted");
    malformed = rows;
    for (auto& row : malformed) row.next_output[0] = maximum;
    rejects([&] { (void)Model(Mode::current_input, malformed); }, "RHS accumulation overflow accepted");
    malformed = rows;
    for (auto& row : malformed) row.next_output[0] = 100 * row.state[0];
    Model large_gain(Mode::current_input, malformed);
    auto extreme = rows[0].state;
    extreme[0] = maximum;
    rejects([&] { (void)large_gain.predict(extreme, 0); }, "prediction overflow accepted");
    rejects([&] { (void)model.forecast(rows[0].state, std::array<double, 2>{0, inf}); },
            "nonfinite future command accepted");
    rejects([&] { (void)model.forecast(std::vector<double>(11), {}); },
            "invalid initial state accepted for empty forecast");
    std::array<Frame, 11> frames{};
    std::array<double, 10> requests{};
    rejects([&] { (void)droid::light_prediction_state(Mode::history_input,
            std::span<const Frame>(frames).first(10), requests); }, "incomplete history accepted");
    rejects([&] { (void)droid::light_prediction_state(Mode::history_input, frames, {}); },
            "missing input history accepted");
    rejects([&] { (void)droid::light_prediction_state(Mode::history_no_input, frames, requests); },
            "no-input state retained requests");
    frames[4][7] = inf;
    rejects([&] { (void)droid::light_prediction_state(Mode::history_input, frames, requests); },
            "nonfinite causal frame accepted");
}
} // namespace

int main() {
    const std::vector<std::pair<std::string_view, std::function<void()>>> tests{
        {"analytic MIMO ridge recovery", analytic_mimo_ridge_recovery},
        {"causal state and companion matrices", causal_state_and_companion_matrices},
        {"input ablation and immutable model", input_ablation_and_freezing},
        {"recursive forecast without teacher forcing", recursive_forecast_without_teacher_forcing},
        {"unpenalized bias and degenerate evidence", unpenalized_bias_and_degenerate_evidence},
        {"malformed and overflowing inputs", malformed_and_overflow_rejection}
    };
    std::size_t passed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " light prediction test groups passed\n";
    return 0;
}
