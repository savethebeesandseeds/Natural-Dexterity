#include "droid/light_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;
using Mode = droid::LightContextMode;
using Frame = droid::LightContextFrame;
using Example = droid::LightContextExample;
void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void close(double a, double b, std::string_view message) {
    require(std::isfinite(a) && std::abs(a - b) < 1e-12, message);
}
template<class Function> void rejects(Function&& function, std::string_view message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}
Json observation(bool valid = true, double epoch = 0.0, bool renamed = false) {
    const auto sample = [&](std::string id, std::string family, double period, double latency,
                            double sampled, double age, std::uint64_t sequence, Json channels) {
        return Json{{"module_id", std::move(id)}, {"family_id", std::move(family)}, {"valid", valid},
            {"sequence", valid ? sequence : 0},
            {"sample_time_s", valid ? Json(sampled + epoch) : Json(nullptr)},
            {"delivered_time_s", valid ? Json(sampled + epoch + latency) : Json(nullptr)},
            {"age_s", valid ? Json(age) : Json(nullptr)}, {"sample_period_s", period},
            {"latency_s", latency}, {"observations", std::move(channels)}};
    };
    Json imu = sample(renamed ? "renamed-imu" : "imu-0001", "imu_6axis_v0", .01, .006, .01, .01,
        renamed ? 102 : 2, Json{{"specific_force_m_s2", valid ? Json::array({100.0, -5.0, 9.81}) : Json::array({0.0,0.0,0.0})},
        {"angular_velocity_rad_s", valid ? Json::array({1.0,20.0,-3.0}) : Json::array({0.0,0.0,0.0})}});
    Json light = sample(renamed ? "renamed-light" : "light-0001", "ambient_light_v0", .1, .02, 0, .02,
        renamed ? 101 : 1, Json{{"illuminance_lux", valid ? 420.0 : 0.0}, {"saturated", false}});
    Json motor{{"module_id", renamed ? "renamed-motor" : "motor-0001"},
        {"sku_id", "rotary_dc_gearmotor_v0"}, {"valid", true},
        {"feedback", Json{{"position_rad", .4}, {"velocity_rad_s", 1.2}, {"current_a", .15},
            {"bus_voltage_v", 7.4}, {"temperature_c", 25.0}, {"output_torque_nm", .1},
            {"load_impedance_nm_s_per_rad", .1}, {"stuck_score", 0.0}, {"stuck", false}, {"fault_flags", Json::array()}}}};
    return Json{{"schema_version", "construction_observation_v2"},
        {"reward_sensors", renamed ? Json::array({light, imu}) : Json::array({imu, light})},
        {"actuator_feedback", Json::array({motor})}};
}
void projection_boundary() {
    const Json raw = observation();
    const Json before = raw;
    const Frame projected = droid::project_light_context_frame(raw, .3);
    require(raw == before, "projection mutated input observation");
    close(projected[0], 2.0, "effort was clipped or used the wrong divisor");
    close(projected[1], std::sin(.4), "encoder sine wrong");
    close(projected[2], std::cos(.4), "encoder cosine wrong");
    close(projected[3], .12, "shaft velocity normalization wrong");
    close(projected[4], .3, "current normalization wrong");
    close(projected[5], 2.0, "specific force was clipped or misnormalized");
    close(projected[6], -.1, "specific force order wrong");
    close(projected[7], 9.81/50, "specific force gravity channel wrong");
    close(projected[8], .1, "gyro order wrong");
    close(projected[9], 2.0, "gyro was clipped");
    close(projected[10], -.3, "gyro third axis wrong");
    close(projected[11], .42, "lux normalization wrong");
    require(projected == droid::project_light_context_frame(observation(true, 10, true), .3),
            "timestamps, sequence, IDs or sensor array order leaked into features");
    rejects([&] { (void)droid::project_light_context_frame(observation(false), 0); }, "undelivered sensors accepted");
    rejects([&] { (void)droid::project_light_context_frame(raw, .36); }, "effort outside common envelope accepted");
    rejects([&] { (void)droid::project_light_context_frame(raw, std::numeric_limits<double>::quiet_NaN()); }, "nonfinite effort accepted");
    const auto bad = [&](auto mutate) {
        Json input = raw; mutate(input);
        rejects([&] { (void)droid::project_light_context_frame(input, 0); }, "malformed or privileged observation accepted");
    };
    bad([](Json& value) { value["sun"] = Json::array({1,0,1}); });
    bad([](Json& value) { value["source"] = "A"; });
    bad([](Json& value) { value["body"] = "body-0"; });
    bad([](Json& value) { value["reward"] = 1.0; });
    bad([](Json& value) { value["reward_sensors"][0]["observations"]["world_orientation"] = 0; });
    bad([](Json& value) { value["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"voltage_limited"}); });
    bad([](Json& value) { value["reward_sensors"][0]["age_s"] = .1; });
    bad([](Json& value) { value["reward_sensors"][1]["observations"]["illuminance_lux"] = std::numeric_limits<double>::infinity(); });
    bad([](Json& value) { value["reward_sensors"].erase(0); });
}
void causal_windows_and_light_ablation() {
    std::array<Frame, 11> frames{};
    for (std::size_t t = 0; t < frames.size(); ++t)
        for (std::size_t f = 0; f < frames[t].size(); ++f) frames[t][f] = t * 100 + f;
    const auto history = droid::light_context_features(Mode::history, frames);
    require(history.size() == 132, "history dimension");
    for (std::size_t tap = 0; tap < 11; ++tap)
        for (std::size_t f = 0; f < 12; ++f)
            close(history[tap * 12 + f], frames[10 - tap][f], "history is not latest-to-oldest causal flattening");
    const auto instant = droid::light_context_features(Mode::instantaneous, frames);
    require(std::equal(instant.begin(), instant.end(), frames.back().begin()), "instantaneous used earlier frame");
    const auto light = droid::light_context_features(Mode::light_only, frames);
    require(light.size() == 1 && light[0] == frames.back()[11], "light-only includes extra channels");
    const auto no_light = droid::light_context_features(Mode::no_light_history, frames);
    require(no_light.size() == 121, "no-light history dimension");
    for (std::size_t tap = 0; tap < 11; ++tap)
        for (std::size_t f = 0; f < 11; ++f)
            close(no_light[tap * 11 + f], frames[10 - tap][f], "light ablation changed mechanical coordinate order");
    for (auto& frame : frames) frame[11] += 100000;
    require(droid::light_context_features(Mode::no_light_history, frames) == no_light,
            "light survived no-light history ablation");
    require(droid::light_context_features(Mode::history, frames) != history, "history discarded actual light");
    rejects([&] { (void)droid::light_context_features(Mode::history, std::span(frames).first(10)); }, "incomplete history accepted");
    rejects([&] { (void)droid::light_context_features(Mode::instantaneous, std::span<const Frame>{}); }, "empty instantaneous window accepted");
    frames[0][3] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { (void)droid::light_context_features(Mode::history, frames); }, "nonfinite history accepted");
    rejects([&] { (void)droid::light_context_dimension(static_cast<Mode>(99)); }, "unknown mode accepted");
}
void separation_and_frozen_model() {
    std::vector<Example> training;
    for (int i = 0; i < 5; ++i) training.push_back({{1.0}, true});
    for (int i = 0; i < 5; ++i) training.push_back({{-1.0}, false});
    const droid::LightContextDecoder model(Mode::light_only, training);
    const Json saved = model.serialize();
    const std::array<double, 1> a{1}, b{-1};
    const auto pa = model.predict(a), pb = model.predict(b);
    close(pa.probability_a, 6.0/7.0, "positive class probability wrong");
    close(pb.probability_a, 1.0/7.0, "negative class probability wrong");
    require(pa.predicts_a && !pb.predicts_a && pa.neighbor_count == 5 && pb.neighbor_count == 5,
            "separated classes not decoded");
    training[0].features[0] = -999;
    training[0].is_a = false;
    require(model.serialize() == saved, "fit retained mutable caller training storage");
    for (int i = 0; i < 4; ++i) close(model.predict(a).probability_a, pa.probability_a, "prediction changed model state");
    require(model.serialize() == saved, "prediction mutated frozen model");
    require(saved.at("dimension") == 1 && saved.at("training").size() == 10 &&
            saved.at("training")[0].at("features") == Json::array({1.0}),
            "numeric model archive does not represent actual training data");
}
void cutoff_ties_and_order_symmetry() {
    std::vector<Example> training;
    for (int i = 0; i < 4; ++i) training.push_back({{0.0}, i % 2 == 0});
    for (int i = 0; i < 6; ++i) training.push_back({{i % 2 ? -1.0 : 1.0}, i < 4});
    training.push_back({{2.0}, true});
    const std::array<double, 1> query{0};
    const droid::LightContextDecoder original(Mode::light_only, training);
    const auto prediction = original.predict(query);
    require(prediction.neighbor_count == 10, "exact fifth-distance ties were truncated to five");
    close(prediction.squared_mean_distance_cutoff, 1.0, "wrong fifth-distance cutoff");
    close(prediction.probability_a, 7.0/12.0, "tie-inclusive vote count wrong");
    std::reverse(training.begin(), training.end());
    const droid::LightContextDecoder reversed(Mode::light_only, training);
    const auto reverse_prediction = reversed.predict(query);
    require(reverse_prediction.probability_a == prediction.probability_a &&
            reverse_prediction.neighbor_count == prediction.neighbor_count &&
            reverse_prediction.squared_mean_distance_cutoff == prediction.squared_mean_distance_cutoff,
            "training row order changed tie voting");
    for (auto& example : training) example.is_a = !example.is_a;
    const droid::LightContextDecoder complemented(Mode::light_only, training);
    close(complemented.predict(query).probability_a, 1.0 - prediction.probability_a,
          "label complement violated smoothed probability symmetry");
}
void paired_no_light_negative_control() {
    std::vector<Example> training;
    for (int i = 0; i < 6; ++i) {
        std::array<Frame, 11> a{}, b{};
        for (std::size_t t = 0; t < 11; ++t) {
            a[t][0] = b[t][0] = i * .1;
            a[t][11] = .9; b[t][11] = .1;
        }
        const auto fa = droid::light_context_features(Mode::no_light_history, a);
        const auto fb = droid::light_context_features(Mode::no_light_history, b);
        require(fa == fb, "paired no-light examples differ");
        training.push_back({fa, true}); training.push_back({fb, false});
    }
    const droid::LightContextDecoder model(Mode::no_light_history, training);
    for (const auto& example : training) {
        const auto prediction = model.predict(example.features);
        close(prediction.probability_a, .5, "identical opposite-label pairs acquired false decodability");
        require(prediction.predicts_a && prediction.neighbor_count % 2 == 0,
                "balanced exact ties were not retained with the declared A tie decision");
    }
}
void finite_dimensions_and_incomplete_fit() {
    std::vector<Example> examples(5, Example{{0.0}, true});
    rejects([&] { droid::LightContextDecoder model(Mode::light_only, std::span(examples).first(4)); }, "fewer than five training examples accepted");
    examples[0].features.push_back(1);
    rejects([&] { droid::LightContextDecoder model(Mode::light_only, examples); }, "wrong training dimension accepted");
    examples[0].features = {std::numeric_limits<double>::infinity()};
    rejects([&] { droid::LightContextDecoder model(Mode::light_only, examples); }, "nonfinite training example accepted");
    examples[0].features = {0};
    const droid::LightContextDecoder model(Mode::light_only, examples);
    const Json before = model.serialize();
    const std::array<double, 2> wrong{0,1};
    const std::array<double, 1> nan{std::numeric_limits<double>::quiet_NaN()};
    rejects([&] { (void)model.predict(wrong); }, "wrong query dimension accepted");
    rejects([&] { (void)model.predict(nan); }, "nonfinite query accepted");
    require(model.serialize() == before, "rejected query changed model");
    for (auto& example : examples) example.features[0] = std::numeric_limits<double>::max();
    const droid::LightContextDecoder large(Mode::light_only, examples);
    const std::array<double, 1> negative{-std::numeric_limits<double>::max()};
    rejects([&] { (void)large.predict(negative); }, "nonfinite squared distance produced a prediction");
}
} // namespace
int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"strict local projection", projection_boundary},
        {"causal windows and light ablation", causal_windows_and_light_ablation},
        {"separation and immutable model", separation_and_frozen_model},
        {"cutoff ties and row-order symmetry", cutoff_ties_and_order_symmetry},
        {"paired no-light negative control", paired_no_light_negative_control},
        {"finite dimensions and incomplete fit", finite_dimensions_and_incomplete_fit}};
    for (const auto& [name, test] : tests) {
        try { test(); }
        catch (const std::exception& error) { std::cerr << "FAIL " << name << ": " << error.what() << '\n'; return 1; }
        std::cout << "PASS " << name << '\n';
    }
    std::cout << tests.size() << " native light context tests passed\n";
    return 0;
}