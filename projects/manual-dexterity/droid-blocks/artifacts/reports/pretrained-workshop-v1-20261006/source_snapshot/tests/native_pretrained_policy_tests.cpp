#include "droid/pretrained_policy.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
std::size_t checks{};
void check(bool valid, const std::string& message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template <typename Function>
void rejects(Function&& function, const std::string& message) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
Json observation() {
    return Json{{"schema_version", "construction_observation_v2"},
        {"reward_sensors", Json::array({
            Json{{"family_id", "imu_6axis_v0"}, {"valid", true},
                 {"observations", Json{{"specific_force_m_s2", Json::array({50., -100., 500.})},
                                        {"angular_velocity_rad_s", Json::array({10., -20., 60.})}}}},
            Json{{"family_id", "ambient_light_v0"}, {"valid", true},
                 {"observations", Json{{"illuminance_lux", 1250.}}}}
        })},
        {"actuator_feedback", Json::array({Json{{"feedback", Json{
             {"position_rad", std::acos(-1.) / 2.}, {"velocity_rad_s", 20.}, {"current_a", -.25}}}}})}};
}

std::string tensor_bytes(const Json& data) {
    std::string result;
    for (const auto& value : data) {
        const auto bits = std::bit_cast<std::uint32_t>(value.get<float>());
        for (unsigned shift = 0; shift < 32; shift += 8)
            result.push_back(static_cast<char>((bits >> shift) & 255U));
    }
    return result;
}

class Scratch final {
public:
    Scratch() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("droid-pretrained-policy-test-" + std::to_string(suffix));
        if (!std::filesystem::create_directory(root)) throw std::runtime_error("Cannot create exclusive policy test scratch");
        file = root / "actor.json";
    }
    ~Scratch() {
        // Only these two task-created paths are removed; no recursive cleanup.
        std::error_code unused;
        std::filesystem::remove(file, unused);
        std::filesystem::remove(root, unused);
    }
    void write(const Json& value) const {
        std::ofstream stream(file);
        if (!stream) throw std::runtime_error("Cannot write policy mutation fixture");
        stream << value.dump(2) << '\n';
    }
    std::filesystem::path root, file;
};

void input_tests() {
    const Json obs = observation();
    const auto frame = droid::PretrainedHistory::frame(obs, .075, -.15, .1);
    const std::array<float, 17> expected{.5f, 1.f, 0.f, 2.f, -.5f,
                                       1.f, -2.f, 5.f, 1.f, -2.f, 5.f,
                                       1.25f, -1.f, .5f, 1.f, 1.f, 1.f};
    for (std::size_t index = 0; index < expected.size(); ++index)
        check(std::abs(frame[index] - expected[index]) < 1e-7f, "Local channel scaling/order differs");
    Json masked = obs;
    for (auto& value : masked["reward_sensors"]) {
        value["valid"] = false;
        value.erase("observations");
    }
    const auto absent = droid::PretrainedHistory::frame(masked, 0., 0., 0.);
    for (std::size_t index = 5; index <= 11; ++index) check(absent[index] == 0.f, "Invalid sensor value entered frame");
    check(absent[14] == 0.f && absent[15] == 0.f && absent[16] == 1.f, "Validity/presence masks differ");
    Json diagnostic = obs;
    diagnostic["assembly"] = Json{{"segments", Json::array({999, 888})}, {"powered_hinge", 1}};
    diagnostic["sun"] = Json{{"position_m", Json::array({999., 888., 777.})}};
    diagnostic["passive_joint"] = 12345.;
    check(droid::PretrainedHistory::frame(diagnostic, .075, -.15, .1) == frame,
          "Nonlocal diagnostic fields entered actor input");
    droid::PretrainedHistory uninitialized;
    rejects([&] { (void)uninitialized.values(); }, "Uninitialized history was exposed");
    rejects([&] { uninitialized.append(obs, 0., 0., 0.); }, "Uninitialized history accepted a macro decision");
    droid::PretrainedHistory history(obs);
    check(history.count() == 1, "Initial history count differs");
    check(std::all_of(history.values().begin(), history.values().begin() + 11 * 17,
                      [](float value) { return value == 0.f; }), "Initial absent frames are not zero padded");
    check(history.values()[203] == 1.f && history.values()[204 - 17] == 0.f,
          "Initial frame lacks presence or has previous action experience");
    for (int step = 1; step <= 13; ++step) history.append(obs, 0., 0., step * .01);
    check(history.count() == 12, "History count does not cap at twelve");
    for (int frame_index = 0; frame_index < 12; ++frame_index) {
        const float reward = static_cast<float>((frame_index + 2) * .01 / .2);
        check(history.values()[frame_index * 17 + 13] == reward, "Ring history does not retain oldest-to-newest frames");
    }
    check(history.diagnostics().at("completed_decisions") == 13, "History completed-decision diagnostic differs");
    const auto before = history.values();
    Json invalid = obs;
    invalid["actuator_feedback"][0]["feedback"]["current_a"] = "bad";
    rejects([&] { history.append(invalid, 0., 0., 0.); }, "Invalid local feedback was accepted");
    check(history.values() == before, "Rejected append corrupted existing history");
    rejects([&] { history.reset(invalid); }, "Invalid reset observation was accepted");
    check(history.values() == before && history.count() == 12, "Rejected reset corrupted existing history");
    rejects([&] { (void)droid::PretrainedHistory::frame(obs, std::numeric_limits<double>::infinity(), 0., 0.); },
            "Nonfinite macro effort was accepted");
    history.reset(masked);
    check(history.count() == 1 && history.diagnostics().at("completed_decisions") == 0,
          "Reset retained old-body macro history");
    check(history.diagnostics().at("input_float32_le_sha256").get<std::string>().size() == 64,
          "History diagnostic lacks a byte commitment");
}

void categorical_tests() {
    using Policy = droid::PretrainedPolicy;
    const droid::PretrainedOutput probabilities{.25f, .5f, .25f};
    check(Policy::categorical(probabilities, 0.) == 0, "Categorical start endpoint differs");
    check(Policy::categorical(probabilities, std::nextafter(.25, 0.)) == 0, "CDF left boundary differs");
    check(Policy::categorical(probabilities, .25) == 1, "CDF equality did not choose following action");
    check(Policy::categorical(probabilities, .75) == 2, "Second CDF equality differs");
    check(Policy::categorical(probabilities, std::nextafter(1., 0.)) == 2, "Last positive mass endpoint differs");
    check(Policy::categorical({0.f, .5f, .5f}, 0.) == 1, "Zero first mass was selected");
    check(Policy::categorical({.5f, .5f, 0.f}, std::nextafter(1., 0.)) == 1, "Zero final mass was selected");
    check(Policy::categorical({1.f, 1.f, 1.f}, 1. / 3.) == 1, "Double normalization differs");
    rejects([&] { (void)Policy::categorical(probabilities, -1.); }, "Negative uniform was accepted");
    rejects([&] { (void)Policy::categorical(probabilities, 1.); }, "Closed upper endpoint was accepted");
    rejects([&] { (void)Policy::categorical(probabilities, std::numeric_limits<double>::quiet_NaN()); }, "NaN uniform was accepted");
    rejects([&] { (void)Policy::categorical({0.f, 0.f, 0.f}, .5); }, "Empty probability mass was accepted");
    rejects([&] { (void)Policy::categorical({-.1f, .5f, .5f}, .5); }, "Negative probability was accepted");
    rejects([&] { (void)Policy::categorical({1.1f, 0.f, 0.f}, .5); }, "Probability above one was accepted");
    rejects([&] { (void)Policy::categorical({std::numeric_limits<float>::infinity(), 0.f, 0.f}, .5); }, "Nonfinite probability was accepted");
}

void artifact_tests(const std::filesystem::path& artifact) {
    const droid::PretrainedPolicy policy(artifact);
    const Json identity = policy.metadata();
    check(identity.at("policy_id") == "pooled-seed101-v1" && identity.at("parameters") == 17475,
          "Canonical actor identity differs");
    check(identity.at("weight_updates") == 0 && identity.at("learning") == false,
          "Frozen actor metadata permits learning");
    droid::PretrainedInput input{};
    const auto logits = policy.logits(input);
    const auto probabilities = policy.probabilities(input);
    check(logits == policy.logits(input), "Frozen inference mutates between calls");
    double total = 0.;
    for (float value : probabilities) { check(std::isfinite(value) && value > 0.f, "Invalid canonical softmax probability"); total += value; }
    check(std::abs(total - 1.) < 1e-6, "Softmax probability sum differs");
    check(policy.sample(input, .5) == droid::PretrainedPolicy::categorical(probabilities, .5),
          "Sample does not use exported softmax law");
    input[0] = 5.1f;
    rejects([&] { (void)policy.logits(input); }, "Unclipped actor input was accepted");
    input[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)policy.probabilities(input); }, "Nonfinite actor input was accepted");
    check(policy.metadata() == identity, "Inference changed artifact identity");
    std::ifstream stream(artifact);
    if (!stream) throw std::runtime_error("Cannot read canonical actor test fixture");
    const Json original = Json::parse(stream);
    Scratch scratch;
    // Formatting is not authority: equivalent lossless bytes/provenance remain valid.
    scratch.write(original);
    const droid::PretrainedPolicy reformatted(scratch.file);
    check(reformatted.logits({}) == logits, "JSON reformatting changed canonical inference");
    auto reject_mutation = [&](Json changed, const std::string& reason) {
        scratch.write(changed);
        rejects([&] { const droid::PretrainedPolicy invalid(scratch.file); }, reason);
    };
    Json changed = original;
    changed["dimensions"] = Json::array({204, 64, 63, 3});
    reject_mutation(changed, "Wrong actor dimensions were accepted");
    changed = original;
    changed["provenance"]["checkpoint"]["sha256"] = std::string(64, '0');
    reject_mutation(changed, "Changed checkpoint provenance was accepted");
    changed = original;
    changed["action_requests"][0] = -.2;
    reject_mutation(changed, "Changed native request magnitudes were accepted");
    changed = original;
    changed["tensors"][0]["data"][0] = 0.;
    reject_mutation(changed, "Changed actor bytes with old labels were accepted");
    // Forge self-reported tensor checksums too; trusted actor pins still reject.
    changed["tensors"][0]["sha256"] = droid::sha256_hex(tensor_bytes(changed["tensors"][0]["data"]));
    std::string forged_bytes;
    for (const auto& tensor : changed["tensors"]) forged_bytes += tensor_bytes(tensor["data"]);
    changed["actor_bytes_sha256"] = droid::sha256_hex(forged_bytes);
    reject_mutation(changed, "Forged self-reported tensor hashes bypassed the trusted actor pin");
    changed = original;
    changed["tensors"][1]["shape"] = Json::array({3, 63});
    reject_mutation(changed, "Wrong tensor dimensions were accepted");
    changed = original;
    changed["tensors"][0]["data"][0] = "NaN";
    reject_mutation(changed, "Nonnumeric actor weight was accepted");
    changed = original;
    changed["tensors"][0]["data"][0] = .1;
    reject_mutation(changed, "Lossy float64 actor weight was accepted");
    changed = original;
    changed["stochastic_contract"]["torch_seeded_trajectory_equivalence"] = true;
    reject_mutation(changed, "False Torch trajectory equivalence was accepted");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto artifact = argc > 1 ? std::filesystem::path(argv[1]) :
            std::filesystem::path(DROID_MODULE_CATALOG_PATH).parent_path().parent_path() /
                "artifacts/pretrained/pooled-seed101-v1.json";
        input_tests();
        categorical_tests();
        artifact_tests(artifact);
        std::cout << "pretrained policy: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained policy test: " << error.what() << '\n';
        return 1;
    }
}
