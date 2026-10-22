#include "droid/environment.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Json = nlohmann::json;
using Actions = std::map<std::string, double>;
constexpr std::uint64_t kDevelopmentSeed = 6990021049862345368ULL;
constexpr std::array<std::string_view, 3> kAssemblies{
    droid::kNormalTransmissionAssembly, droid::kReversedTransmissionAssembly,
    droid::kDisconnectedTransmissionAssembly};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            message + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

Actions actions(double first, double rest = 0.0) {
    return {{"motor-0001", first}, {"motor-0002", rest},
            {"motor-0003", rest}, {"motor-0004", rest}};
}

const Json& eye(const Json& observation) {
    for (const Json& sample : observation.at("reward_sensors")) {
        if (sample.at("module_id") == "sensor-0001") return sample;
    }
    throw std::runtime_error("missing eye");
}

Json reset(droid::DroidEnvironment& env, std::string_view assembly, bool recording = false) {
    return env.reset(assembly, kDevelopmentSeed, recording,
                     droid::kBodyDiscoveryEpisodeProfile);
}

void test_nonintegral_clock_rejected(const std::filesystem::path& path) {
    std::ifstream source(path, std::ios::binary);
    require(source.good(), "must read canonical model for timing rejection fixture");
    std::string xml((std::istreambuf_iterator<char>(source)),
                    std::istreambuf_iterator<char>());
    constexpr std::string_view original = "timestep=\"0.002\"";
    const std::size_t offset = xml.find(original);
    require(offset != std::string::npos, "fixture must find canonical timestep");
    xml.replace(offset, original.size(), "timestep=\"0.003\"");
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto fixture = std::filesystem::temp_directory_path() /
        ("droid-body-nonintegral-clock-" + std::to_string(suffix) + ".xml");
    require(!std::filesystem::exists(fixture), "temporary fixture must be new");
    struct RemoveFixture {
        std::filesystem::path path;
        ~RemoveFixture() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    } cleanup{fixture};
    {
        std::ofstream output(fixture, std::ios::binary);
        output << xml;
        require(output.good(), "must write temporary timing fixture");
    }
    droid::DroidEnvironment env(fixture);
    (void)env.reset("demo_rover_v0", kDevelopmentSeed, false,
                    droid::kLightSearch1dEpisodeProfile);
    const Json before = env.visualization_state();
    bool rejected = false;
    try { (void)reset(env, kAssemblies[0]); }
    catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("integral physics steps") !=
            std::string::npos;
    }
    require(rejected, "successor must reject clocks that would round latency early");
    require(env.visualization_state() == before,
            "clock rejection must preserve model, active episode and state");
    const Json transition = env.step(actions(0.0), 0.003);
    require(!transition.at("terminated").get<bool>() &&
            transition.at("info").at("episode_profile") == droid::kLightSearch1dEpisodeProfile,
            "original episode must remain usable after rejected body reset");
}

void test_opt_in_and_restoration(const std::filesystem::path& path) {
    droid::DroidEnvironment env(path);
    droid::DroidEnvironment reference(path);
    const Json frozen_spec = env.spec();
    const Json descriptor = env.body_experiment_spec();
    require(descriptor.at("supported_assembly_ids").size() == 3,
            "successor descriptor must enumerate three explicit assemblies");
    for (const auto& [assembly, profile] : std::vector<std::pair<std::string, std::string>>{
        {std::string(kAssemblies[0]), "light_search_1d_v1"},
        {"demo_rover_v0", std::string(droid::kBodyDiscoveryEpisodeProfile)},
        {"arbitrary_body", std::string(droid::kBodyDiscoveryEpisodeProfile)}}) {
        bool rejected = false;
        try { (void)env.reset(assembly, kDevelopmentSeed, false, profile); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "successor requires both explicit profile and assembly");
    }
    for (const std::string_view assembly : kAssemblies) {
        const Json result = reset(env, assembly);
        require(env.spec() == frozen_spec, "legacy environment spec must remain exact");
        require(result.at("observation").size() == 2,
                "body identity must not enter observation");
        for (int step = 0; step < 20; ++step) (void)env.step(actions(0.2), 0.02);
        const Json actual_reset = env.reset("demo_rover_v0", kDevelopmentSeed,
                                          false, droid::kLightSearch1dEpisodeProfile);
        const Json expected_reset = reference.reset("demo_rover_v0", kDevelopmentSeed,
                                                   false, droid::kLightSearch1dEpisodeProfile);
        require(actual_reset == expected_reset, "legacy reset must restore exact observation");
        require(!env.visualization_state().contains("body_experiment_physics"),
                "legacy state must not acquire successor diagnostics");
        for (int step = 0; step < 35; ++step) {
            require(env.step(actions(0.15, 0.15), 0.02) ==
                    reference.step(actions(0.15, 0.15), 0.02),
                    "restored legacy transition must match untouched native simulation");
            require(env.visualization_state() == reference.visualization_state(),
                    "restored legacy visualization must match untouched simulation");
        }
    }
}

void test_sensor_clock(const std::filesystem::path& path) {
    droid::DroidEnvironment env(path);
    const Json initial = reset(env, kAssemblies[0], true);
    const Json& initial_eye = eye(initial.at("observation"));
    require(!initial_eye.at("valid").get<bool>() && initial_eye.at("sequence") == 0 &&
            initial_eye.at("sample_time_s").is_null(),
            "first light sample must wait for physical delivery latency");
    std::map<std::uint64_t, double> sampled_lux;
    sampled_lux[0] = env.visualization_state().at("body_experiment_physics")
                        .at("instantaneous_illuminance_lux");
    for (std::uint64_t tick = 1; tick <= 220; ++tick) {
        const Json transition = env.step(actions(0.3, 0.3), 0.002);
        const Json& sample = eye(transition.at("observation"));
        const Json visual = env.visualization_state();
        if (tick % 50 == 0) {
            sampled_lux[tick] = visual.at("body_experiment_physics")
                                      .at("instantaneous_illuminance_lux");
        }
        const std::uint64_t sequence = tick < 20 ? 0 : (tick - 20) / 50 + 1;
        require(sample.at("sequence") == sequence,
                "light sequence must advance only on delivery ticks");
        require(sample.at("valid").get<bool>() == (sequence != 0),
                "validity must reflect availability of a delivered sample");
        require(!transition.at("terminated").get<bool>(),
                "initial light warm-up must not trigger a safety stop");
        if (sequence == 0) continue;
        const std::uint64_t sampled_tick = (sequence - 1) * 50;
        near(sample.at("sample_time_s"), sampled_tick * 0.002, 1e-9,
             "sampling timestamp must record acquisition time");
        near(sample.at("delivered_time_s"), sampled_tick * 0.002 + 0.04, 1e-9,
             "delivery timestamp must include latency");
        near(sample.at("age_s"), (tick - sampled_tick) * 0.002, 1e-9,
             "age must grow while a sample is held");
        near(sample.at("observations").at("illuminance_lux"),
             sampled_lux.at(sampled_tick), 1e-6,
             "delayed sample must equal the earlier actual mounted-site reading");
        near(visual.at("sensors").at("ambient_light_lux"),
             sample.at("observations").at("illuminance_lux"), 1e-9,
             "visible eye must display the same held reading as the controller");
    }
    require(std::abs(sampled_lux.at(200) - sampled_lux.at(0)) > 0.01,
            "clock test must exercise changing real light, not a stationary fixture");
    const Json replay = env.replay_and_verify(env.trace());
    require(replay.at("ok").get<bool>() && replay.at("verified_steps") == 220,
            "sample queue, metadata, rewards and physics must replay deterministically");
}

void test_mechanical_transmission(const std::filesystem::path& path) {
    droid::DroidEnvironment env(path);
    std::array<double, 3> wheel_speed{};
    std::array<double, 3> shaft_speed{};
    for (std::size_t index = 0; index < kAssemblies.size(); ++index) {
        (void)reset(env, kAssemblies[index], true);
        for (int step = 0; step < 30; ++step) {
            const Json transition = env.step(actions(0.22), 0.02);
            require(!transition.at("terminated").get<bool>(),
                    "bounded isolated motor probe must not hit a safety stop");
        }
        const Json state = env.visualization_state();
        const Json& physics = state.at("body_experiment_physics");
        wheel_speed[index] = physics.at("wheel_velocity_rad_s");
        shaft_speed[index] = physics.at("shaft_velocity_rad_s");
        require(shaft_speed[index] > 0.1,
                "unchanged positive shaft command must spin local shaft in every assembly");
        near(state.at("actuators").at("commands").at(0), 0.22, 1e-9,
             "mechanical variant must never swap or zero the requested motor action");
        if (index < 2) {
            const double sign = index == 0 ? 1.0 : -1.0;
            near(shaft_speed[index], sign * wheel_speed[index], 0.04,
                 "actual wheel and shaft must obey the signed mechanical constraint");
            near(physics.at("constraint_position_error_rad"), 0.0, 0.01,
                 "connected shaft must retain the declared position constraint");
        } else {
            require(!physics.at("connected").get<bool>(), "disconnect must release constraint");
            require(shaft_speed[index] > std::abs(wheel_speed[index]) + 1.0,
                    "disconnected shaft must turn independently of its unpowered wheel");
        }
        require(env.replay_and_verify(env.trace()).at("ok").get<bool>(),
                "each mechanical variant must replay exactly");
    }
    require(wheel_speed[0] > 0.1 && wheel_speed[1] < -0.1,
            "reversing the transmission must reverse actual wheel rotation");

    // Evidence for downstream probe design: report actual free-body changes,
    // including yaw, without assuming a solo wheel causes only translation.
    for (const std::string_view assembly : kAssemblies) {
        for (const double effort : {0.22, -0.22}) {
            (void)reset(env, assembly);
            const double initial_lux = env.visualization_state()
                .at("body_experiment_physics").at("instantaneous_illuminance_lux");
            for (int step = 0; step < 50; ++step) (void)env.step(actions(effort), 0.02);
            const Json state = env.visualization_state();
            std::cout << Json{{"assembly_id", assembly}, {"effort", effort},
                {"position_m", state.at("robot").at("position")},
                {"yaw_rad", state.at("robot").at("yaw")},
                {"light_change_lux", state.at("body_experiment_physics")
                    .at("instantaneous_illuminance_lux").get<double>() - initial_lux},
                {"shaft_velocity_rad_s", state.at("body_experiment_physics")
                    .at("shaft_velocity_rad_s")}}.dump() << '\n';
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path path = argc > 1 ? argv[1] : "models/droid.xml";
    try {
        test_nonintegral_clock_rejected(path);
        test_opt_in_and_restoration(path);
        test_sensor_clock(path);
        test_mechanical_transmission(path);
        std::cout << "PASS native body mechanics, timing, isolation and replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL body experiment: " << error.what() << '\n';
        return 1;
    }
}
