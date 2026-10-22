#include "droid/environment.hpp"
#include "droid/simulation.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using Actions = std::map<std::string, double>;

[[nodiscard]] std::filesystem::path default_model_path() {
#if defined(DROID_TEST_MODEL_PATH)
    return DROID_TEST_MODEL_PATH;
#elif defined(DROID_MODEL_PATH)
    return DROID_MODEL_PATH;
#else
    return "models/droid.xml";
#endif
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
    try {
        std::forward<Callable>(callable)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

[[nodiscard]] Actions actions(
    double motor_1,
    double motor_2,
    double motor_3,
    double motor_4) {
    return {
        {"motor-0001", motor_1},
        {"motor-0002", motor_2},
        {"motor-0003", motor_3},
        {"motor-0004", motor_4},
    };
}

[[nodiscard]] Actions uniform_actions(double effort) {
    return actions(effort, effort, effort, effort);
}

[[nodiscard]] bool contains_key_recursive(
    const Json& value,
    const std::set<std::string>& keys) {
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) {
            if (keys.contains(key) || contains_key_recursive(child, keys)) {
                return true;
            }
        }
    } else if (value.is_array()) {
        for (const Json& child : value) {
            if (contains_key_recursive(child, keys)) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] double light_source_x(const droid::DroidEnvironment& environment) {
    const Json state = environment.visualization_state();
    const Json& position = state.at("world").at("light_source_position");
    require(
        position.is_array() && position.size() == 3,
        "privileged light source position must be a three-vector");
    return position.at(0).get<double>();
}

[[nodiscard]] const Json& ambient_sensor_sample(const Json& observation) {
    const Json& samples = observation.at("reward_sensors");
    for (const Json& sample : samples) {
        if (sample.at("module_id") == "sensor-0001") {
            return sample;
        }
    }
    throw std::runtime_error("ambient sensor-0001 is missing");
}

[[nodiscard]] double ambient_lux(const Json& observation) {
    return ambient_sensor_sample(observation)
        .at("observations")
        .at("illuminance_lux")
        .get<double>();
}

[[nodiscard]] const Json& ambient_reward_component(const Json& transition) {
    for (const Json& component : transition.at("reward_components")) {
        if (component.at("module_id") == "sensor-0001") {
            return component;
        }
    }
    throw std::runtime_error("ambient reward component is missing");
}

[[nodiscard]] std::filesystem::path temporary_trace_path() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    return std::filesystem::temp_directory_path() /
        ("droid-environment-trace-" + std::to_string(suffix) + ".json");
}

class TemporaryFile final {
public:
    explicit TemporaryFile(std::filesystem::path path)
        : path_(std::move(path)) {}

    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void test_machine_readable_spec(const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    const Json spec = environment.spec();

    require(
        spec.at("api_version") == "agent_environment_v0",
        "agent API version is missing or unexpected");
    require(
        spec.at("engine").at("id") == "mujoco" &&
            spec.at("engine").at("version").is_string() &&
            !spec.at("engine").at("version").get_ref<const std::string&>().empty(),
        "agent spec must bind the MuJoCo engine identity and version");
    require(
        spec.at("supported_assembly_ids") ==
            Json::array({"demo_rover_v0"}),
        "v0 must expose only the fixed demo assembly");
    require(
        spec.at("action").at("motor_ids") ==
            Json::array({
                "motor-0001", "motor-0002", "motor-0003", "motor-0004"}),
        "action spec must expose the four opaque motor IDs");
    require(
        spec.at("physics_timestep_s").get<double>() == 0.002,
        "physics timestep must be machine-readable");
    require(
        spec.at("nominal_control_dt_s").get<double>() == 0.02,
        "nominal control interval must be machine-readable");
    require(
        spec.at("default_episode_profile") ==
            std::string(droid::kFixedDemoEpisodeProfile),
        "fixed demo must remain the default episode profile");
    const Json& profiles = spec.at("episode_profiles");
    require(
        profiles.is_array() && profiles.size() == 2,
        "spec must declare exactly the fixed and light-search profiles");
    const auto profile_by_id = [&profiles](std::string_view profile_id)
        -> const Json& {
        for (const Json& profile : profiles) {
            if (profile.at("id") == std::string(profile_id)) {
                return profile;
            }
        }
        throw std::runtime_error(
            "episode profile is absent from machine-readable spec");
    };
    const Json& fixed = profile_by_id(droid::kFixedDemoEpisodeProfile);
    require(
        fixed.at("seed_affects_realization") == false &&
            fixed.at("light_source_x_m") == 16.0,
        "fixed profile semantics changed");
    const Json& search = profile_by_id(droid::kLightSearch1dEpisodeProfile);
    require(
        search.at("distribution").at("kind") ==
                "balanced_sign_uniform_24bit_distance_grid" &&
            search.at("distribution").at("side_support") ==
                Json::array({"negative_x", "positive_x"}),
        "light-search distribution must declare its balanced source sides");
    const Json& distance =
        search.at("distribution").at("absolute_distance_m");
    require(
        distance.at("minimum_inclusive_m") == 12.0 &&
            distance.at("maximum_exclusive_m") == 20.0 &&
            distance.at("step_m").get<double>() == 0x1p-21 &&
            distance.at("bucket_count").get<std::uint32_t>() == (1U << 24U),
        "light-search distance grid is not the exact 24-bit [12,20) grid");
    require(
        search.at("sensor_field").at("sample_position") ==
                "mounted_sensor_site_world_xy" &&
            search.at("sensor_field").at("source_y_m") == 0.0,
        "light-search spec must declare site-based radial sensing");
    const Json& mapping = search.at("seed_mapping");
    require(
        mapping.at("integer_mixer") == "splitmix64_finalizer" &&
            mapping.at("input") == "seed XOR 0x4c49474854563100" &&
            mapping.at("side_bit") == 63 &&
            mapping.at("side_bit_0") == "negative_x" &&
            mapping.at("side_bit_1") == "positive_x" &&
            mapping.at("distance_bucket_expression") ==
                "mixed & 0x00ffffff" &&
            mapping.at("distance_formula_m") == "12 + bucket * 2^-21" &&
            mapping.at("source_x_formula_m") ==
                "(side_bit ? +1 : -1) * distance" &&
            mapping.at("realization_key_expression") ==
                "(side_bit << 24) | bucket" &&
            mapping.at("realization_key_bits") == 25 &&
            mapping.at("integer_only") == true,
        "light-search seed mapping is not fully specified");
    require(
        search.at("realization_policy_visible") == false,
        "search-profile realization must be hidden from policy input");
}

void test_light_realization_mapping() {
    const droid::LightSearch1dRealization seed_zero =
        droid::light_search_1d_realization(0);
    require(
        seed_zero.positive_side && seed_zero.distance_bucket == 0x00e742acU &&
            seed_zero.source_x_m == 19.2268886566162109375 &&
            droid::light_search_1d_realization_key(0) == 0x01e742acU,
        "golden light-search realization changed for seed 0");

    const droid::LightSearch1dRealization seed_one =
        droid::light_search_1d_realization(1);
    require(
        !seed_one.positive_side && seed_one.distance_bucket == 0x00f9da24U &&
            seed_one.source_x_m == -19.8078784942626953125 &&
            droid::light_search_1d_realization_key(1) == 0x00f9da24U,
        "golden light-search realization changed for seed 1");

    const std::uint64_t maximum_seed =
        std::numeric_limits<std::uint64_t>::max();
    const droid::LightSearch1dRealization maximum =
        droid::light_search_1d_realization(maximum_seed);
    require(
        maximum.positive_side && maximum.distance_bucket == 0x00a4e32eU &&
            maximum.source_x_m == 17.15273189544677734375 &&
            droid::light_search_1d_realization_key(maximum_seed) ==
                0x01a4e32eU,
        "golden light-search realization changed for UINT64_MAX");

    std::set<std::uint32_t> sampled_keys;
    bool saw_negative = false;
    bool saw_positive = false;
    for (std::uint64_t seed = 0; seed < 4'096; ++seed) {
        const droid::LightSearch1dRealization realization =
            droid::light_search_1d_realization(seed);
        const std::uint32_t key =
            droid::light_search_1d_realization_key(seed);
        const double distance_m = std::abs(realization.source_x_m);
        require(
            sampled_keys.emplace(key).second,
            "sampled light-search seeds produced a duplicate realization key");
        require(
            distance_m >= 12.0 && distance_m < 20.0 &&
                distance_m == 12.0 +
                    static_cast<double>(realization.distance_bucket) * 0x1p-21,
            "light-search realization left its exact distance grid");
        require(
            ((key >> 24U) != 0U) == realization.positive_side &&
                (key & 0x00ff'ffffU) == realization.distance_bucket,
            "light-search realization key does not encode side and distance");
        saw_negative = saw_negative || !realization.positive_side;
        saw_positive = saw_positive || realization.positive_side;
    }
    require(
        saw_negative && saw_positive,
        "sampled light-search mapping did not cover both source sides");
}

void test_deterministic_policy_safe_reset(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    const Json first = environment.reset("demo_rover_v0", 0xC0FFEE, false);
    const Json second = environment.reset("demo_rover_v0", 0xC0FFEE, false);
    require(
        first.dump() == second.dump(),
        "same assembly and seed must produce byte-identical reset JSON");

    const Json& observation = first.at("observation");
    require(
        observation.size() == 2 && observation.contains("reward_sensors") &&
            observation.contains("actuator_feedback"),
        "policy observation must contain only the two module collections");
    require(
        observation.at("reward_sensors").size() == 3,
        "the v0 assembly must expose three mounted reward sensors");
    require(
        observation.at("actuator_feedback").size() == 4,
        "the v0 assembly must expose four actuator feedback samples");

    const std::set<std::string> forbidden_policy_keys{
        "robot",
        "world",
        "controller",
        "position",
        "quaternion",
        "yaw",
        "speed",
        "distance",
        "contact_count",
        "privileged_diagnostics",
        "targets",
        "target_velocity_rad_s",
        "direction",
        "light_source_position",
        "light_source_x_m",
        "source_side",
        "episode_profile",
        "seed",
    };
    require(
        !contains_key_recursive(observation, forbidden_policy_keys),
        "policy observation leaked privileged visualization state");
    require(
        !contains_key_recursive(
            observation.at("actuator_feedback"),
            {"reward", "reward_rate", "transition_reward", "reward_vote"}),
        "actuator feedback must never carry reward authority");

    const Json stepped_observation =
        environment.step(uniform_actions(0.25), 0.02).at("observation");
    require(
        !contains_key_recursive(stepped_observation, forbidden_policy_keys),
        "stepped policy observation leaked privileged visualization state");
    require(
        !contains_key_recursive(
            stepped_observation.at("actuator_feedback"),
            {"reward", "reward_rate", "transition_reward", "reward_vote"}),
        "stepped actuator feedback acquired reward authority");

    const Json visualization = environment.visualization_state();
    require(
        visualization.contains("robot") &&
            visualization.contains("controller"),
        "privileged visualization must remain available on its separate path");
}

void test_seeded_episode_profiles(const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);

    const Json implicit_fixed =
        environment.reset("demo_rover_v0", 0, false);
    require(
        light_source_x(environment) == 16.0 &&
            implicit_fixed.at("info").at("episode_profile") ==
                std::string(droid::kFixedDemoEpisodeProfile),
        "default reset did not select the canonical fixed profile");
    const Json explicit_fixed = environment.reset(
        "demo_rover_v0", 0, false, droid::kFixedDemoEpisodeProfile);
    require(
        explicit_fixed.dump() == implicit_fixed.dump(),
        "implicit and explicit fixed profiles differ");
    const Json other_seed_fixed = environment.reset(
        "demo_rover_v0", 1, false, droid::kFixedDemoEpisodeProfile);
    require(
        other_seed_fixed.at("observation").dump() ==
            explicit_fixed.at("observation").dump() &&
            light_source_x(environment) == 16.0,
        "fixed profile incorrectly used its seed to change the world");

    const Json positive = environment.reset(
        "demo_rover_v0", 0, false, droid::kLightSearch1dEpisodeProfile);
    const std::string positive_state = environment.visualization_state().dump();
    require(
        light_source_x(environment) ==
            droid::light_search_1d_realization(0).source_x_m,
        "light-search golden seed 0 selected the wrong source realization");
    const Json positive_again = environment.reset(
        "demo_rover_v0", 0, false, droid::kLightSearch1dEpisodeProfile);
    require(
        positive_again.dump() == positive.dump() &&
            environment.visualization_state().dump() == positive_state,
        "same light-search seed was not byte-exact at reset");

    const Json negative = environment.reset(
        "demo_rover_v0", 1, false, droid::kLightSearch1dEpisodeProfile);
    require(
        light_source_x(environment) ==
            droid::light_search_1d_realization(1).source_x_m,
        "light-search golden seed 1 selected the wrong source realization");
    require(
        positive.at("info").at("episode_profile") ==
                std::string(droid::kLightSearch1dEpisodeProfile) &&
            negative.at("info").at("episode_profile") ==
                std::string(droid::kLightSearch1dEpisodeProfile),
        "reset info omitted the selected episode profile");
    require(
        ambient_lux(positive.at("observation")) !=
            ambient_lux(negative.at("observation")),
        "mounted light sensor did not distinguish seeded source realizations");
    require(
        !contains_key_recursive(
            positive.at("observation"),
            {"world",
             "light_source_position",
             "light_source_x_m",
             "source_side",
             "episode_profile",
             "seed"}) &&
            !contains_key_recursive(
                negative.at("observation"),
                {"world",
                 "light_source_position",
                 "light_source_x_m",
                 "source_side",
                 "episode_profile",
                 "seed"}),
        "episode realization leaked into a policy observation");
    require(
        !contains_key_recursive(
            negative.at("info"),
            {"light_source_position", "light_source_x_m", "source_side"}),
        "reset info leaked the hidden source realization");

    const std::uint64_t full_width_seed =
        std::numeric_limits<std::uint64_t>::max();
    const Json full_width = environment.reset(
        "demo_rover_v0",
        full_width_seed,
        false,
        droid::kLightSearch1dEpisodeProfile);
    require(
        light_source_x(environment) ==
                droid::light_search_1d_realization(full_width_seed).source_x_m &&
            full_width.at("info").at("seed").is_number_unsigned() &&
            full_width.at("info").at("seed").get<std::uint64_t>() ==
                full_width_seed,
        "light-search mapping did not preserve the full unsigned seed domain");

    const std::string state_before_rejection =
        environment.visualization_state().dump();
    require_throws(
        [&] {
            (void)environment.reset(
                "demo_rover_v0", 1, false, "unknown_profile");
        },
        "unknown episode profile must be rejected");
    require(
        environment.visualization_state().dump() == state_before_rejection,
        "rejected profile partially mutated the live environment");
}

void test_seeded_distance_light_action_sensitivity(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment positive(model_path);
    droid::DroidEnvironment negative(model_path);
    const Json positive_reset = positive.reset(
        "demo_rover_v0", 0, false, droid::kLightSearch1dEpisodeProfile);
    const Json negative_reset = negative.reset(
        "demo_rover_v0", 1, false, droid::kLightSearch1dEpisodeProfile);
    const double positive_initial_lux =
        ambient_lux(positive_reset.at("observation"));
    const double negative_initial_lux =
        ambient_lux(negative_reset.at("observation"));

    Json positive_transition;
    Json negative_transition;
    Json positive_first_transition;
    Json negative_first_transition;
    for (std::size_t index = 0; index < 100; ++index) {
        positive_transition = positive.step(uniform_actions(0.5), 0.02);
        negative_transition = negative.step(uniform_actions(0.5), 0.02);
        if (index == 0) {
            positive_first_transition = positive_transition;
            negative_first_transition = negative_transition;
        }
        require(
            !positive_transition.at("terminated").get<bool>() &&
                !negative_transition.at("terminated").get<bool>(),
            "safe action-sensitivity probe terminated unexpectedly");
    }

    require(
        ambient_lux(positive_transition.at("observation")) >
            positive_initial_lux,
        "positive-x drive did not increase illumination toward positive source");
    require(
        ambient_lux(negative_transition.at("observation")) <
            negative_initial_lux,
        "positive-x drive did not decrease illumination away from negative source");
    require(
        ambient_reward_component(positive_transition).at("reward_rate") >
                ambient_reward_component(positive_first_transition)
                    .at("reward_rate") &&
            ambient_reward_component(negative_transition).at("reward_rate") <
                ambient_reward_component(negative_first_transition)
                    .at("reward_rate"),
        "seeded-distance sensor rewards are not directionally action-sensitive");

    for (const Json* transition :
         {&positive_transition, &negative_transition}) {
        require(
            transition->at("info").at("episode_profile") ==
                std::string(droid::kLightSearch1dEpisodeProfile),
            "step info omitted the selected episode profile");
        require(
            transition->at("reward_components").size() == 3 &&
                transition->at("observation").at("reward_sensors").size() ==
                    3 &&
                transition->at("observation")
                        .at("actuator_feedback")
                        .size() == 4,
            "episode variation changed module topology");
        long double component_sum = 0.0L;
        for (const Json& component : transition->at("reward_components")) {
            component_sum += static_cast<long double>(
                component.at("transition_reward").get<double>());
        }
        require(
            transition->at("reward").get<double>() ==
                static_cast<double>(component_sum),
            "profiled reward no longer reconciles to sensor votes");
        require(
            !contains_key_recursive(
                transition->at("observation"),
                {"world",
                 "light_source_position",
                 "light_source_x_m",
                 "source_side",
                 "episode_profile",
                 "seed"}),
            "step observation leaked the episode realization");
    }
}

void test_external_actions_change_the_world(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment idle(model_path);
    droid::DroidEnvironment driven(model_path);
    require(
        idle.reset("demo_rover_v0", 71, false).dump() ==
            driven.reset("demo_rover_v0", 71, false).dump(),
        "comparison environments did not start identically");

    Json idle_transition;
    Json driven_transition;
    for (int index = 0; index < 5; ++index) {
        idle_transition = idle.step(uniform_actions(0.0), 0.02);
        driven_transition = driven.step(uniform_actions(0.8), 0.02);
    }
    require(
        idle_transition.at("observation").dump() !=
            driven_transition.at("observation").dump(),
        "external motor efforts did not change the policy observation");
    require(
        driven.visualization_state().at("controller").at("id") ==
            "external_effort_v0",
        "agent step must bypass the privileged demo controller");
}

void test_reward_aggregation(const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    (void)environment.reset("demo_rover_v0", 17, false);
    const Json transition = environment.step(uniform_actions(0.35), 0.02);

    const Json& components = transition.at("reward_components");
    require(
        components.size() == 3,
        "exactly the three mounted sensor instances must vote");
    long double integrated_sum = 0.0L;
    for (const Json& component : components) {
        require(
            component.contains("module_id") &&
                component.contains("family_id") &&
                component.contains("reward_rate") &&
                component.contains("transition_reward"),
            "reward component lacks auditable provenance");
        integrated_sum += static_cast<long double>(
            component.at("transition_reward").get<double>());
    }
    require(
        transition.at("reward").get<double>() ==
            static_cast<double>(integrated_sum),
        "transition reward must exactly equal the ordered component sum");
    require(
        transition.at("info").at("physics_steps_executed") == 10,
        "nominal control transition must contain ten physics steps");
    require(
        transition.at("info").at("elapsed_control_dt_s") == 0.02,
        "reported elapsed control interval is incorrect");
}

void test_record_save_load_and_exact_replay(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    const std::uint64_t full_width_seed =
        std::numeric_limits<std::uint64_t>::max();
    const Json reset =
        environment.reset("demo_rover_v0", full_width_seed, true);
    const Json first = environment.step(
        actions(0.2, 0.25, 0.3, 0.35), 0.02);
    const Json second = environment.step(
        actions(-0.1, 0.0, 0.1, 0.2), 0.04);
    (void)first;
    (void)second;

    const Json trace = environment.trace();
    require(environment.recording(), "recording flag was not retained");
    require(
        trace.at("schema_version") ==
            droid::kEnvironmentTraceSchemaVersion,
        "trace schema version is missing");
    require(
        trace.at("episode_profile") ==
                std::string(droid::kFixedDemoEpisodeProfile) &&
            trace.at("reset").at("info").at("episode_profile") ==
                std::string(droid::kFixedDemoEpisodeProfile),
        "v2 trace did not retain its episode profile");
    require(trace.at("reset").dump() == reset.dump(), "reset was not recorded");
    require(trace.at("steps").size() == 2, "trace lost an ordered action");
    require(
        trace.at("seed").is_number_unsigned() &&
            trace.at("seed").get<std::uint64_t>() == full_width_seed,
        "trace did not preserve the full unsigned seed domain");
    require(
        trace.at("steps").at(0).at("index") == 0 &&
            trace.at("steps").at(1).at("index") == 1,
        "trace step indices are not ordered");

    TemporaryFile saved(temporary_trace_path());
    environment.save_trace(saved.path());
    const Json loaded = droid::DroidEnvironment::load_trace(saved.path());
    require(
        loaded.dump() == trace.dump(),
        "save/load did not preserve canonical trace JSON");

    require_throws(
        [&] { environment.save_trace(saved.path()); },
        "save_trace must refuse to overwrite user trace data");
    require(
        droid::DroidEnvironment::load_trace(saved.path()).dump() ==
            trace.dump(),
        "refused overwrite changed the existing trace");

    const std::string live_state_before =
        environment.visualization_state().dump();
    const Json report = environment.replay_and_verify_file(saved.path());
    require(report.at("ok").get<bool>(), "exact replay did not verify");
    require(
        report.at("verified_steps") == 2,
        "replay verified the wrong number of transitions");
    require(
        environment.visualization_state().dump() == live_state_before,
        "fresh replay mutated the calling environment");
}

void test_profiled_and_legacy_trace_replay(
    const std::filesystem::path& model_path) {
    for (const std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{1}}) {
        droid::DroidEnvironment environment(model_path);
        (void)environment.reset(
            "demo_rover_v0",
            seed,
            true,
            droid::kLightSearch1dEpisodeProfile);
        (void)environment.step(uniform_actions(0.25), 0.02);
        (void)environment.step(uniform_actions(-0.15), 0.04);
        const Json trace = environment.trace();
        require(
            trace.at("episode_profile") ==
                std::string(droid::kLightSearch1dEpisodeProfile),
            "profiled trace omitted its episode profile");
        const Json report = environment.replay_and_verify(trace);
        require(
            report.at("ok").get<bool>() &&
                report.at("schema_version") ==
                    droid::kEnvironmentTraceSchemaVersion &&
                report.at("verified_steps") == 2,
            "exact replay failed for one side of the profiled world");
    }

    droid::DroidEnvironment legacy_environment(model_path);
    (void)legacy_environment.reset(
        "demo_rover_v0",
        77,
        true,
        droid::kFixedDemoEpisodeProfile);
    (void)legacy_environment.step(uniform_actions(0.1), 0.02);
    Json legacy_trace = legacy_environment.trace();
    legacy_trace["schema_version"] =
        droid::kLegacyEnvironmentTraceSchemaVersion;
    legacy_trace.erase("episode_profile");
    legacy_trace.at("reset").at("info").erase("episode_profile");
    for (Json& entry : legacy_trace.at("steps")) {
        entry.at("transition").at("info").erase("episode_profile");
    }
    const Json legacy_report =
        legacy_environment.replay_and_verify(legacy_trace);
    require(
        legacy_report.at("ok").get<bool>() &&
            legacy_report.at("schema_version") ==
                droid::kLegacyEnvironmentTraceSchemaVersion &&
            legacy_report.at("verified_steps") == 1,
        "legacy v1 trace was not read as the fixed demo profile");

    TemporaryFile legacy_file(temporary_trace_path());
    {
        std::ofstream output(legacy_file.path(), std::ios::binary);
        require(
            static_cast<bool>(output),
            "unable to create temporary legacy trace fixture");
        output << legacy_trace.dump();
        require(
            static_cast<bool>(output),
            "unable to write temporary legacy trace fixture");
    }
    require(
        droid::DroidEnvironment::load_trace(legacy_file.path()).dump() ==
            legacy_trace.dump(),
        "legacy v1 trace loader changed the document");
    const Json legacy_file_report =
        legacy_environment.replay_and_verify_file(legacy_file.path());
    require(
        legacy_file_report.dump() == legacy_report.dump(),
        "legacy v1 file replay differs from in-memory replay");
}

void test_replay_reports_first_exact_mismatch(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    (void)environment.reset("demo_rover_v0", 9, true);
    (void)environment.step(uniform_actions(0.15), 0.02);
    Json altered = environment.trace();
    Json& reward =
        altered.at("steps").at(0).at("transition").at("reward");
    reward = reward.get<double>() + 0.125;

    const Json report = environment.replay_and_verify(altered);
    require(!report.at("ok").get<bool>(), "altered replay unexpectedly passed");
    require(
        report.at("verified_steps") == 0,
        "mismatch report counted the altered transition as verified");
    const Json& mismatch = report.at("mismatch");
    require(
        mismatch.at("phase") == "step" &&
            mismatch.at("transition_index") == 0,
        "mismatch report did not identify the first transition");
    require(
        mismatch.at("path").get<std::string>().ends_with("/reward"),
        "mismatch report did not identify the exact JSON path");
    require(
        mismatch.contains("expected") && mismatch.contains("actual"),
        "mismatch report omitted the differing values");
}

void test_action_and_timing_validation(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    require_throws(
        [&] { (void)environment.step(uniform_actions(0.0), 0.02); },
        "step without reset must fail");
    require_throws(
        [&] { (void)environment.reset("unknown_assembly", 1, false); },
        "unknown assembly must fail");
    (void)environment.reset("demo_rover_v0", 1, false);

    Actions missing = uniform_actions(0.0);
    missing.erase("motor-0004");
    require_throws(
        [&] { (void)environment.step(missing, 0.02); },
        "missing motor action must fail");

    Actions unknown = uniform_actions(0.0);
    unknown.emplace("motor-9999", 0.0);
    require_throws(
        [&] { (void)environment.step(unknown, 0.02); },
        "unknown motor action must fail");

    Actions non_finite = uniform_actions(0.0);
    non_finite.at("motor-0002") =
        std::numeric_limits<double>::quiet_NaN();
    require_throws(
        [&] { (void)environment.step(non_finite, 0.02); },
        "non-finite motor action must fail");
    require_throws(
        [&] { (void)environment.step(uniform_actions(0.0), 0.0); },
        "zero control interval must fail");
    require_throws(
        [&] { (void)environment.step(uniform_actions(0.0), 0.003); },
        "non-integral physics interval must fail");
    require_throws(
        [&] { (void)environment.step(uniform_actions(0.0), 1.002); },
        "control interval above the advertised maximum must fail");

    const Json clamped = environment.step(uniform_actions(4.0), 0.02);
    require(
        clamped.at("info").at("clamped_action_ids").size() == 4,
        "motor driver must report each clamped external action");
}

void test_optional_recording_and_mode_guards(
    const std::filesystem::path& model_path) {
    droid::DroidEnvironment environment(model_path);
    (void)environment.reset("demo_rover_v0", 5, false);
    const Json transition = environment.step(uniform_actions(0.0), 0.02);
    require(
        transition.at("truncated") == false,
        "v0 must reserve truncation for an external horizon");
    require(!environment.recording(), "recording must remain opt-in");
    require_throws(
        [&] { (void)environment.trace(); },
        "trace() must reject an unrecorded episode");

    if (transition.at("terminated").get<bool>()) {
        require_throws(
            [&] { (void)environment.step(uniform_actions(0.0), 0.02); },
            "step after termination must require reset");
    }

    (void)environment.reset(
        "demo_rover_v0",
        1,
        false,
        droid::kLightSearch1dEpisodeProfile);
    require(
        light_source_x(environment) ==
            droid::light_search_1d_realization(1).source_x_m,
        "search-profile setup did not select its golden negative realization");
    environment.reset_visualization();
    const Json reset_visualization = environment.visualization_state();
    require(
        light_source_x(environment) == 16.0 &&
            reset_visualization.at("world").at("episode_profile") ==
                std::string(droid::kFixedDemoEpisodeProfile),
        "visualization reset did not restore the canonical fixed world");
    require_throws(
        [&] { (void)environment.step(uniform_actions(0.0), 0.02); },
        "visualization reset must not silently begin an agent episode");

    droid::DroidEnvironment shared(model_path, true);
    (void)shared.reset(
        "demo_rover_v0",
        1,
        false,
        droid::kLightSearch1dEpisodeProfile);
    require(
        !shared.health().at("running").get<bool>() &&
            light_source_x(shared) ==
                droid::light_search_1d_realization(1).source_x_m,
        "agent reset must pause the realtime demo worker");
    shared.set_demo_running(true);
    require(
        shared.health().at("running").get<bool>() &&
            light_source_x(shared) == 16.0 &&
            shared.visualization_state().at("world").at("episode_profile") ==
                std::string(droid::kFixedDemoEpisodeProfile),
        "demo mode must restore the canonical fixed reward field");
    (void)shared.reset(
        "demo_rover_v0",
        0,
        false,
        droid::kLightSearch1dEpisodeProfile);
    require(
        !shared.health().at("running").get<bool>(),
        "a subsequent evaluation reset must pause demo mode again");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path model_path =
            argc > 1 ? std::filesystem::path(argv[1]) : default_model_path();
        if (argc > 2) {
            throw std::invalid_argument(
                "usage: droid-native-environment-tests [model.xml]");
        }

        const std::vector<std::pair<std::string, std::function<void()>>> tests{
            {"machine-readable spec",
             [&] { test_machine_readable_spec(model_path); }},
            {"light realization mapping", test_light_realization_mapping},
            {"deterministic policy-safe reset",
             [&] { test_deterministic_policy_safe_reset(model_path); }},
            {"seeded episode profiles",
             [&] { test_seeded_episode_profiles(model_path); }},
            {"seeded-distance light action sensitivity",
             [&] { test_seeded_distance_light_action_sensitivity(model_path); }},
            {"external actions change world",
             [&] { test_external_actions_change_the_world(model_path); }},
            {"sensor reward aggregation",
             [&] { test_reward_aggregation(model_path); }},
            {"record save load exact replay",
             [&] { test_record_save_load_and_exact_replay(model_path); }},
            {"profiled and legacy trace replay",
             [&] { test_profiled_and_legacy_trace_replay(model_path); }},
            {"first exact replay mismatch",
             [&] { test_replay_reports_first_exact_mismatch(model_path); }},
            {"action and timing validation",
             [&] { test_action_and_timing_validation(model_path); }},
            {"optional recording and mode guards",
             [&] { test_optional_recording_and_mode_guards(model_path); }},
        };

        for (const auto& [name, test] : tests) {
            test();
            std::cout << "[PASS] " << name << '\n';
        }
        std::cout << tests.size() << " native environment tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] native environment: " << error.what() << '\n';
        return 1;
    }
}
