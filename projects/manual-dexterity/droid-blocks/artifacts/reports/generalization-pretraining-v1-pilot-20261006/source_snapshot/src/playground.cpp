#include "droid/playground.hpp"
#include "droid/body_controller.hpp"

#include "droid/learner.hpp"
#include "droid/module_catalog.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace droid {
namespace {

using Json = nlohmann::json;

[[nodiscard]] PolicyArtifactData validated_artifact(
    DroidEnvironment& environment,
    const std::filesystem::path& model_path,
    const std::filesystem::path& artifact_path) {
    // Pin both the exact preserved file and its canonical integrity identity.
    // Neither checksum is presented as an authenticity proof.
    if (sha256_file(artifact_path) != kPlaygroundArtifactFileSha256) {
        throw std::runtime_error("playground policy file differs from the frozen v3 artifact");
    }
    PolicyArtifactData artifact = load_policy_artifact(artifact_path);
    if (artifact.self_sha256 != kPlaygroundArtifactSha256) {
        throw std::runtime_error("playground policy identity differs from the frozen v3 artifact");
    }
    const Json spec = environment.spec();
    const PolicyCompatibility actual{
        .environment_api_version = spec.at("api_version").get<std::string>(),
        .assembly_id = spec.at("assembly_id").get<std::string>(),
        .motor_ids = spec.at("action").at("motor_ids").get<std::vector<std::string>>(),
        .control_dt_s = kPlaygroundControlDtS,
        .model_sha256 = sha256_file(model_path),
        .module_catalog_sha256 = sha256_hex(catalog_json().dump()),
        .environment_spec_sha256 = sha256_hex(spec.dump()),
        .feature_schema_sha256 = policy_feature_schema_sha256(),
    };
    require_policy_compatible(artifact, actual);

    // Derivation is a metadata check, never a challenge-world rollout. Only
    // these two published training worlds can reach reset below.
    const MirroredSeedSplits splits = make_mirrored_seed_splits();
    if (seed_list_sha256(splits.train) != artifact.training.train_seeds_sha256 ||
        artifact.training.max_control_steps != kPlaygroundMaxSteps ||
        artifact.training.scenario_profile_id != kLightSearch1dEpisodeProfile) {
        throw std::runtime_error("playground training provenance does not match the frozen artifact");
    }
    for (const std::uint64_t seed : {kPlaygroundLeftSeed, kPlaygroundRightSeed}) {
        if (std::find(splits.train.begin(), splits.train.end(), seed) == splits.train.end() ||
            std::find(splits.validation.begin(), splits.validation.end(), seed) != splits.validation.end() ||
            std::find(splits.test.begin(), splits.test.end(), seed) != splits.test.end()) {
            throw std::runtime_error("playground world is not an approved training world");
        }
    }
    if (light_search_seed_has_positive_side(kPlaygroundLeftSeed) ||
        !light_search_seed_has_positive_side(kPlaygroundRightSeed)) {
        throw std::runtime_error("playground training-world sides do not match");
    }
    return artifact;
}

[[nodiscard]] double observed_lux(const Json& observation) {
    for (const Json& sensor : observation.at("reward_sensors")) {
        if (sensor.at("family_id") == "ambient_light_v0" &&
            sensor.at("valid").get<bool>()) {
            return sensor.at("observations").at("illuminance_lux").get<double>();
        }
    }
    throw std::runtime_error("playground light sensor has no valid observation");
}

}  // namespace

struct PlaygroundSession::Impl {
    Impl(DroidEnvironment& borrowed_environment,
         const std::filesystem::path& model,
         const std::filesystem::path& artifact,
         bool realtime)
        : environment(borrowed_environment), model_path(model), artifact_path(artifact) {
        environment.set_demo_running(false);
        reset_learned_locked("left");
        if (realtime) {
            worker = std::thread([this] {
                std::unique_lock lock(mutex);
                while (!wake.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopping; })) {
                    tick_locked();
                }
            });
        }
    }

    ~Impl() {
        {
            std::scoped_lock lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void fail_locked(std::string message) {
        status = "error";
        error = std::move(message);
        environment.set_demo_running(false);
    }

    void reset_learned_locked(std::string_view requested_side) {
        // Validate before touching the simulation or clearing existing memory.
        // An error retains the last physical state for inspection.
        try {
            auto next_policy = std::make_unique<SharedLinearMemoryPolicy>(
                validated_artifact(environment, model_path, artifact_path));
            const auto next_motor_ids = next_policy->artifact().compatibility.motor_ids;
            next_policy->reset(next_motor_ids);
            const Json reset = environment.reset(
                "demo_rover_v0",
                requested_side == "left" ? kPlaygroundLeftSeed : kPlaygroundRightSeed,
                false,
                kLightSearch1dEpisodeProfile);
            const double next_lux = observed_lux(reset.at("observation"));
            policy = std::move(next_policy);
            motor_ids = next_motor_ids;
            observation = reset.at("observation");
            side = requested_side;
            mode = "learned";
            status = "ready";
            step = 0;
            initial_lux = next_lux;
            current_lux = next_lux;
            elapsed_s = 0.0;
            error.clear();
            ++session_id;
        } catch (const std::exception& exception) {
            // The requested controller remains selected, with a visible error.
            mode = "learned";
            fail_locked(exception.what());
        }
    }

    void reset_body_locked(std::string_view requested_side,
                           std::string_view requested_variant,
                           std::string_view requested_strategy) {
        try {
            auto next_controller = std::make_unique<BodyProbeController>();
            const auto next_motor_ids = environment.spec().at("action").at("motor_ids")
                .get<std::vector<std::string>>();
            next_controller->reset(next_motor_ids, requested_strategy);
            const std::string assembly = "transmission_" + std::string(requested_variant) + "_v1";
            const Json reset = environment.reset(
                assembly, requested_side == "left" ? kPlaygroundLeftSeed : kPlaygroundRightSeed,
                false, "body_discovery_1d_v1");
            body_controller = std::move(next_controller);
            motor_ids = next_motor_ids;
            observation = reset.at("observation");
            side = requested_side;
            body_variant = requested_variant;
            body_strategy = requested_strategy;
            mode = "body_lab";
            status = "ready";
            step = 0;
            elapsed_s = 0.0;
            // No delivered sample exists during the sensor's initial latency.
            initial_lux = 0.0;
            current_lux = 0.0;
            body_has_light = false;
            error.clear();
            ++session_id;
        } catch (const std::exception& exception) {
            // Failed construction leaves the previous owner, scene and its
            // retained evidence selected. Never label that scene as a new body.
            fail_locked(exception.what());
        }
    }

    void tick_locked() {
        if (mode == "demo") {
            const Json native_health = environment.health();
            if (native_health.at("status") == "error") {
                fail_locked(native_health.value("error", "The native demo stopped unexpectedly."));
            }
            return;
        }
        if ((mode != "learned" && mode != "body_lab") || status != "running") {
            return;
        }
        try {
            // Complete inference boundary: physical observations + opaque IDs.
            // No seed, side, reward, info, pose, or display metadata reaches act.
            const auto actions = mode == "body_lab"
                ? body_controller->act(observation, motor_ids)
                : policy->act(observation, motor_ids);
            const Json transition = environment.step(actions, kPlaygroundControlDtS);
            observation = transition.at("observation");
            if (mode == "body_lab") {
                for (const Json& sensor : observation.at("reward_sensors")) {
                    if (sensor.at("family_id") == "ambient_light_v0" && sensor.at("valid") == true) {
                        current_lux = sensor.at("observations").at("illuminance_lux").get<double>();
                        if (!body_has_light) { initial_lux = current_lux; body_has_light = true; }
                    }
                }
            } else {
                current_lux = observed_lux(observation);
            }
            ++step;
            elapsed_s = static_cast<double>(step) * kPlaygroundControlDtS;
            if (transition.at("terminated").get<bool>() ||
                transition.at("truncated").get<bool>()) {
                fail_locked("The native environment stopped the episode before its playground horizon.");
            } else if (step >= (mode == "body_lab" ? kBodyPlaygroundMaxSteps : kPlaygroundMaxSteps)) {
                status = "completed";
            }
        } catch (const std::exception& exception) {
            fail_locked(exception.what());
        }
    }

    [[nodiscard]] Json state_locked() const {
        Json state = environment.visualization_state();
        const bool body = mode == "body_lab";
        const std::size_t horizon = body ? kBodyPlaygroundMaxSteps : kPlaygroundMaxSteps;
        Json metadata{
            {"mode", mode}, {"status", status}, {"side", side},
            {"elapsed_s", mode == "demo" ? state.at("sim_time") : Json(elapsed_s)},
            {"duration_s", horizon * kPlaygroundControlDtS},
            {"step", step}, {"max_steps", horizon},
            {"session_id", session_id},
            {"light_position_m", state.at("world").at("light_source_position")},
            {"initial_illuminance_lux", initial_lux},
            {"illuminance_lux", (mode == "learned" || body) ? Json(current_lux) : state.at("sensors").at("ambient_light_lux")},
            {"artifact_sha256", mode == "learned" && policy ? Json(policy->artifact().self_sha256) : Json(nullptr)},
            {"policy_id", body ? Json(kBodyControllerId) : mode == "learned" ? Json("shared_linear_memory_v2") : mode == "demo" ? Json("demo_cruise_v0") : Json("external_effort_v0")},
            {"exposure", body ? Json("known_development_bodies_and_training_worlds") : mode == "learned" ? Json("known_training_world") : Json(nullptr)},
        };
        if (body) {
            Json detail = body_controller ? body_controller->diagnostics() : Json::object();
            detail["id"] = "transmission_discovery_v1";
            detail["variant"] = body_variant;
            detail["strategy"] = body_strategy;
            detail["sensor_period_s"] = 0.1;
            detail["sensor_latency_s"] = 0.04;
            detail["light_sample_valid"] = body_has_light;
            if (status == "ready" || status == "completed") detail["phase"] = status;
            metadata["body_experiment"] = std::move(detail);
        }
        if (!error.empty()) {
            metadata["error"] = error;
        }
        state["playground"] = std::move(metadata);
        return state;
    }

    void control_locked(std::string_view action, std::string_view requested_side,
                        std::string_view variant = {}, std::string_view strategy = {}) {
        if (!requested_side.empty() && ((action != "reset" && action != "body") ||
            (requested_side != "left" && requested_side != "right"))) {
            throw std::invalid_argument("side must be left or right and is accepted only with reset or body");
        }
        if (action != "body" && (!variant.empty() || !strategy.empty())) {
            throw std::invalid_argument("variant and strategy are accepted only with body");
        }
        if (action == "body") {
            if ((!variant.empty() && variant != "normal" && variant != "reversed" && variant != "disconnected") ||
                (!strategy.empty() && strategy != "discover" && strategy != "together")) {
                throw std::invalid_argument("body needs a normal/reversed/disconnected variant and discover/together strategy");
            }
            reset_body_locked(requested_side.empty() ? side : requested_side,
                variant.empty() ? body_variant : variant, strategy.empty() ? body_strategy : strategy);
            return;
        }
        if (action == "reset") {
            reset_learned_locked(requested_side.empty() ? side : requested_side);
            return;
        }
        if (action == "demo") {
            environment.set_demo_running(false);
            environment.reset_visualization();
            mode = "demo";
            status = "paused";
            step = 0;
            elapsed_s = 0.0;
            initial_lux = environment.visualization_state().at("sensors").at("ambient_light_lux").get<double>();
            error.clear();
            ++session_id;
            return;
        }
        if (action != "play" && action != "pause") {
            throw std::invalid_argument("action must be play, pause, reset, body, or demo");
        }
        if (mode == "external") {
            throw std::logic_error("external agent owns control; explicitly reset the playground or select demo");
        }
        if (status == "error") {
            throw std::logic_error("playground has an error; reset to retry or explicitly select demo");
        }
        if (action == "pause") {
            if (mode == "demo") {
                environment.set_demo_running(false);
            }
            if (status == "running") {
                status = "paused";
            }
            return;
        }
        if (status == "completed") {
            throw std::logic_error("playground episode is complete; reset before playing again");
        }
        if (mode == "demo") {
            environment.set_demo_running(true);
        }
        status = "running";
    }

    DroidEnvironment& environment;
    std::filesystem::path model_path;
    std::filesystem::path artifact_path;
    mutable std::mutex mutex;
    std::condition_variable wake;
    bool stopping{false};
    std::thread worker;
    std::unique_ptr<SharedLinearMemoryPolicy> policy;
    std::unique_ptr<BodyProbeController> body_controller;
    std::string body_variant{"normal"};
    std::string body_strategy{"discover"};
    bool body_has_light{false};
    std::vector<std::string> motor_ids;
    Json observation;
    std::string mode{"learned"};
    std::string status{"error"};
    std::string side{"left"};
    std::string error;
    std::uint64_t session_id{0};
    std::size_t step{0};
    double elapsed_s{0.0};
    double initial_lux{0.0};
    double current_lux{0.0};
};

PlaygroundSession::PlaygroundSession(
    DroidEnvironment& environment,
    const std::filesystem::path& model_path,
    const std::filesystem::path& artifact_path,
    bool realtime)
    : impl_(std::make_unique<Impl>(environment, model_path, artifact_path, realtime)) {}

PlaygroundSession::~PlaygroundSession() = default;

Json PlaygroundSession::state() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->state_locked();
}

Json PlaygroundSession::health() const {
    std::scoped_lock lock(impl_->mutex);
    Json health = impl_->environment.health();
    health["playground"] = Json{{"mode", impl_->mode}, {"status", impl_->status}};
    if (impl_->status == "error") {
        health["status"] = "error";
        health["error"] = impl_->error;
        health["playground"]["error"] = impl_->error;
    }
    return health;
}

Json PlaygroundSession::catalog() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->environment.catalog();
}

Json PlaygroundSession::spec() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->environment.spec();
}

Json PlaygroundSession::trace() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->environment.trace();
}

Json PlaygroundSession::control(std::string_view action, std::string_view side,
                               std::string_view variant, std::string_view strategy) {
    std::scoped_lock lock(impl_->mutex);
    impl_->control_locked(action, side, variant, strategy);
    return impl_->state_locked();
}

Json PlaygroundSession::legacy_control(std::string_view action) {
    std::scoped_lock lock(impl_->mutex);
    if (action != "play" && action != "pause" && action != "reset") {
        throw std::invalid_argument("action must be play, pause, or reset");
    }
    if (impl_->mode == "external") {
        throw std::logic_error("external agent owns control; explicitly reset the playground or select demo");
    }
    impl_->control_locked(action == "reset" ? (impl_->mode == "demo" ? "demo" : impl_->mode == "body_lab" ? "body" : "reset") : action, {});
    return impl_->state_locked();
}

void PlaygroundSession::tick() {
    std::scoped_lock lock(impl_->mutex);
    impl_->tick_locked();
}

Json PlaygroundSession::agent_reset(
    std::string_view assembly_id, std::uint64_t seed, bool recording,
    std::string_view episode_profile) {
    std::scoped_lock lock(impl_->mutex);
    Json reset = impl_->environment.reset(assembly_id, seed, recording, episode_profile);
    impl_->mode = "external";
    impl_->status = "stopped";
    impl_->step = 0;
    impl_->elapsed_s = 0.0;
    impl_->initial_lux = 0.0;
    for (const auto& sensor : reset.at("observation").at("reward_sensors")) {
        if (sensor.at("family_id") == "ambient_light_v0" && sensor.at("valid") == true)
            impl_->initial_lux = sensor.at("observations").at("illuminance_lux").get<double>();
    }
    impl_->error.clear();
    ++impl_->session_id;
    return reset;
}

Json PlaygroundSession::agent_step(
    const std::map<std::string, double>& actions, double control_dt_s) {
    std::scoped_lock lock(impl_->mutex);
    if (impl_->mode != "external") {
        throw std::logic_error("agent step requires an explicit agent reset to take control");
    }
    Json transition = impl_->environment.step(actions, control_dt_s);
    ++impl_->step;
    impl_->elapsed_s += control_dt_s;
    if (transition.at("terminated").get<bool>() || transition.at("truncated").get<bool>()) {
        impl_->status = "completed";
    }
    return transition;
}

}  // namespace droid
