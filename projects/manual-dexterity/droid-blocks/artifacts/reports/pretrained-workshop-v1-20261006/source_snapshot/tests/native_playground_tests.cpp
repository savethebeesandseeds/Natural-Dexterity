#include "droid/playground.hpp"
#include "droid/learner.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;
constexpr std::string_view kModel{"models/droid.xml"};
constexpr std::string_view kArtifact{"artifacts/light-search-policy-v3-seed0.json"};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function>
void require_throws(Function&& function, std::string_view message) {
    try {
        std::invoke(std::forward<Function>(function));
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

[[nodiscard]] Json physical_state(Json state) {
    state.erase("playground");
    return state;
}

void test_approved_worlds_and_frozen_identity() {
    const auto splits = droid::make_mirrored_seed_splits();
    require(splits.train.at(0) == droid::kPlaygroundLeftSeed &&
                splits.train.at(1) == droid::kPlaygroundRightSeed,
            "playground worlds changed from the published first training pair");
    for (const auto seed : {droid::kPlaygroundLeftSeed, droid::kPlaygroundRightSeed}) {
        require(std::find(splits.train.begin(), splits.train.end(), seed) != splits.train.end(),
                "playground seed is absent from training");
        require(std::find(splits.validation.begin(), splits.validation.end(), seed) == splits.validation.end() &&
                    std::find(splits.test.begin(), splits.test.end(), seed) == splits.test.end(),
                "playground seed overlaps a protected split");
    }
    require(!droid::light_search_seed_has_positive_side(droid::kPlaygroundLeftSeed) &&
                droid::light_search_seed_has_positive_side(droid::kPlaygroundRightSeed),
            "playground side mapping changed");
    require(droid::sha256_file(kArtifact) == droid::kPlaygroundArtifactFileSha256,
            "frozen artifact bytes changed");
    require(droid::load_policy_artifact(kArtifact).self_sha256 == droid::kPlaygroundArtifactSha256,
            "frozen artifact canonical identity changed");
}

void test_native_inference_both_sides() {
    for (const std::string side : {"left", "right"}) {
        droid::DroidEnvironment environment(kModel, false);
        droid::PlaygroundSession playground(environment);
        Json state = playground.control("reset", side);
        require(state.at("playground").at("status") == "ready", "learned reset failed");
        require(state.at("playground").at("mode") == "learned", "reset did not select learned");
        require(state.at("playground").at("exposure") == "known_training_world", "world exposure missing");
        const auto session_id = state.at("playground").at("session_id").get<std::uint64_t>();
        const Json reset_physics = physical_state(state);
        playground.tick();
        require(playground.state() == state, "ready state advanced without play");

        // Independent execution of the actual frozen policy is an oracle for
        // every action/transition. It receives the exact public observation
        // boundary, never playground side, pose, reward, or display metadata.
        droid::DroidEnvironment reference(kModel, false);
        droid::SharedLinearMemoryPolicy policy(droid::load_policy_artifact(kArtifact));
        const auto ids = reference.spec().at("action").at("motor_ids").get<std::vector<std::string>>();
        policy.reset(ids);
        Json observation = reference.reset(
            "demo_rover_v0", side == "left" ? droid::kPlaygroundLeftSeed : droid::kPlaygroundRightSeed,
            false, droid::kLightSearch1dEpisodeProfile).at("observation");
        require(reset_physics == reference.visualization_state(), "learned reset differs from native reset");
        const double initial_lux = state.at("playground").at("initial_illuminance_lux");
        (void)playground.control("play");
        std::vector<Json> first_states;
        for (std::size_t step = 0; step < droid::kPlaygroundMaxSteps; ++step) {
            if (step == 37) {
                (void)playground.legacy_control("pause");
                const Json paused = playground.state();
                for (int tick = 0; tick < 8; ++tick) {
                    playground.tick();
                }
                require(playground.state() == paused, "paused world or policy advanced");
                require(paused.at("playground").at("status") == "paused", "pause status missing");
                (void)playground.legacy_control("play");
            }
            const auto actions = policy.act(observation, ids);
            const Json transition = reference.step(actions, droid::kPlaygroundControlDtS);
            observation = transition.at("observation");
            require(!transition.at("terminated").get<bool>() && !transition.at("truncated").get<bool>(),
                    "known training-world reference stopped unexpectedly");
            playground.tick();
            state = playground.state();
            require(physical_state(state).dump() == reference.visualization_state().dump(),
                    "playground inference diverged from observation-only frozen policy execution");
            require(state.at("playground").at("step") == step + 1, "playground step accounting differs");
            require(std::abs(state.at("playground").at("elapsed_s").get<double>() -
                             static_cast<double>(step + 1) * droid::kPlaygroundControlDtS) < 1e-12,
                    "playground time accounting differs");
            if (step < 80) {
                first_states.push_back(physical_state(state));
            }
        }
        const Json metadata = state.at("playground");
        require(metadata.at("status") == "completed", "playground did not stop at horizon");
        require(metadata.at("elapsed_s") == 20.0, "playground horizon is not twenty seconds");
        const double final_lux = metadata.at("illuminance_lux");
        require(final_lux > initial_lux + 1.0, "known training world did not finish in brighter light");
        std::cout << "known training world " << side << ": " << initial_lux
                  << " -> " << final_lux << " lux in 1000 native steps\n";
        playground.tick();
        require(playground.state() == state, "completed world advanced");
        require_throws([&] { (void)playground.control("play"); }, "completed world resumed without reset");

        const Json restarted = playground.legacy_control("reset");
        require(restarted.at("playground").at("status") == "ready" &&
                    restarted.at("playground").at("side") == side,
                "legacy reset did not preserve learned world and mode");
        require(restarted.at("playground").at("session_id").get<std::uint64_t>() > session_id,
                "reset did not advance session identity");
        require(physical_state(restarted) == reset_physics, "reset did not restore exact initial physics");
        (void)playground.control("play");
        for (const Json& expected : first_states) {
            playground.tick();
            require(physical_state(playground.state()) == expected, "reset did not clear learned policy memory");
        }
    }
}

void test_control_ownership_and_unchanged_boundaries() {
    droid::DroidEnvironment environment(kModel, false);
    const Json original_spec = environment.spec();
    droid::PlaygroundSession playground(environment);
    require(playground.spec() == original_spec, "playground modified frozen agent spec");
    const Json before = playground.state();
    require_throws([&] { (void)playground.control("play", "right"); }, "play accepted a hidden reset");
    require_throws([&] { (void)playground.control("reset", "test"); }, "reset accepted an unapproved world");
    require_throws([&] { (void)playground.control("train"); }, "playground accepted training action");
    require_throws([&] { (void)playground.agent_step({}, 0.02); }, "agent stole learned control without reset");
    require(playground.state() == before, "rejected commands mutated learned state");

    (void)playground.control("play");
    playground.tick();
    const Json reset = playground.agent_reset(
        "demo_rover_v0", droid::kPlaygroundLeftSeed, true, droid::kLightSearch1dEpisodeProfile);
    require(!reset.contains("playground") && !reset.at("observation").contains("playground"),
            "playground metadata leaked into agent reset");
    const Json external = playground.state();
    require(external.at("playground").at("mode") == "external", "agent reset did not relinquish learned mode");
    playground.tick();
    require(playground.state() == external, "learned worker advanced external episode");
    require_throws([&] { (void)playground.legacy_control("play"); }, "legacy play stole external control");
    require_throws([&] { (void)playground.legacy_control("reset"); }, "legacy reset stole external control");
    std::map<std::string, double> zero;
    for (const auto& id : original_spec.at("action").at("motor_ids")) {
        zero.emplace(id.get<std::string>(), 0.0);
    }
    const Json transition = playground.agent_step(zero, 0.02);
    require(!transition.contains("playground"), "playground metadata leaked into agent step");
    require(playground.trace().at("steps").size() == 1, "external trace was not preserved");

    const Json demo = playground.control("demo");
    require(demo.at("playground").at("mode") == "demo" &&
                demo.at("playground").at("status") == "paused", "explicit demo did not prepare paused demo");
    playground.tick();
    require(playground.state() == demo, "learned worker advanced demo");
    require(playground.legacy_control("reset").at("playground").at("mode") == "demo",
            "legacy reset changed demo mode");
    require_throws([&] { (void)playground.agent_step(zero, 0.02); }, "agent stole demo ownership");
    require(playground.control("reset", "right").at("playground").at("mode") == "learned",
            "explicit playground reset did not recover learned ownership");
}

class TemporaryFile final {
public:
    TemporaryFile() : path(std::filesystem::temp_directory_path() /
        ("droid-playground-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}
    ~TemporaryFile() { std::error_code error; std::filesystem::remove(path, error); }
    std::filesystem::path path;
};

void test_fail_closed_and_recovery() {
    TemporaryFile artifact;
    droid::DroidEnvironment environment(kModel, false);
    droid::PlaygroundSession playground(environment, kModel, artifact.path);
    const Json failed = playground.state();
    require(failed.at("playground").at("status") == "error" &&
                failed.at("playground").contains("error"), "missing artifact did not fail visibly");
    require(playground.health().at("status") == "error", "policy failure reported healthy");
    playground.tick();
    require(playground.state() == failed, "failed policy advanced physics");
    require_throws([&] { (void)playground.control("play"); }, "failed policy allowed play");
    std::filesystem::copy_file(kArtifact, artifact.path);
    require(playground.control("reset").at("playground").at("status") == "ready", "valid artifact did not recover on reset");
    require(playground.health().at("status") == "ok", "recovered policy still reported unhealthy");
    (void)playground.control("play");
    playground.tick();
    const Json preserved = physical_state(playground.state());
    {
        std::ofstream output(artifact.path, std::ios::binary | std::ios::app);
        output << '\n';
    }
    const Json altered = playground.control("reset");
    require(altered.at("playground").at("status") == "error", "changed frozen file bytes were accepted");
    require(physical_state(altered) == preserved, "failed reset erased inspectable physical state");
    require_throws([&] { (void)playground.control("play"); }, "failed reset resumed potentially advanced memory");

    TemporaryFile model;
    std::filesystem::copy_file(kModel, model.path);
    {
        std::ofstream output(model.path, std::ios::binary | std::ios::app);
        output << '\n';
    }
    droid::DroidEnvironment changed_environment(model.path, false);
    droid::PlaygroundSession changed(changed_environment, model.path, kArtifact);
    require(changed.state().at("playground").at("status") == "error", "incompatible actual model accepted frozen policy");
}

void test_inference_failure_requires_reset() {
    droid::DroidEnvironment environment(kModel, false);
    droid::PlaygroundSession playground(environment);
    (void)playground.control("play");
    playground.tick();
    const Json first_step = physical_state(playground.state());

    // Deliberately violate the session's exclusive-ownership contract to
    // inject an engine lifecycle failure after act() advances policy memory.
    // Production HTTP routes cannot reach the environment this way.
    environment.reset_visualization();
    const Json preserved = environment.visualization_state();
    playground.tick();
    const Json failed = playground.state();
    require(failed.at("playground").at("status") == "error", "failed native step did not stop learned inference");
    require(physical_state(failed) == preserved, "failed native step advanced physical state");
    playground.tick();
    require(playground.state() == failed, "error state resumed learned inference");
    require_throws([&] { (void)playground.control("play"); }, "error resumed with policy memory ahead of physics");
    (void)playground.control("reset");
    (void)playground.control("play");
    playground.tick();
    require(physical_state(playground.state()) == first_step, "explicit reset did not recover both physics and policy memory");
}

template <typename Predicate>
void wait_until(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline, "native worker did not advance within timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void test_realtime_worker_handoffs() {
    droid::DroidEnvironment environment(kModel, true);
    droid::PlaygroundSession playground(environment, kModel, kArtifact, true);
    (void)playground.control("play");
    wait_until([&] { return playground.state().at("playground").at("step").get<int>() >= 3; });
    const Json paused = playground.control("pause");
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    require(playground.state() == paused, "native learned worker advanced during pause");
    (void)playground.control("play");
    wait_until([&] { return playground.state().at("playground").at("step") > paused.at("playground").at("step"); });

    (void)playground.control("demo");
    (void)playground.control("play");
    wait_until([&] { return playground.state().at("sim_time").get<double>() > 0.01; });
    const Json reset = playground.control("reset", "right");
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    require(playground.state() == reset, "demo worker advanced learned ready state after handoff");
    (void)playground.control("play");
    wait_until([&] { return playground.state().at("playground").at("step").get<int>() >= 2; });
    (void)playground.agent_reset("demo_rover_v0", droid::kPlaygroundLeftSeed, false, droid::kLightSearch1dEpisodeProfile);
    const Json external = playground.state();
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    require(playground.state() == external, "learned worker advanced external agent after ownership handoff");
}

void test_body_experiment_lifecycle() {
    droid::DroidEnvironment environment(kModel, false);
    droid::PlaygroundSession playground(environment);
    const Json original = physical_state(playground.state());
    const Json original_spec = playground.spec();
    const Json ready = playground.control("body", "left", "reversed", "discover");
    require(ready.at("playground").at("mode") == "body_lab", "body reset did not select experiment");
    require(ready.at("playground").at("artifact_sha256").is_null(), "body claimed frozen policy provenance");
    require(ready.at("playground").at("duration_s") == 40.0, "wrong body horizon");
    require(ready.at("playground").at("body_experiment").at("light_sample_valid") == false, "body received future initial sample");
    require(playground.spec() == original_spec, "body changed frozen environment spec");
    require_throws([&] { (void)playground.control("body", "left", "invented", "discover"); }, "invalid body accepted");
    require_throws([&] { (void)playground.control("body", "left", "normal", "invented"); }, "invalid strategy accepted");
    require_throws([&] { (void)playground.control("play", {}, "normal"); }, "variant accepted on play");
    require(playground.state() == ready, "invalid request changed body state");
    (void)playground.control("play");
    for (int i = 0; i < 200; ++i) playground.tick();
    const Json paused = playground.control("pause");
    require(paused.at("playground").at("status") == "paused", "body did not pause");
    for (int i = 0; i < 20; ++i) playground.tick();
    require(playground.state() == paused, "paused body advanced memory or physics");
    const Json repeat_ready = playground.legacy_control("reset");
    require(repeat_ready.at("playground").at("mode") == "body_lab", "legacy reset lost body mode");
    (void)playground.control("play");
    for (int i = 0; i < 200; ++i) playground.tick();
    const Json repeated = playground.control("pause");
    require(physical_state(repeated) == physical_state(paused), "body reset did not reproduce physics");
    require(repeated.at("playground").at("body_experiment") == paused.at("playground").at("body_experiment"), "body reset did not reproduce probe memory");
    (void)playground.control("play");
    for (int i = 200; i < 2000; ++i) playground.tick();
    const Json complete = playground.state();
    require(complete.at("playground").at("status") == "completed", "body failed to complete");
    playground.tick();
    require(playground.state() == complete, "completed body advanced");
    require_throws([&] { (void)playground.control("play"); }, "completed body resumed without reset");
    (void)playground.agent_reset("transmission_disconnected_v1", droid::kPlaygroundLeftSeed, false, "body_discovery_1d_v1");
    require(playground.state().at("playground").at("mode") == "external", "body external reset failed on initial invalid light");
    require_throws([&] { (void)playground.control("play"); }, "body external owner was stolen");
    const Json restored = playground.control("reset", "left");
    require(restored.at("playground").at("status") == "ready", "original learned policy could not recover after body");
    require(physical_state(restored) == original, "original model not restored after body experiment");
}

void test_failed_body_reset_preserves_identity() {
    TemporaryFile model;
    std::filesystem::copy_file(kModel, model.path);
    droid::DroidEnvironment environment(model.path, false);
    droid::PlaygroundSession playground(environment, model.path, kArtifact);
    const Json previous = playground.state();
    { std::ofstream output(model.path, std::ios::trunc); output << "invalid XML"; }
    const Json failed = playground.control("body", "right", "disconnected", "discover");
    require(failed.at("playground").at("mode") == previous.at("playground").at("mode"), "failed body reset falsely changed scene identity");
    require(failed.at("playground").at("status") == "error", "failed body reset did not surface error");
    require(!failed.at("playground").contains("body_experiment"), "failed body reset exposed stale probe evidence");
    require(physical_state(failed) == physical_state(previous), "failed model compilation replaced preserved scene");
    require_throws([&] { (void)playground.control("play"); }, "failed body reset allowed autonomous play");
}

}  // namespace

int main() {
    try {
        test_approved_worlds_and_frozen_identity();
        test_native_inference_both_sides();
        test_control_ownership_and_unchanged_boundaries();
        test_fail_closed_and_recovery();
        test_inference_failure_requires_reset();
        test_realtime_worker_handoffs();
        test_body_experiment_lifecycle();
        test_failed_body_reset_preserves_identity();
        std::cout << "native playground tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native playground tests failed: " << error.what() << '\n';
        return 1;
    }
}
