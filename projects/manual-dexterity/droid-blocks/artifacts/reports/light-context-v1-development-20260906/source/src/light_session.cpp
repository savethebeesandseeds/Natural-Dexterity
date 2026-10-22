#include "droid/light_session.hpp"
#include "droid/construction.hpp"
#include "droid/light_learner.hpp"
#include "droid/policy.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr double kDt = .02;
constexpr std::size_t kRunSteps = 6000;
bool valid_mode(const std::string& mode) {
    return mode == "learner" || mode == "zero" || mode == "rhythm" || mode == "random";
}
void fields(const Json& request, const std::string& action) {
    for (const auto& [key, value] : request.items()) {
        (void)value;
        if (key == "action" || (action == "build" && key == "assembly") ||
            (action == "start" && key == "mode") || (action == "sun" && key == "sun")) continue;
        throw std::invalid_argument("unexpected light experiment field: " + key);
    }
}
}
namespace {
void baseline_continuity(const Json& observation, const Json& previous, std::size_t steps) {
    const double time=steps*kDt;
    const auto& sensors=observation.at("reward_sensors");
    for (const auto& sensor:sensors) {
        if (sensor.at("valid")==true) {
            const double sample_time=sensor.at("sample_time_s").get<double>();
            if (std::abs(sample_time+sensor.at("age_s").get<double>()-time)>2e-6)
                throw std::invalid_argument("baseline observation differs from expected control time");
        } else if (steps!=0) throw std::invalid_argument("baseline sensor is invalid after initial delivery");
        if (previous.empty()) continue;
        const Json* last=nullptr;
        for (const auto& candidate:previous.at("reward_sensors"))
            if (candidate.at("family_id")==sensor.at("family_id")) last=&candidate;
        if (!last || last->at("module_id")!=sensor.at("module_id"))
            throw std::invalid_argument("baseline sensor identity changed without reset");
        if (last->at("valid")!=true) continue;
        const auto sequence=sensor.at("sequence").get<std::uint64_t>();
        const auto old_sequence=last->at("sequence").get<std::uint64_t>();
        if (sequence<old_sequence) throw std::invalid_argument("baseline sensor sequence moved backwards");
        if (sequence==old_sequence) {
            for (const std::string field:{"sample_time_s","delivered_time_s","observations"})
                if (sensor.at(field)!=last->at(field)) throw std::invalid_argument("baseline held sensor sample changed");
        } else if (sensor.at("sample_time_s").get<double>()<=last->at("sample_time_s").get<double>())
            throw std::invalid_argument("baseline new sample did not advance time");
    }
    if (!previous.empty() && observation.at("actuator_feedback").at(0).at("module_id")!=
        previous.at("actuator_feedback").at(0).at("module_id"))
        throw std::invalid_argument("baseline motor identity changed without reset");
}
}
struct LightControl::Impl {
    LightLearner learner;
    std::mt19937_64 random{1};
    std::string mode{"learner"};
    std::size_t steps{};
    double effort{}, requested{};
    bool pending{};
    Json last_observation;
};
LightControl::LightControl() : impl_(std::make_unique<Impl>()) {}
LightControl::~LightControl() = default;
void LightControl::reset(const std::string& mode, std::uint64_t seed) {
    if (!valid_mode(mode)) throw std::invalid_argument("unknown light controller");
    impl_->learner.reset(seed);
    impl_->random.seed(seed);
    impl_->mode = mode;
    impl_->steps = 0;
    impl_->effort = impl_->requested = 0;
    impl_->pending = false;
    impl_->last_observation = Json();
}
double LightControl::act(const Json& observation) {
    if (impl_->pending) throw std::logic_error("Observe the last action before choosing another.");
    double effort{};
    if (impl_->mode == "learner") effort = impl_->learner.act(observation);
    else {
        (void)LightLearner::bounded_effort(0, impl_->effort, observation);
        baseline_continuity(observation,impl_->last_observation,impl_->steps);
        double request = impl_->requested;
        if (impl_->steps < 5 || impl_->mode == "zero") request = 0;
        else if (impl_->mode == "rhythm") request = .15 * std::sin(2 * std::acos(-1.0) * .5 * impl_->steps * kDt);
        else if ((impl_->steps - 5) % 20 == 0) request = (static_cast<int>(impl_->random() % 3) - 1) * .15;
        effort = LightLearner::bounded_effort(request, impl_->effort, observation);
        impl_->requested = request;
        impl_->last_observation = observation;
    }
    impl_->effort = effort;
    impl_->pending = true;
    return effort;
}
void LightControl::observe(const Json& transition) {
    if (!impl_->pending) throw std::logic_error("Choose an action before observing its consequence.");
    if (impl_->mode == "learner") impl_->learner.observe(transition);
    else {
        if (transition.at("terminated").get<bool>() || transition.at("safety").at("veto").get<bool>() ||
            !transition.at("safety").at("flags").empty() ||
            !transition.at("observation").at("actuator_feedback").at(0).at("feedback").at("fault_flags").empty())
            throw std::runtime_error("Native feedback stopped this control. Reset before another run.");
        (void)LightLearner::bounded_effort(0, 0, transition.at("observation"));
        if (transition.at("truncated")!=false || transition.at("info").at("physics_steps_executed")!=10 ||
            std::abs(transition.at("info").at("elapsed_control_dt_s").get<double>()-kDt)>2e-6)
            throw std::invalid_argument("baseline transition duration mismatch");
        baseline_continuity(transition.at("observation"),impl_->last_observation,impl_->steps+1);
        impl_->last_observation=transition.at("observation");
    }
    impl_->pending = false;
    ++impl_->steps;
}
void LightControl::freeze_learning() {
    if (impl_->mode != "learner") throw std::logic_error("Only learner control has values to freeze.");
    if (impl_->pending) throw std::logic_error("Finish the pending act/observe pair before freezing learning.");
    impl_->learner.freeze_learning();
}
LightControl::ValueCheckpoint LightControl::capture_values() const {
    if (impl_->mode != "learner") throw std::logic_error("Only learner control has values to capture.");
    if (impl_->pending) throw std::logic_error("Finish the pending act/observe pair before capturing values.");
    return impl_->learner.capture_values();
}
void LightControl::restore_values(const ValueCheckpoint& checkpoint) {
    if (impl_->mode != "learner") throw std::logic_error("Only learner control has values to restore.");
    if (impl_->pending) throw std::logic_error("Finish the pending act/observe pair before restoring values.");
    impl_->learner.restore_values(checkpoint);
}
Json LightControl::diagnostics() const {
    if (impl_->mode == "learner") {
        Json result = impl_->learner.diagnostics();
        std::ostringstream random_state;
        random_state << impl_->random;
        Json state{{"schema", "light-control-complete-state-v1"},
            {"learner", result.at("state_fingerprint_sha256")}, {"random", random_state.str()},
            {"mode", impl_->mode}, {"steps", impl_->steps},
            {"effort_binary64", std::bit_cast<std::uint64_t>(impl_->effort)},
            {"requested_binary64", std::bit_cast<std::uint64_t>(impl_->requested)}, {"pending", impl_->pending},
            {"last_observation", impl_->last_observation}};
        result["state_fingerprint_sha256"] = sha256_hex(state.dump());
        state["schema"] = "light-control-nonparameter-state-v1";
        state["learner"] = result.at("nonparameter_state_fingerprint_sha256");
        result["nonparameter_state_fingerprint_sha256"] = sha256_hex(state.dump());
        return result;
    }
    return Json{{"algorithm_id", "light_comparison_controls_v1"}, {"mode", impl_->mode},
        {"action_label", impl_->mode == "zero" ? "Coast" : impl_->mode == "rhythm" ? "Fixed rhythm" : "Random pulse"},
        {"effort", impl_->effort}, {"decision_count", impl_->mode == "random" ? impl_->steps / 20 : 0},
        {"history_size", 0}, {"updates", 0}, {"learning", false}};
}
Json LightControl::specification() {
    return Json{{"schema", "light_controls_v1"}, {"learner", LightLearner::specification()},
        {"modes", Json::array({"learner", "zero", "rhythm", "random"})},
        {"baseline_warmup_s", .1}, {"baseline_amplitude", .15}, {"rhythm_frequency_hz", .5},
        {"random_hold_s", .4}, {"random_seed_mapping", "mt19937_64 output modulo 3: -1,0,+1"},
        {"shared_governor", "LightLearner::bounded_effort"},
        {"budget", "equal 120s simulated horizon and actuator bounds; actual energy is measured, not forced equal"}};
}
struct LightSession::Impl {
    explicit Impl(bool realtime) : threaded(realtime) {
        world.rebuild(ConstructionWorld::light_assembly());
        sun_events.push_back(Json{{"time_s",0.0},{"sun",world.state().at("sun")}});
        if (threaded) worker = std::thread([this] {
            std::unique_lock lock(mutex);
            while (!wake.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopping; })) tick_locked();
        });
    }
    ~Impl() {
        { std::scoped_lock lock(mutex); stopping = true; }
        wake.notify_all();
        if (worker.joinable()) worker.join();
    }
    void clear_locked() {
        controller.reset(mode);
        status = "ready";
        error.clear();
        steps = 0;
        history = Json::array();
        sun_events = Json::array({Json{{"time_s",0.0},{"sun",world.state().at("sun")}}});
        ++session_id;
    }
    void tick_locked() {
        if (status != "running") return;
        try {
            const double effort = controller.act(world.observation());
            const Json transition = world.step(effort);
            ++steps;
            controller.observe(transition);
            if (steps % 5 == 0) {
                const Json physical = world.state();
                const Json sensor = physical.at("light_sensor");
                history.push_back(Json{{"time_s", steps * kDt},
                    {"illuminance_lux", sensor.at("valid") == true ? sensor.at("observations").at("illuminance_lux") : Json(nullptr)},
                    {"reward_rate", physical.at("reward").at("rate")}, {"effort", effort}});
                if (history.size() > 1200) history.erase(history.begin());
            }
            if (steps >= kRunSteps) status = "completed";
        } catch (const std::exception& exception) { status = "error"; error = exception.what(); }
    }
    Json state_locked() const {
        return Json{{"schema", "light_session_v1"}, {"status", status}, {"mode", mode},
            {"session_id", session_id}, {"assembly_revision", assembly_revision},
            {"assembly", world.assembly()}, {"physics", world.state()}, {"observation", world.observation()},
            {"learner", controller.diagnostics()}, {"history", history}, {"sun_events", sun_events},
            {"elapsed_s", steps * kDt}, {"duration_s", kRunSteps * kDt}, {"error", error}};
    }
    Json control_locked(const Json& request) {
        if (!request.is_object() || !request.contains("action") || !request.at("action").is_string())
            throw std::invalid_argument("light experiment action must be a string");
        const std::string action = request.at("action").get<std::string>();
        fields(request, action);
        if (action == "sun") {
            if (!request.contains("sun")) throw std::invalid_argument("sun requires its position and intensity");
            world.set_sun(request.at("sun"));
            sun_events.push_back(Json{{"time_s", steps * kDt}, {"sun", request.at("sun")}});
            if (sun_events.size() > 1200) sun_events.erase(sun_events.begin());
        } else if (action == "pause") {
            if (status == "running") status = "paused";
        } else if (action == "play") {
            if (status != "paused") throw std::logic_error("Start a light experiment first.");
            status = "running";
        } else if (action == "reset") {
            world.reset();
            clear_locked();
        } else if (action == "build") {
            if (status == "running") throw std::logic_error("Pause before changing the body.");
            if (!request.contains("assembly") || !request.at("assembly").is_object() ||
                request.at("assembly").value("schema", "") != "construction_kit_v2")
                throw std::invalid_argument("This experiment needs a construction_kit_v2 body with a light sensor.");
            const Json sun = world.state().at("sun");
            world.rebuild(request.at("assembly"));
            world.set_sun(sun);
            ++assembly_revision;
            clear_locked();
        } else if (action == "start") {
            if (status == "running") throw std::logic_error("Pause before beginning another run.");
            if (!request.contains("mode") || !request.at("mode").is_string() || !valid_mode(request.at("mode")))
                throw std::invalid_argument("start requires learner, zero, rhythm or random mode");
            mode = request.at("mode").get<std::string>();
            world.reset();
            clear_locked();
            status = "running";
        } else throw std::invalid_argument("unknown light experiment action");
        return state_locked();
    }
    mutable std::mutex mutex;
    std::condition_variable wake;
    bool threaded{}, stopping{};
    ConstructionWorld world;
    LightControl controller;
    std::string status{"ready"}, mode{"learner"}, error;
    std::size_t steps{}, session_id{1}, assembly_revision{1};
    Json history = Json::array(), sun_events = Json::array();
    std::thread worker;
};
LightSession::LightSession(bool realtime) : impl_(std::make_unique<Impl>(realtime)) {}
LightSession::~LightSession() = default;
Json LightSession::state() const { std::scoped_lock lock(impl_->mutex); return impl_->state_locked(); }
Json LightSession::control(const Json& request) { std::scoped_lock lock(impl_->mutex); return impl_->control_locked(request); }
Json LightSession::specification() {
    return Json{{"schema", "light_session_v1"}, {"physics", ConstructionWorld::light_specification()},
        {"controller", LightControl::specification()}, {"duration_s", kRunSteps * kDt},
        {"moving_sun", "does not reset physics, pending samples or learner memory"},
        {"reset_and_rebuild", "fresh physics and fresh learner; current sun retained"},
        {"policy_inputs", "timestamped local sensors, actuator feedback and summed transition sensor reward"}};
}
void LightSession::advance_steps(std::size_t count) {
    std::scoped_lock lock(impl_->mutex);
    if (impl_->threaded) throw std::logic_error("Explicit stepping requires a worker-free light session.");
    for (std::size_t i = 0; i < count && impl_->status == "running"; ++i) impl_->tick_locked();
}
} // namespace droid
