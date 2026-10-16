#include "droid/pretrained_session.hpp"
#include "droid/pretrained_policy.hpp"
#include "droid/construction.hpp"
#include "droid/light_learner.hpp"
#include "droid/policy.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr double kDt = .02;
constexpr std::size_t kRunSteps = 3200;
constexpr std::array<double,3> kRequests{-.15,0.,.15};

Json configuration(const std::string& body, const std::string& motor, const std::string& lamp) {
    if (body != "long_upper" && body != "long_lower" && body != "weighted")
        throw std::invalid_argument("Choose long_upper, long_lower or weighted body");
    if (motor != "base" && motor != "elbow") throw std::invalid_argument("Choose base or elbow motor");
    if (lamp != "left" && lamp != "right") throw std::invalid_argument("Choose left or right lamp");
    const Json lengths = body == "long_upper" ? Json{5,3} : body == "long_lower" ? Json{3,5} : Json{5,5};
    const Json blocks = body == "weighted"
        ? Json::array({Json{{"id","block-1"},{"segment",1},{"slot",0},{"side",-1}}}) : Json::array();
    return Json{{"assembly", {{"schema","construction_kit_v2"},
        {"name","Pretrained sun-seeker"},{"segments",lengths},
        {"powered_hinge",motor == "base" ? 0 : 1},{"blocks",blocks},
        {"sensor",{{"segment",1},{"slot",2},{"side",-1}}},
        {"light",{{"segment",1},{"slot",2},{"side",1},{"face",1}}}}},
        {"sun",{{"position_m",{lamp == "left" ? -.3 : .3,0.,-.3}},{"intensity_lux",1000.}}}};
}

std::unique_ptr<ConstructionWorld> fresh_world(const Json& config) {
    auto result = std::make_unique<ConstructionWorld>();
    result->rebuild(config.at("assembly"));
    result->set_sun(config.at("sun"));
    result->reset(); // Queue the initial sample under the configured lamp.
    return result;
}

Json reading(const Json& observation) {
    for (const auto& sensor : observation.at("reward_sensors")) {
        if (sensor.at("family_id") == "ambient_light_v0") {
            const bool valid = sensor.at("valid").get<bool>();
            return Json{{"light_valid",valid}, {"illuminance_lux", valid
                ? sensor.at("observations").at("illuminance_lux") : Json(nullptr)}};
        }
    }
    throw std::runtime_error("Missing local light sensor");
}
} // namespace

struct PretrainedSession::Impl {
    explicit Impl(bool realtime, const std::filesystem::path& actor)
        : threaded(realtime), policy(actor) {
        config = configuration("long_upper","base","left");
        selected = Json{{"body","long_upper"},{"motor","base"},{"lamp","left"}};
        world = fresh_world(config);
        clear_locked();
        if (threaded) worker = std::thread([this] {
            std::unique_lock lock(mutex);
            while (!wake.wait_for(lock,std::chrono::milliseconds(20),[this] { return stopping; })) tick_locked();
        });
    }
    ~Impl() {
        { std::scoped_lock lock(mutex); stopping = true; }
        wake.notify_all();
        if (worker.joinable()) worker.join();
    }
    void clear_locked() {
        frames.reset(world->observation());
        status = "ready";
        error.clear();
        steps = decisions = pending_steps = 0;
        previous_effort = request = pending_reward = total_reward = 0.;
        rng_state = 1;
        history = Json::array();
        completed_trace = Json::array();
        pending_trace = Json(nullptr);
        audit.clear();
        ++session_id;
    }
    // Portable SplitMix64-v1; one top-53-bit uniform per actor decision.
    // This preserves categorical sampling, not Torch's seeded action stream.
    double uniform() {
        std::uint64_t value = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
        value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
        value ^= value >> 31;
        return static_cast<double>(value >> 11) * 0x1.0p-53;
    }
    void tick_locked() {
        if (status != "running") return;
        try {
            if (pending_steps == 0) {
                const double draw = uniform();
                const auto probabilities = policy.probabilities(frames.values());
                const std::size_t action = PretrainedPolicy::categorical(probabilities,draw);
                request = kRequests.at(action);
                ++decisions;
                pending_trace = Json{{"action",action},{"uniform",draw},{"probabilities",probabilities},
                    {"request",request},{"start_step",steps},{"efforts",Json::array()},
                    {"input_float32_le_sha256",frames.diagnostics().at("input_float32_le_sha256")}};
                pending_reward = 0.;
                audit.clear();
            }
            previous_effort = LightLearner::bounded_effort(request,previous_effort,world->observation());
            const Json transition = world->step(previous_effort);
            ++steps;
            ++pending_steps;
            pending_reward += transition.at("reward").get<double>();
            total_reward += transition.at("reward").get<double>();
            pending_trace["efforts"].push_back(previous_effort);
            audit += Json{{"effort",previous_effort},{"transition",transition}}.dump() + '\n';
            const auto& feedback = transition.at("observation").at("actuator_feedback").at(0).at("feedback");
            const bool stopped = transition.at("terminated").get<bool>() ||
                transition.at("safety").at("veto").get<bool>() || !feedback.at("fault_flags").empty();
            if (stopped) status = "stopped";
            else if (steps >= kRunSteps) status = "completed";
            if (pending_steps == 5 || stopped) {
                const Json observation = world->observation();
                frames.append(observation,previous_effort,request,pending_reward);
                Json sample = reading(observation);
                sample.update(Json{{"time_s",steps*kDt},{"effort",previous_effort},
                    {"request",request},{"reward",pending_reward}});
                history.push_back(std::move(sample));
                pending_trace.update(Json{{"reward",pending_reward},{"native_steps",pending_steps},
                    {"elapsed_s",kDt*pending_steps},{"time_s",steps*kDt},{"terminated",stopped},
                    {"observation_sha256",sha256_hex(observation.dump())},
                    {"transition_sha256",sha256_hex(audit)}});
                completed_trace.push_back(std::move(pending_trace));
                pending_trace = Json(nullptr);
                pending_steps = 0;
                pending_reward = 0.;
                audit.clear();
            }
        } catch (const std::exception& exception) {
            status = "error";
            error = exception.what(); // Hold the failed prefix. No automatic reset.
        }
    }
    Json policy_metadata() const {
        Json value = policy.metadata();
        value["actor_sha256"] = value.at("actor_bytes_sha256");
        value["source_checkpoint_sha256"] = "4c7ce76f644dd07d41340ef311d5f797578fa58f0470e2b89aa49c0d3b93082c";
        value.update(Json{{"id","pooled-seed101-v1"},{"source_seed",101},{"training","pooled"},
            {"frozen",true},{"parameters",17475},{"sampler","splitmix64-top53-categorical-v1"},
            {"sampler_seed",1},{"torch_seeded_trajectory_identity",false}});
        return value;
    }
    Json state_locked() const {
        const Json physical = world->state();
        return Json{{"schema","pretrained_session_v1"},{"session_id",session_id},
            {"assembly_revision",assembly_revision},{"status",status},{"running",status == "running"},
            {"case",selected},{"policy",policy_metadata()},{"assembly",world->assembly()},
            {"physics",physical},{"observation",world->observation()},{"history",history},
            {"elapsed_s",steps*kDt},{"duration_s",kRunSteps*kDt},{"native_steps",steps},
            {"macro_decisions",decisions},{"history_frames",frames.count()},
            {"brain_history",frames.diagnostics()},{"pending_native_steps",pending_steps},
            {"previous_effort",previous_effort},{"previous_request",request},
            {"total_reward",total_reward},{"safety",physical.at("safety")},{"error",error}};
    }
    Json trace_locked() const {
        return Json{{"schema","pretrained_session_trace_v1"},{"session_id",session_id},
            {"case",config},{"selection",selected},{"policy",policy_metadata()},
            {"sampler_seed",1},{"status",status},{"native_steps",steps},
            {"decisions",completed_trace},{"pending_decision",pending_trace},
            {"total_reward",total_reward},{"final_observation",world->observation()},
            {"final_physics",world->state()}};
    }
    Json control_locked(const Json& input) {
        if (!input.is_object() || !input.contains("action") || !input.at("action").is_string())
            throw std::invalid_argument("pretrained action must be a string");
        const std::string action = input.at("action").get<std::string>();
        if (action != "wake" && action != "pause" && action != "rebuild")
            throw std::invalid_argument("Choose wake, pause or rebuild");
        for (auto it = input.begin(); it != input.end(); ++it) {
            const auto& field = it.key();
            if (field != "action" && field != "expected_session_id" &&
                !(action == "rebuild" && (field == "body" || field == "motor" || field == "lamp")))
                throw std::invalid_argument("Unknown pretrained control field: " + field);
        }
        if (!input.contains("expected_session_id") || !input.at("expected_session_id").is_number_integer() ||
            input.at("expected_session_id") <= 0)
            throw std::invalid_argument("Control requires a positive integer expected_session_id");
        if (input.at("expected_session_id") != session_id)
            throw std::logic_error("This body has changed. Refresh before controlling it.");
        if (action == "wake") {
            if (status != "ready" && status != "paused")
                throw std::logic_error("Pause a running body or rebuild a finished body first.");
            status = "running";
        } else if (action == "pause") {
            if (status == "running") status = "paused";
        } else {
            if (status == "running") throw std::logic_error("Pause before rebuilding this body.");
            for (const auto& key : {"body","motor","lamp"})
                if (!input.contains(key) || !input.at(key).is_string())
                    throw std::invalid_argument("Rebuild requires body, motor and lamp choices");
            const Json replacement_config = configuration(input.at("body"),input.at("motor"),input.at("lamp"));
            auto replacement = fresh_world(replacement_config);
            // Validate the initial stack before replacing any live state.
            PretrainedHistory replacement_frames(replacement->observation());
            world = std::move(replacement);
            config = replacement_config;
            selected = Json{{"body",input.at("body")},{"motor",input.at("motor")},{"lamp",input.at("lamp")}};
            ++assembly_revision;
            clear_locked();
        }
        return state_locked();
    }
    mutable std::mutex mutex;
    std::condition_variable wake;
    bool threaded{}, stopping{};
    const PretrainedPolicy policy;
    PretrainedHistory frames;
    std::unique_ptr<ConstructionWorld> world;
    Json selected, config, history, completed_trace, pending_trace;
    std::string status{"ready"}, error, audit;
    std::size_t steps{}, decisions{}, pending_steps{}, session_id{}, assembly_revision{1};
    std::uint64_t rng_state{1};
    double previous_effort{}, request{}, pending_reward{}, total_reward{};
    std::thread worker;
};

PretrainedSession::PretrainedSession(bool realtime,const std::filesystem::path& actor)
    : impl_(std::make_unique<Impl>(realtime,actor)) {}
PretrainedSession::~PretrainedSession() = default;
Json PretrainedSession::state() const { std::scoped_lock lock(impl_->mutex); return impl_->state_locked(); }
Json PretrainedSession::trace() const { std::scoped_lock lock(impl_->mutex); return impl_->trace_locked(); }
Json PretrainedSession::control(const Json& request) { std::scoped_lock lock(impl_->mutex); return impl_->control_locked(request); }
Json PretrainedSession::specification() {
    return Json{{"schema","pretrained_session_v1"},{"duration_s",64.},
        {"bodies",{"long_upper","long_lower","weighted"}},{"motors",{"base","elbow"}},
        {"lamps",{"left","right"}},{"decision_dt_s",.1},{"native_dt_s",.02},
        {"actor","pooled-seed101-v1; frozen categorical PPO actor, no optimizer"},
        {"rebuild","fresh physics, local history and sampler; identical pretrained weights"},
        {"pause","preserves physics, local history, sampler and incomplete decision"},
        {"scope","Anchored planar two-link simulation, one motor, one IMU, one light; these layouts have already been evaluated"},
        {"safety","Native governor every .02s; any veto or motor fault stops without reset rescue"},
        {"physics",ConstructionWorld::light_specification()}};
}
void PretrainedSession::advance_steps(std::size_t count) {
    std::scoped_lock lock(impl_->mutex);
    if (impl_->threaded) throw std::logic_error("Explicit stepping requires a worker-free pretrained session");
    if (count > kRunSteps) throw std::invalid_argument("At most 3200 native steps per call");
    for (std::size_t i = 0; i < count && impl_->status == "running"; ++i) impl_->tick_locked();
}
} // namespace droid
