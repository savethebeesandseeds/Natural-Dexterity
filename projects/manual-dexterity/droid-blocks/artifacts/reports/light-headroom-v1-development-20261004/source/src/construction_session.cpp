#include "droid/construction_session.hpp"
#include "droid/construction.hpp"
#include "droid/motion_explorer.hpp"

#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace droid {
namespace {
using Json = nlohmann::json;
constexpr double kDt = .02;

void require_fields(const Json& request, const std::string& action) {
    for (const auto& [key, value] : request.items()) {
        (void)value;
        if (key == "action" || (action == "build" && key == "assembly") ||
            (action == "replay" && key == "pattern_id")) continue;
        throw std::invalid_argument("unexpected construction field: " + key);
    }
}
}

struct ConstructionSession::Impl {
    explicit Impl(bool realtime) : threaded(realtime) {
        if (threaded) worker = std::thread([this] {
            std::unique_lock lock(mutex);
            while (!wake.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopping; })) {
                tick_locked();
            }
        });
    }
    ~Impl() {
        { std::scoped_lock lock(mutex); stopping = true; }
        wake.notify_all();
        if (worker.joinable()) worker.join();
    }

    void begin_trial_locked(const std::string& id) {
        world.reset();
        explorer.reset(id);
        active_pattern = id;
        trial_steps = 0;
        trial_efforts.clear();
        trail = Json::array();
        trail.push_back(Json{{"time_s", 0.0}, {"position", world.state().at("tip_position_m")}});
    }

    void tick_locked() {
        if (status != "running") return;
        try {
            const double effort = explorer.act(world.observation());
            if (mode == "replay") {
                const auto& original = recorded_efforts.at(active_pattern);
                if (trial_steps >= original.size() || effort != original[trial_steps])
                    throw std::runtime_error("Replay differed from the recorded motor sequence.");
            }
            const Json transition = world.step(effort);
            ++trial_steps;
            ++total_steps;
            trial_efforts.push_back(effort);
            // Diagnostic poses are never passed to the explorer.
            const Json physical = world.state();
            if (trial_steps % 2 == 0)
                trail.push_back(Json{{"time_s", trial_steps * kDt}, {"position", physical.at("tip_position_m")}});
            if (transition.at("terminated").get<bool>() || transition.at("safety").at("veto").get<bool>())
                throw std::runtime_error("Native safety stopped this motion. Inspect machine details.");
            explorer.observe(transition.at("observation"));
            if (!explorer.complete()) return;
            if (mode == "explore") {
                Json card = explorer.result();
                card["tip_trail"] = trail;
                repertoire.push_back(std::move(card));
                recorded_efforts[active_pattern] = trial_efforts;
                if (trial_index + 1 < patterns.size()) {
                    ++trial_index;
                    begin_trial_locked(patterns.at(trial_index).at("pattern_id").get<std::string>());
                    return;
                }
            }
            status = "completed";
        } catch (const std::exception& exception) {
            // No additional physics is advanced after an observation/controller fault.
            status = "error";
            error = exception.what();
        }
    }

    Json state_locked() const {
        double duration = 0.0;
        if (mode == "explore") {
            for (const auto& pattern : patterns) duration += pattern.at("duration_s").get<double>();
        } else if (mode == "replay") {
            for (const auto& pattern : patterns)
                if (pattern.at("pattern_id") == active_pattern) duration = pattern.at("duration_s").get<double>();
        }
        return Json{
            {"schema", "construction_session_v1"}, {"status", status}, {"mode", mode},
            {"session_id", session_id}, {"assembly_revision", assembly_revision},
            {"trial_index", trial_index}, {"trial_count", mode == "explore" ? patterns.size() : std::size_t{1}},
            {"active_pattern_id", active_pattern}, {"trial_steps", trial_steps},
            {"assembly", world.assembly()}, {"physics", world.state()},
            {"observation", world.observation()},
            {"explorer", active_pattern.empty() ? Json::object() : explorer.diagnostics()},
            {"patterns", patterns}, {"repertoire", repertoire}, {"trail", trail},
            {"elapsed_s", total_steps * kDt}, {"total_s", duration}, {"error", error},
        };
    }

    Json control_locked(const Json& request) {
        if (!request.is_object() || !request.contains("action") || !request.at("action").is_string())
            throw std::invalid_argument("construction action must be a string");
        const std::string action = request.at("action").get<std::string>();
        require_fields(request, action);
        if (action == "pause") {
            if (status == "running") status = "paused";
        } else if (action == "play") {
            if (status != "paused") throw std::logic_error("Choose Explore motions or a recorded motion to start.");
            status = "running";
        } else if (action == "build") {
            if (status == "running") throw std::logic_error("Pause before rebuilding the creature.");
            if (!request.contains("assembly")) throw std::invalid_argument("build requires an assembly");
            world.rebuild(request.at("assembly")); // Validates and compiles before committing.
            ++assembly_revision;
            reset_session_locked(true);
        } else if (action == "reset") {
            world.reset();
            reset_session_locked(false);
        } else if (action == "explore") {
            if (status == "running") throw std::logic_error("Pause before starting another exploration.");
            begin_trial_locked(patterns.at(0).at("pattern_id").get<std::string>());
            repertoire = Json::array();
            recorded_efforts.clear();
            mode = "explore";
            status = "running";
            error.clear();
            total_steps = 0;
            trial_index = 0;
            ++session_id;
        } else if (action == "replay") {
            if (!request.contains("pattern_id") || !request.at("pattern_id").is_string())
                throw std::invalid_argument("replay requires a pattern_id");
            const auto id = request.at("pattern_id").get<std::string>();
            if (!recorded_efforts.contains(id)) throw std::invalid_argument("This body has not recorded that motion.");
            if (status == "running") throw std::logic_error("Pause before replaying another motion.");
            begin_trial_locked(id);
            mode = "replay";
            status = "running";
            error.clear();
            trial_index = 0;
            total_steps = 0;
            ++session_id;
        } else {
            throw std::invalid_argument("unknown construction action");
        }
        return state_locked();
    }

    void reset_session_locked(bool forget) {
        status = "ready";
        mode = "idle";
        error.clear();
        active_pattern.clear();
        trial_index = trial_steps = total_steps = 0;
        trail = Json::array();
        trial_efforts.clear();
        if (forget) { repertoire = Json::array(); recorded_efforts.clear(); }
        ++session_id;
    }

    mutable std::mutex mutex;
    std::condition_variable wake;
    bool threaded{false};
    bool stopping{false};
    ConstructionWorld world;
    MotionExplorer explorer;
    const Json patterns = MotionExplorer::patterns();
    Json repertoire = Json::array();
    Json trail = Json::array();
    std::map<std::string, std::vector<double>> recorded_efforts;
    std::vector<double> trial_efforts;
    std::string status{"ready"}, mode{"idle"}, active_pattern, error;
    std::size_t session_id{1}, assembly_revision{1};
    std::size_t trial_index{0}, trial_steps{0}, total_steps{0};
    std::thread worker;
};

ConstructionSession::ConstructionSession(bool realtime) : impl_(std::make_unique<Impl>(realtime)) {}
ConstructionSession::~ConstructionSession() = default;
Json ConstructionSession::state() const { std::scoped_lock lock(impl_->mutex); return impl_->state_locked(); }
Json ConstructionSession::control(const Json& request) { std::scoped_lock lock(impl_->mutex); return impl_->control_locked(request); }
Json ConstructionSession::specification() {
    return Json{{"schema", "construction_session_v1"}, {"physics", ConstructionWorld::specification()},
                {"patterns", MotionExplorer::patterns()}, {"explorer", motion_explorer_spec()}, {"reward_optimization", false},
                {"rebuild_memory", "fresh"}, {"reset_preserves_repertoire", true}};
}
void ConstructionSession::advance_steps(std::size_t count) {
    std::scoped_lock lock(impl_->mutex);
    if (impl_->threaded) throw std::logic_error("Explicit stepping requires a worker-free construction session.");
    for (std::size_t step = 0; step < count && impl_->status == "running"; ++step) impl_->tick_locked();
}
} // namespace droid
