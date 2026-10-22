#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace droid {

// Canonical, policy-hidden realization for light_search_1d_v1. The source
// distance lies on an exact 24-bit binary grid in [12, 20) metres. The compact
// key places the side bit above the distance bucket and is therefore 25 bits.
struct LightSearch1dRealization {
    bool positive_side{false};
    std::uint32_t distance_bucket{0};
    double source_x_m{0.0};
};

[[nodiscard]] LightSearch1dRealization light_search_1d_realization(
    std::uint64_t seed) noexcept;
[[nodiscard]] std::uint32_t light_search_1d_realization_key(
    std::uint64_t seed) noexcept;

// Thread-safe owner of the MuJoCo model and the reference demo environment.
// Passing realtime=false creates a paused, worker-free instance for exact
// deterministic stepping in tests and conformance probes.
class DroidSimulation final {
public:
    explicit DroidSimulation(
        const std::filesystem::path& model_path,
        bool realtime = true);
    ~DroidSimulation();

    DroidSimulation(const DroidSimulation&) = delete;
    DroidSimulation& operator=(const DroidSimulation&) = delete;
    DroidSimulation(DroidSimulation&&) = delete;
    DroidSimulation& operator=(DroidSimulation&&) = delete;

    [[nodiscard]] bool running() const;
    void set_running(bool running);
    void reset();

    // Policy-safe synchronous-agent boundary. reset_agent() pauses a realtime
    // worker and selects direct effort control. step_agent() accepts one
    // finite effort for every opaque motor instance and advances an integral
    // number of MuJoCo physics transitions. set_running(true) returns a
    // realtime instance to the privileged visualization-only demo controller.
    [[nodiscard]] nlohmann::json agent_spec() const;
    // Separate opt-in successor descriptor; agent_spec() remains the frozen
    // legacy contract in every mode.
    [[nodiscard]] nlohmann::json body_experiment_spec() const;
    [[nodiscard]] nlohmann::json reset_agent(
        std::string_view assembly_id,
        std::uint64_t seed,
        std::string_view episode_profile = "fixed_demo_v0");
    [[nodiscard]] nlohmann::json step_agent(
        const std::map<std::string, double>& actions,
        double control_dt_s);
    [[nodiscard]] nlohmann::json policy_observation() const;

    // Advance exactly count physics transitions while holding the simulation
    // lock. This is available to worker-free instances and paused realtime
    // instances; an active realtime worker is rejected.
    void advance_steps(std::size_t count);

    [[nodiscard]] nlohmann::json state() const;
    [[nodiscard]] nlohmann::json catalog() const;
    [[nodiscard]] nlohmann::json health() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using Simulation = DroidSimulation;

}  // namespace droid
