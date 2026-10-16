#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kEnvironmentTraceSchemaVersion{
    "droid-blocks.environment-trace.v2"};
inline constexpr std::string_view kLegacyEnvironmentTraceSchemaVersion{
    "droid-blocks.environment-trace.v1"};
inline constexpr std::string_view kFixedDemoEpisodeProfile{"fixed_demo_v0"};
inline constexpr std::string_view kLightSearch1dEpisodeProfile{
    "light_search_1d_v1"};
inline constexpr std::string_view kBodyDiscoveryEpisodeProfile{
    "body_discovery_1d_v1"};
inline constexpr std::string_view kNormalTransmissionAssembly{
    "transmission_normal_v1"};
inline constexpr std::string_view kReversedTransmissionAssembly{
    "transmission_reversed_v1"};
inline constexpr std::string_view kDisconnectedTransmissionAssembly{
    "transmission_disconnected_v1"};
inline constexpr std::string_view kDefaultEpisodeProfile{
    kFixedDemoEpisodeProfile};

// A synchronous learning boundary over one native MuJoCo simulation.
//
// Worker-free instances (the default) are intended for high-throughput
// training. A realtime instance may be shared with the visualization server;
// reset()/step() switch it into paused external-agent mode, while
// set_demo_running(true) hands control back to the visual demo controller.
class DroidEnvironment final {
public:
    explicit DroidEnvironment(
        const std::filesystem::path& model_path = "models/droid.xml",
        bool realtime = false);
    ~DroidEnvironment();

    DroidEnvironment(const DroidEnvironment&) = delete;
    DroidEnvironment& operator=(const DroidEnvironment&) = delete;
    DroidEnvironment(DroidEnvironment&&) = delete;
    DroidEnvironment& operator=(DroidEnvironment&&) = delete;

    [[nodiscard]] nlohmann::json spec() const;
    [[nodiscard]] nlohmann::json body_experiment_spec() const;

    [[nodiscard]] nlohmann::json reset(
        std::string_view assembly_id,
        std::uint64_t seed,
        bool recording = false,
        std::string_view episode_profile = kDefaultEpisodeProfile);
    [[nodiscard]] nlohmann::json step(
        const std::map<std::string, double>& actions,
        double control_dt_s);

    [[nodiscard]] bool recording() const;
    [[nodiscard]] nlohmann::json trace() const;
    void save_trace(const std::filesystem::path& path) const;

    [[nodiscard]] static nlohmann::json load_trace(
        const std::filesystem::path& path);

    // Replay never touches this environment's simulation. It creates a fresh,
    // worker-free environment from the same model and compares canonical JSON
    // at reset and after every recorded transition.
    [[nodiscard]] nlohmann::json replay_and_verify(
        const nlohmann::json& recorded_trace) const;
    [[nodiscard]] nlohmann::json replay_and_verify_file(
        const std::filesystem::path& path) const;

    // Narrow pass-throughs used by the visualization/evaluation server. These
    // are deliberately separate from the policy-safe observation boundary.
    [[nodiscard]] nlohmann::json visualization_state() const;
    [[nodiscard]] nlohmann::json catalog() const;
    [[nodiscard]] nlohmann::json health() const;
    void set_demo_running(bool running);
    void reset_visualization();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace droid
