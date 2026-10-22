#pragma once

#include "droid/environment.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::uint64_t kPlaygroundLeftSeed{6106049681611768608ULL};
inline constexpr std::uint64_t kPlaygroundRightSeed{6990021049862345368ULL};
inline constexpr std::size_t kPlaygroundMaxSteps{1'000};
inline constexpr std::size_t kBodyPlaygroundMaxSteps{2'000};
inline constexpr double kPlaygroundControlDtS{0.02};
inline constexpr std::string_view kPlaygroundArtifactSha256{
    "4dd4d6ccadecb0b3b69d3b3d412af83eff15208b19058701538dbdf83a4c32b5"};
inline constexpr std::string_view kPlaygroundArtifactFileSha256{
    "6f7d5069bdb884d75f1241629d437c039a2f7a486633d9ef2d7f6b70db0a5582"};

// Owns control of a borrowed environment. All HTTP access goes through this
// boundary so a reset, an external-agent request, and learned inference cannot
// interleave. The environment must outlive the session and have no other users.
// The optional worker advances live native inference; worker-free tests invoke
// tick() through exactly the same code path.
class PlaygroundSession final {
public:
    explicit PlaygroundSession(
        DroidEnvironment& environment,
        const std::filesystem::path& model_path = "models/droid.xml",
        const std::filesystem::path& artifact_path =
            "artifacts/light-search-policy-v3-seed0.json",
        bool realtime = false);
    ~PlaygroundSession();

    PlaygroundSession(const PlaygroundSession&) = delete;
    PlaygroundSession& operator=(const PlaygroundSession&) = delete;

    [[nodiscard]] nlohmann::json state() const;
    [[nodiscard]] nlohmann::json health() const;
    [[nodiscard]] nlohmann::json catalog() const;
    [[nodiscard]] nlohmann::json spec() const;
    [[nodiscard]] nlohmann::json trace() const;

    // side is accepted with reset or body. reset prepares the original learned
    // mode; body prepares the explicit experiment with optional variant and
    // strategy. demo selects the fixed scripted demo, paused.
    [[nodiscard]] nlohmann::json control(
        std::string_view action, std::string_view side = {},
        std::string_view variant = {}, std::string_view strategy = {});
    [[nodiscard]] nlohmann::json legacy_control(std::string_view action);
    void tick();

    // Explicit reset is the only way for an HTTP agent to take ownership.
    [[nodiscard]] nlohmann::json agent_reset(
        std::string_view assembly_id,
        std::uint64_t seed,
        bool recording,
        std::string_view episode_profile);
    [[nodiscard]] nlohmann::json agent_step(
        const std::map<std::string, double>& actions, double control_dt_s);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace droid
