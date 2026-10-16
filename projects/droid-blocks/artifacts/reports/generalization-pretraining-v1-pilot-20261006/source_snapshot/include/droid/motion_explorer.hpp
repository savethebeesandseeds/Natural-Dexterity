#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kMotionExplorerId{"bounded_rhythm_probe_v1"};
inline constexpr double kMotionExplorerControlDtS{0.02};
inline constexpr std::size_t kMotionExplorerMaxSteps{300};
inline constexpr double kMotionExplorerDurationS{6.0};

[[nodiscard]] nlohmann::json motion_explorer_spec();

// A fixed, documented collection of bounded motor excitations. Sensor-derived
// summaries describe the resulting experience; they are not reward scores or
// measurements of world-space reachability. No trained parameters are loaded.
class MotionExplorer final {
public:
    MotionExplorer();
    ~MotionExplorer();
    MotionExplorer(const MotionExplorer&) = delete;
    MotionExplorer& operator=(const MotionExplorer&) = delete;

    [[nodiscard]] static nlohmann::json patterns();
    void reset(std::string pattern_id);

    // Exactly one act -> native physics step(.02 s) -> observe cycle per
    // transition. act starts from the reset observation at t=0; observe commits
    // the pending effort and its poststep sample. A failed call changes no
    // explorer state. The session must stop a failed trial and require reset.
    [[nodiscard]] double act(const nlohmann::json& observation);
    void observe(const nlohmann::json& observation);

    [[nodiscard]] bool complete() const;
    [[nodiscard]] nlohmann::json result() const;
    [[nodiscard]] nlohmann::json diagnostics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace droid
