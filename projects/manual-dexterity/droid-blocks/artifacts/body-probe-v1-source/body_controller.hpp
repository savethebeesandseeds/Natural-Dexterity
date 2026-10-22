#pragma once

#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::string_view kBodyControllerId{"paired_light_probe_v1"};
inline constexpr double kBodyControllerControlDtS{0.02};
inline constexpr double kBodyControllerEffortLimit{0.35};
inline constexpr double kBodyControllerSlewLimit{0.1};

[[nodiscard]] nlohmann::json body_controller_spec();

// A handwritten, run-local adaptation baseline. It discovers the sign of a
// motor's effect on the light reading, not the motor's physical transmission.
// The inference boundary deliberately has no body variant, source, world pose,
// seed, reward, simulator callback, or frozen-policy artifact.
class BodyProbeController final {
public:
    BodyProbeController();
    ~BodyProbeController();
    BodyProbeController(const BodyProbeController&) = delete;
    BodyProbeController& operator=(const BodyProbeController&) = delete;

    // "discover" probes separately, in the supplied enumeration order.
    // "together" broadcasts the same paired probes and retains one shared
    // preference. Renaming IDs preserves enumeration; sorting IDs is forbidden.
    void reset(std::span<const std::string> motor_ids,
               std::string_view strategy = "discover");

    // Call once per .02 s transition, first with the reset observation at t=0.
    // Bad/stale observations throw without advancing adaptation state. The
    // session must stop on failure and require a reset before further actions.
    [[nodiscard]] std::map<std::string, double> act(
        const nlohmann::json& observation,
        std::span<const std::string> motor_ids);

    [[nodiscard]] nlohmann::json diagnostics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace droid
