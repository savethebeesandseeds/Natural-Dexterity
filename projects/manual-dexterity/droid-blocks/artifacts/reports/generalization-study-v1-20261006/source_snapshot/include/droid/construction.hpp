#pragma once

#include <memory>
#include <nlohmann/json.hpp>

namespace droid {

// Isolated, worker-free two-link fixture. The optional assembly.powered_hinge
// selects one powered coordinate: 0 is the anchored base (the legacy default),
// 1 is the elbow; the other coordinate is passive. Local observations never
// disclose this assembly field. Rebuild validates and compiles a replacement
// before replacing any active state.
class ConstructionWorld final {
public:
    ConstructionWorld();
    ~ConstructionWorld();
    ConstructionWorld(const ConstructionWorld&) = delete;
    ConstructionWorld& operator=(const ConstructionWorld&) = delete;

    void rebuild(const nlohmann::json& assembly);
    void reset();
    // V2 only. Moving the sun preserves mechanics, time and already acquired
    // samples. Reset and a v2-to-v2 rebuild preserve the selected sun.
    void set_sun(const nlohmann::json& sun);
    [[nodiscard]] nlohmann::json assembly() const;
    [[nodiscard]] nlohmann::json observation() const;
    [[nodiscard]] nlohmann::json state() const;
    // Advances exactly .02 s, with ten .002 s physical transitions. A hard
    // safety event removes powered torque for the remaining transitions and
    // requires reset before another step. Non-finite requests are rejected.
    [[nodiscard]] nlohmann::json step(double effort);
    [[nodiscard]] static nlohmann::json default_assembly();
    [[nodiscard]] static nlohmann::json specification();
    [[nodiscard]] static nlohmann::json light_assembly();
    [[nodiscard]] static nlohmann::json light_specification();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace droid
