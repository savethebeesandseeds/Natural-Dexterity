#pragma once

#include <cstdint>
#include <memory>

#include <nlohmann/json.hpp>

namespace droid {

// Online, history-conditioned control of the continuous construction world.
// reset explicitly forgets values and experience; moving the lamp must not call
// reset. Action selection receives only the declared physical observation.
class LightLearner final {
public:
    explicit LightLearner(std::uint64_t seed = 1);
    ~LightLearner();
    LightLearner(const LightLearner&) = delete;
    LightLearner& operator=(const LightLearner&) = delete;

    void reset(std::uint64_t seed = 1);
    [[nodiscard]] double act(const nlohmann::json& observation);
    void observe(const nlohmann::json& transition);
    [[nodiscard]] nlohmann::json diagnostics() const;
    [[nodiscard]] static nlohmann::json specification();

    // Shared baseline envelope. Validates both sensors and local feedback;
    // initial undelivered readings request zero. State-owning callers must also
    // enforce their cadence and continuity, which a stateless helper cannot.
    [[nodiscard]] static double bounded_effort(
        double requested, double previous,
        const nlohmann::json& observation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace droid
