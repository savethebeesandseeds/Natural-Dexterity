#pragma once

#include <cstdint>
#include <memory>

#include <nlohmann/json.hpp>

namespace droid {

// Online, history-conditioned control of the continuous construction world.
// reset explicitly forgets values and experience; moving the lamp must not call
// reset. Action selection receives only the declared physical observation.
// v2 initializes every action's value equally from the fully delivered quiet
// reward intervals; subsequent TD updates still use the raw summed reward.
class LightLearner final {
public:
    // Immutable, copyable value-only snapshot. It has no default constructor,
    // mutable fields, JSON parser or physical/controller-state payload.
    class ValueCheckpoint final {
    public:
        ValueCheckpoint(const ValueCheckpoint&) = default;
        ValueCheckpoint& operator=(const ValueCheckpoint&) = default;
    private:
        struct Data;
        std::shared_ptr<const Data> data_;
        explicit ValueCheckpoint(std::shared_ptr<const Data> data);
        friend class LightLearner;
    };

    explicit LightLearner(std::uint64_t seed = 1);
    ~LightLearner();
    LightLearner(const LightLearner&) = delete;
    LightLearner& operator=(const LightLearner&) = delete;

    void reset(std::uint64_t seed = 1);
    [[nodiscard]] double act(const nlohmann::json& observation);
    void observe(const nlohmann::json& transition);
    // One-way intervention after quiet initialization and between act/observe
    // pairs. Only parameter writes stop; experience and exploration continue.
    // Repeated calls are idempotent. reset starts an unfrozen learner again.
    void freeze_learning();
    // Both require initialized values and no pending act/observe pair.
    // Restore changes only weights; it preserves the learning-freeze flag,
    // history, traces, replay, RNG, counters and the current option.
    [[nodiscard]] ValueCheckpoint capture_values() const;
    void restore_values(const ValueCheckpoint& checkpoint);
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
