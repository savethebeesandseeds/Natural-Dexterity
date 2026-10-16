#pragma once
#include "droid/light_learner.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

namespace droid {
// Common driver used by the real-time session and development comparisons.
class LightControl final {
public:
    using ValueCheckpoint = LightLearner::ValueCheckpoint;
    LightControl();
    ~LightControl();
    void reset(const std::string& mode, std::uint64_t seed = 1);
    double act(const nlohmann::json& observation);
    void observe(const nlohmann::json& transition);
    // Development intervention, available only in initialized learner mode
    // between complete act/observe pairs. Not exposed by the browser API.
    void freeze_learning();
    [[nodiscard]] ValueCheckpoint capture_values() const;
    void restore_values(const ValueCheckpoint& checkpoint);
    nlohmann::json diagnostics() const;
    static nlohmann::json specification();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class LightSession final {
public:
    explicit LightSession(bool realtime = false);
    ~LightSession();
    LightSession(const LightSession&) = delete;
    LightSession& operator=(const LightSession&) = delete;
    nlohmann::json state() const;
    nlohmann::json control(const nlohmann::json& request);
    static nlohmann::json specification();
    void advance_steps(std::size_t count);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace droid
