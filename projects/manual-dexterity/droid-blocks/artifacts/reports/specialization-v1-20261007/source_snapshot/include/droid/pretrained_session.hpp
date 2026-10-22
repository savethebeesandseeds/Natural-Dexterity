#pragma once
#include <cstddef>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>

namespace droid {
// Isolated native world with a frozen, provenance-bound actor. No optimizer.
class PretrainedSession final {
public:
    explicit PretrainedSession(bool realtime = false,
        const std::filesystem::path& actor = "artifacts/pretrained/pooled-seed101-v1.json");
    ~PretrainedSession();
    PretrainedSession(const PretrainedSession&) = delete;
    PretrainedSession& operator=(const PretrainedSession&) = delete;
    nlohmann::json state() const;
    nlohmann::json trace() const;
    nlohmann::json control(const nlohmann::json& request);
    static nlohmann::json specification();
    void advance_steps(std::size_t count);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace droid
