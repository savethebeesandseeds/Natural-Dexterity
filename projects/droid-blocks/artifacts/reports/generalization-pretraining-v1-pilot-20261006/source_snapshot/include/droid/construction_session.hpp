#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>

namespace droid {

// Separate successor workbench. Never borrows or resets the frozen rover world.
class ConstructionSession final {
public:
    explicit ConstructionSession(bool realtime = false);
    ~ConstructionSession();
    ConstructionSession(const ConstructionSession&) = delete;
    ConstructionSession& operator=(const ConstructionSession&) = delete;
    [[nodiscard]] nlohmann::json state() const;
    [[nodiscard]] nlohmann::json control(const nlohmann::json& request);
    [[nodiscard]] static nlohmann::json specification();
    void advance_steps(std::size_t count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace droid
