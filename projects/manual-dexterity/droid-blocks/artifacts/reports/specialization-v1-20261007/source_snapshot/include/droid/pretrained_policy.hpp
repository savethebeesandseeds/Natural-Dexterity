#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>

#include <nlohmann/json.hpp>

namespace droid {

inline constexpr std::size_t kPretrainedFrameSize = 17;
inline constexpr std::size_t kPretrainedFrameCount = 12;
inline constexpr std::size_t kPretrainedInputSize =
    kPretrainedFrameSize * kPretrainedFrameCount;
inline constexpr std::size_t kPretrainedActionCount = 3;
using PretrainedInput = std::array<float, kPretrainedInputSize>;
using PretrainedOutput = std::array<float, kPretrainedActionCount>;

// Versioned frozen seed101 actor only. The loader independently validates the
// lossless float32 bytes against pinned trusted digests and source provenance.
// Evaluating the actor never constructs physics or mutates its weights.
class PretrainedPolicy final {
public:
    explicit PretrainedPolicy(const std::filesystem::path& path);
    ~PretrainedPolicy();
    PretrainedPolicy(const PretrainedPolicy&) = delete;
    PretrainedPolicy& operator=(const PretrainedPolicy&) = delete;

    [[nodiscard]] nlohmann::json metadata() const;
    [[nodiscard]] PretrainedOutput logits(const PretrainedInput& input) const;
    [[nodiscard]] PretrainedOutput probabilities(const PretrainedInput& input) const;
    [[nodiscard]] int sample(const PretrainedInput& input, double uniform) const;
    // Normalize the three float32 probabilities in double precision, then use
    // strict u < cumulative_mass. A boundary belongs to the following action;
    // zero-mass actions are skipped. This sampler is independent of Torch RNG.
    [[nodiscard]] static int categorical(const PretrainedOutput& probabilities,
                                         double uniform);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The sole actor input boundary: native local samples and completed macro
// action/reward only. Initial reset inserts eleven absent zero frames and one
// current local frame. Each append is one completed .1s macro decision, not a
// .02s physical control. Pausing never calls reset or append.
class PretrainedHistory final {
public:
    PretrainedHistory() = default;
    explicit PretrainedHistory(const nlohmann::json& initial_observation);
    void reset(const nlohmann::json& initial_observation);
    void append(const nlohmann::json& observation, double executed_effort,
                double previous_request, double reward);
    [[nodiscard]] const PretrainedInput& values() const;
    [[nodiscard]] std::size_t count() const noexcept;
    [[nodiscard]] nlohmann::json diagnostics() const;
    [[nodiscard]] static std::array<float, kPretrainedFrameSize> frame(
        const nlohmann::json& observation, double executed_effort,
        double previous_request, double reward);

private:
    PretrainedInput input_{};
    std::size_t count_{};
    std::size_t completed_decisions_{};
};

} // namespace droid
