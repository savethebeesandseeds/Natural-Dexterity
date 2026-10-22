#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace atari {

inline constexpr int kObservationWidth = 84;
inline constexpr int kObservationHeight = 84;
inline constexpr int kFrameStack = 4;
inline constexpr int kFrameSkip = 4;
inline constexpr std::size_t kFramePixels = kObservationWidth * kObservationHeight;
inline constexpr std::size_t kObservationBytes = kFrameStack * kFramePixels;
using Observation = std::array<std::uint8_t, kObservationBytes>;

struct EnvConfig {
  float sticky_probability = 0.25F;
  int max_episode_frames = 108000;
  int noop_max = 30;
  bool fire_reset = true;
};

struct Step {
  float reward = 0.0F;           // Sum of original game rewards; never clipped.
  bool done = false;            // A real game-over, never a lost-life pseudoepisode.
  bool truncated = false;       // Separate time limit, for value bootstrapping.
  float episode_return = 0.0F;  // Original score, including reset-prefix actions.
  int episode_raw_frames = 0;   // ALE episode frames, including reset-prefix actions.
};

// A single, independent native ALE instance. Use one owner thread per instance.
// Observations are channel-first, oldest frame first, grayscale uint8. Each step
// repeats an action for up to four raw frames, max-pools the last two available
// grayscale frames and bilinearly resizes the full screen to 84x84. reset() fills
// all four stack entries with the current image. rgb() is full-resolution RGB24.
class AtariEnv {
 public:
  explicit AtariEnv(const std::string& rom_path, std::uint64_t seed,
                    EnvConfig config = {});
  ~AtariEnv();
  AtariEnv(AtariEnv&&) noexcept;
  AtariEnv& operator=(AtariEnv&&) noexcept;
  AtariEnv(const AtariEnv&) = delete;
  AtariEnv& operator=(const AtariEnv&) = delete;

  const Observation& reset();
  // Recreates ALE's random stream as well as the reset-prefix RNG. Ordinary
  // training should use reset() so that every episode is not identical.
  const Observation& reset(std::uint64_t seed);
  Step step(int minimal_action_index);
  const Observation& observation() const;
  std::vector<std::uint8_t> rgb() const;
  int width() const;
  int height() const;
  int actions() const;
  float episode_return() const;
  int episode_raw_frames() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace atari
