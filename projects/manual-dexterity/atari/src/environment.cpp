#include "atari/environment.hpp"

#include <ale/ale_interface.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace atari {
namespace {

int ale_seed(std::uint64_t seed) {
  // ALE accepts a nonnegative signed 32-bit seed. Mix all user seed bits before
  // narrowing, so that seeds separated by 2^32 do not always create duplicates.
  seed += 0x9e3779b97f4a7c15ULL;
  seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
  seed ^= seed >> 31;
  return static_cast<int>(seed & 0x7fffffffULL);
}

void validate_config(const EnvConfig& config) {
  if (!std::isfinite(config.sticky_probability) || config.sticky_probability < 0.0F ||
      config.sticky_probability > 1.0F) {
    throw std::invalid_argument("Sticky-action probability must be in [0, 1]");
  }
  if (config.max_episode_frames < 1 || config.noop_max < 0) {
    throw std::invalid_argument("Episode frame limit must be positive and noop_max nonnegative");
  }
  const auto prefix_max = static_cast<std::int64_t>(config.noop_max) +
                          (config.fire_reset ? kFrameSkip : 0);
  if (prefix_max >= config.max_episode_frames) {
    throw std::invalid_argument("Episode frame limit must exceed the maximum reset prefix");
  }
}

}  // namespace

struct AtariEnv::Impl {
  std::string rom_path;
  EnvConfig config;
  ale::ALEInterface emulator;
  ale::ActionVect action_set;
  std::mt19937_64 reset_rng;
  Observation observation{};
  std::vector<std::uint8_t> previous_gray;
  std::vector<std::uint8_t> current_gray;
  std::vector<std::uint8_t> pooled_gray;
  float score = 0.0F;
  bool ended = false;

  Impl(std::string path, std::uint64_t seed, EnvConfig settings)
      : rom_path(std::move(path)), config(settings), reset_rng(seed) {
    validate_config(config);
    if (!std::filesystem::is_regular_file(rom_path)) {
      throw std::invalid_argument("Atari ROM is not a regular file: " + rom_path);
    }
    emulator.setInt("random_seed", ale_seed(seed));
    emulator.setInt("frame_skip", 1); // Our wrapper owns frame skip and pooling.
    emulator.setInt("max_num_frames_per_episode", config.max_episode_frames);
    emulator.setFloat("repeat_action_probability", config.sticky_probability);
    emulator.setBool("color_averaging", false);
    emulator.setBool("truncate_on_loss_of_life", false);
    emulator.setBool("display_screen", false);
    emulator.setBool("sound", false);
    emulator.setInt("reward_min", std::numeric_limits<int>::min());
    emulator.setInt("reward_max", std::numeric_limits<int>::max());
    emulator.loadROM(rom_path);
    action_set = emulator.getMinimalActionSet();
    if (action_set.empty()) {
      throw std::runtime_error("ALE returned an empty minimal action set");
    }
    const auto pixels = static_cast<std::size_t>(width()) * height();
    if (pixels == 0) {
      throw std::runtime_error("ALE returned an empty screen");
    }
    previous_gray.resize(pixels);
    current_gray.resize(pixels);
    pooled_gray.resize(pixels);
    reset();
  }

  int width() const { return static_cast<int>(emulator.getScreen().width()); }
  int height() const { return static_cast<int>(emulator.getScreen().height()); }

  void resize_frame(std::uint8_t* destination) const {
    const int source_width = width();
    const int source_height = height();
    // Half-pixel bilinear sampling; includes the scoreboard and never crops.
    for (int y = 0; y < kObservationHeight; ++y) {
      const float source_y = std::clamp(
          (static_cast<float>(y) + 0.5F) * source_height / kObservationHeight - 0.5F,
          0.0F, static_cast<float>(source_height - 1));
      const int y0 = static_cast<int>(source_y);
      const int y1 = std::min(y0 + 1, source_height - 1);
      const float dy = source_y - y0;
      for (int x = 0; x < kObservationWidth; ++x) {
        const float source_x = std::clamp(
            (static_cast<float>(x) + 0.5F) * source_width / kObservationWidth - 0.5F,
            0.0F, static_cast<float>(source_width - 1));
        const int x0 = static_cast<int>(source_x);
        const int x1 = std::min(x0 + 1, source_width - 1);
        const float dx = source_x - x0;
        const float top = pooled_gray[y0 * source_width + x0] * (1.0F - dx) +
                          pooled_gray[y0 * source_width + x1] * dx;
        const float bottom = pooled_gray[y1 * source_width + x0] * (1.0F - dx) +
                             pooled_gray[y1 * source_width + x1] * dx;
        destination[y * kObservationWidth + x] =
            static_cast<std::uint8_t>(std::lround(top * (1.0F - dy) + bottom * dy));
      }
    }
  }

  const Observation& reset() {
    // A pathological ROM that terminates during its reset prefix gets a bounded
    // retry, rather than returning an already-terminal observation to the learner.
    for (int attempt = 0; attempt < 8; ++attempt) {
      emulator.reset_game();
      score = 0.0F;
      const int noops = std::uniform_int_distribution<int>(0, config.noop_max)(reset_rng);
      for (int frame = 0; frame < noops && !emulator.game_over(); ++frame) {
        score += static_cast<float>(emulator.act(ale::PLAYER_A_NOOP));
      }
      const auto fire = std::find(action_set.begin(), action_set.end(), ale::PLAYER_A_FIRE);
      if (config.fire_reset && fire != action_set.end()) {
        for (int frame = 0; frame < kFrameSkip && !emulator.game_over(); ++frame) {
          score += static_cast<float>(emulator.act(*fire));
        }
      }
      if (emulator.game_over()) {
        continue;
      }
      ended = false;
      emulator.getScreenGrayscale(current_gray);
      previous_gray = current_gray;
      pooled_gray = current_gray;
      resize_frame(observation.data());
      for (int frame = 1; frame < kFrameStack; ++frame) {
        std::copy_n(observation.begin(), kFramePixels,
                    observation.begin() + frame * kFramePixels);
      }
      return observation;
    }
    throw std::runtime_error("Game repeatedly ended during reset-prefix actions");
  }

  Step step(int action_index) {
    if (action_index < 0 || action_index >= static_cast<int>(action_set.size())) {
      throw std::out_of_range("Minimal Atari action index is out of range");
    }
    if (ended) {
      throw std::logic_error("Atari episode ended; call reset() before step()");
    }
    float reward = 0.0F;
    for (int frame = 0; frame < kFrameSkip; ++frame) {
      previous_gray.swap(current_gray);
      reward += static_cast<float>(emulator.act(action_set[action_index]));
      emulator.getScreenGrayscale(current_gray);
      if (emulator.game_over()) {
        break;
      }
    }
    // Pool only images from the actual final pair, even if a terminal frame
    // interrupts the repeat. In the one-frame case, the prior screen is valid.
    std::transform(previous_gray.begin(), previous_gray.end(), current_gray.begin(),
                   pooled_gray.begin(), [](std::uint8_t a, std::uint8_t b) {
                     return std::max(a, b);
                   });
    std::memmove(observation.data(), observation.data() + kFramePixels,
                 (kFrameStack - 1) * kFramePixels);
    resize_frame(observation.data() + (kFrameStack - 1) * kFramePixels);
    score += reward;
    const bool done = emulator.game_over(false);
    const bool truncated = emulator.game_truncated();
    ended = done || truncated;
    return {reward, done, truncated, score, emulator.getEpisodeFrameNumber()};
  }
};

AtariEnv::AtariEnv(const std::string& rom_path, std::uint64_t seed, EnvConfig config)
    : impl_(std::make_unique<Impl>(rom_path, seed, config)) {}
AtariEnv::~AtariEnv() = default;
AtariEnv::AtariEnv(AtariEnv&&) noexcept = default;
AtariEnv& AtariEnv::operator=(AtariEnv&&) noexcept = default;

const Observation& AtariEnv::reset() { return impl_->reset(); }
const Observation& AtariEnv::reset(std::uint64_t seed) {
  // Settings take effect on loadROM, not reset_game. Reconstructing guarantees
  // the ALE sticky-action RNG matches an independently constructed instance.
  impl_ = std::make_unique<Impl>(impl_->rom_path, seed, impl_->config);
  return impl_->observation;
}
Step AtariEnv::step(int action_index) { return impl_->step(action_index); }
const Observation& AtariEnv::observation() const { return impl_->observation; }
std::vector<std::uint8_t> AtariEnv::rgb() const {
  std::vector<std::uint8_t> pixels;
  impl_->emulator.getScreenRGB(pixels);
  return pixels;
}
int AtariEnv::width() const { return impl_->width(); }
int AtariEnv::height() const { return impl_->height(); }
int AtariEnv::actions() const { return static_cast<int>(impl_->action_set.size()); }
float AtariEnv::episode_return() const { return impl_->score; }
int AtariEnv::episode_raw_frames() const { return impl_->emulator.getEpisodeFrameNumber(); }

}  // namespace atari
