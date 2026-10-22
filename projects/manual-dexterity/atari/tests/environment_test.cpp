#include "atari/environment.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const char* message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void deterministic_rollout(const std::string& rom) {
  atari::AtariEnv first(rom, 12345), second(rom, 12345);
  require(first.observation() == second.observation(), "Seeded reset differs");
  require(first.actions() > 0, "Empty action set");
  require(first.width() > 0 && first.height() > 0, "Empty render dimensions");
  require(first.rgb().size() == static_cast<std::size_t>(first.width() * first.height() * 3),
          "RGB24 render has incorrect size");
  require(first.rgb() == second.rgb(), "Seeded RGB images differ");
  for (int frame = 1; frame < atari::kFrameStack; ++frame) {
    require(std::equal(first.observation().begin(),
                       first.observation().begin() + atari::kFramePixels,
                       first.observation().begin() + frame * atari::kFramePixels),
            "Reset frame stack was not initialized completely");
  }
  require_throws<std::out_of_range>([&] { first.step(-1); }, "Negative action accepted");
  require_throws<std::out_of_range>([&] { first.step(first.actions()); },
                                   "Oversized action accepted");
  for (int step = 0; step < 200; ++step) {
    const auto before = first.observation();
    const int previous_frames = first.episode_raw_frames();
    const int action = (step * 17) % first.actions();
    const auto a = first.step(action), b = second.step(action);
    require(a.reward == b.reward && a.done == b.done && a.truncated == b.truncated &&
                a.episode_return == b.episode_return &&
                a.episode_raw_frames == b.episode_raw_frames,
            "Seeded transition differs");
    require(first.observation() == second.observation(), "Seeded observation differs");
    require(a.episode_raw_frames > previous_frames &&
                a.episode_raw_frames - previous_frames <= atari::kFrameSkip,
            "Action-repeat frame count is wrong");
    require(std::equal(before.begin() + atari::kFramePixels, before.end(),
                       first.observation().begin()), "Temporal stack did not shift");
    if (a.done || a.truncated) {
      first.reset();
      second.reset();
    }
  }
  first.reset(99999);
  atari::AtariEnv reseeded(rom, 99999);
  require(first.observation() == reseeded.observation(), "Explicit reseed reset differs");
  for (int step = 0; step < 20; ++step) {
    first.step(step % first.actions());
    reseeded.step(step % reseeded.actions());
    require(first.observation() == reseeded.observation(), "Explicit reseed RNG differs");
  }
}

void truncation(const std::string& rom) {
  atari::EnvConfig config;
  config.noop_max = 0;
  config.fire_reset = false;
  config.max_episode_frames = 7;
  atari::AtariEnv env(rom, 777, config);
  auto transition = env.step(0);
  require(!transition.done && !transition.truncated && transition.episode_raw_frames == 4,
          "Episode truncated before its frame limit");
  transition = env.step(0);
  require(!transition.done && transition.truncated && transition.episode_raw_frames == 7,
          "Time-limit truncation was not separated from game-over");
  require_throws<std::logic_error>([&] { env.step(0); }, "Step accepted after truncation");
  env.reset();
  require(env.episode_raw_frames() == 0 && env.episode_return() == 0,
          "Reset did not clear episode accounting");
}

void full_pong_episode(const std::string& rom) {
  // This test is deliberately Pong-specific: a natural game ends at 21 points.
  atari::EnvConfig config;
  config.noop_max = 0;
  atari::AtariEnv env(rom, 42, config);
  float total_reward = env.episode_return();
  for (int step = 0; step < 27000; ++step) {
    const auto transition = env.step(0);
    total_reward += transition.reward;
    require(total_reward == transition.episode_return, "Raw score accounting differs");
    if (transition.done || transition.truncated) {
      require(transition.done && !transition.truncated, "Pong failed to terminate naturally");
      require(transition.episode_return >= -21 && transition.episode_return <= 21,
              "Pong score is outside its legal range");
      require_throws<std::logic_error>([&] { env.step(0); }, "Step accepted after game-over");
      return;
    }
  }
  throw std::runtime_error("No complete Pong episode observed");
}
}  // namespace

int main(int argc, char** argv) {
  const char* rom = argc > 1 ? argv[1] : std::getenv("ATARI_TEST_ROM");
  if (rom == nullptr || *rom == '\0') {
    std::cout << "SKIP: pass a Pong ROM path or set ATARI_TEST_ROM\n";
    return 77;
  }
  try {
    deterministic_rollout(rom);
    truncation(rom);
    full_pong_episode(rom);
    std::cout << "PASS: seeded native ALE, preprocessing, rendering, truncation, and Pong scoring\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
