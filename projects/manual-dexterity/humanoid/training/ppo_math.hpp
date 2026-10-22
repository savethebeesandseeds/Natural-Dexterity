#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace humanoid::ppo {

// Flattened arrays are [time, environment]. next_value is evaluated before any
// reset. True terminations disable bootstrapping; external time limits retain it.
// Both episode endings stop the GAE recursion across the subsequent reset.
inline std::vector<float> generalized_advantages(
    const std::vector<float>& reward, const std::vector<float>& value,
    const std::vector<float>& next_value,
    const std::vector<unsigned char>& terminated,
    const std::vector<unsigned char>& episode_end,
    std::size_t horizon, std::size_t environments,
    float gamma = 0.99f, float lambda = 0.95f) {
  const auto count = horizon * environments;
  if (!horizon || !environments || reward.size() != count || value.size() != count ||
      next_value.size() != count || terminated.size() != count || episode_end.size() != count) {
    throw std::invalid_argument("Invalid GAE rollout dimensions");
  }
  std::vector<float> advantage(count), running(environments, 0.0f);
  for (std::size_t time = horizon; time-- > 0;) {
    for (std::size_t env = 0; env < environments; ++env) {
      const auto index = time * environments + env;
      const float bootstrap = terminated[index] ? 0.0f : next_value[index];
      const float delta = reward[index] + gamma * bootstrap - value[index];
      running[env] = delta + gamma * lambda * (episode_end[index] ? 0.0f : running[env]);
      advantage[index] = running[env];
    }
  }
  return advantage;
}

}  // namespace humanoid::ppo
