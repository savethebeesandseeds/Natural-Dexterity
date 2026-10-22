#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace humanoid::goal {

// Frozen acceptance criteria for the user-approved PPO milestone. This gate
// complements, rather than replaces, inspection of the actual contact gait.
inline std::vector<std::string> failures(const nlohmann::json& state,
                                         double seconds, double command) {
  std::vector<std::string> result;
  auto measured = [&](const char* key) {
    return state.contains(key) && state[key].is_number()
        ? state[key].get<double>() : std::nan("");
  };
  auto at_least = [&](const char* key, double limit, const char* reason) {
    const double value = measured(key);
    if (!std::isfinite(value) || value + 1e-8 < limit) result.emplace_back(reason);
  };
  if (!state.contains("fall") || !state["fall"].is_boolean() || state["fall"].get<bool>())
    result.emplace_back("fall or missing fall observation");
  at_least("time", seconds, "incomplete continuous duration");
  at_least("distance", .7 * command * seconds, "forward distance below 70% of command");
  at_least("left_steps", 5, "fewer than five left touchdowns");
  at_least("right_steps", 5, "fewer than five right touchdowns");
  at_least("alternating_count", 8, "fewer than eight alternating touchdowns");
  const double flight = measured("airborne_fraction");
  if (!std::isfinite(flight) || flight < 0 || flight > .1 + 1e-10)
    result.emplace_back("airborne fraction exceeds 10% or is missing");
  return result;
}

inline bool aggregate_pass(const std::vector<int>& passed, int trials_per_group) {
  if (passed.size() != 9 || trials_per_group != 5) return false;
  int sum = 0;
  for (const int count : passed) {
    if (count < 4 || count > trials_per_group) return false;
    sum += count;
  }
  return 10 * sum >= 9 * 45;
}
}  // namespace humanoid::goal
