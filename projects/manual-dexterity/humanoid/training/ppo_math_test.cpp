#include "ppo_math.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void near(float actual, float expected) {
  if (std::abs(actual - expected) > 1e-5f) {
    throw std::runtime_error("GAE mismatch: " + std::to_string(actual) + " vs " + std::to_string(expected));
  }
}
}

int main() {
  try {
    using humanoid::ppo::generalized_advantages;
    // Same final transition, different reason for ending: truncation must keep V(s').
    auto terminal = generalized_advantages({1}, {2}, {10}, {1}, {1}, 1, 1, .9f, .8f);
    auto timeout = generalized_advantages({1}, {2}, {10}, {0}, {1}, 1, 1, .9f, .8f);
    near(terminal[0], -1);
    near(timeout[0], 8);
    // Never propagate the next episode's advantage across a reset.
    auto reset = generalized_advantages({1, 100}, {2, 0}, {10, 0}, {0, 1}, {1, 1}, 2, 1, .9f, .8f);
    near(reset[0], 8);
    near(reset[1], 100);
    // Two environments interleaved in memory remain independent. The rollout
    // boundary is not a terminal: the final bootstrap values still contribute.
    auto batch = generalized_advantages({1, 2, 3, 4}, {0, 0, 0, 0}, {0, 0, 5, 6},
                                        {0, 0, 0, 0}, {0, 0, 0, 0}, 2, 2, .9f, .8f);
    near(batch[2], 7.5f);
    near(batch[3], 9.4f);
    near(batch[0], 1 + .72f * 7.5f);
    near(batch[1], 2 + .72f * 9.4f);
    bool rejected = false;
    try { generalized_advantages({}, {}, {}, {}, {}, 1, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Malformed rollout was not rejected");
    std::cout << "GAE termination, timeout, reset isolation and batching checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
