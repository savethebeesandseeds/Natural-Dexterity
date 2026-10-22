#include "humanoid/goal_criteria.hpp"
#include <iostream>
#include <stdexcept>

int main() {
  using humanoid::goal::failures;
  auto require = [](bool condition, const char* message) { if (!condition) throw std::runtime_error(message); };
  nlohmann::json walk = {{"time", 60.0}, {"distance", 21.0}, {"fall", false},
    {"left_steps", 60}, {"right_steps", 60}, {"alternating_count", 119}, {"airborne_fraction", .05}};
  require(failures(walk, 60, .5).empty(), "A complete threshold gait must qualify");
  auto stand = walk; stand["distance"] = .2; stand["left_steps"] = 0; stand["alternating_count"] = 0;
  require(failures(stand, 60, .5).size() >= 3, "Standing must not pass despite survival");
  auto collapse = walk; collapse["fall"] = true;
  require(!failures(collapse, 60, .5).empty(), "A late fall must fail despite prior distance");
  auto partial = walk; partial["time"] = 59.98;
  require(!failures(partial, 60, .5).empty(), "A missing final control interval must fail");
  auto missing = walk; missing.erase("airborne_fraction");
  require(!failures(missing, 60, .5).empty(), "Absent measurement must fail closed");
  auto hopping = walk; hopping["airborne_fraction"] = .1001;
  require(!failures(hopping, 60, .5).empty(), "Excessive flight must fail");
  require(!humanoid::goal::aggregate_pass({5,5,5,5,5,5,5,5,3}, 5), "Overall success must not conceal a bad group");
  require(!humanoid::goal::aggregate_pass({5,5,5,5,4,4,4,4,4}, 5), "40/45 is below 90 percent");
  require(humanoid::goal::aggregate_pass({5,5,5,5,5,4,4,4,4}, 5), "41/45 with every group >=4 must qualify numerically");
  std::cout << "Standing, falls, incomplete trials, missing data, flight and aggregate gate checks passed\n";
}
