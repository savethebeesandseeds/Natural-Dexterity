#pragma once

#include <memory>
#include <nlohmann/json.hpp>

namespace helicopter {

enum class ControllerMode { Hinf, Pid };

// A reduced rigid-body helicopter, not a full aerodynamic rotor model.
// Each step advances MuJoCo by 0.01 seconds. Feedback reads simulated state.
class Simulation {
 public:
  explicit Simulation(unsigned seed = 0, double mass_scale = 1.0,
                      bool autopilot = true, ControllerMode mode = ControllerMode::Hinf,
                      double sensor_noise_scale = 1.0);
  ~Simulation();
  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;
  void reset(unsigned seed = 0, double mass_scale = 1.0, bool autopilot = true,
             ControllerMode mode = ControllerMode::Hinf, double sensor_noise_scale = 1.0);
  void step();
  void gust();
  // Public model description uses the same constants as the dynamics/controller.
  [[nodiscard]] static nlohmann::json model();
  [[nodiscard]] nlohmann::json state() const;
  [[nodiscard]] bool finished() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace helicopter
