#pragma once

#include "rotor_model.hpp"
#include <memory>
#include <vector>

namespace helicopter {

struct EntropyResult {
  rotor::Input<double> command{};
  bool accepted=false;
  nlohmann::json diagnostics;
};

// Native CasADi/IPOPT constrained direct multiple shooting. Numerical candidates
// are checked against the original constraints before they may drive the plant.
class EntropyController {
 public:
  static constexpr int horizon_steps=20;
  static constexpr double interval=.1;
  static constexpr double update_period=.1;
  static constexpr double planning_tilt_max_rad=.40;
  static constexpr double planning_airspeed_max_m_s=4.7;
  explicit EntropyController(const rotor::Params& parameters);
  ~EntropyController();
  EntropyController(const EntropyController&)=delete;
  EntropyController& operator=(const EntropyController&)=delete;
  void reset();
  // A causal estimate for the declared common mass/inertia scaling family;
  // the compiled NLP accepts ratios in [0.5, 1.5].
  EntropyResult solve(const rotor::State<double>& initial,
                      const rotor::Input<double>& previous,
                      const rotor::Vec3<double>& wind,
                      const std::vector<rotor::Reference>& reference,
                      double time,double mass_ratio=1.0);
  // Record one scheduled PID landing-support decision without running IPOPT.
  // The returned command is not executable: accepted is always false, and the
  // caller supplies its explicit landing-support controller's command.
  EntropyResult landing_support(double time);
  // Record the first observation interval without running IPOPT. The caller
  // holds the prepared trim command; the returned command is not executable.
  EntropyResult observation_hold(double time);
  static nlohmann::json description();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
