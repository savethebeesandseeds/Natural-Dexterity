#pragma once

#include "rotor_model.hpp"
#include <array>
#include <memory>

namespace helicopter {

struct HinfResult {
  rotor::Input<double> command{};
  nlohmann::json diagnostics;
};

// Central continuous-time DGKF minimum-entropy H-infinity output feedback,
// realized digitally. The certificate belongs to the published linear hover
// generalized plant; nonlinear reference handling and limits are separate.
class HinfController {
 public:
  static constexpr int plant_states=20, controller_states=23, measurements=12;
  static constexpr double sample_period=.01;
  static constexpr std::array<double,measurements> sensor_scales{
    .005,.005,.005,.02,.02,.02,.002,.002,.002,.01,.01,.01};
  explicit HinfController(const rotor::Params& nominal=rotor::Params{});
  ~HinfController();
  HinfController(const HinfController&)=delete;
  HinfController& operator=(const HinfController&)=delete;
  void reset();
  void reset(const rotor::Input<double>& held_command);
  HinfResult update(const rotor::State<double>& measured,const rotor::Reference& reference,
                    double dt=sample_period);
  static nlohmann::json design();
  static nlohmann::json description();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helicopter
