#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace humanoid {

struct TrainingMetrics {
  double x, y, z, vx, vy, vz, upright;
  bool fallen, left_contact, right_contact;
  double command_vx = 0, command_vy = 0, command_yaw_rate = 0;
  // Pelvis-origin velocities in pelvis axes (linear m/s, angular rad/s).
  double body_vx = 0, body_vy = 0, body_vz = 0;
  double body_wx = 0, body_wy = 0, body_wz = 0, yaw = 0;
  // Actual actuator effort over the last control interval, after model limits.
  double applied_torque_sq_mean = 0, mechanical_power_abs = 0;
  std::array<double, 12> joint_velocity{}, applied_torque{};
  // Lowest sole collision-sphere surface above the z=0 floor, in metres.
  double left_foot_height = 0, right_foot_height = 0;
  // Force-weighted mean tangential contact-point speed; zero without support.
  // Use the support flags when aggregating so swing samples are excluded.
  double left_contact_slip_speed = 0, right_contact_slip_speed = 0;
  bool left_support = false, right_support = false;
};

// One independent MuJoCo model/data pair. All methods, including destruction,
// must run on the same thread when rendering is used.
class WalkingSimulation {
 public:
  explicit WalkingSimulation(std::string asset_root = "", std::string device = "cuda",
                             int width = 960, int height = 640,
                             bool load_pretrained = true);
  ~WalkingSimulation() noexcept;
  WalkingSimulation(const WalkingSimulation&) = delete;
  WalkingSimulation& operator=(const WalkingSimulation&) = delete;

  nlohmann::json reset(std::uint64_t seed = 0, double command_x = 0.5,
                       const std::string& policy = "pretrained");
  nlohmann::json step();
  nlohmann::json state() const;
  nlohmann::json provenance();
  void load_policy(const std::string& checkpoint);
  std::vector<unsigned char> render_rgb();
  std::vector<unsigned char> render_jpeg(int quality = 85);
  void close() noexcept;
  double control_dt() const noexcept;

  // External-control mode never loads or calls the published policy. Actions
  // are the same 12 normalized position offsets used by the upstream controller.
  std::vector<float> observation() const;
  void step_action(const std::vector<float>& action);
  bool fallen() const noexcept;
  TrainingMetrics training_metrics() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

nlohmann::json evaluate(WalkingSimulation& simulation, std::size_t trials = 5,
                        double seconds = 20.0, double command_x = 0.5);
nlohmann::json evaluate(WalkingSimulation& simulation, std::size_t trials,
                        double seconds, double command_x,
                        const std::string& policy_path);

}  // namespace humanoid
