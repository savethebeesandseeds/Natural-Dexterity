// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Native adaptation of Unitree's deploy_mujoco.py: the observation, policy and
// PD timing are preserved. Additions: telemetry, seeded trials, CPU headless
// rendering, and an external-action interface for separately trained policies.

#include "humanoid/simulation.hpp"

#include <GL/osmesa.h>
#include <mujoco/mujoco.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <yaml-cpp/yaml.h>
#include <openssl/evp.h>
#include <cstdio>
#include <jpeglib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <csetjmp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace humanoid {
namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;
constexpr const char* kCommit = "276801e46c5d433564f24658bac64f254b7d2d4b";
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<const char*, 12> kActionJointOrder{
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"};

std::string environment(const char* key, const std::string& fallback) {
  const char* value = std::getenv(key);
  return value && *value ? value : fallback;
}

struct JpegFailure {
  jpeg_error_mgr manager{};
  std::jmp_buf jump;
  char message[JMSG_LENGTH_MAX]{};
};

void jpeg_failure(j_common_ptr common) {
  auto* error = reinterpret_cast<JpegFailure*>(common->err);
  common->err->format_message(common, error->message);
  std::longjmp(error->jump, 1);
}

struct JpegEncoder {
  jpeg_compress_struct compressor{};
  JpegFailure error;
  unsigned char* buffer = nullptr;
  unsigned long length = 0;
  ~JpegEncoder() {
    jpeg_destroy_compress(&compressor);
    std::free(buffer);
  }
};

std::string digest_bytes(const unsigned char* bytes, std::size_t size) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (!EVP_Digest(bytes, size, digest.data(), &length, EVP_sha256(), nullptr))
    throw std::runtime_error("SHA-256 failed");
  std::ostringstream output;
  for (unsigned int i = 0; i < length; ++i)
    output << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return output.str();
}

std::string file_sha256(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot hash asset: " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx || !EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr))
    throw std::runtime_error("SHA-256 initialization failed");
  std::array<char, 65536> buffer{};
  while (stream.read(buffer.data(), buffer.size()) || stream.gcount()) {
    if (!EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(stream.gcount())))
      throw std::runtime_error("SHA-256 update failed");
  }
  if (!stream.eof()) throw std::runtime_error("Asset read failed: " + path.string());
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (!EVP_DigestFinal_ex(ctx.get(), digest.data(), &length))
    throw std::runtime_error("SHA-256 finalization failed");
  std::ostringstream output;
  for (unsigned int i = 0; i < length; ++i)
    output << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return output.str();
}

template <std::size_t N>
std::array<float, N> float_array(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != N)
    throw std::runtime_error("Unexpected G1 configuration vector size");
  std::array<float, N> values{};
  for (std::size_t i = 0; i < N; ++i) values[i] = node[i].as<float>();
  return values;
}
}  // namespace

struct WalkingSimulation::Impl {
  fs::path root, config_path, policy_path, xml_path, trained_policy_path;
  YAML::Node config;
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{nullptr, mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data{nullptr, mj_deleteData};
  torch::Device device;
  std::unique_ptr<torch::jit::Module> policy;
  std::array<float, 12> kp{}, kd{}, nominal{}, action{}, target{};
  std::array<float, 3> command{}, command_scale{};
  int decimation = 10, width, height, pelvis = -1, floor = -1;
  std::array<int, 2> foot{};
  std::array<std::array<int, 4>, 2> sole_geoms{};
  TrainingMetrics physical{};
  std::array<double, 2> support_force{}, slip_speed{}, support_seconds{};
  std::array<double, 2> slip_distance{}, slip_squared_integral{}, max_slip_speed{};
  std::uint64_t seed = 0, counter = 0;
  double dt = 0.02, action_scale = .25, angular_scale = .25, position_scale = 1, velocity_scale = .05;
  std::string policy_name;
  std::array<double, 12> initial_noise{};
  std::array<double, 3> start{}, previous{};
  std::array<bool, 2> contacts{};
  std::array<double, 2> absent_time{}, contact_time{};
  std::array<int, 2> touchdowns{};
  int last_touchdown = -1, alternating_count = 0;
  bool fall = false;
  double fall_time = -1, path_distance = 0, min_height = 0, min_upright = 1;
  double airborne_time = 0, flight_time = 0, max_flight_time = 0, speed_error_sum = 0;
  double physics_ms = 0, inference_ms = 0, step_ms = 0, scene_ms = 0;
  double draw_ms = 0, readback_ms = 0, render_ms = 0, jpeg_ms = 0;
  std::size_t samples = 0;
  std::array<double, 12> action_max_abs_by_joint{};
  double action_max_abs = 0;
  std::uint64_t action_intervals = 0, action_near_one_values = 0;
  struct TouchdownEvent { double time; int side; };
  std::vector<TouchdownEvent> touchdown_events;
  json cached_provenance;
  OSMesaContext gl = nullptr;
  std::vector<unsigned char> gl_buffer;
  mjvCamera camera{};
  mjvOption visual_options{};
  mjvScene scene{};
  mjrContext render_context{};
  bool scene_ready = false, render_ready = false;

  Impl(std::string asset_root, std::string inference_device, int w, int h)
      : device(inference_device), width(w), height(h) {
    if (width < 16 || height < 16 || width > 4096 || height > 4096)
      throw std::invalid_argument("Invalid render dimensions");
    root = fs::canonical(asset_root.empty() ? environment("HUMANOID_ASSET_ROOT", "/opt/humanoid/unitree_rl_gym") : asset_root);
    config_path = root / "deploy/deploy_mujoco/configs/g1.yaml";
    config = YAML::LoadFile(config_path.string());
    policy_path = asset_path(config["policy_path"].as<std::string>());
    xml_path = asset_path(config["xml_path"].as<std::string>());
    std::array<char, 2048> error{};
    model.reset(mj_loadXML(xml_path.c_str(), nullptr, error.data(), error.size()));
    if (!model) throw std::runtime_error("MuJoCo model load failed: " + std::string(error.data()));
    model->opt.timestep = config["simulation_dt"].as<double>();
    decimation = config["control_decimation"].as<int>();
    dt = model->opt.timestep * decimation;
    model->vis.global.offwidth = width;
    model->vis.global.offheight = height;
    // CPU OpenGL needs no multisampled or large shadow framebuffer. These are
    // display settings only; collision geometry and dynamics are unchanged.
    model->vis.quality.offsamples = 0;
    model->vis.quality.shadowsize = 256;
    data.reset(mj_makeData(model.get()));
    if (!data) throw std::runtime_error("MuJoCo data allocation failed");
    if (device.is_cuda() && !torch::cuda::is_available())
      throw std::runtime_error("CUDA inference requested, but LibTorch cannot access CUDA");
    kp = float_array<12>(config["kps"]); kd = float_array<12>(config["kds"]);
    nominal = float_array<12>(config["default_angles"]);
    command_scale = float_array<3>(config["cmd_scale"]);
    action_scale = config["action_scale"].as<double>();
    angular_scale = config["ang_vel_scale"].as<double>();
    position_scale = config["dof_pos_scale"].as<double>();
    velocity_scale = config["dof_vel_scale"].as<double>();
    pelvis = id(mjOBJ_BODY, "pelvis"); floor = id(mjOBJ_GEOM, "floor");
    foot = {id(mjOBJ_BODY, "left_ankle_roll_link"), id(mjOBJ_BODY, "right_ankle_roll_link")};
    validate_contract();
    for (int side = 0; side < 2; ++side) {
      std::size_t count = 0;
      for (int geom = 0; geom < model->ngeom; ++geom) {
        if (model->geom_bodyid[geom] != foot[side] || model->geom_type[geom] != mjGEOM_SPHERE ||
            !(model->geom_contype[geom] || model->geom_conaffinity[geom])) continue;
        if (count == sole_geoms[side].size()) throw std::runtime_error("Unexpected G1 sole collision geometry");
        sole_geoms[side][count++] = geom;
      }
      if (count != sole_geoms[side].size()) throw std::runtime_error("Expected four G1 sole collision spheres per foot");
    }
    // More than a 20-second episode can produce with the 60ms touchdown
    // debounce. Training records POD events, constructing JSON only on demand.
    touchdown_events.reserve(1024);
    mjv_defaultCamera(&camera); mjv_defaultOption(&visual_options);
    // Unitree supplies both visual meshes (group 1) and overlapping collision
    // meshes (group 0). Draw each robot surface once; keep the floor visible.
    model->geom_group[floor] = 2;
    visual_options.geomgroup[0] = 0;
    mjv_defaultScene(&scene); mjr_defaultContext(&render_context);
    camera.type = mjCAMERA_FREE; camera.distance = 3.1; camera.azimuth = 135; camera.elevation = -18;
  }

  ~Impl() noexcept { close(); }

  fs::path asset_path(std::string value) const {
    const std::string placeholder = "{LEGGED_GYM_ROOT_DIR}";
    const auto found = value.find(placeholder);
    if (found != std::string::npos) value.replace(found, placeholder.size(), root.string());
    const fs::path path = fs::weakly_canonical(value);
    const fs::path relative = path.lexically_relative(root);
    if (relative.empty() || *relative.begin() == "..")
      throw std::runtime_error("Configured asset is outside asset root");
    return path;
  }

  int id(mjtObj kind, const char* name) const {
    const int value = mj_name2id(model.get(), kind, name);
    if (value < 0) throw std::runtime_error("Missing G1 model element: " + std::string(name));
    return value;
  }

  void validate_contract() const {
    if (model->nq != 19 || model->nv != 18 || model->nu != 12 || model->neq != 0 ||
        config["num_actions"].as<int>() != 12 || config["num_obs"].as<int>() != 47 || std::abs(dt - .02) > 1e-12)
      throw std::runtime_error("Expected official free-base G1 12-joint/47-observation/50Hz model");
    const int base = id(mjOBJ_JOINT, "floating_base_joint");
    if (model->jnt_type[base] != mjJNT_FREE) throw std::runtime_error("Base must be free");
    int index = 0;
    for (const char* side : {"left", "right"}) {
      for (const char* joint : {"hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"}) {
        const auto name = std::string(side) + "_" + joint + "_joint";
        const int j = id(mjOBJ_JOINT, name.c_str()), a = id(mjOBJ_ACTUATOR, name.c_str());
        if (model->jnt_qposadr[j] != index + 7 || model->jnt_dofadr[j] != index + 6 ||
            a != index || model->actuator_trnid[2 * a] != j)
          throw std::runtime_error("Unexpected G1 joint/actuator order: " + name);
        ++index;
      }
    }
  }

  void reset(std::uint64_t trial_seed, double command_x, const std::string& kind) {
    if (kind != "pretrained" && kind != "zero" && kind != "external" && kind != "trained")
      throw std::invalid_argument("Policy must be pretrained, zero, external or trained");
    if (kind == "trained" && trained_policy_path.empty())
      throw std::logic_error("Call load_policy(checkpoint) before resetting a trained policy");
    if (!std::isfinite(command_x) || std::abs(command_x) > 1)
      throw std::invalid_argument("Forward command must be within [-1,1] m/s");
    seed = trial_seed; policy_name = kind; command = {float(command_x), 0, 0};
    mj_resetData(model.get(), data.get());
    initial_noise.fill(0);
    if (seed != 0) {
      std::mt19937_64 generator(seed);
      std::uniform_real_distribution<double> distribution(-.005, .005);
      for (int i = 0; i < 12; ++i) data->qpos[7 + i] += initial_noise[i] = distribution(generator);
    }
    mj_forward(model.get(), data.get());
    policy.reset();
    if (kind == "pretrained" || kind == "trained") {
      // The exported G1 recurrent module cannot flatten its cuDNN parameters;
      // that backend otherwise repacks its weights on every 50Hz action.
      // Use ATen's CUDA recurrent operators instead. This retains GPU inference
      // and avoids the measured repeated allocation/compaction, not its warning.
      if (device.is_cuda()) at::globalContext().setUserEnabledCuDNN(false);
      const auto& checkpoint = kind == "pretrained" ? policy_path : trained_policy_path;
      policy = std::make_unique<torch::jit::Module>(torch::jit::load(checkpoint.string(), device));
      policy->eval();
    }
    action.fill(0); target = nominal; counter = 0;
    std::copy_n(data->qpos, 3, start.begin()); previous = start;
    path_distance = 0; fall = false; fall_time = -1;
    min_height = data->qpos[2]; min_upright = 1;
    contacts = feet_contacts(); absent_time.fill(0); contact_time.fill(0); touchdowns.fill(0);
    last_touchdown = -1; alternating_count = 0;
    airborne_time = flight_time = max_flight_time = speed_error_sum = 0; samples = 0;
    action_max_abs_by_joint.fill(0); action_max_abs = 0;
    action_intervals = action_near_one_values = 0;
    touchdown_events.clear();
    physical = {};
    support_seconds.fill(0); slip_distance.fill(0);
    slip_squared_integral.fill(0); max_slip_speed.fill(0);
    update_physical_metrics(false);
    // Locally trained policies choose a_0 from the reset observation, matching
    // step_action(). The published baseline keeps its upstream startup timing.
    if (kind == "trained") infer_action();
  }

  std::vector<float> observation() const {
    std::vector<float> obs(47, 0);
    const auto* q = data->qpos + 3;
    const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    for (int i = 0; i < 3; ++i) {
      obs[i] = float(data->qvel[3 + i] * angular_scale);
      obs[6 + i] = command[i] * command_scale[i];
    }
    obs[3] = float(2 * (-qz * qx + qw * qy));
    obs[4] = float(-2 * (qz * qy + qw * qx));
    obs[5] = float(1 - 2 * (qw * qw + qz * qz));
    for (int i = 0; i < 12; ++i) {
      obs[9 + i] = float((data->qpos[7 + i] - nominal[i]) * position_scale);
      obs[21 + i] = float(data->qvel[6 + i] * velocity_scale);
      obs[33 + i] = action[i];
    }
    const double phase = std::fmod(counter * model->opt.timestep, .8) / .8;
    obs[45] = float(std::sin(2 * kPi * phase)); obs[46] = float(std::cos(2 * kPi * phase));
    return obs;
  }

  void set_action(const std::vector<float>& values) {
    if (values.size() != 12) throw std::invalid_argument("Expected 12 policy actions");
    for (int i = 0; i < 12; ++i) {
      if (!std::isfinite(values[i])) throw std::runtime_error("Non-finite policy action");
      action[i] = values[i]; target[i] = float(action[i] * action_scale + nominal[i]);
    }
  }

  void infer_action() {
    inference_ms = 0;
    if (!policy) return;
    const auto started = std::chrono::steady_clock::now();
    c10::InferenceMode inference_guard;
    auto obs = observation();
    auto input = torch::from_blob(obs.data(), {1, 47}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    auto output = policy->forward({input}).toTensor().to(torch::kCPU).contiguous();
    if (output.numel() != 12 || output.scalar_type() != torch::kFloat32)
      throw std::runtime_error("Policy must return 12 float32 actions");
    const float* values = output.data_ptr<float>();
    set_action(std::vector<float>(values, values + 12));
    inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  }

  void advance(bool infer_next) {
    if (fall) return;
    // Observe the action whose targets are about to drive this PD interval.
    // In playback infer_action() below prepares the NEXT interval; counting
    // there could include an action never applied after the final fall.
    // Zero-torque control bypasses PD and has no position-target actions.
    if (policy_name != "zero") {
      for (std::size_t joint = 0; joint < action.size(); ++joint) {
        const double magnitude = std::abs(static_cast<double>(action[joint]));
        action_max_abs_by_joint[joint] = std::max(action_max_abs_by_joint[joint], magnitude);
        action_max_abs = std::max(action_max_abs, magnitude);
        action_near_one_values += magnitude >= .95;
      }
      ++action_intervals;
    }
    const auto started = std::chrono::steady_clock::now();
    physical.applied_torque.fill(0);
    physical.applied_torque_sq_mean = physical.mechanical_power_abs = 0;
    for (int substep = 0; substep < decimation; ++substep) {
      std::array<double, 12> velocity_before{};
      for (int i = 0; i < 12; ++i)
        data->ctrl[i] = policy_name == "zero" ? 0 : (target[i] - data->qpos[7 + i]) * kp[i] - data->qvel[6 + i] * kd[i];
      std::copy_n(data->qvel + 6, 12, velocity_before.begin());
      mj_step(model.get(), data.get());
      // qfrc_actuator is the actual generalized actuator force, including the
      // joint actuator-force limits. ctrl is only the requested PD torque.
      // Under the model's Euler integrator, these forces act at pre-step qvel.
      for (int i = 0; i < 12; ++i) {
        const double torque = data->qfrc_actuator[6 + i];
        physical.applied_torque[i] += torque / decimation;
        physical.applied_torque_sq_mean += torque * torque / (12 * decimation);
        physical.mechanical_power_abs += std::abs(torque * velocity_before[i]) / decimation;
      }
      ++counter;
    }
    for (int i = 0; i < model->nq; ++i)
      if (!std::isfinite(data->qpos[i])) throw std::runtime_error("Non-finite MuJoCo position");
    for (int i = 0; i < model->nv; ++i)
      if (!std::isfinite(data->qvel[i])) throw std::runtime_error("Non-finite MuJoCo velocity");
    if (std::abs(data->time - counter * model->opt.timestep) > 1e-7)
      throw std::runtime_error("MuJoCo unexpectedly reset its clock; refusing success result");
    physics_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    inference_ms = 0;
    if (infer_next) infer_action();
    const auto forward_started = std::chrono::steady_clock::now();
    mj_forward(model.get(), data.get());
    physics_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - forward_started).count();
    update_metrics();
    step_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  }

  std::array<bool, 2> feet_contacts() const {
    std::array<bool, 2> result{};
    for (int i = 0; i < data->ncon; ++i) {
      const auto& contact = data->contact[i];
      if (contact.dist > 0) continue;
      const int other = contact.geom[0] == floor ? contact.geom[1] : contact.geom[1] == floor ? contact.geom[0] : -1;
      if (other >= 0)
        for (int side = 0; side < 2; ++side)
          if (model->geom_bodyid[other] == foot[side]) result[side] = true;
    }
    return result;
  }

  void update_physical_metrics(bool accumulate) noexcept {
    physical.x = data->qpos[0]; physical.y = data->qpos[1]; physical.z = data->qpos[2];
    physical.vx = data->qvel[0]; physical.vy = data->qvel[1]; physical.vz = data->qvel[2];
    physical.upright = data->xmat[9 * pelvis + 8]; physical.fallen = fall;
    physical.left_contact = contacts[0]; physical.right_contact = contacts[1];
    physical.command_vx = command[0]; physical.command_vy = command[1]; physical.command_yaw_rate = command[2];
    mjtNum velocity[6]{};
    // XBODY selects the pelvis body origin/axes; BODY would use its different
    // inertial COM frame. MuJoCo returns rotation first, translation second.
    mj_objectVelocity(model.get(), data.get(), mjOBJ_XBODY, pelvis, velocity, 1);
    physical.body_wx = velocity[0]; physical.body_wy = velocity[1]; physical.body_wz = velocity[2];
    physical.body_vx = velocity[3]; physical.body_vy = velocity[4]; physical.body_vz = velocity[5];
    physical.yaw = std::atan2(data->xmat[9 * pelvis + 3], data->xmat[9 * pelvis]);
    std::copy_n(data->qvel + 6, 12, physical.joint_velocity.begin());
    std::array<double, 2> foot_height{};
    for (int side = 0; side < 2; ++side) {
      foot_height[side] = std::numeric_limits<double>::infinity();
      for (int geom : sole_geoms[side])
        foot_height[side] = std::min(foot_height[side],
            data->geom_xpos[3 * geom + 2] - model->geom_size[3 * geom]);
    }
    physical.left_foot_height = foot_height[0]; physical.right_foot_height = foot_height[1];
    support_force.fill(0); slip_speed.fill(0);
    for (int i = 0; i < data->ncon; ++i) {
      const auto& contact = data->contact[i];
      if (contact.dist > 0 || contact.efc_address < 0) continue;
      const int other = contact.geom[0] == floor ? contact.geom[1] : contact.geom[1] == floor ? contact.geom[0] : -1;
      if (other < 0) continue;
      const int body = model->geom_bodyid[other];
      const int side = body == foot[0] ? 0 : body == foot[1] ? 1 : -1;
      if (side < 0) continue;
      mjtNum force[6]{};
      mj_contactForce(model.get(), data.get(), i, force);
      if (force[0] <= 1e-6) continue;
      // Velocity of the material point on the foot at the contact position,
      // not the foot COM and not the motion of the geometric contact locus.
      // The model floor is static, so this is also floor-relative velocity.
      std::array<mjtNum, 3 * 18> jacobian{};
      mj_jac(model.get(), data.get(), jacobian.data(), nullptr, contact.pos, body);
      mjtNum point_velocity[3]{};
      for (int axis = 0; axis < 3; ++axis)
        for (int dof = 0; dof < 18; ++dof)
          point_velocity[axis] += jacobian[18 * axis + dof] * data->qvel[dof];
      const double normal_speed = mju_dot3(point_velocity, contact.frame);
      double tangential_squared = 0;
      for (int axis = 0; axis < 3; ++axis) {
        const double tangent = point_velocity[axis] - normal_speed * contact.frame[axis];
        tangential_squared += tangent * tangent;
      }
      support_force[side] += force[0];
      slip_speed[side] += force[0] * std::sqrt(tangential_squared);
    }
    for (int side = 0; side < 2; ++side) {
      if (support_force[side] <= 0) continue;
      slip_speed[side] /= support_force[side];
      if (accumulate) {
        support_seconds[side] += dt;
        slip_distance[side] += slip_speed[side] * dt;
        slip_squared_integral[side] += slip_speed[side] * slip_speed[side] * dt;
        max_slip_speed[side] = std::max(max_slip_speed[side], slip_speed[side]);
      }
    }
    physical.left_support = support_force[0] > 0; physical.right_support = support_force[1] > 0;
    physical.left_contact_slip_speed = slip_speed[0]; physical.right_contact_slip_speed = slip_speed[1];
  }

  void update_metrics() {
    path_distance += std::hypot(data->qpos[0] - previous[0], data->qpos[1] - previous[1]);
    std::copy_n(data->qpos, 3, previous.begin());
    const double upright = data->xmat[9 * pelvis + 8];
    min_height = std::min(min_height, data->qpos[2]); min_upright = std::min(min_upright, upright);
    if (data->qpos[2] < .45 || upright < .5) { fall = true; fall_time = data->time; }
    const auto now = feet_contacts();
    std::array<int, 2> events{};
    std::size_t event_count = 0;
    for (int side = 0; side < 2; ++side) {
      if (now[side]) {
        if (!contacts[side] && absent_time[side] >= .06 - 1e-9) {
          ++touchdowns[side]; events[event_count++] = side;
          touchdown_events.push_back({data->time, side});
        }
        absent_time[side] = 0; contact_time[side] += dt;
      } else { absent_time[side] += dt; }
    }
    if (event_count == 1) {
      if (last_touchdown >= 0 && last_touchdown != events[0]) ++alternating_count;
      last_touchdown = events[0];
    } else if (event_count == 2) { last_touchdown = -1; }
    contacts = now;
    if (!now[0] && !now[1]) {
      airborne_time += dt; flight_time += dt; max_flight_time = std::max(max_flight_time, flight_time);
    } else { flight_time = 0; }
    const double error = data->qvel[0] - command[0]; speed_error_sum += error * error; ++samples;
    update_physical_metrics(true);
  }

  json state() const {
    const double elapsed = data->time, distance = data->qpos[0] - start[0];
    json events = json::array(), slips = json::object();
    for (const auto& event : touchdown_events)
      events.push_back({{"time", event.time}, {"side", event.side == 0 ? "left" : "right"}});
    const auto slip_summary = [](double duration, double travelled, double squared, double maximum) {
      return json{{"support_seconds", duration}, {"slip_distance", travelled},
                  {"mean_speed", duration > 0 ? json(travelled / duration) : json(nullptr)},
                  {"rms_speed", duration > 0 ? json(std::sqrt(squared / duration)) : json(nullptr)},
                  {"max_speed", duration > 0 ? json(maximum) : json(nullptr)}};
    };
    for (int side = 0; side < 2; ++side) {
      auto entry = slip_summary(support_seconds[side], slip_distance[side],
                                slip_squared_integral[side], max_slip_speed[side]);
      entry["support"] = support_force[side] > 0;
      entry["contact_slip_speed"] = slip_speed[side];
      entry["normal_force"] = support_force[side];
      slips[side == 0 ? "left" : "right"] = std::move(entry);
    }
    slips["combined"] = slip_summary(support_seconds[0] + support_seconds[1],
        slip_distance[0] + slip_distance[1], slip_squared_integral[0] + slip_squared_integral[1],
        std::max(max_slip_speed[0], max_slip_speed[1]));
    slips["definition"] = "At each 20ms endpoint, each supporting foot's speed is the normal-force-weighted mean tangential speed of its material contact points relative to the static floor. Support requires touching, active constraint, and normal force >1e-6 N. Swing samples are excluded. Integrals use endpoint speed times 20ms; RMS is over these per-foot speeds, and double support contributes two foot intervals. No support yields null aggregate speeds.";
    slips["units"] = {{"speed", "m/s"}, {"slip_distance", "m"}, {"support_seconds", "s"}, {"normal_force", "N"}};
    return {
      {"time", elapsed}, {"seed", seed}, {"policy", policy_name},
      {"policy_origin", policy_name == "pretrained" ? "Unitree pretrained" : policy_name == "zero" ? "zero motor torques" : policy_name == "trained" ? "locally trained TorchScript" : "external policy actions"},
      {"command_x", command[0]}, {"distance", distance}, {"path_distance", path_distance},
      {"speed", data->qvel[0]}, {"mean_speed", elapsed > 0 ? distance / elapsed : 0},
      {"velocity", {data->qvel[0], data->qvel[1], data->qvel[2]}},
      {"torso_height", data->qpos[2]}, {"height_reference", "pelvis floating-base origin above z=0 ground"},
      {"upright", data->xmat[9 * pelvis + 8]}, {"fall", fall}, {"fall_time", fall ? json(fall_time) : json(nullptr)},
      {"min_height", min_height}, {"min_upright", min_upright},
      {"feet_contacts", {{"left", contacts[0]}, {"right", contacts[1]}}},
      {"foot_slip", std::move(slips)},
      {"physical_metrics", {
        {"command", {physical.command_vx, physical.command_vy, physical.command_yaw_rate}},
        {"body_velocity", {physical.body_vx, physical.body_vy, physical.body_vz}},
        {"body_angular_velocity", {physical.body_wx, physical.body_wy, physical.body_wz}},
        {"yaw", physical.yaw}, {"joint_velocity", physical.joint_velocity},
        {"applied_torque", physical.applied_torque},
        {"applied_torque_sq_mean", physical.applied_torque_sq_mean},
        {"mechanical_power_abs", physical.mechanical_power_abs},
        {"left_foot_height", physical.left_foot_height}, {"right_foot_height", physical.right_foot_height}
      }},
      {"left_steps", touchdowns[0]}, {"right_steps", touchdowns[1]}, {"alternating_count", alternating_count},
      {"airborne_fraction", elapsed > 0 ? airborne_time / elapsed : 0}, {"max_flight_time", max_flight_time},
      {"speed_tracking_rmse", samples ? std::sqrt(speed_error_sum / samples) : 0},
      {"position", {data->qpos[0], data->qpos[1], data->qpos[2]}},
      {"physics_device", "cpu"}, {"inference_device", device.str()},
      {"timing_ms", {{"physics", physics_ms}, {"inference", inference_ms}, {"step_total", step_ms},
                     {"scene_update", scene_ms}, {"draw", draw_ms}, {"readback", readback_ms},
                     {"render_rgb", render_ms}, {"jpeg_encode", jpeg_ms}}},
      {"applied_action_diagnostics", {
        {"control_intervals", action_intervals}, {"action_values", action_intervals * action.size()},
        {"max_abs_normalized_action", action_max_abs},
        {"per_joint_max_abs", action_max_abs_by_joint}, {"joint_order", kActionJointOrder},
        {"near_one_threshold", .95},
        {"near_one_fraction", action_intervals
            ? json(static_cast<double>(action_near_one_values) / (static_cast<double>(action_intervals) * action.size()))
            : json(nullptr)},
        {"joint_position_offset_scale_rad", action_scale},
        {"scope", "One sample per joint per applied PD control interval since reset; zero-torque control excluded"},
        {"interpretation", "near_one_fraction counts |normalized action| >=0.95; this is not saturation for an unbounded published policy"}
      }},
      {"initial_joint_noise_rad", initial_noise}, {"touchdown_events", std::move(events)}
    };
  }

  void init_renderer() {
    if (gl) return;
    gl = OSMesaCreateContextExt(OSMESA_RGBA, 24, 8, 0, nullptr);
    if (!gl) throw std::runtime_error("OSMesa context creation failed");
    gl_buffer.resize(static_cast<std::size_t>(width) * height * 4);
    if (!OSMesaMakeCurrent(gl, gl_buffer.data(), GL_UNSIGNED_BYTE, width, height)) {
      OSMesaDestroyContext(gl); gl = nullptr;
      throw std::runtime_error("OSMesaMakeCurrent failed");
    }
    mjv_makeScene(model.get(), &scene, 4000); scene_ready = true;
    mjr_makeContext(model.get(), &render_context, mjFONTSCALE_100); render_ready = true;
    mjr_setBuffer(mjFB_OFFSCREEN, &render_context);
    if (render_context.currentBuffer != mjFB_OFFSCREEN)
      throw std::runtime_error("MuJoCo offscreen framebuffer unavailable");
  }

  std::vector<unsigned char> render_rgb() {
    const auto started = std::chrono::steady_clock::now();
    init_renderer();
    if (!OSMesaMakeCurrent(gl, gl_buffer.data(), GL_UNSIGNED_BYTE, width, height))
      throw std::runtime_error("OSMesaMakeCurrent failed");
    camera.lookat[0] = data->qpos[0]; camera.lookat[1] = data->qpos[1]; camera.lookat[2] = .7;
    const auto scene_started = std::chrono::steady_clock::now();
    mjv_updateScene(model.get(), data.get(), &visual_options, nullptr, &camera, mjCAT_ALL, &scene);
    scene.flags[mjRND_SHADOW] = 0;
    scene.flags[mjRND_REFLECTION] = 0;
    scene.flags[mjRND_FOG] = 0;
    const auto draw_started = std::chrono::steady_clock::now();
    scene_ms = std::chrono::duration<double, std::milli>(draw_started - scene_started).count();
    const mjrRect viewport{0, 0, width, height};
    mjr_render(viewport, &scene, &render_context);
    const auto readback_started = std::chrono::steady_clock::now();
    draw_ms = std::chrono::duration<double, std::milli>(readback_started - draw_started).count();
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
    mjr_readPixels(pixels.data(), nullptr, viewport, &render_context);
    readback_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - readback_started).count();
    const auto stride = static_cast<std::size_t>(width) * 3;
    for (int y = 0; y < height / 2; ++y)
      for (std::size_t x = 0; x < stride; ++x)
        std::swap(pixels[y * stride + x], pixels[(height - y - 1) * stride + x]);
    render_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return pixels;
  }

  json provenance() {
    if (cached_provenance.is_null()) {
      std::vector<fs::path> files{config_path, policy_path};
      for (const auto& entry : fs::recursive_directory_iterator(xml_path.parent_path())) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (ext == ".xml" || ext == ".stl" || ext == ".obj") files.push_back(entry.path());
      }
      std::sort(files.begin(), files.end());
      json hashes = json::object();
      for (const auto& path : files) hashes[path.lexically_relative(root).generic_string()] = file_sha256(path);
      const auto serialized = hashes.dump();
      cached_provenance = {
        {"source", "https://github.com/unitreerobotics/unitree_rl_gym"},
        {"source_commit", environment("UNITREE_COMMIT", kCommit)},
        {"policy_training", "Published Unitree policy; not trained by this project"},
        {"source_contract", "deploy/deploy_mujoco/deploy_mujoco.py and configs/g1.yaml"},
        {"asset_sha256", hashes},
        {"asset_manifest_sha256", digest_bytes(reinterpret_cast<const unsigned char*>(serialized.data()), serialized.size())},
        {"asset_manifest_scope", "config, policy, and every XML/STL/OBJ in the G1 model resource directory"},
        {"versions", {{"mujoco", mj_versionString()}, {"libtorch", TORCH_VERSION}, {"compiler", __VERSION__}, {"language", "C++17"}}},
        {"simulation_dt", model->opt.timestep}, {"control_dt", dt},
        {"physical_metric_definitions", {
          {"velocity", "Pelvis body-origin linear m/s and angular rad/s velocity expressed in pelvis axes; mj_objectVelocity(mjOBJ_XBODY, local=1). Legacy velocity remains world-frame free-base linear velocity."},
          {"yaw", "World heading in radians of the pelvis x-axis, atan2(R_yx,R_xx)."},
          {"foot_height", "Minimum world z of each of four sole collision sphere centres minus its radius; metres above the z=0 floor, negative during penetration."},
          {"joint_velocity", "End-of-control-interval joint velocity, rad/s, official actuator order."},
          {"applied_torque", "Mean actual generalized joint actuator force qfrc_actuator[6:18] across the ten physics substeps, N m; includes model joint actuator-force limits."},
          {"applied_torque_sq_mean", "Mean squared actual joint actuator torque over twelve joints and ten physics substeps, (N m)^2."},
          {"mechanical_power_abs", "Mean over ten physics substeps of sum_j |actual actuator torque_j * pre-integration joint velocity_j|, watts; mechanical activity proxy, not electrical power consumption."},
          {"foot_slip", "Endpoint 50Hz force-weighted tangential material contact-point speed from mj_jac*qvel. Supporting foot-floor contacts only; per-foot and combined support-duration-weighted integrals in state(). This is a sampled slip diagnostic, not an absence-of-slip proof."}
        }},
        {"dimensions", {{"nq", model->nq}, {"nv", model->nv}, {"nu", model->nu}}},
        {"physics_device", "cpu"}, {"inference_device", device.str()}, {"render_backend", "OSMesa CPU OpenGL"},
        {"cudnn_enabled", at::globalContext().userEnabledCuDNN()},
        {"render_settings", {{"width", width}, {"height", height}, {"multisamples", 0},
                             {"shadows", false}, {"reflections", false}, {"collision_visuals", false}}},
        {"model_limitations", "12 actuated leg joints; upper-body joints are fixed; level ground only"}
      };
      const fs::path source = environment("HUMANOID_SOURCE_ROOT", "/workspace");
      if (fs::exists(source / "src/simulation.cpp"))
        cached_provenance["simulation_code_sha256"] = file_sha256(source / "src/simulation.cpp");
      if (fs::exists("/proc/self/exe")) cached_provenance["executable_sha256"] = file_sha256("/proc/self/exe");
    }
    auto result = cached_provenance;
    if (policy_name == "trained") {
      result["policy_training"] = "Locally trained TorchScript policy; consult the accompanying training run manifest";
      result["active_policy_path"] = trained_policy_path.string();
      result["active_policy_sha256"] = file_sha256(trained_policy_path);
    }
    return result;
  }

  void close() noexcept {
    if (gl) {
      OSMesaMakeCurrent(gl, gl_buffer.data(), GL_UNSIGNED_BYTE, width, height);
      if (render_ready) mjr_freeContext(&render_context);
      if (scene_ready) mjv_freeScene(&scene);
      OSMesaDestroyContext(gl); gl = nullptr;
    }
    render_ready = scene_ready = false;
    gl_buffer.clear();
  }
};

WalkingSimulation::WalkingSimulation(std::string root, std::string device, int width, int height, bool load_pretrained)
  : impl_(std::make_unique<Impl>(std::move(root), std::move(device), width, height)) {
  impl_->reset(0, .5, load_pretrained ? "pretrained" : "external");
}
WalkingSimulation::~WalkingSimulation() noexcept = default;
json WalkingSimulation::reset(std::uint64_t seed, double command, const std::string& policy) {
  impl_->reset(seed, command, policy); return impl_->state();
}
json WalkingSimulation::step() { impl_->advance(true); return impl_->state(); }
json WalkingSimulation::state() const { return impl_->state(); }
json WalkingSimulation::provenance() { return impl_->provenance(); }
void WalkingSimulation::load_policy(const std::string& checkpoint) {
  impl_->trained_policy_path = std::filesystem::canonical(checkpoint);
  impl_->reset(impl_->seed, impl_->command[0], "trained");
}
std::vector<unsigned char> WalkingSimulation::render_rgb() { return impl_->render_rgb(); }
void WalkingSimulation::close() noexcept { impl_->close(); }
double WalkingSimulation::control_dt() const noexcept { return impl_->dt; }
std::vector<float> WalkingSimulation::observation() const { return impl_->observation(); }
bool WalkingSimulation::fallen() const noexcept { return impl_->fall; }
TrainingMetrics WalkingSimulation::training_metrics() const noexcept {
  return impl_->physical;
}
void WalkingSimulation::step_action(const std::vector<float>& values) {
  if (impl_->policy_name != "external") throw std::logic_error("step_action requires external policy mode");
  impl_->set_action(values); impl_->advance(false);
}

std::vector<unsigned char> WalkingSimulation::render_jpeg(int quality) {
  if (quality < 1 || quality > 100) throw std::invalid_argument("JPEG quality must be 1..100");
  auto rgb = render_rgb();
  const auto encode_started = std::chrono::steady_clock::now();
  // Libjpeg's default handler exits the process. Translate failures into an
  // exception instead, keeping mutable encoder state on the heap across longjmp.
  auto encoder = std::make_unique<JpegEncoder>();
  auto& compressor = encoder->compressor;
  compressor.err = jpeg_std_error(&encoder->error.manager);
  encoder->error.manager.error_exit = jpeg_failure;
  if (setjmp(encoder->error.jump))
    throw std::runtime_error("JPEG encoding failed: " + std::string(encoder->error.message));
  jpeg_create_compress(&compressor);
  jpeg_mem_dest(&compressor, &encoder->buffer, &encoder->length);
  compressor.image_width = impl_->width; compressor.image_height = impl_->height;
  compressor.input_components = 3; compressor.in_color_space = JCS_RGB;
  jpeg_set_defaults(&compressor); jpeg_set_quality(&compressor, quality, TRUE);
  jpeg_start_compress(&compressor, TRUE);
  while (compressor.next_scanline < compressor.image_height) {
    JSAMPROW row = rgb.data() + static_cast<std::size_t>(compressor.next_scanline) * impl_->width * 3;
    jpeg_write_scanlines(&compressor, &row, 1);
  }
  jpeg_finish_compress(&compressor);
  impl_->jpeg_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - encode_started).count();
  return std::vector<unsigned char>(encoder->buffer, encoder->buffer + encoder->length);
}

json evaluate(WalkingSimulation& sim, std::size_t trials, double seconds, double command_x) {
  return evaluate(sim, trials, seconds, command_x, "");
}

json evaluate(WalkingSimulation& sim, std::size_t trials, double seconds, double command_x,
              const std::string& policy_path) {
  if (trials < 5 || !std::isfinite(seconds) || seconds < 20 || !std::isfinite(command_x) || command_x <= 0)
    throw std::invalid_argument("Evaluation requires >=5 trials, >=20 seconds, positive command");
  const auto started = std::chrono::steady_clock::now();
  const bool custom = !policy_path.empty();
  const std::string evaluated_policy = custom ? "trained" : "pretrained";
  fs::path checkpoint;
  std::string checkpoint_hash;
  if (custom) {
    checkpoint = fs::canonical(policy_path);
    if (!fs::is_regular_file(checkpoint)) throw std::invalid_argument("Policy checkpoint must be a regular local file");
    checkpoint_hash = file_sha256(checkpoint);
    sim.load_policy(checkpoint.string());
  }
  sim.reset(0, command_x, evaluated_policy);
  // Capture the evaluated policy now, before the paired zero-control resets.
  auto policy_provenance = sim.provenance();
  if (custom) {
    policy_provenance["source_scope"] = "Unitree robot-model assets and controller observation/action contract";
    policy_provenance["policy_training"] = "Caller-supplied local TorchScript checkpoint; consult its training run manifest for training provenance";
    policy_provenance["evaluated_policy"] = {
      {"kind", "explicit local checkpoint"}, {"path", checkpoint.string()}, {"sha256", checkpoint_hash},
      {"published_reference_weights_used", false}
    };
  }
  json results = json::array(), summary = json::object();
  for (const std::string& policy : std::array<std::string, 2>{evaluated_policy, "zero"}) {
    std::size_t passed = 0;
    for (std::size_t seed = 0; seed < trials; ++seed) {
      if (custom && file_sha256(checkpoint) != checkpoint_hash)
        throw std::runtime_error("Selected checkpoint changed during evaluation; refusing mixed-policy results");
      auto result = sim.reset(seed, command_x, policy);
      while (result["time"].get<double>() + 1e-9 < seconds && !sim.fallen()) result = sim.step();
      const bool success = !result["fall"].get<bool>() && result["time"].get<double>() >= seconds - sim.control_dt()
        && result["distance"].get<double>() >= command_x * seconds * .5
        && result["left_steps"].get<int>() >= 5 && result["right_steps"].get<int>() >= 5
        && result["alternating_count"].get<int>() >= 8 && result["airborne_fraction"].get<double>() <= .1;
      result["walking_criteria_pass"] = success; passed += success;
      json progress;
      for (const char* key : {"policy", "seed", "time", "distance", "fall", "left_steps", "right_steps", "alternating_count", "walking_criteria_pass"})
        progress[key] = result[key];
      std::cout << progress.dump() << std::endl;
      results.push_back(std::move(result));
    }
    summary[policy] = {{"passed", passed}, {"trials", trials}};
  }
  if (custom && file_sha256(checkpoint) != checkpoint_hash)
    throw std::runtime_error("Selected checkpoint changed during evaluation; refusing mixed-policy results");
  return {
    {"schema_version", 1}, {"provenance", policy_provenance},
    {"experiment", {
      {"requested_seconds_per_trial", seconds}, {"trials_per_policy", trials}, {"command_x", command_x},
      {"seed_definition", "seed 0 canonical; others mt19937_64 uniform +/-0.005 rad initial leg-joint noise; exact vector recorded"},
      {"zero_baseline", "All 12 motor torques zero; no PD, policy, or external support"},
      {"fall_definition", "pelvis height <0.45m OR pelvis up-axis cosine <0.5; trial then stops"},
      {"step_definition", "foot-floor touchdown after >=60ms absence; simultaneous landings do not alternate"},
      {"success_criteria", "full duration without fall; distance >=50% commanded; >=5 touchdowns per foot; >=8 alternations; <=10% airborne samples"},
      {"limitations", {
        custom ? "Initial-condition trials of one caller-selected checkpoint, not independently trained seeds."
               : "Initial-condition trials of one published policy, not independently trained seeds.",
        "Foot-contact telemetry supports visual gait inspection; absence of slipping is not established.",
        "A finite flat-ground rollout is not a stability proof or evidence of hardware safety.",
        custom ? "This evaluator measures checkpoint behavior; training provenance must be established separately."
               : "This baseline evaluation makes no policy-training or sim-to-real claim."
      }}
    }},
    {"results", results}, {"summary", summary},
    {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}
  };
}
}  // namespace humanoid
