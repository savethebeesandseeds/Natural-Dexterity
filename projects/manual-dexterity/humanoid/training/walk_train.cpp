#include "humanoid/simulation.hpp"
#include "walk_policy.hpp"
#include "ppo_math.hpp"
#include "swing_diagnostics.hpp"
#include <torch/cuda.h>
#include <torch/version.h>
#include <ATen/CPUGeneratorImpl.h>
#include <openssl/evp.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace humanoid::walk;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void stop(int) { interrupted = 1; }
struct Options {
  int envs = 64, workers = 8, horizon = 64, epochs = 5, minibatch = 512;
  int diagnostic_samples = 1, diagnostic_noise_hold = 1;
  int64_t steps = 1000000, checkpoint_every = 262144;
  uint64_t seed = 1, diagnostic_seed = 1729;
  bool seed_set = false, reward_version_set = false;
  double seconds = 1800, episode_seconds = 20, eval_seconds = 20;
  double command_min = .25, command_max = .75, learning_rate = .0003, entropy = .003;
  double diagnostic_noise_scale = 0;
  std::string device = "cuda", assets, resume, diagnose, reward_version = "v2";
  fs::path output;
};
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "From-scratch native history PPO; no expert weights, labels or motion templates.\n"
        "--steps 1000000 (ADDITIONAL transitions) --envs 64 --workers 8 --horizon 64\n"
        "--epochs 5 --minibatch 512 --seed 1 --device cuda|cpu --output NEW_DIRECTORY\n"
        "--max-seconds 1800 --episode-seconds 20 --eval-seconds 20 --checkpoint-every 262144\n"
        "--command-min .25 --command-max .75 --learning-rate .0003 --entropy .003\n"
        "--reward-version v2|v3|v4|v5|v6|v7|v8 (v2 default; inherited on resume unless explicitly set)\n"
        "--resume CHECKPOINT_DIRECTORY --diagnose CHECKPOINT_DIRECTORY --assets ASSET_ROOT\n"
        "Load-only diagnostics: --diagnostic-samples 1 --diagnostic-noise-scale 0\n"
        "--diagnostic-noise-hold 1 --diagnostic-seed 1729\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[i];
    if (key == "--steps") o.steps = std::stoll(value);
    else if (key == "--envs") o.envs = std::stoi(value);
    else if (key == "--workers") o.workers = std::stoi(value);
    else if (key == "--horizon") o.horizon = std::stoi(value);
    else if (key == "--epochs") o.epochs = std::stoi(value);
    else if (key == "--minibatch") o.minibatch = std::stoi(value);
    else if (key == "--seed") { o.seed = std::stoull(value); o.seed_set = true; }
    else if (key == "--device") o.device = value;
    else if (key == "--output") o.output = value;
    else if (key == "--assets") o.assets = value;
    else if (key == "--resume") o.resume = value;
    else if (key == "--diagnose") o.diagnose = value;
    else if (key == "--diagnostic-samples") o.diagnostic_samples = std::stoi(value);
    else if (key == "--diagnostic-noise-scale") o.diagnostic_noise_scale = std::stod(value);
    else if (key == "--diagnostic-noise-hold") o.diagnostic_noise_hold = std::stoi(value);
    else if (key == "--diagnostic-seed") {
      if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument("--diagnostic-seed requires an unsigned decimal integer");
      o.diagnostic_seed = std::stoull(value);
    }
    else if (key == "--reward-version") { o.reward_version = value; o.reward_version_set = true; }
    else if (key == "--max-seconds") o.seconds = std::stod(value);
    else if (key == "--episode-seconds") o.episode_seconds = std::stod(value);
    else if (key == "--eval-seconds") o.eval_seconds = std::stod(value);
    else if (key == "--checkpoint-every") o.checkpoint_every = std::stoll(value);
    else if (key == "--command-min") o.command_min = std::stod(value);
    else if (key == "--command-max") o.command_max = std::stod(value);
    else if (key == "--learning-rate") o.learning_rate = std::stod(value);
    else if (key == "--entropy") o.entropy = std::stod(value);
    else throw std::invalid_argument("Unknown option " + key);
  }
  if (o.steps < 1 || o.steps > 1000000000 || o.envs < 1 || o.envs > 512 || o.workers < 1 || o.workers > 24 ||
      o.horizon < 8 || o.horizon > 512 || o.epochs < 1 || o.epochs > 20 || o.minibatch < 32 ||
      o.checkpoint_every < 1 || o.seed > 100000 ||
      !std::isfinite(o.seconds) || o.seconds <= 0 || !std::isfinite(o.episode_seconds) || o.episode_seconds < 2 || o.episode_seconds > 60 ||
      !std::isfinite(o.eval_seconds) || o.eval_seconds < 0 || o.eval_seconds > 60 ||
      !std::isfinite(o.command_min) || !std::isfinite(o.command_max) || o.command_min < .25 || o.command_max > .75 || o.command_min > o.command_max ||
      !std::isfinite(o.learning_rate) || o.learning_rate < 1e-6 || o.learning_rate > .003 ||
      !std::isfinite(o.entropy) || o.entropy < 0 || o.entropy > .1 || (o.device != "cuda" && o.device != "cpu") ||
      (o.reward_version != "v2" && o.reward_version != "v3" && o.reward_version != "v4" && o.reward_version != "v5" && o.reward_version != "v6" && o.reward_version != "v7" && o.reward_version != "v8") || (!o.resume.empty() && !o.diagnose.empty()) ||
      (!o.diagnose.empty() && o.eval_seconds <= 0))
    throw std::invalid_argument("Invalid bounds or device");
  if (o.diagnostic_samples < 1 || o.diagnostic_samples > 5 || o.diagnostic_noise_hold < 1 || o.diagnostic_noise_hold > 8 ||
      !std::isfinite(o.diagnostic_noise_scale) || o.diagnostic_noise_scale < 0 || o.diagnostic_noise_scale > 3)
    throw std::invalid_argument("Invalid diagnostic sample count, noise scale or hold");
  if (o.diagnose.empty() && (o.diagnostic_samples != 1 || o.diagnostic_noise_hold != 1 || o.diagnostic_noise_scale != 0 || o.diagnostic_seed != 1729))
    throw std::invalid_argument("Nondefault diagnostic options require --diagnose; training is unchanged");
  if (o.output.empty()) o.output = fs::path("/workspace/artifacts/walk-ppo") / std::to_string(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  o.output = fs::absolute(o.output);
  if (fs::exists(o.output)) throw std::invalid_argument("Output directory must be new");
  return o;
}
void write_json(const fs::path& path, const Json& value) {
  std::ofstream stream(path); stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream << value.dump(2) << '\n';
}
Json read_json(const fs::path& path) {
  std::ifstream in(path); if (!in) throw std::runtime_error("Cannot read " + path.string());
  Json result; in >> result; return result;
}
std::string sha256(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot hash " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("SHA256 initialization failed");
  std::array<char, 65536> bytes{};
  while (input.read(bytes.data(), bytes.size()) || input.gcount())
    if (EVP_DigestUpdate(context.get(), bytes.data(), static_cast<size_t>(input.gcount())) != 1) throw std::runtime_error("SHA256 update failed");
  if (!input.eof()) throw std::runtime_error("SHA256 read failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{}; unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) throw std::runtime_error("SHA256 final failed");
  std::ostringstream out;
  for (unsigned i = 0; i < length; ++i) out << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return out.str();
}
// Each world has one owner per job; no rendering or shared MuJoCo data in workers.
class Pool {
  std::vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable ready, done;
  std::function<void(int)> task;
  std::exception_ptr failure;
  size_t generation = 0;
  int outstanding = 0, worlds;
  bool stopping = false;
 public:
  Pool(int count, int environments) : worlds(environments) {
    for (int worker = 0; worker < count; ++worker) threads.emplace_back([this, worker, count] {
      size_t observed = 0;
      for (;;) {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return stopping || generation != observed; });
        if (stopping) return;
        observed = generation; auto current = task; lock.unlock();
        try { for (int env = worker; env < worlds; env += count) current(env); }
        catch (...) { std::lock_guard<std::mutex> guard(mutex); if (!failure) failure = std::current_exception(); }
        lock.lock(); if (--outstanding == 0) done.notify_one();
      }
    });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> guard(mutex); stopping = true; }
    ready.notify_all(); for (auto& thread : threads) thread.join();
  }
  void run(std::function<void(int)> job) {
    std::unique_lock<std::mutex> lock(mutex);
    task = std::move(job); failure = nullptr; outstanding = static_cast<int>(threads.size()); ++generation;
    ready.notify_all(); done.wait(lock, [&] { return outstanding == 0; }); task = nullptr;
    if (failure) std::rethrow_exception(failure);
  }
};
struct Episode {
  History history;
  std::array<float, kActions> previous_action{};
  std::array<double, 2> air_time{};
  std::array<double, 2> peak_unsupported_sole_clearance{};  // v7/v8 only; zeroed with every episode/support interval.
  int steps = 0, last_touch = -1, last_touch_step = -1000;
  double reward = 0, command = .5;
};
struct Reward {
  float total = 0, tracking = 0, airborne = 0, slip = 0;
  int alternations = 0;
};
Json reward_recipe(const std::string& version) {
  const bool v3 = version == "v3" || version == "v4" || version == "v5" || version == "v6" || version == "v7" || version == "v8";
  const bool v4 = version == "v4" || version == "v5" || version == "v6" || version == "v7" || version == "v8";
  const bool upright_events = version == "v6" || version == "v7" || version == "v8";
  const bool clearance_events = version == "v7" || version == "v8";
  Json recipe = {{"version", version}, {"control_dt", .02}, {"survival_rate", v3 ? 1. : .1}, {"fall_penalty", version == "v5" ? 10. : v3 ? 5. : 1.},
    {"forward_lateral_frame", v3 ? "world" : "pelvis"}, {"progress_upright_gate", v3}, {"motion_contact_upright_gate", v3},
    {"tracking_rate", 4.}, {"tracking_sigma", "0.15 +0.15*command"}, {"stationary_tracking_baseline_subtracted", true},
    {"tracking_upright_gate", true}, {"progress_rate", .5}, {"single_support_rate", .25}, {"no_support_cost_rate", .5},
    {"single_support_multiplier", v3 ? "clamp(world_vx/command,0,1)*upright" : "clamp(body_vx/command,0,1)"},
    {"airtime_event_multiplier", upright_events ? "upright" : v3 ? "clamp(world_vx/command,0,1)*upright" : "clamp(body_vx/command,0,1)"},
    {"alternation_event_multiplier", upright_events ? "upright" : v3 ? "clamp(world_vx/command,0,1)*upright" : "clamp(body_vx/command,0,1)"},
    {"events_require_forward_speed", !upright_events}, {"clearance_reward_gate", clearance_events},
    {"touchdown_airtime_multiplier", clearance_events ? 3.5 : .35}, {"airtime_threshold_seconds", .06}, {"airtime_reward_offset_seconds", .04},
    {"airtime_reward_cap_seconds", .3}, {"alternation_bonus", clearance_events ? 1.0 : .1}, {"alternation_interval_seconds", .16},
    {"lateral_cost_rate", .5}, {"pelvis_vertical_cost_rate", .5}, {"roll_pitch_cost_rate", .05}, {"yaw_cost_rate", .25},
    {"tilt_cost_rate", 1.}, {"slip_squared_cost_rate", v4 ? 4. : .4}, {"slip_squared_cap", 25.},
    {"slip_squared_normalization", v4 ? "divide raw sum by command_x squared before applying cap" : "none; cap raw sum"},
    {"slip_cost_rate_formula", v4 ? "4.0*min(sum_support_contact_speed_squared/(command_x*command_x),25.0)" : "0.4*min(sum_support_contact_speed_squared,25.0)"},
    {"slip_telemetry", "mean_slip_squared and progress.csv slip remain raw summed support contact speed squared, in m^2/s^2"},
    {"actual_torque_squared_cost_rate", .00001}, {"absolute_power_cost_rate", .0001}, {"action_slew_cost_rate", .015}};
  if (clearance_events) {
    recipe["touchdown_peak_sole_clearance_threshold_m"] = .01;
    recipe["touchdown_clearance_definition"] = "Peak sampled minimum sole clearance during the force-unsupported interval, observed at50Hz; reset on support. Both airtime>=0.06s and peak>=0.01m are required to recognize a " + version + " reward touchdown.";
  }
  if (version == "v8") {
    recipe["airtime_requires_qualifying_alternation"] = true;
    recipe["airtime_credit_definition"] = "Computed v7 airtime bonus is credited only when this interval has alternate==1: one qualifying opposite-foot touchdown with the existing minimum0.16s separation. First-foot, same-foot and simultaneous touchdowns receive no airtime bonus.";
  }
  return recipe;
}
Reward reward(const humanoid::TrainingMetrics& m, const std::vector<float>& action, Episode& e, int version) {
  constexpr double dt = .02;
  const bool v3 = version >= 3;
  const double target = e.command, sigma = .15 + .15 * target;
  const double forward = v3 ? m.vx : m.body_vx, lateral = v3 ? m.vy : m.body_vy;
  // Subtract the stationary tracking score so standing cannot earn command tracking reward.
  const double tracking = std::exp(-std::pow((forward - target) / sigma, 2)) - std::exp(-std::pow(target / sigma, 2));
  const double upright = std::clamp(m.upright, 0., 1.);
  const double motion_gate = v3 ? upright : 1.;
  const double moving = std::clamp(forward / target, 0., 1.) * motion_gate;
  const double event_multiplier = version == 6 || version == 7 || version == 8 ? upright : moving;
  double slew = 0;
  for (int j = 0; j < kActions; ++j) { slew += std::pow(action[j] - e.previous_action[j], 2); e.previous_action[j] = action[j]; }
  slew /= kActions;
  const std::array<bool, 2> contact{m.left_support, m.right_support};
  std::array<bool, 2> touchdown{};
  double air_bonus = 0;
  for (int foot = 0; foot < 2; ++foot) {
    if (contact[foot]) {
      touchdown[foot] = e.air_time[foot] >= .06;
      if (version >= 7) touchdown[foot] = touchdown[foot] && e.peak_unsupported_sole_clearance[foot] >= .01;
      if (touchdown[foot]) air_bonus += (version >= 7 ? 3.5 : .35) * std::clamp(e.air_time[foot] - .04, 0., .3) * event_multiplier;
      e.air_time[foot] = 0;
      if (version >= 7) e.peak_unsupported_sole_clearance[foot] = 0;
    } else {
      e.air_time[foot] += dt;
      if (version >= 7) e.peak_unsupported_sole_clearance[foot] = std::max(e.peak_unsupported_sole_clearance[foot], foot == 0 ? m.left_foot_height : m.right_foot_height);
    }
  }
  int alternate = 0;
  if (touchdown[0] != touchdown[1]) {
    const int foot = touchdown[0] ? 0 : 1;
    if (e.last_touch >= 0 && foot != e.last_touch && e.steps - e.last_touch_step >= 8) alternate = 1;
    e.last_touch = foot; e.last_touch_step = e.steps;
  }
  if (version == 8 && alternate == 0) air_bonus = 0;
  const double slip = (m.left_support ? m.left_contact_slip_speed * m.left_contact_slip_speed : 0.) +
                      (m.right_support ? m.right_contact_slip_speed * m.right_contact_slip_speed : 0.);
  const double slip_cost = version >= 4 ? 4. * std::min(slip / (target * target), 25.) : .4 * std::min(slip, 25.);
  const double rate = 4. * tracking * upright + .5 * std::clamp(forward / target, -1., 1.) * motion_gate + (v3 ? 1. : .1) * upright
    - .5 * lateral * lateral - .5 * m.body_vz * m.body_vz
    - .05 * (m.body_wx * m.body_wx + m.body_wy * m.body_wy) - .25 * m.body_wz * m.body_wz
    - 1. * (1. - upright) * (1. - upright) - slip_cost
    - .00001 * m.applied_torque_sq_mean - .0001 * m.mechanical_power_abs - .015 * slew
    + .25 * (contact[0] != contact[1]) * moving - .5 * (!contact[0] && !contact[1]);
  const double result = dt * rate + air_bonus + (version >= 7 ? 1.0 : .1) * alternate * event_multiplier - (m.fallen ? (version == 5 ? 10. : v3 ? 5. : 1.) : 0.);
  if (!std::isfinite(result)) throw std::runtime_error("Nonfinite physical reward");
  return {static_cast<float>(result), static_cast<float>(tracking), static_cast<float>(air_bonus), static_cast<float>(slip), alternate};
}
std::vector<float> critic_input(const History& history, const humanoid::TrainingMetrics& m) {
  auto result = history.values;
  const std::array<double, kPrivileged> extra{m.body_vx, m.body_vy, m.body_vz, .25*m.body_wx, .25*m.body_wy, .25*m.body_wz,
    m.z, m.upright, 5*m.left_foot_height, 5*m.right_foot_height, double(m.left_support), double(m.right_support),
    m.left_contact_slip_speed, m.right_contact_slip_speed, m.command_vx, .01*std::sqrt(std::max(0., m.applied_torque_sq_mean))};
  for (double value : extra) {
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite privileged observation");
    result.push_back(static_cast<float>(value));
  }
  return result;
}
Json screen(ActorCritic& model, const Options& o, const torch::Device& device, Json* simulator_provenance = nullptr) {
  Json results = Json::array();
  if (o.eval_seconds <= 0) return results;
  torch::NoGradGuard guard;
  const bool diagnostic = !o.diagnose.empty();
  const bool stochastic = diagnostic && o.diagnostic_noise_scale > 0;
  const int samples = diagnostic ? o.diagnostic_samples : 1;
  const int target_steps = static_cast<int>(std::round(o.eval_seconds / .02));
  torch::Tensor playback_std;
  std::vector<float> model_std, effective_std;
  if (diagnostic) {
    playback_std = model->stddev();
    model_std = vector(playback_std);
    effective_std = vector(o.diagnostic_noise_scale * playback_std);
  }
  int command_index = 0;
  for (double command : {.25, .5, .75}) {
   for (int trial_index = 0; trial_index < samples; ++trial_index) {
    // Unsigned wrap is deliberate and recorded; scale/hold do not affect pairing.
    const uint64_t noise_seed = o.diagnostic_seed + uint64_t(command_index) * 1000003ULL + uint64_t(trial_index) * 10007ULL;
    std::vector<float> noise_rows;
    if (stochastic) {
      std::mt19937_64 local_rng(noise_seed);
      std::normal_distribution<float> standard_normal(0.f, 1.f);
      noise_rows.resize(static_cast<size_t>(target_steps) * kActions);
      for (auto& value : noise_rows) value = standard_normal(local_rng);
    }
    torch::Tensor held_noise;
    std::array<double, kActions> action_delta_squared{};
    uint64_t near_bound_values = 0;
    humanoid::WalkingSimulation sim(o.assets, "cpu", 64, 64, false);
    sim.reset(8000000000ULL + o.seed, command, "external");
    SwingDiagnostics swing_observer;
    const auto initial = sim.training_metrics();
    swing_observer.reset(0., {initial.left_support, initial.right_support}, {initial.left_foot_height, initial.right_foot_height});
    if (simulator_provenance && simulator_provenance->empty()) {
      *simulator_provenance = sim.provenance();
      (*simulator_provenance)["policy_training"] = "External actor restored from this project's random-initialized PPO checkpoint; no expert weights or labels";
      (*simulator_provenance)["external_actor_inference_device"] = o.device;
      (*simulator_provenance)["simulator_policy_loaded"] = false;
    }
    History history;
    std::array<Episode, 7> reward_states;
    for (auto& state : reward_states) state.command = command;
    std::array<double, 7> discounted{}, undiscounted{};
    double discount = 1.;
    int intervals = 0;
    const int late_start_step = static_cast<int>(std::round(5. / sim.control_dt()));
    bool reached_five_seconds = false;
    double late_start_x = 0, late_final_x = 0, late_speed_error_squared = 0, late_slip_squared = 0;
    int late_intervals = 0, late_support_observations = 0;
    for (int step = 0; step < target_steps && !sim.fallen() && !interrupted; ++step) {
      history.append(sim.observation());
      std::vector<float> action;
      if (stochastic) {
        if (step % o.diagnostic_noise_hold == 0) {
          const std::vector<float> innovation(noise_rows.begin() + step*kActions, noise_rows.begin() + (step+1)*kActions);
          held_noise = tensor(innovation, kActions, device);
        }
        const auto mean = model->actor(tensor(history.values, kHistory, device));
        action = vector(kActionLimit * torch::tanh(mean + o.diagnostic_noise_scale * playback_std * held_noise));
        const auto mean_action = vector(kActionLimit * torch::tanh(mean));
        for (int joint = 0; joint < kActions; ++joint) action_delta_squared[joint] += std::pow(action[joint] - mean_action[joint], 2);
      } else {
        // Preserve the previous deterministic expression, with no added tensor/noise arithmetic.
        action = vector(kActionLimit * torch::tanh(model->actor(tensor(history.values, kHistory, device))));
      }
      if (diagnostic) for (float value : action) near_bound_values += std::abs(value) >= .95*kActionLimit;
      sim.step_action(action);
      const auto metrics = sim.training_metrics();
      swing_observer.observe((step + 1) * sim.control_dt(), {metrics.left_support, metrics.right_support},
        {metrics.left_foot_height, metrics.right_foot_height});
      if (diagnostic && step + 1 == late_start_step) {
        reached_five_seconds = true; late_start_x = metrics.x;
      }
      if (diagnostic && step + 1 > late_start_step) {
        ++late_intervals; late_final_x = metrics.x;
        late_speed_error_squared += std::pow(metrics.vx - command, 2);
        if (metrics.left_support) {
          late_slip_squared += std::pow(metrics.left_contact_slip_speed, 2); ++late_support_observations;
        }
        if (metrics.right_support) {
          late_slip_squared += std::pow(metrics.right_contact_slip_speed, 2); ++late_support_observations;
        }
      }
      for (int version = 0; version < 7; ++version) {
        auto& state = reward_states[version]; ++state.steps;
        const double observed_reward = reward(metrics, action, state, version + 2).total;
        undiscounted[version] += observed_reward;
        discounted[version] += discount * observed_reward;
      }
      discount *= .99; ++intervals;
    }
    auto result = sim.state(); result["policy_origin"] = "from-scratch PPO";
    result["swing_diagnostics"] = swing_observer.report();
    result["external_actor_inference_device"] = o.device; result["automatic_resets"] = false;
    if (diagnostic) {
      std::array<double, kActions> action_rms{}, target_rms{};
      for (int joint = 0; joint < kActions; ++joint) {
        action_rms[joint] = intervals ? std::sqrt(action_delta_squared[joint] / intervals) : 0.;
        target_rms[joint] = .25 * action_rms[joint];
      }
      const double combined_action_rms = intervals ? std::sqrt(std::accumulate(action_delta_squared.begin(), action_delta_squared.end(), 0.) / (intervals*kActions)) : 0.;
      result["diagnostic_playback"] = {{"method", stochastic ? "stochastic diagnostic playback" : "deterministic mean playback"},
        {"new_trained_policy", false}, {"learned_gait_claim", false}, {"noise_scale", o.diagnostic_noise_scale},
        {"model_latent_stddev", model_std}, {"effective_latent_noise_stddev", effective_std},
        {"noise_hold_control_steps", o.diagnostic_noise_hold}, {"noise_hold_seconds", o.diagnostic_noise_hold * sim.control_dt()},
        {"control_hz", 1. / sim.control_dt()}, {"base_noise_seed", o.diagnostic_seed}, {"derived_noise_seed", noise_seed},
        {"command_index", command_index}, {"trial_index", trial_index}, {"trial", trial_index + 1}, {"samples_per_command", samples},
        {"physical_reset_seed", 8000000000ULL + o.seed}, {"noise_generator", "local std::mt19937_64 + std::normal_distribution<float>"},
        {"noise_seed_formula", "base +1000003*command_index +10007*trial_index modulo2^64; zero-based indices; independent of scale/hold"},
        {"noise_schedule", "Pre-generated control-step rows; hold boundary step t uses row t, other rows discarded; shared boundary innovations across hold settings"},
        {"action_formula", "3*tanh(mean_at_current_50Hz_observation + noise_scale*model_stddev*held_standard_normal)"},
        {"noise_rows_generated", stochastic ? target_steps : 0},
        {"noise_refreshes_applied", stochastic ? (intervals + o.diagnostic_noise_hold - 1) / o.diagnostic_noise_hold : 0},
        {"same_state_action_delta_rms_per_joint", action_rms}, {"same_state_target_delta_rms_rad_per_joint", target_rms},
        {"same_state_action_delta_rms_combined", combined_action_rms}, {"same_state_target_delta_rms_rad_combined", .25 * combined_action_rms},
        {"action_delta_definition", "Actual post-tanh noisy action minus mean action evaluated from the same current history; RMS over observed intervals, combined also averages12 joints; target delta=0.25*action delta radians"},
        {"near_action_bound_threshold", .95*kActionLimit},
        {"near_action_bound_fraction", intervals ? Json(double(near_bound_values)/(intervals*kActions)) : Json(nullptr)},
        {"interpretation", "Injected exploration diagnostic of a frozen model; motion/lifts do not establish a new trained controller or learned walking gait"}};
      result["after_five_seconds"] = {{"reached_five_seconds", reached_five_seconds}, {"control_intervals", late_intervals},
        {"alive_after_5_exposure_seconds", late_intervals * sim.control_dt()},
        {"distance_x_m", late_intervals ? Json(late_final_x - late_start_x) : Json(nullptr)},
        {"mean_forward_speed_m_s", late_intervals ? Json((late_final_x - late_start_x) / (late_intervals * sim.control_dt())) : Json(nullptr)},
        {"speed_rmse_m_s", late_intervals ? Json(std::sqrt(late_speed_error_squared / late_intervals)) : Json(nullptr)},
        {"mean_raw_slip_squared_m2_s2", late_intervals ? Json(late_slip_squared / late_intervals) : Json(nullptr)},
        {"supporting_foot_speed_rms_m_s", late_support_observations ? Json(std::sqrt(late_slip_squared / late_support_observations)) : Json(nullptr)},
        {"supporting_foot_endpoint_observations", late_support_observations}, {"fell_in_window", late_intervals > 0 && sim.fallen()},
        {"definition", "Available control intervals after nominal5s, ending at fall/stop; includes a terminal-fall interval and assumes no exposure afterward. Distance is world x change from5s; speed RMSE uses50Hz endpoint world vx minus command. Raw slip is summed squared force-support foot speeds per interval; supporting-foot RMS instead divides by supported-foot endpoint count. No observations means null statistics."}};
    }
    result["counterfactual_returns"] = {{"v2", {{"discounted", discounted[0]}, {"undiscounted", undiscounted[0]}}},
      {"v3", {{"discounted", discounted[1]}, {"undiscounted", undiscounted[1]}}},
      {"v4", {{"discounted", discounted[2]}, {"undiscounted", undiscounted[2]}}},
      {"v5", {{"discounted", discounted[3]}, {"undiscounted", undiscounted[3]}}},
      {"v6", {{"discounted", discounted[4]}, {"undiscounted", undiscounted[4]}}},
      {"v7", {{"discounted", discounted[5]}, {"undiscounted", undiscounted[5]}}},
      {"v8", {{"discounted", discounted[6]}, {"undiscounted", undiscounted[6]}}}, {"gamma", .99}, {"control_intervals", intervals},
      {"bootstrap_included", false}, {"scope", "Same actual actions and physical states; separate reward histories; no control feedback; finite rollout sums"}};
    result["scope"] = stochastic ? "stochastic diagnostic playback of frozen checkpoint; not a trained policy or walking gate; reserved goal seeds excluded" : "selection screen; reserved goal seeds excluded";
    results.push_back(result);
   }
   ++command_index;
  }
  return results;
}
int run(Options o) {
  if (o.device == "cuda" && !torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable; no automatic fallback");
  const torch::Device device(o.device == "cuda" ? torch::kCUDA : torch::kCPU);
  const bool diagnosing = !o.diagnose.empty();
  const fs::path input_checkpoint = diagnosing ? o.diagnose : o.resume;
  Json parent = Json::object();
  std::string parent_recipe;
  if (!input_checkpoint.empty()) {
    parent = read_json(input_checkpoint / "metadata.json");
    if (parent.at("schema").get<int>() != 1 || parent.at("initialization") != "random; no pretrained weights" ||
        parent.at("expert_assistance").get<bool>() || parent.at("teacher_parameters_copied").get<bool>() || (!diagnosing && parent.at("envs") != o.envs))
      throw std::runtime_error("Checkpoint provenance/schema/environment-count mismatch");
    const auto original_seed = parent.at("seed").get<uint64_t>();
    if (original_seed > 100000 || (o.seed_set && original_seed != o.seed)) throw std::runtime_error("Checkpoint must preserve original training seed");
    o.seed = original_seed;
    parent_recipe = parent.value("reward_version", "v2");  // Existing frozen v2 checkpoints predate this field.
    if (parent_recipe != "v2" && parent_recipe != "v3" && parent_recipe != "v4" && parent_recipe != "v5" && parent_recipe != "v6" && parent_recipe != "v7" && parent_recipe != "v8") throw std::runtime_error("Unknown parent reward version");
    if (!o.reward_version_set) o.reward_version = parent_recipe;
    if (sha256(input_checkpoint / "model.pt") != parent.at("model_sha256") ||
        sha256(input_checkpoint / "policy.pt") != parent.at("policy_sha256")) throw std::runtime_error("Checkpoint model/policy hash mismatch");
    if (!diagnosing && (sha256(input_checkpoint / "optimizer.pt") != parent.at("optimizer_sha256") ||
        sha256(input_checkpoint / "rng.pt") != parent.at("rng_sha256"))) throw std::runtime_error("Resume optimizer/RNG hash mismatch");
  }
  torch::set_num_threads(1); torch::manual_seed(o.seed);
  const int recipe_number = o.reward_version == "v2" ? 2 : o.reward_version == "v3" ? 3 : o.reward_version == "v4" ? 4 : o.reward_version == "v5" ? 5 : o.reward_version == "v6" ? 6 : o.reward_version == "v7" ? 7 : 8;
  fs::create_directories(o.output / "source");
  Json report = {{"schema", 1}, {"status", "initializing"}, {"algorithm", "native history PPO-Clip with GAE and privileged critic"},
    {"seed", o.seed}, {"initialization", "random; no pretrained weights"}, {"expert_assistance", false}, {"teacher_parameters_copied", false},
    {"teacher_labels", false}, {"scripted_joint_trajectories", false}, {"external_support", false}, {"walking_proven", false},
    {"device", o.device}, {"physics_device", "cpu"}, {"libtorch", TORCH_VERSION}, {"envs", o.envs}, {"workers", o.workers},
    {"horizon", o.horizon}, {"epochs", o.epochs}, {"minibatch", o.minibatch}, {"requested_additional_steps", o.steps},
    {"max_seconds", o.seconds}, {"episode_seconds", o.episode_seconds}, {"command_min", o.command_min}, {"command_max", o.command_max},
    {"actor", "376-256ELU-128ELU-12; 8 causal frames; 3*tanh action"}, {"critic", "392-256ELU-128ELU-1; history plus16 physical features"},
    {"history_frames", kFrames}, {"public_observations", kObs}, {"action_limit", kActionLimit}, {"joint_offset_scale_rad", .25},
    {"normalization", "fixed simulator observation scales, input clamp +-10, per-rollout standardized advantages"},
    {"gamma", .99}, {"gae_lambda", .95}, {"clip", .2}, {"entropy_proxy_coefficient", o.entropy},
    {"reward", "physical reward rates times0.02s plus debounced contact events; versioned recipe below"},
    {"reward_version", o.reward_version}, {"reward_recipe", reward_recipe(o.reward_version)},
    {"reward_version_selection", o.reward_version_set ? "explicit CLI" : parent.empty() ? "default v2" : "inherited from checkpoint (legacy absence means v2)"},
    {"parent_reward_version", parent.empty() ? Json(nullptr) : Json(parent_recipe)},
    {"reward_version_changed_from_parent", !parent.empty() && parent_recipe != o.reward_version},
    {"reserved_goal_reset_seeds", "9000000000..9000000004 excluded"}, {"compile_time", std::string(__DATE__) + " " + __TIME__}};
  for (const auto& source : {fs::path(__FILE__), fs::path(__FILE__).parent_path() / "walk_policy.hpp",
      fs::path(__FILE__).parent_path() / "ppo_math.hpp", fs::path(__FILE__).parent_path() / "swing_diagnostics.hpp"}) {
    if (!fs::exists(source)) throw std::runtime_error("Source snapshot missing " + source.string());
    fs::copy_file(source, o.output / "source" / source.filename());
    report["source_files"][source.filename().string()] = sha256(source);
  }
  report["source_sha256"] = sha256(__FILE__);
  report["source_on_disk_sha256"] = report["source_sha256"];
  report["source_binding"] = "Source files snapshotted at startup; executable hashed separately; build/launch coordinator verifies source unchanged";
  report["executable_sha256"] = sha256("/proc/self/exe");
  write_json(o.output / "run.json", report);
  ActorCritic model; model->to(device);
  if (diagnosing) {
    // This branch precedes optimizer construction, the environment pool and all rollout allocation.
    torch::load(model, (input_checkpoint / "model.pt").string(), device); model->eval();
    report["mode"] = "checkpoint_reward_diagnostic"; report["status"] = "diagnosing";
    report["training_performed"] = false; report["optimizer_updates"] = 0; report["additional_steps"] = 0;
    report["envs"] = 1; report["workers"] = 0; report["requested_additional_steps"] = 0;
    report["completed_steps"] = parent.at("completed_steps"); report["policy_sha256"] = parent.at("policy_sha256");
    report["completed_steps_scope"] = "Inherited source checkpoint count; this diagnostic performs no training transitions";
    report["diagnostic_configuration"] = {{"method", o.diagnostic_noise_scale > 0 ? "stochastic diagnostic playback" : "deterministic mean playback"},
      {"samples_per_command", o.diagnostic_samples}, {"noise_scale", o.diagnostic_noise_scale}, {"noise_hold_control_steps", o.diagnostic_noise_hold},
      {"base_noise_seed", o.diagnostic_seed}, {"physical_reset_seed", 8000000000ULL + o.seed}, {"eval_seconds", o.eval_seconds},
      {"noise_seed_formula", "base +1000003*zero_based_command_index +10007*zero_based_trial_index modulo2^64; independent of scale/hold"},
      {"global_rng_used_for_diagnostic_noise", false}, {"checkpoint_parameters_modified", false},
      {"new_trained_policy", false}, {"learned_gait_claim", false}};
    report["checkpoint"] = {{"path", fs::absolute(input_checkpoint).string()}, {"metadata_sha256", sha256(input_checkpoint / "metadata.json")},
      {"model_sha256", parent.at("model_sha256")}, {"policy_sha256", parent.at("policy_sha256")},
      {"seed", o.seed}, {"completed_steps", parent.at("completed_steps")}, {"optimizer_updates", parent.at("optimizer_updates")},
      {"reward_version", parent_recipe}, {"source_sha256", parent.at("source_sha256")}};
    report["counterfactual_recipes"] = {{"v2", reward_recipe("v2")}, {"v3", reward_recipe("v3")},
      {"v4", reward_recipe("v4")}, {"v5", reward_recipe("v5")}, {"v6", reward_recipe("v6")}, {"v7", reward_recipe("v7")}, {"v8", reward_recipe("v8")}};
    write_json(o.output / "run.json", report);
    const auto started = Clock::now();
    Json simulator_provenance;
    report["screens"] = screen(model, o, device, &simulator_provenance);
    report["model_provenance"] = simulator_provenance;
    report["model_provenance"]["external_actor_checkpoint_path"] = fs::absolute(input_checkpoint / "model.pt").string();
    report["model_provenance"]["external_actor_checkpoint_sha256"] = parent.at("model_sha256");
    write_json(o.output / "screen.json", report["screens"]);
    report["seconds"] = std::chrono::duration<double>(Clock::now() - started).count();
    report["status"] = interrupted ? "interrupted" : "diagnosed";
    write_json(o.output / "run.json", report); std::cout << report.dump(2) << std::endl;
    return interrupted ? 130 : 0;
  }
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(o.learning_rate).eps(1e-5));
  int64_t completed = 0, updates = 0; double prior_seconds = 0, learning_rate = o.learning_rate;
  std::vector<uint64_t> reset_counts(o.envs, 0);
  std::vector<std::mt19937_64> generators;
  for (int env = 0; env < o.envs; ++env) generators.emplace_back(o.seed * 100003ULL + env);
  if (!o.resume.empty()) {
    torch::load(model, (fs::path(o.resume) / "model.pt").string(), device);
    torch::serialize::InputArchive archive; archive.load_from((fs::path(o.resume) / "optimizer.pt").string(), device); optimizer.load(archive);
    torch::Tensor rng; torch::load(rng, (fs::path(o.resume) / "rng.pt").string());
    auto generator = at::detail::getDefaultCPUGenerator(); generator.set_state(rng);
    completed = parent.at("completed_steps").get<int64_t>(); updates = parent.at("optimizer_updates").get<int64_t>();
    learning_rate = parent.at("learning_rate").get<double>(); prior_seconds = parent.at("cumulative_training_seconds").get<double>();
    reset_counts = parent.at("reset_counts").get<std::vector<uint64_t>>();
    if (reset_counts.size() != static_cast<size_t>(o.envs) || parent.at("command_rng").size() != static_cast<size_t>(o.envs))
      throw std::runtime_error("Resume environment state count mismatch");
    for (int env = 0; env < o.envs; ++env) { std::istringstream state(parent.at("command_rng").at(env).get<std::string>()); state >> generators[env]; if (!state) throw std::runtime_error("Invalid command RNG"); }
    report["resume_parent"] = {{"path", fs::absolute(o.resume).string()}, {"metadata_sha256", sha256(fs::path(o.resume) / "metadata.json")},
      {"completed_steps", completed}, {"model_sha256", parent.at("model_sha256")}, {"source_sha256", parent.at("source_sha256")},
      {"reward_version", parent_recipe}, {"reward_version_changed", parent_recipe != o.reward_version}};
    report["resume_semantics"] = "model/Adam/CPU random stream/accounting restored; physics episodes restart with fresh reset counters, not bitwise trajectory continuation";
  }
  const int64_t starting_steps = completed, target_steps = completed + o.steps;
  std::vector<std::unique_ptr<humanoid::WalkingSimulation>> worlds;
  std::vector<Episode> episodes(o.envs);
  for (int env = 0; env < o.envs; ++env) worlds.push_back(std::make_unique<humanoid::WalkingSimulation>(o.assets, "cpu", 64, 64, false));
  report["model_provenance"] = worlds.front()->provenance();
  report["model_provenance"]["policy_training"] = "No published policy loaded; random initialization and PPO reward updates only";
  Pool pool(std::min(o.workers, o.envs), o.envs);
  std::vector<float> observations(o.envs * kHistory), privileged(o.envs * kCritic);
  auto reset_world = [&](int env) {
    auto& episode = episodes[env]; episode = Episode{};
    episode.command = std::uniform_real_distribution<double>(o.command_min, o.command_max)(generators[env]);
    ++reset_counts[env];
    // All training reset seeds remain below the separately reserved 9e9 range.
    const uint64_t reset_seed = (o.seed * 1000003ULL + env * 100003ULL + reset_counts[env]) % 7000000000ULL;
    worlds[env]->reset(reset_seed, episode.command, "external");
    episode.history.append(worlds[env]->observation());
    std::copy(episode.history.values.begin(), episode.history.values.end(), observations.begin() + env * kHistory);
    const auto extra = critic_input(episode.history, worlds[env]->training_metrics());
    std::copy(extra.begin(), extra.end(), privileged.begin() + env * kCritic);
  };
  pool.run(reset_world);
  const auto initial_parameters = [&] { std::vector<torch::Tensor> parts; for (const auto& p : model->parameters()) parts.push_back(p.detach().flatten()); return torch::cat(parts).clone(); }();
  const auto started = Clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
  report["completed_steps"] = completed; report["optimizer_updates"] = updates;
  report["initial_screen"] = screen(model, o, device);
  std::ofstream progress(o.output / "progress.csv");
  progress << "completed_steps,additional_steps,seconds,sps,reward,tracking,body_vx,slip,alternations,falls,timeouts,episode_return,policy_loss,value_loss,kl,learning_rate,stddev,updates\n";
  auto checkpoint = [&]() {
    const auto directory = o.output / "checkpoints" / std::to_string(completed);
    if (fs::exists(directory)) return directory;
    fs::create_directories(directory);
    torch::save(model, (directory / "model.pt").string());
    torch::serialize::OutputArchive archive; optimizer.save(archive); archive.save_to((directory / "optimizer.pt").string());
    torch::save(at::detail::getDefaultCPUGenerator().get_state(), (directory / "rng.pt").string());
    model->export_policy(directory / "policy.pt");
    Json meta = {{"schema", 1}, {"seed", o.seed}, {"initialization", "random; no pretrained weights"}, {"expert_assistance", false},
      {"teacher_parameters_copied", false}, {"envs", o.envs}, {"completed_steps", completed}, {"optimizer_updates", updates},
      {"reward_version", o.reward_version}, {"reward_recipe", reward_recipe(o.reward_version)},
      {"parent_reward_version", parent.empty() ? Json(nullptr) : Json(parent_recipe)},
      {"reward_version_changed_from_parent", !parent.empty() && parent_recipe != o.reward_version},
      {"learning_rate", learning_rate}, {"cumulative_training_seconds", prior_seconds + elapsed()}, {"reset_counts", reset_counts},
      {"command_rng", Json::array()}, {"source_sha256", report.at("source_sha256")}, {"executable_sha256", report.at("executable_sha256")}};
    for (const auto& generator : generators) { std::ostringstream state; state << generator; meta["command_rng"].push_back(state.str()); }
    for (const auto& name : {"model", "optimizer", "rng", "policy"}) meta[std::string(name) + "_sha256"] = sha256(directory / (std::string(name) + ".pt"));
    write_json(directory / "metadata.json", meta);
    fs::copy_file(directory / "policy.pt", o.output / "policy.pt", fs::copy_options::overwrite_existing);
    report["policy_sha256"] = meta["policy_sha256"]; report["latest_checkpoint"] = directory.string();
    write_json(directory / "screen.json", screen(model, o, device));
    return directory;
  };
  checkpoint();
  int64_t next_checkpoint = completed + o.checkpoint_every;
  report["status"] = "training"; write_json(o.output / "run.json", report);
  const int n = o.envs, h = o.horizon, count = n * h;
  while (completed < target_steps && !interrupted && elapsed() < o.seconds) {
    std::vector<float> obs(count * kHistory), critic(count * kCritic), latent(count * kActions), logp(count), values(count), rewards(count), next_values(count);
    std::vector<unsigned char> terminated(count), ended(count);
    int falls = 0, timeouts = 0, alternate_count = 0, finished = 0;
    double episode_returns = 0, tracking_sum = 0, vx_sum = 0, slip_sum = 0;
    for (int t = 0; t < h; ++t) {
      torch::NoGradGuard guard;
      const int offset = t * n;
      std::copy(observations.begin(), observations.end(), obs.begin() + offset * kHistory);
      std::copy(privileged.begin(), privileged.end(), critic.begin() + offset * kCritic);
      const auto input = tensor(observations, kHistory, device);
      const auto means = model->actor(input);
      // All random draws use the checkpointed CPU generator, including CUDA runs.
      const auto z = means + model->stddev() * torch::randn({n, kActions}, torch::kFloat32).to(device);
      const auto action = vector(kActionLimit * torch::tanh(z));
      const auto zhost = vector(z), probabilities = vector(model->log_probability(input, z));
      const auto predictions = vector(model->value(tensor(privileged, kCritic, device)));
      std::copy(zhost.begin(), zhost.end(), latent.begin() + offset * kActions);
      std::copy(probabilities.begin(), probabilities.end(), logp.begin() + offset);
      std::copy(predictions.begin(), predictions.end(), values.begin() + offset);
      std::vector<Reward> terms(n);
      std::vector<double> velocities(n);
      pool.run([&](int env) {
        const int index = offset + env;
        const std::vector<float> applied(action.begin() + env*kActions, action.begin() + (env+1)*kActions);
        auto& episode = episodes[env];
        worlds[env]->step_action(applied);
        const auto metrics = worlds[env]->training_metrics();
        ++episode.steps; terms[env] = reward(metrics, applied, episode, recipe_number); rewards[index] = terms[env].total;
        episode.reward += rewards[index]; velocities[env] = metrics.body_vx;
        terminated[index] = metrics.fallen;
        ended[index] = metrics.fallen || episode.steps >= static_cast<int>(std::round(o.episode_seconds / .02));
        episode.history.append(worlds[env]->observation());
        std::copy(episode.history.values.begin(), episode.history.values.end(), observations.begin() + env*kHistory);
        const auto after = critic_input(episode.history, metrics);
        std::copy(after.begin(), after.end(), privileged.begin() + env*kCritic);
      });
      // Preserve the actual post-step history and privileged state for timeout bootstrapping.
      const auto next = vector(model->value(tensor(privileged, kCritic, device)));
      std::copy(next.begin(), next.end(), next_values.begin() + offset);
      for (int env = 0; env < n; ++env) {
        tracking_sum += terms[env].tracking; slip_sum += terms[env].slip; alternate_count += terms[env].alternations; vx_sum += velocities[env];
        if (ended[offset+env]) { ++finished; episode_returns += episodes[env].reward; falls += terminated[offset+env]; timeouts += !terminated[offset+env]; }
      }
      pool.run([&](int env) { if (ended[offset+env]) reset_world(env); });
    }
    const auto advantage = humanoid::ppo::generalized_advantages(rewards, values, next_values, terminated, ended, h, n);
    std::vector<float> targets(count);
    for (int i = 0; i < count; ++i) targets[i] = values[i] + advantage[i];
    const auto xb = tensor(obs, kHistory, device), cb = tensor(critic, kCritic, device), zb = tensor(latent, kActions, device);
    const auto pb = tensor(logp, 1, device).flatten(), tb = tensor(targets, 1, device).flatten(), vb = tensor(values, 1, device).flatten();
    auto ab = tensor(advantage, 1, device).flatten(); ab = (ab - ab.mean()) / (ab.std(false) + 1e-8);
    double policy_sum = 0, value_sum = 0, kl_sum = 0; int iteration_updates = 0; bool excessive_kl = false;
    for (int epoch = 0; epoch < o.epochs && !excessive_kl; ++epoch) {
      const auto order = torch::randperm(count, torch::TensorOptions().dtype(torch::kInt64)).to(device);
      for (int begin = 0; begin < count; begin += o.minibatch) {
        const auto indices = order.slice(0, begin, std::min(begin + o.minibatch, count));
        const auto ratio_log = model->log_probability(xb.index_select(0, indices), zb.index_select(0, indices)) - pb.index_select(0, indices);
        const auto ratio = ratio_log.exp(), adv = ab.index_select(0, indices);
        const auto pl = -torch::minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean();
        const auto value = model->value(cb.index_select(0, indices));
        const auto old_value = vb.index_select(0, indices), target = tb.index_select(0, indices);
        const auto vl = .5 * torch::maximum((value-target).square(), (old_value + (value-old_value).clamp(-.2,.2)-target).square()).mean();
        const double kl = (ratio - 1. - ratio_log).mean().item<double>();
        if (!std::isfinite(kl)) throw std::runtime_error("Nonfinite PPO KL");
        if (kl > .06) { excessive_kl = true; break; }
        const auto loss = pl + vl - o.entropy * model->entropy_proxy();
        if (!torch::isfinite(loss).item<bool>()) throw std::runtime_error("Nonfinite PPO loss");
        optimizer.zero_grad(); loss.backward();
        if (!std::isfinite(torch::nn::utils::clip_grad_norm_(model->parameters(), 1.))) throw std::runtime_error("Nonfinite PPO gradient");
        optimizer.step(); policy_sum += pl.item<double>(); value_sum += vl.item<double>(); kl_sum += kl; ++iteration_updates; ++updates;
      }
    }
    const double mean_kl = kl_sum / std::max(iteration_updates, 1);
    if (excessive_kl || mean_kl > .02) learning_rate = std::max(1e-5, learning_rate / 1.5);
    else if (mean_kl < .005) learning_rate = std::min(.001, learning_rate * 1.5);
    for (auto& group : optimizer.param_groups()) static_cast<torch::optim::AdamOptions&>(group.options()).lr(learning_rate);
    completed += count;
    const double mean_reward = std::accumulate(rewards.begin(), rewards.end(), 0.) / count;
    const double seconds = elapsed(), sps = (completed-starting_steps) / seconds;
    const double stddev = model->stddev().mean().item<double>();
    progress << completed << ',' << completed-starting_steps << ',' << seconds << ',' << sps << ',' << mean_reward << ','
      << tracking_sum/count << ',' << vx_sum/count << ',' << slip_sum/count << ',' << alternate_count << ',' << falls << ',' << timeouts << ','
      << episode_returns/std::max(finished,1) << ',' << policy_sum/std::max(iteration_updates,1) << ',' << value_sum/std::max(iteration_updates,1) << ','
      << mean_kl << ',' << learning_rate << ',' << stddev << ',' << iteration_updates << '\n'; progress.flush();
    report["completed_steps"] = completed; report["additional_steps"] = completed-starting_steps; report["optimizer_updates"] = updates;
    report["cumulative_training_seconds"] = prior_seconds+seconds; report["seconds"] = seconds; report["learning_rate"] = learning_rate;
    report["latest_rollout"] = {{"mean_reward",mean_reward},{"mean_tracking",tracking_sum/count},{"mean_body_vx",vx_sum/count},
      {"falls",falls},{"timeouts",timeouts},{"alternations",alternate_count},{"mean_slip_squared",slip_sum/count},
      {"mean_slip_squared_is_raw",true},{"mean_slip_squared_units","m^2/s^2"},{"stddev",stddev},{"kl",mean_kl}};
    std::cout << "steps=" << completed << " reward=" << mean_reward << " vx=" << vx_sum/count << " falls=" << falls
      << " timeouts=" << timeouts << " alternations=" << alternate_count << " sps=" << sps << std::endl;
    if (completed >= next_checkpoint) { checkpoint(); next_checkpoint = completed + o.checkpoint_every; }
    write_json(o.output / "run.json", report);
  }
  checkpoint();
  std::vector<torch::Tensor> pieces; for (const auto& p : model->parameters()) pieces.push_back(p.detach().flatten());
  report["parameter_change_l2_this_invocation"] = (torch::cat(pieces)-initial_parameters).norm().item<double>();
  report["completed_steps"] = completed; report["additional_steps"] = completed-starting_steps; report["optimizer_updates"] = updates;
  report["seconds"] = elapsed(); report["cumulative_training_seconds"] = prior_seconds+elapsed();
  report["status"] = interrupted ? "interrupted" : completed >= target_steps ? "completed" : "time_limit";
  report["policy_sha256"] = sha256(o.output / "policy.pt");
  report["final_screen"] = screen(model, o, device);
  write_json(o.output / "run.json", report);
  std::cout << report.dump(2) << std::endl;
  return interrupted ? 130 : completed >= target_steps ? 0 : 124;
}
}  // namespace
int main(int argc, char** argv) {
  fs::path output;
  try { std::signal(SIGINT, stop); std::signal(SIGTERM, stop); auto o = parse(argc,argv); output=o.output; return run(o); }
  catch (const std::exception& error) {
    std::cerr << "Walk PPO failed: " << error.what() << '\n';
    if (!output.empty() && fs::is_directory(output)) {
      try {
        write_json(output/"failure.json", {{"status","failed"},{"error",error.what()}});
        Json report = fs::exists(output/"run.json") ? read_json(output/"run.json") : Json::object();
        report["status"]="failed"; report["error"]=error.what(); write_json(output/"run.json",report);
      } catch (...) {}
    }
    return 1;
  }
}
