#include "humanoid/simulation.hpp"
#include "policy.hpp"
#include "ppo_math.hpp"

#include <torch/cuda.h>
#include <torch/version.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using humanoid::WalkingSimulation;
using humanoid::ppo::ActorCritic;
using humanoid::ppo::kActions;
using humanoid::ppo::kObservations;

namespace {
volatile std::sig_atomic_t interrupted = 0;
void request_stop(int) { interrupted = 1; }

struct Options {
  std::int64_t steps = 200000;
  int environments = 16, horizon = 128, epochs = 4, minibatch = 256;
  std::uint64_t seed = 1;
  double max_seconds = 1800, evaluation_seconds = 20;
  std::string device = "cuda", assets, evaluate_only;
  fs::path output;
};

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Native MuJoCo/LibTorch PPO. Starts with random weights unless --eval-only is used.\n"
                   "  --steps N           requested transitions, rounded to complete rollouts (200000)\n"
                   "  --envs N            independent CPU MuJoCo simulations (16; maximum 64)\n"
                   "  --seed N            network, sampling and reset seed (1)\n"
                   "  --output DIRECTORY  new artifact directory\n"
                   "  --assets DIRECTORY  shared Unitree asset root\n"
                   "  --device cuda|cpu   batched neural-network device (cuda; no automatic fallback)\n"
                   "  --max-seconds N     training wall-clock cap checked between rollouts (1800)\n"
                   "  --eval-seconds N    continuous evaluation duration, 5..60 (20)\n"
                   "  --eval-only FILE    restore our model.pt and evaluate without updates\n";
      std::exit(0);
    }
    if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--steps") options.steps = std::stoll(value);
    else if (key == "--envs") options.environments = std::stoi(value);
    else if (key == "--seed") options.seed = std::stoull(value);
    else if (key == "--output") options.output = value;
    else if (key == "--assets") options.assets = value;
    else if (key == "--device") options.device = value;
    else if (key == "--max-seconds") options.max_seconds = std::stod(value);
    else if (key == "--eval-seconds") options.evaluation_seconds = std::stod(value);
    else if (key == "--eval-only") options.evaluate_only = value;
    else throw std::invalid_argument("Unknown option " + key);
  }
  if (options.steps <= 0 || options.environments < 1 || options.environments > 64 ||
      !std::isfinite(options.max_seconds) || options.max_seconds <= 0 ||
      !std::isfinite(options.evaluation_seconds) || options.evaluation_seconds < 5 || options.evaluation_seconds > 60) {
    throw std::invalid_argument("Invalid steps, environment count, or duration");
  }
  if (options.device != "cuda" && options.device != "cpu") throw std::invalid_argument("--device must be cuda or cpu");
  if (options.output.empty()) {
    const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    options.output = fs::path("/workspace/artifacts/training") /
                     (std::to_string(timestamp) + "-seed" + std::to_string(options.seed));
  }
  options.output = fs::absolute(options.output);
  if (fs::exists(options.output)) throw std::invalid_argument("Output must be a new directory: " + options.output.string());
  return options;
}

void write_json(const fs::path& path, const Json& value) {
  std::ofstream stream(path);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  stream << value.dump(2) << '\n';
}

torch::Tensor tensor(const std::vector<float>& data, std::vector<std::int64_t> shape,
                     const torch::Device& device) {
  return torch::from_blob(const_cast<float*>(data.data()), shape, torch::kFloat32).clone().to(device);
}

std::vector<float> vector(const torch::Tensor& tensor) {
  auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  const auto* first = cpu.data_ptr<float>();
  return {first, first + cpu.numel()};
}

bool finite_observation(const std::vector<float>& observation) {
  return observation.size() == kObservations &&
         std::all_of(observation.begin(), observation.end(), [](float value) { return std::isfinite(value); });
}

void copy_observation(const std::vector<float>& source, std::vector<float>& destination, std::size_t offset) {
  if (!finite_observation(source)) throw std::runtime_error("Simulation returned a nonfinite or incorrectly shaped observation");
  std::copy(source.begin(), source.end(), destination.begin() + static_cast<std::ptrdiff_t>(offset));
}

struct RewardState {
  std::vector<float> previous_action = std::vector<float>(kActions, 0);
  int elapsed = 0, last_support = -1, last_switch = 0;
  double episode_return = 0;
};

float shaped_reward(const humanoid::TrainingMetrics& metrics, const std::vector<float>& action,
                    RewardState& state) {
  double magnitude = 0, slew = 0;
  for (int i = 0; i < kActions; ++i) {
    magnitude += action[i] * action[i];
    const auto difference = action[i] - state.previous_action[i];
    slew += difference * difference;
  }
  magnitude /= kActions;
  slew /= kActions;
  const double track = std::exp(-std::pow(metrics.vx - .5, 2) / .25);
  double cadence = 0;
  if (metrics.left_contact != metrics.right_contact) {
    const int support = metrics.left_contact ? 0 : 1;
    if (state.last_support >= 0 && support != state.last_support && state.elapsed - state.last_switch >= 8) {
      cadence = .2;
      state.last_switch = state.elapsed;
    }
    state.last_support = support;
  }
  state.previous_action = action;
  // The action terms regularize target magnitude and slew; they are not torque or energy measurements.
  return static_cast<float>(.5 + 1.5 * track + .5 * std::clamp(metrics.upright, 0.0, 1.0)
                            + cadence - .2 * metrics.vy * metrics.vy - .02 * magnitude - .05 * slew);
}

Json evaluate(ActorCritic& network, const Options& options, const torch::Device& device,
              const std::string& label, std::uint64_t seed) {
  WalkingSimulation simulation(options.assets, "cpu", 64, 64, false);
  simulation.reset(seed, .5, "external");
  const auto start = simulation.training_metrics();
  auto previous = start;
  const double control_dt = simulation.control_dt();
  const int target_steps = static_cast<int>(std::round(options.evaluation_seconds / control_dt));
  std::ofstream trajectory(options.output / (label + "-trajectory.csv"));
  trajectory << "time,x,y,z,vx,vy,upright,left_contact,right_contact,fallen\n";
  int frames = 0, upright_frames = 0, left_touchdowns = 0, right_touchdowns = 0;
  bool fallen = false;
  double path_length = 0, min_height = start.z;
  torch::NoGradGuard guard;
  network->eval();
  for (int i = 0; i < target_steps && !interrupted; ++i) {
    const auto observation = simulation.observation();
    if (!finite_observation(observation)) throw std::runtime_error("Nonfinite evaluation observation");
    const auto action = vector(torch::tanh(network->actor(tensor(observation, {1, kObservations}, device))));
    simulation.step_action(action);
    const auto current = simulation.training_metrics();
    path_length += std::hypot(current.x - previous.x, current.y - previous.y);
    min_height = std::min(min_height, current.z);
    upright_frames += current.upright >= .8 && current.z >= .55;
    left_touchdowns += current.left_contact && !previous.left_contact;
    right_touchdowns += current.right_contact && !previous.right_contact;
    ++frames;
    trajectory << frames * control_dt << ',' << current.x << ',' << current.y << ',' << current.z << ','
               << current.vx << ',' << current.vy << ',' << current.upright << ',' << current.left_contact
               << ',' << current.right_contact << ',' << current.fallen << '\n';
    previous = current;
    if (current.fallen) { fallen = true; break; }
  }
  const double duration = frames * control_dt;
  const double displacement = std::hypot(previous.x - start.x, previous.y - start.y);
  const double forward_displacement = previous.x - start.x;
  const double upright_fraction = frames ? static_cast<double>(upright_frames) / frames : 0;
  Json result = {{"label", label}, {"seed", seed}, {"deterministic", true}, {"automatic_resets", false},
      {"requested_seconds", options.evaluation_seconds}, {"seconds", duration}, {"fallen", fallen},
      {"net_displacement_m", displacement}, {"forward_displacement_m", forward_displacement},
      {"path_length_m", path_length}, {"mean_path_speed_m_s", duration ? path_length / duration : 0},
      {"min_base_height_m", min_height}, {"upright_fraction", upright_fraction},
      {"left_touchdowns", left_touchdowns}, {"right_touchdowns", right_touchdowns}};
  result["applied_action_diagnostics"] = simulation.state().at("applied_action_diagnostics");
  result["candidate_walking"] = duration >= 20.0 && !fallen && forward_displacement >= 3 &&
                                  upright_fraction >= .9 && left_touchdowns >= 3 && right_touchdowns >= 3;
  result["interpretation"] = "An engineering screen; inspect the live policy for stepping rather than sliding or hopping.";
  write_json(options.output / (label + "-evaluation.json"), result);
  network->train();
  return result;
}

fs::path save_checkpoint(ActorCritic& network, torch::optim::Adam& optimizer,
                         const Options& options, std::int64_t steps) {
  const auto directory = options.output / "checkpoints" / std::to_string(steps);
  if (!fs::exists(directory)) {
    fs::create_directories(directory);
    torch::save(network, (directory / "model.pt").string());
    torch::serialize::OutputArchive archive;
    optimizer.save(archive);
    archive.save_to((directory / "optimizer.pt").string());
    write_json(directory / "metadata.json", {{"steps", steps}, {"seed", options.seed},
        {"obs_size", kObservations}, {"action_size", kActions}, {"libtorch", TORCH_VERSION},
        {"initialization", "random; no pretrained weights"}});
  }
  return directory;
}

int run(const Options& options) {
  if (options.device == "cuda" && !torch::cuda::is_available()) {
    throw std::runtime_error("CUDA requested but unavailable; verify container GPU access and LibTorch libraries");
  }
  const torch::Device device(options.device == "cuda" ? torch::kCUDA : torch::kCPU);
  torch::manual_seed(options.seed);
  if (device.is_cuda()) torch::cuda::manual_seed_all(options.seed);
  torch::set_num_threads(4);
  fs::create_directories(options.output);
  Json report = {{"status", "initializing"}, {"algorithm", "native C++ tanh-Gaussian PPO-Clip + GAE"},
      {"model", "Unitree G1; 47 observations, 12 position-target actions"}, {"libtorch", TORCH_VERSION},
      {"device", options.device}, {"physics_device", "cpu"}, {"seed", options.seed},
      {"requested_steps", options.steps}, {"environments", options.environments},
      {"horizon", options.horizon}, {"epochs", options.epochs}, {"minibatch", options.minibatch},
      {"max_training_seconds", options.max_seconds}, {"gamma", .99}, {"gae_lambda", .95},
      {"learning_rate", .0003}, {"clip_epsilon", .2}, {"target_kl", .03},
      {"adam_epsilon", 1e-5}, {"max_gradient_norm", .5}, {"gaussian_entropy_coefficient", .005},
      {"action_transform", "tanh"}, {"normalized_action_limit", 1.0},
      {"joint_position_offset_scale_rad", .25}, {"control_period_seconds", .02},
      {"observation_normalization", "fixed shared-simulator scales; no running normalization"},
      {"walking_proven", false}, {"initialization", options.evaluate_only.empty() ? "random; no pretrained weights" : "explicit evaluation checkpoint"}};
  const auto report_path = options.output / "run.json";
  write_json(report_path, report);
  ActorCritic network;
  network->to(device);
  torch::optim::Adam optimizer(network->parameters(), torch::optim::AdamOptions(.0003).eps(1e-5));
  if (!options.evaluate_only.empty()) {
    torch::load(network, options.evaluate_only, device);
    report["evaluation"] = evaluate(network, options, device, "restored", options.seed + 10001);
    network->export_policy(options.output / "policy.pt");
    report["status"] = "evaluated";
    write_json(report_path, report);
    return 0;
  }
  std::cout << "Native PPO on " << options.device << "; MuJoCo physics on CPU; artifacts " << options.output << std::endl;
  report["initial_evaluation"] = evaluate(network, options, device, "initial", options.seed + 10001);
  save_checkpoint(network, optimizer, options, 0);
  const auto initial_parameters = [&]() {
    std::vector<torch::Tensor> pieces;
    for (const auto& parameter : network->parameters()) pieces.push_back(parameter.detach().flatten());
    return torch::cat(pieces).clone();
  }();

  const int n = options.environments, horizon = options.horizon, count = n * horizon;
  std::vector<std::unique_ptr<WalkingSimulation>> simulations;
  std::vector<RewardState> episode(n);
  std::vector<std::uint64_t> reset_counts(n, 0);
  std::vector<float> observations(n * kObservations);
  for (int env = 0; env < n; ++env) {
    simulations.push_back(std::make_unique<WalkingSimulation>(options.assets, "cpu", 64, 64, false));
    simulations.back()->reset(options.seed + static_cast<std::uint64_t>(env) * 100003, .5, "external");
    copy_observation(simulations.back()->observation(), observations, env * kObservations);
  }
  std::ofstream metrics(options.output / "progress.csv");
  metrics << "steps,seconds,transitions_per_second,mean_step_reward,episodes,mean_episode_return,falls,timeouts,policy_loss,value_loss,approx_kl,updates\n";
  std::int64_t total_steps = 0;
  int iteration = 0, total_updates = 0;
  const auto started = Clock::now();
  report["status"] = "training";
  write_json(report_path, report);
  bool limited = false;
  while (total_steps < options.steps && !interrupted) {
    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    if (elapsed >= options.max_seconds) { limited = true; break; }
    std::vector<float> stored_obs(count * kObservations), latent(count * kActions), old_logp(count),
        rewards(count), values(count), next_values(count);
    std::vector<unsigned char> terminated(count), ended(count);
    std::vector<double> returns;
    int falls = 0, timeouts = 0;
    for (int time = 0; time < horizon; ++time) {
      torch::NoGradGuard guard;
      const int offset = time * n;
      std::copy(observations.begin(), observations.end(), stored_obs.begin() + offset * kObservations);
      const auto obs_tensor = tensor(observations, {n, kObservations}, device);
      const auto means = network->actor(obs_tensor);
      const auto latent_tensor = means + network->standard_deviation() * torch::randn_like(means);
      const auto actions = vector(torch::tanh(latent_tensor));
      const auto sampled_latent = vector(latent_tensor);
      const auto probabilities = vector(network->log_probability(obs_tensor, latent_tensor));
      const auto predictions = vector(network->value(obs_tensor));
      std::copy(sampled_latent.begin(), sampled_latent.end(), latent.begin() + offset * kActions);
      std::copy(probabilities.begin(), probabilities.end(), old_logp.begin() + offset);
      std::copy(predictions.begin(), predictions.end(), values.begin() + offset);
      std::vector<float> next_observations(n * kObservations);
      for (int env = 0; env < n; ++env) {
        const int index = offset + env;
        const std::vector<float> action(actions.begin() + env * kActions, actions.begin() + (env + 1) * kActions);
        simulations[env]->step_action(action);
        const auto state = simulations[env]->training_metrics();
        auto after = simulations[env]->observation();
        const bool valid = finite_observation(after) && std::isfinite(state.vx) && std::isfinite(state.vy) && std::isfinite(state.upright);
        const bool terminal = state.fallen || !valid;
        const bool timeout = ++episode[env].elapsed >= 1000 && !terminal;
        rewards[index] = terminal ? -5.0f : shaped_reward(state, action, episode[env]);
        terminated[index] = terminal;
        ended[index] = terminal || timeout;
        episode[env].episode_return += rewards[index];
        if (!valid) after.assign(kObservations, 0);
        // This post-step observation is preserved for timeout bootstrapping before resetting.
        copy_observation(after, next_observations, env * kObservations);
        if (ended[index]) {
          returns.push_back(episode[env].episode_return);
          falls += terminal;
          timeouts += timeout;
          episode[env] = RewardState{};
        }
      }
      const auto next_prediction = vector(network->value(tensor(next_observations, {n, kObservations}, device)));
      std::copy(next_prediction.begin(), next_prediction.end(), next_values.begin() + offset);
      for (int env = 0; env < n; ++env) {
        if (ended[offset + env]) {
          ++reset_counts[env];
          simulations[env]->reset(options.seed + env * 100003ULL + reset_counts[env], .5, "external");
          copy_observation(simulations[env]->observation(), observations, env * kObservations);
        } else {
          std::copy_n(next_observations.begin() + env * kObservations, kObservations,
                      observations.begin() + env * kObservations);
        }
      }
    }

    auto advantages = humanoid::ppo::generalized_advantages(rewards, values, next_values, terminated, ended, horizon, n);
    std::vector<float> targets(count);
    for (int i = 0; i < count; ++i) targets[i] = advantages[i] + values[i];
    const auto observation_batch = tensor(stored_obs, {count, kObservations}, device);
    const auto latent_batch = tensor(latent, {count, kActions}, device);
    const auto probability_batch = tensor(old_logp, {count}, device);
    const auto target_batch = tensor(targets, {count}, device);
    auto advantage_batch = tensor(advantages, {count}, device);
    advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std(false) + 1e-8);
    double policy_loss_sum = 0, value_loss_sum = 0, kl_sum = 0;
    int updates = 0;
    bool stop_epoch = false;
    for (int epoch = 0; epoch < options.epochs && !stop_epoch; ++epoch) {
      const auto permutation = torch::randperm(count, torch::TensorOptions().dtype(torch::kInt64).device(device));
      for (int begin = 0; begin < count; begin += options.minibatch) {
        const auto indices = permutation.slice(0, begin, std::min(begin + options.minibatch, count));
        const auto obs = observation_batch.index_select(0, indices);
        const auto z = latent_batch.index_select(0, indices);
        const auto log_ratio = network->log_probability(obs, z) - probability_batch.index_select(0, indices);
        const auto ratio = log_ratio.exp();
        const auto advantage = advantage_batch.index_select(0, indices);
        const auto policy_loss = -torch::minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean();
        const auto value_loss = .5 * (network->value(obs) - target_batch.index_select(0, indices)).square().mean();
        const auto approximate_kl = (ratio - 1 - log_ratio).mean();
        const double kl = approximate_kl.detach().item<double>();
        if (!std::isfinite(kl)) throw std::runtime_error("Nonfinite PPO KL; preserving existing checkpoints");
        if (kl > .03) { stop_epoch = true; break; }
        const auto loss = policy_loss + .5 * value_loss - .005 * network->gaussian_entropy();
        if (!torch::isfinite(loss).item<bool>()) throw std::runtime_error("Nonfinite PPO loss; preserving existing checkpoints");
        optimizer.zero_grad();
        loss.backward();
        const double gradient_norm = torch::nn::utils::clip_grad_norm_(network->parameters(), .5);
        if (!std::isfinite(gradient_norm)) throw std::runtime_error("Nonfinite PPO gradients; preserving existing checkpoints");
        optimizer.step();
        policy_loss_sum += policy_loss.detach().item<double>();
        value_loss_sum += value_loss.detach().item<double>();
        kl_sum += kl;
        ++updates;
      }
    }
    total_updates += updates;
    total_steps += count;
    ++iteration;
    const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
    const double mean_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0) / count;
    const double mean_return = returns.empty() ? 0 : std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    const auto divisor = std::max(updates, 1);
    metrics << total_steps << ',' << seconds << ',' << total_steps / seconds << ',' << mean_reward << ','
            << returns.size() << ',' << mean_return << ',' << falls << ',' << timeouts << ','
            << policy_loss_sum / divisor << ',' << value_loss_sum / divisor << ',' << kl_sum / divisor << ',' << updates << '\n';
    metrics.flush();
    std::cout << "steps=" << total_steps << " reward=" << mean_reward << " falls=" << falls
              << " updates=" << updates << " sps=" << total_steps / seconds << std::endl;
    report["steps"] = total_steps;
    report["optimizer_updates"] = total_updates;
    report["training_seconds"] = seconds;
    if (iteration % 20 == 0) report["latest_checkpoint"] = save_checkpoint(network, optimizer, options, total_steps).string();
    write_json(report_path, report);
  }
  report["latest_checkpoint"] = save_checkpoint(network, optimizer, options, total_steps).string();
  network->export_policy(options.output / "policy.pt");
  std::vector<torch::Tensor> final_pieces;
  for (const auto& parameter : network->parameters()) final_pieces.push_back(parameter.detach().flatten());
  report["parameter_change_l2"] = (torch::cat(final_pieces) - initial_parameters).norm().item<double>();
  report["optimizer_updates"] = total_updates;
  report["steps"] = total_steps;
  report["status"] = interrupted ? "interrupted" : limited ? "time_limit" : "completed";
  report["policy"] = (options.output / "policy.pt").string();
  if (!interrupted) {
    report["final_evaluation"] = evaluate(network, options, device, "final", options.seed + 10001);
    report["heldout_evaluation"] = evaluate(network, options, device, "heldout", options.seed + 20001);
    report["candidate_walking"] = report["final_evaluation"]["candidate_walking"].get<bool>() &&
                                  report["heldout_evaluation"]["candidate_walking"].get<bool>();
  }
  write_json(report_path, report);
  std::cout << report.dump(2) << '\n';
  std::cout << "Saved " << options.output / "policy.pt" << "; inspect physical stepping before claiming walking.\n";
  return interrupted ? 130 : limited ? 124 : 0;
}
}  // namespace

int main(int argc, char** argv) {
  fs::path output;
  try {
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    const auto options = parse(argc, argv);
    output = options.output;
    return run(options);
  } catch (const std::exception& error) {
    std::cerr << "Training failed: " << error.what() << '\n';
    if (!output.empty() && fs::is_directory(output)) {
      try {
        write_json(output / "failure.json", {{"status", "failed"}, {"error", error.what()}, {"walking_proven", false}});
        std::ifstream previous(output / "run.json");
        if (previous) {
          Json record;
          previous >> record;
          record["status"] = "failed";
          record["error"] = error.what();
          write_json(output / "run.json", record);
        }
      }
      catch (...) {}
    }
    return 1;
  }
}
