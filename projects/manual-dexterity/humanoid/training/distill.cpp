#include "humanoid/simulation.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <ATen/Context.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void stop(int) { interrupted = 1; }
constexpr int kObs = 47, kHistory = 8, kInput = kObs * kHistory, kAct = 12, kCapacity = 100000, kBatch = 256;
struct Options {
  int rounds = 5, steps = 4000, epochs = 10;
  double seconds = 600;
  std::uint64_t seed = 1;
  std::string device = "cuda", assets = "/opt/humanoid/unitree_rl_gym";
  fs::path output;
};
Options parse(int argc, char** argv) {
  Options o;
  if (const auto* root = std::getenv("HUMANOID_ASSET_ROOT")) o.assets = root;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help") {
      std::cout << "Expert-supervised action imitation / DAgger; not PPO or independent RL.\n"
        "--rounds 5 --steps-per-round 4000 --epochs 10 --max-seconds 600\n"
        "--output NEW_DIRECTORY --device cuda|cpu --seed 1 --asset-root PATH\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("Missing value for " + flag);
    const std::string value = argv[i];
    if (flag == "--rounds") o.rounds = std::stoi(value);
    else if (flag == "--steps-per-round") o.steps = std::stoi(value);
    else if (flag == "--epochs") o.epochs = std::stoi(value);
    else if (flag == "--max-seconds") o.seconds = std::stod(value);
    else if (flag == "--seed") o.seed = std::stoull(value);
    else if (flag == "--output") o.output = value;
    else if (flag == "--device") o.device = value;
    else if (flag == "--asset-root") o.assets = value;
    else throw std::invalid_argument("Unknown argument " + flag);
  }
  if (o.rounds < 1 || o.rounds > 20 || o.steps < 1 || o.steps > 100000 || o.epochs < 1 || o.epochs > 100 ||
      !std::isfinite(o.seconds) || o.seconds <= 0 || (o.device != "cpu" && o.device != "cuda"))
    throw std::invalid_argument("Invalid round, step, epoch, time limit, or device");
  if (o.output.empty()) o.output = fs::path("/workspace/artifacts/imitation") /
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
  o.output = fs::absolute(o.output);
  if (fs::exists(o.output)) throw std::invalid_argument("Output directory must be new");
  return o;
}
std::string sha256(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot hash " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("SHA init failed");
  std::array<char, 65536> buffer{};
  while (file.read(buffer.data(), buffer.size()) || file.gcount())
    if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(file.gcount())) != 1)
      throw std::runtime_error("SHA update failed");
  if (!file.eof()) throw std::runtime_error("SHA file read failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) throw std::runtime_error("SHA final failed");
  std::ostringstream result;
  for (unsigned i = 0; i < length; ++i) result << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return result.str();
}
void save_json(const fs::path& path, const Json& value) {
  std::ofstream out(path); out << value.dump(2) << '\n';
  if (!out) throw std::runtime_error("Cannot write " + path.string());
}
torch::Tensor as_tensor(const std::vector<float>& v, int width, const torch::Device& device) {
  return torch::from_blob(const_cast<float*>(v.data()), {static_cast<std::int64_t>(v.size() / width), width},
                          torch::kFloat32).clone().to(device);
}
std::vector<float> as_vector(const torch::Tensor& t, int expected) {
  auto host = t.detach().to(torch::kCPU).contiguous();
  if (host.numel() != expected || host.scalar_type() != torch::kFloat32 || !torch::isfinite(host).all().item<bool>())
    throw std::runtime_error("Invalid policy output");
  return {host.data_ptr<float>(), host.data_ptr<float>() + host.numel()};
}
struct History {
  std::vector<float> values = std::vector<float>(kInput, 0.f);
  void reset() { std::fill(values.begin(), values.end(), 0.f); }
  void append(const std::vector<float>& observation) {
    if (observation.size() != kObs || !std::all_of(observation.begin(), observation.end(), [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("Invalid current pre-action observation");
    std::move(values.begin() + kObs, values.end(), values.begin());
    std::copy(observation.begin(), observation.end(), values.end() - kObs);
  }
};
struct StudentImpl : torch::nn::Module {
  torch::nn::Linear a{nullptr}, b{nullptr}, c{nullptr};
  StudentImpl() {
    a = register_module("a", torch::nn::Linear(kInput, 256));
    b = register_module("b", torch::nn::Linear(256, 256));
    c = register_module("c", torch::nn::Linear(256, kAct));
    torch::NoGradGuard guard;
    torch::nn::init::orthogonal_(a->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(b->weight, std::sqrt(2.0));
    torch::nn::init::orthogonal_(c->weight, .01);
    for (auto* layer : {&a, &b, &c}) torch::nn::init::zeros_((*layer)->bias);
  }
  torch::Tensor forward(const torch::Tensor& observation) {
    return 3 * torch::tanh(c->forward(torch::tanh(b->forward(torch::tanh(a->forward(observation))))));
  }
  void export_policy(const fs::path& path) {
    torch::NoGradGuard guard;
    torch::jit::Module module("ExpertGuidedStudent");
    for (const auto& entry : named_parameters())
      module.register_parameter(entry.key()[0] + std::string(entry.key().find("weight") != std::string::npos ? "w" : "b"),
                                entry.value().detach().to(torch::kCPU).clone(), false);
    module.register_buffer("history", torch::zeros({1, kInput}, torch::kFloat32));
    module.define(R"JIT(
def forward(self, observation):
    assert observation.dim() == 2 and observation.size(0) == 1 and observation.size(1) == 47, "Expected one current observation with shape [1, 47]"
    self.history.copy_(torch.cat([self.history[:, 47:], observation], 1))
    x = torch.tanh(torch.matmul(self.history, self.aw.t()) + self.ab)
    x = torch.tanh(torch.matmul(x, self.bw.t()) + self.bb)
    return 3.0 * torch.tanh(torch.matmul(x, self.cw.t()) + self.cb)
)JIT");
    // Twelve nonzero frames exercise zero padding, ordering, and rollover beyond eight frames.
    const auto sequence = torch::linspace(-1.13, .97, 12 * kObs, torch::kFloat32).reshape({12, kObs});
    auto verify_sequence = [&](torch::jit::Module& candidate) {
      if (candidate.attr("history").toTensor().count_nonzero().item<int64_t>() != 0)
        throw std::runtime_error("Exported student must start with zero observation history");
      History history;
      for (int frame = 0; frame < 12; ++frame) {
        const auto observation = sequence.slice(0, frame, frame + 1);
        history.append(as_vector(observation, kObs));
        const auto reference = forward(as_tensor(history.values, kInput, a->weight.device())).to(torch::kCPU);
        const auto actual = candidate.forward({observation}).toTensor();
        if (!torch::allclose(reference, actual, 1e-4, 1e-5) ||
            !torch::equal(candidate.attr("history").toTensor(), as_tensor(history.values, kInput, torch::kCPU)))
          throw std::runtime_error("Student TorchScript causal-history sequence mismatch");
      }
    };
    verify_sequence(module);
    module.attr("history").toTensor().zero_();  // Never serialize the validation probe's state.
    module.save(path.string());
    auto restored = torch::jit::load(path.string());
    verify_sequence(restored);  // A fresh native episode must recover zero history from disk.
  }
};
TORCH_MODULE(Student);
struct Buffer {
  std::vector<float> observations, labels;
  std::uint64_t seen = 0;
  std::mt19937_64 generator;
  explicit Buffer(std::uint64_t seed) : generator(seed) {}
  void add(const std::vector<float>& obs, const std::vector<float>& target) {
    if (obs.size() != kInput || !std::all_of(obs.begin(), obs.end(), [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("Invalid pre-action observation history");
    const auto rows = observations.size() / kInput;
    ++seen;
    if (rows < kCapacity) {
      observations.insert(observations.end(), obs.begin(), obs.end()); labels.insert(labels.end(), target.begin(), target.end());
    } else {
      const auto slot = std::uniform_int_distribution<std::uint64_t>(0, seen - 1)(generator);
      if (slot < kCapacity) {
        std::copy(obs.begin(), obs.end(), observations.begin() + slot * kInput);
        std::copy(target.begin(), target.end(), labels.begin() + slot * kAct);
      }
    }
  }
};
Json screen(Student& student, const Options& o, const torch::Device& device, const fs::path& directory) {
  humanoid::WalkingSimulation sim(o.assets, "cpu", 64, 64, false);
  sim.reset(o.seed + 10001, .5, "external");
  History history;
  torch::NoGradGuard guard;
  for (int step = 0; step < 1000 && !sim.fallen() && !interrupted; ++step) {
    history.append(sim.observation());
    sim.step_action(as_vector(student->forward(as_tensor(history.values, kInput, device)), kAct));
  }
  auto result = sim.state();
  result["external_student_inference_device"] = o.device;
  result["policy_origin"] = "supervised imitation";
  result["automatic_resets"] = false;
  result["requested_seconds"] = 20;
  result["candidate_walking"] = !result["fall"].get<bool>() && result["time"].get<double>() >= 19.999 &&
      result["distance"].get<double>() >= 5 && result["left_steps"].get<int>() >= 5 && result["right_steps"].get<int>() >= 5 &&
      result["alternating_count"].get<int>() >= 8 && result["airborne_fraction"].get<double>() <= .1;
  save_json(directory / "evaluation.json", result);
  return result;
}
int run(const Options& o) {
  if (o.device == "cuda" && !torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable; no automatic CPU fallback");
  const torch::Device device(o.device == "cuda" ? torch::kCUDA : torch::kCPU);
  torch::manual_seed(o.seed); torch::set_num_threads(4);
  if (device.is_cuda()) { torch::cuda::manual_seed_all(o.seed); at::globalContext().setUserEnabledCuDNN(false); }
  const fs::path teacher_path = fs::path(o.assets) / "deploy/pre_train/g1/motion.pt";
  const auto teacher_hash = sha256(teacher_path);
  if (teacher_hash != "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d")
    throw std::runtime_error("Teacher differs from the pinned official G1 checkpoint");
  fs::create_directories(o.output);
  humanoid::WalkingSimulation sim(o.assets, "cpu", 64, 64, false);
  Json report = {{"status", "collecting"}, {"method", "supervised expert-action imitation with DAgger data aggregation"},
    {"reinforcement_learning", false}, {"expert_assistance", true}, {"teacher_parameters_copied", false},
    {"student_initialization", "random weights"}, {"teacher_path", teacher_path.string()}, {"teacher_sha256", teacher_hash},
    {"libtorch", TORCH_VERSION}, {"seed", o.seed}, {"device", o.device}, {"physics_device", "cpu"},
    {"rounds_requested", o.rounds}, {"steps_per_round", o.steps}, {"epochs", o.epochs}, {"max_training_seconds", o.seconds},
    {"student", "376-256tanh-256tanh-12; output 3*tanh; nominal positions +0.25*action radians"},
    {"observation_history_frames", kHistory}, {"observation_history_seconds", .16},
    {"observation_history_order", "oldest to current; zero padding on reset; append current 47 observations before action"},
    {"teacher_observations", kObs}, {"student_training_inputs", kInput}, {"exported_policy_inputs", kObs},
    {"export_validation", "12-frame nonzero causal sequence including rollover; zero history before save and on fresh reload"},
    {"loss", "mean squared error against raw teacher actions; no clipped labels"}, {"adam_learning_rate", .001},
    {"buffer_capacity", kCapacity}, {"walking_proven", false}, {"rounds", Json::array()}, {"model_provenance", sim.provenance()}};
  report["model_provenance"]["policy_training"] = "Published weights provide supervised action labels; student parameters are learned locally";
  if (fs::exists(__FILE__)) report["source_sha256"] = sha256(__FILE__);
  if (fs::exists("/proc/self/exe")) report["executable_sha256"] = sha256("/proc/self/exe");
  save_json(o.output / "run.json", report);
  Student student; student->to(device);
  torch::optim::Adam optimizer(student->parameters(), torch::optim::AdamOptions(.001));
  Buffer buffer(o.seed);
  std::ofstream progress(o.output / "progress.csv");
  progress << "round,teacher_blend,seen,buffer_rows,collection_falls,epochs_completed,updates,mse,seconds,eval_seconds,distance,alternations,candidate\n";
  const auto started = Clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
  auto expired = [&] { return interrupted || elapsed() >= o.seconds; };
  std::uint64_t reset = 0, outside = 0, label_values = 0, updates = 0;
  double label_max = 0, best_score = -1e30;
  bool best_pass = false;
  for (int round = 0; round < o.rounds && !expired(); ++round) {
    const double beta = o.rounds == 1 ? 1. : 1. - double(round) / (o.rounds - 1);
    std::unique_ptr<torch::jit::Module> teacher;
    History history;
    int episode_steps = 0, falls = 0;
    auto reset_episode = [&] {
      sim.reset(o.seed + ++reset, .5, "external");
      history.reset();
      teacher = std::make_unique<torch::jit::Module>(torch::jit::load(teacher_path.string(), device));
      teacher->eval(); episode_steps = 0;
    };
    reset_episode();
    for (int step = 0; step < o.steps && !expired(); ++step) {
      torch::NoGradGuard guard;
      const auto observation = sim.observation();  // BEFORE teacher forward or applying an action.
      history.append(observation);
      const auto input = as_tensor(observation, kObs, device);
      const auto label = as_vector(teacher->forward({input}).toTensor(), kAct);
      buffer.add(history.values, label);  // Only o_{t-7}..o_t; the current expert label has not been applied.
      const auto prediction = as_vector(student->forward(as_tensor(history.values, kInput, device)), kAct);
      std::vector<float> applied(kAct);
      for (int joint = 0; joint < kAct; ++joint) {
        applied[joint] = static_cast<float>(beta * label[joint] + (1 - beta) * prediction[joint]);
        label_max = std::max(label_max, std::abs(double(label[joint])));
        outside += std::abs(label[joint]) > 3; ++label_values;
      }
      sim.step_action(applied);
      if (sim.fallen() || ++episode_steps >= 1000) { falls += sim.fallen(); reset_episode(); }
      if ((step + 1) % 1000 == 0) std::cout << "round=" << round + 1 << " collected=" << step + 1 << " beta=" << beta << std::endl;
    }
    if (buffer.observations.empty()) break;
    const int rows = static_cast<int>(buffer.observations.size() / kInput);
    const auto x = as_tensor(buffer.observations, kInput, device), y = as_tensor(buffer.labels, kAct, device);
    double loss_sum = 0; int round_updates = 0, completed_epochs = 0;
    for (int epoch = 0; epoch < o.epochs && !expired(); ++epoch) {
      const auto order = torch::randperm(rows, torch::TensorOptions().dtype(torch::kInt64).device(device));
      for (int begin = 0; begin < rows && !expired(); begin += kBatch) {
        const auto indices = order.slice(0, begin, std::min(begin + kBatch, rows));
        const auto loss = (student->forward(x.index_select(0, indices)) - y.index_select(0, indices)).square().mean();
        if (!torch::isfinite(loss).item<bool>()) throw std::runtime_error("Nonfinite imitation loss");
        optimizer.zero_grad(); loss.backward();
        if (!std::isfinite(torch::nn::utils::clip_grad_norm_(student->parameters(), 1.0))) throw std::runtime_error("Nonfinite imitation gradient");
        optimizer.step(); loss_sum += loss.item<double>(); ++round_updates; ++updates;
      }
      if (!expired()) ++completed_epochs;
    }
    const auto directory = o.output / ("round-" + std::to_string(round + 1)); fs::create_directories(directory);
    torch::save(student, (directory / "model.pt").string()); student->export_policy(directory / "policy.pt");
    auto evaluation = screen(student, o, device, directory);
    const bool pass = evaluation["candidate_walking"].get<bool>();
    const double score = evaluation["time"].get<double>() + std::clamp(evaluation["distance"].get<double>(), 0., 10.) +
                         .2 * std::min(evaluation["alternating_count"].get<int>(), 40);
    if ((pass && !best_pass) || (pass == best_pass && score > best_score)) {
      best_pass = pass; best_score = score;
      fs::copy_file(directory / "policy.pt", o.output / "policy.pt", fs::copy_options::overwrite_existing);
      report["selected_round"] = round + 1; report["selected_evaluation"] = evaluation;
    }
    Json result = {{"round", round + 1}, {"teacher_blend", beta}, {"seen", buffer.seen}, {"buffer_rows", rows},
      {"collection_falls", falls}, {"completed_epochs", completed_epochs}, {"optimizer_updates", round_updates},
      {"mean_minibatch_mse", round_updates ? Json(loss_sum / round_updates) : Json(nullptr)},
      {"evaluation", evaluation}, {"policy_sha256", sha256(directory / "policy.pt")}, {"model_sha256", sha256(directory / "model.pt")}};
    report["rounds"].push_back(result); report["updates"] = updates; report["seconds"] = elapsed();
    report["teacher_label_max_abs"] = label_max;
    report["teacher_labels_outside_student_range_fraction"] = label_values ? double(outside) / label_values : 0;
    save_json(o.output / "run.json", report);
    progress << round + 1 << ',' << beta << ',' << buffer.seen << ',' << rows << ',' << falls << ',' << completed_epochs << ','
      << round_updates << ',' << (round_updates ? loss_sum / round_updates : 0) << ',' << elapsed() << ',' << evaluation["time"] << ','
      << evaluation["distance"] << ',' << evaluation["alternating_count"] << ',' << pass << '\n'; progress.flush();
    std::cout << result.dump() << std::endl;
  }
  if (!buffer.observations.empty()) torch::save(std::vector<torch::Tensor>{as_tensor(buffer.observations, kInput, torch::kCPU),
      as_tensor(buffer.labels, kAct, torch::kCPU)}, (o.output / "dataset.pt").string());
  report["status"] = interrupted ? "interrupted" : elapsed() >= o.seconds ? "time_limit" : "completed";
  report["seconds"] = elapsed(); report["candidate_walking"] = best_pass;
  report["evaluation_scope"] = "One fixed reset seed per round used for selection; requires independent multi-seed shared evaluation before a walking claim";
  if (fs::exists(o.output / "policy.pt")) report["selected_policy_sha256"] = sha256(o.output / "policy.pt");
  save_json(o.output / "run.json", report); std::cout << report.dump(2) << std::endl;
  return interrupted ? 130 : elapsed() >= o.seconds ? 124 : 0;
}
}  // namespace
int main(int argc, char** argv) {
  fs::path output;
  try { std::signal(SIGINT, stop); std::signal(SIGTERM, stop); const auto options = parse(argc, argv); output = options.output; return run(options); }
  catch (const std::exception& error) {
    std::cerr << "Imitation training failed: " << error.what() << '\n';
    if (!output.empty() && fs::is_directory(output)) {
      try { save_json(output / "failure.json", {{"status", "failed"}, {"error", error.what()}, {"method", "supervised expert imitation"}}); } catch (...) {}
      try {
        Json report = Json::object();
        const auto report_path = output / "run.json";
        if (fs::exists(report_path)) {
          std::ifstream previous(report_path);
          previous >> report;
          if (!report.is_object()) throw std::runtime_error("Existing run report is not a JSON object");
        }
        report["status"] = "failed";
        report["error"] = error.what();
        save_json(report_path, report);  // Preserve prior round results and provenance.
      } catch (const std::exception& report_error) {
        std::cerr << "Could not update failed run status: " << report_error.what() << '\n';
      }
    }
    return 1;
  }
}
