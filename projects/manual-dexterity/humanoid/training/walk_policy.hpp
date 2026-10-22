#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace humanoid::walk {
constexpr int kObs = 47, kFrames = 8, kHistory = kObs * kFrames, kActions = 12;
constexpr int kPrivileged = 16, kCritic = kHistory + kPrivileged;
constexpr double kActionLimit = 3.0;
struct History {
  std::vector<float> values = std::vector<float>(kHistory, 0.f);
  void reset() { std::fill(values.begin(), values.end(), 0.f); }
  void append(const std::vector<float>& observation) {
    if (observation.size() != kObs || !std::all_of(observation.begin(), observation.end(), [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("Invalid 47-value observation");
    std::move(values.begin() + kObs, values.end(), values.begin());
    std::copy(observation.begin(), observation.end(), values.end() - kObs);
  }
};
inline torch::Tensor tensor(const std::vector<float>& values, int width, const torch::Device& device) {
  if (values.empty() || values.size() % width) throw std::runtime_error("Invalid tensor row width");
  return torch::from_blob(const_cast<float*>(values.data()), {static_cast<int64_t>(values.size() / width), width},
                          torch::kFloat32).clone().to(device);
}
inline std::vector<float> vector(const torch::Tensor& value) {
  const auto host = value.detach().to(torch::kCPU).contiguous();
  if (host.scalar_type() != torch::kFloat32 || !torch::isfinite(host).all().item<bool>())
    throw std::runtime_error("Nonfinite policy tensor");
  return {host.data_ptr<float>(), host.data_ptr<float>() + host.numel()};
}
// Explicit ELU avoids a Python resolver in the native TorchScript compiler.
inline torch::Tensor native_elu(const torch::Tensor& value) {
  return torch::where(value > 0., value, value.clamp(-50., 0.).exp() - 1.);
}
struct ActorCriticImpl : torch::nn::Module {
  torch::nn::Linear a1{nullptr}, a2{nullptr}, a3{nullptr}, c1{nullptr}, c2{nullptr}, c3{nullptr};
  torch::Tensor log_std;
  ActorCriticImpl() {
    a1 = register_module("a1", torch::nn::Linear(kHistory, 256));
    a2 = register_module("a2", torch::nn::Linear(256, 128));
    a3 = register_module("a3", torch::nn::Linear(128, kActions));
    c1 = register_module("c1", torch::nn::Linear(kCritic, 256));
    c2 = register_module("c2", torch::nn::Linear(256, 128));
    c3 = register_module("c3", torch::nn::Linear(128, 1));
    log_std = register_parameter("log_std", torch::full({kActions}, -1.0f));
    torch::NoGradGuard guard;
    for (auto* layer : {&a1, &a2, &c1, &c2}) {
      torch::nn::init::orthogonal_((*layer)->weight, std::sqrt(2.));
      torch::nn::init::zeros_((*layer)->bias);
    }
    torch::nn::init::orthogonal_(a3->weight, .01); torch::nn::init::zeros_(a3->bias);
    torch::nn::init::orthogonal_(c3->weight, 1.); torch::nn::init::zeros_(c3->bias);
  }
  torch::Tensor actor(const torch::Tensor& history) {
    return a3->forward(native_elu(a2->forward(native_elu(a1->forward(history.clamp(-10., 10.))))));
  }
  torch::Tensor value(const torch::Tensor& privileged) {
    return c3->forward(native_elu(c2->forward(native_elu(c1->forward(privileged.clamp(-10., 10.)))))).squeeze(-1);
  }
  torch::Tensor stddev() { return log_std.clamp(-3., .3).exp(); }
  torch::Tensor log_probability(const torch::Tensor& history, const torch::Tensor& latent) {
    const auto ls = log_std.clamp(-3., .3);
    const auto normal = -.5 * ((latent - actor(history)) / ls.exp()).square() - ls - .5 * std::log(2. * 3.141592653589793);
    const auto jacobian = std::log(kActionLimit) + 2. * (std::log(2.) - latent - torch::softplus(-2. * latent));
    return (normal - jacobian).sum(-1);
  }
  torch::Tensor entropy_proxy() {
    return (log_std.clamp(-3., .3) + .5 * std::log(2. * 3.141592653589793 * std::exp(1.))).sum();
  }
  void export_policy(const std::filesystem::path& path) {
    torch::NoGradGuard guard;
    torch::jit::Module module("FromScratchHistoryPPO");
    for (int i = 0; i < 3; ++i) {
      auto& layer = i == 0 ? a1 : i == 1 ? a2 : a3;
      module.register_parameter("w" + std::to_string(i + 1), layer->weight.detach().cpu().clone(), false);
      module.register_parameter("b" + std::to_string(i + 1), layer->bias.detach().cpu().clone(), false);
    }
    module.register_buffer("history", torch::zeros({1, kHistory}, torch::kFloat32));
    module.define(R"JIT(
def forward(self, observation):
    assert observation.dim() == 2 and observation.size(0) == 1 and observation.size(1) == 47
    self.history.copy_(torch.cat([self.history[:, 47:], observation], 1))
    x = torch.matmul(torch.clamp(self.history, -10.0, 10.0), self.w1.t()) + self.b1
    x = torch.where(x > 0.0, x, torch.exp(torch.clamp(x, -50.0, 0.0)) - 1.0)
    x = torch.matmul(x, self.w2.t()) + self.b2
    x = torch.where(x > 0.0, x, torch.exp(torch.clamp(x, -50.0, 0.0)) - 1.0)
    return 3.0 * torch.tanh(torch.matmul(x, self.w3.t()) + self.b3)
)JIT");
    const auto frames = torch::linspace(-1.1, .9, 12 * kObs, torch::kFloat32).reshape({12, kObs});
    auto verify = [&](torch::jit::Module& candidate) {
      if (candidate.attr("history").toTensor().count_nonzero().item<int64_t>() != 0)
        throw std::runtime_error("Export starts with nonzero history");
      History history;
      for (int i = 0; i < 12; ++i) {
        const auto obs = frames.slice(0, i, i + 1);
        history.append(vector(obs));
        const auto expected = (kActionLimit * torch::tanh(actor(tensor(history.values, kHistory, a1->weight.device())))).cpu();
        if (!torch::allclose(expected, candidate.forward({obs}).toTensor(), 1e-4, 1e-5) ||
            !torch::equal(candidate.attr("history").toTensor(), tensor(history.values, kHistory, torch::kCPU)))
          throw std::runtime_error("Export history/inference mismatch");
      }
    };
    verify(module);
    module.attr("history").toTensor().zero_();
    module.save(path.string());
    auto restored = torch::jit::load(path.string());
    verify(restored);
  }
};
TORCH_MODULE(ActorCritic);
}  // namespace humanoid::walk
