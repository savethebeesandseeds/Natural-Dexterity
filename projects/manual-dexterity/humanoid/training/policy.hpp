#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <filesystem>

namespace humanoid::ppo {

constexpr int kObservations = 47;
constexpr int kActions = 12;

struct ActorCriticImpl : torch::nn::Module {
  torch::nn::Linear actor1{nullptr}, actor2{nullptr}, mean{nullptr};
  torch::nn::Linear critic1{nullptr}, critic2{nullptr}, critic3{nullptr};
  torch::Tensor log_std;

  ActorCriticImpl() {
    actor1 = register_module("actor1", torch::nn::Linear(kObservations, 128));
    actor2 = register_module("actor2", torch::nn::Linear(128, 128));
    mean = register_module("mean", torch::nn::Linear(128, kActions));
    critic1 = register_module("critic1", torch::nn::Linear(kObservations, 128));
    critic2 = register_module("critic2", torch::nn::Linear(128, 128));
    critic3 = register_module("critic3", torch::nn::Linear(128, 1));
    log_std = register_parameter("log_std", torch::full({kActions}, -0.7f));
    torch::NoGradGuard guard;
    for (auto* layer : {&actor1, &actor2, &critic1, &critic2}) {
      torch::nn::init::orthogonal_((*layer)->weight, std::sqrt(2.0));
      torch::nn::init::zeros_((*layer)->bias);
    }
    torch::nn::init::orthogonal_(mean->weight, .01);
    torch::nn::init::zeros_(mean->bias);
    torch::nn::init::orthogonal_(critic3->weight, 1.0);
    torch::nn::init::zeros_(critic3->bias);
  }

  torch::Tensor actor(const torch::Tensor& observation) {
    return mean->forward(torch::tanh(actor2->forward(torch::tanh(actor1->forward(observation)))));
  }

  torch::Tensor value(const torch::Tensor& observation) {
    return critic3->forward(torch::tanh(critic2->forward(torch::tanh(critic1->forward(observation))))).squeeze(-1);
  }

  torch::Tensor standard_deviation() { return log_std.clamp(-4.0, .7).exp(); }

  // Exact tanh-Gaussian density evaluated from retained pre-tanh samples.
  // The stable Jacobian formula avoids log(1-tanh(z)^2) cancellation.
  torch::Tensor log_probability(const torch::Tensor& observation, const torch::Tensor& latent) {
    const auto bounded_log_std = log_std.clamp(-4.0, .7);
    const auto normalized = (latent - actor(observation)) / bounded_log_std.exp();
    const auto normal = -.5 * normalized.square() - bounded_log_std - .5 * std::log(2.0 * 3.14159265358979323846);
    const auto jacobian = 2.0 * (std::log(2.0) - latent - torch::softplus(-2.0 * latent));
    return (normal - jacobian).sum(-1);
  }

  // Analytic entropy of the unsquashed Gaussian, an explicit exploration proxy.
  torch::Tensor gaussian_entropy() {
    return (log_std.clamp(-4.0, .7) + .5 * std::log(2.0 * 3.14159265358979323846 * std::exp(1.0))).sum();
  }

  void export_policy(const std::filesystem::path& path) {
    torch::NoGradGuard guard;
    torch::jit::Module policy("HumanoidWalkingPolicy");
    policy.register_parameter("w1", actor1->weight.detach().to(torch::kCPU).clone(), false);
    policy.register_parameter("b1", actor1->bias.detach().to(torch::kCPU).clone(), false);
    policy.register_parameter("w2", actor2->weight.detach().to(torch::kCPU).clone(), false);
    policy.register_parameter("b2", actor2->bias.detach().to(torch::kCPU).clone(), false);
    policy.register_parameter("w3", mean->weight.detach().to(torch::kCPU).clone(), false);
    policy.register_parameter("b3", mean->bias.detach().to(torch::kCPU).clone(), false);
    policy.define(R"JIT(
def forward(self, observation):
    x = torch.tanh(torch.matmul(observation, self.w1.t()) + self.b1)
    x = torch.tanh(torch.matmul(x, self.w2.t()) + self.b2)
    return torch.tanh(torch.matmul(x, self.w3.t()) + self.b3)
)JIT");
    // Verify the exported inference graph before saving it for the live viewer.
    auto probe = torch::linspace(-1.0, 1.0, 2 * kObservations, actor1->weight.options()).reshape({2, kObservations});
    const auto expected = torch::tanh(actor(probe)).to(torch::kCPU);
    const auto actual = policy.forward({probe.to(torch::kCPU)}).toTensor();
    if (!torch::allclose(expected, actual, 1e-4, 1e-5)) {
      throw std::runtime_error("TorchScript policy export changed inference outputs");
    }
    policy.save(path.string());
    auto restored = torch::jit::load(path.string());
    if (!torch::allclose(actual, restored.forward({probe.to(torch::kCPU)}).toTensor(), 1e-5, 1e-6)) {
      throw std::runtime_error("Saved TorchScript policy failed its reload check");
    }
  }
};
TORCH_MODULE(ActorCritic);

}  // namespace humanoid::ppo
