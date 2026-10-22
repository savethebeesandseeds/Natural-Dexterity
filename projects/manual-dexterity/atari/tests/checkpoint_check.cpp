#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    using json = nlohmann::json;
    try {
        if (argc != 2) throw std::invalid_argument("Usage: atari-checkpoint-check CHECKPOINT.pt");
        torch::set_num_threads(1);
        torch::serialize::InputArchive checkpoint, model, actor, critic;
        checkpoint.load_from(argv[1], torch::Device(torch::kCPU));
        torch::Tensor steps, action_count, actor_bias, critic_bias;
        checkpoint.read("steps", steps);
        checkpoint.read("action_count", action_count);
        checkpoint.read("model", model);
        model.read("actor", actor);
        model.read("critic", critic);
        actor.read("bias", actor_bias);
        critic.read("bias", critic_bias);
        if (steps.numel() != 1 || action_count.numel() != 1 ||
            steps.scalar_type() != torch::kInt64 || action_count.scalar_type() != torch::kInt64)
            throw std::runtime_error("Checkpoint metadata must contain scalar int64 tensors");
        const auto recorded_steps = steps.item<std::int64_t>();
        const auto actions = action_count.item<std::int64_t>();
        if (recorded_steps <= 0 || actions < 1 || actions > 18 ||
            actor_bias.dim() != 1 || actor_bias.numel() != actions ||
            critic_bias.dim() != 1 || critic_bias.numel() != 1 ||
            !actor_bias.is_floating_point() || !critic_bias.is_floating_point())
            throw std::runtime_error("Invalid checkpoint metadata or policy bias shape");
        const bool finite = torch::isfinite(actor_bias).all().item<bool>() &&
                            torch::isfinite(critic_bias).all().item<bool>();
        const double actor_max = actor_bias.abs().max().item<double>();
        const double critic_max = critic_bias.abs().max().item<double>();
        // NetworkImpl initializes every bias to exactly zero. A finite nonzero
        // policy/value bias demonstrates parameter movement in the saved model.
        const bool updated = finite && (actor_max > 0 || critic_max > 0);
        std::cout << json({{"checkpoint", argv[1]}, {"recorded_steps", recorded_steps},
            {"action_count", actions}, {"actor_bias_max_abs", actor_max},
            {"critic_bias_max_abs", critic_max}, {"finite", finite},
            {"parameters_updated_from_zero_initialization", updated}}).dump(2) << '\n';
        return updated ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << json({{"finite", false}, {"parameters_updated_from_zero_initialization", false},
            {"error", error.what()}}).dump(2) << '\n';
        return 1;
    }
}
