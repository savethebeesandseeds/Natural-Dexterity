#pragma once

#include "atari/games.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace torch::nn { class Module; }

namespace atari {

struct TrainOptions {
    GameSpec game;
    std::string rom_path;
    std::string run_dir;
    std::string resume_checkpoint_path; // Optional weights + Adam continuation into a new run.
    std::string transfer_checkpoint_path; // Encoder only; target heads and Adam start fresh.
    GameSpec transfer_game;
    std::uint64_t seed = 1;
    std::int64_t max_steps = 5'000'000; // Additional decisions in this run, each repeating four frames.
    int max_seconds = 10'800;
    int num_envs = 8;
    int rollout_steps = 128;
    int epochs = 4;
    int minibatch_size = 256;
    int torch_threads = 2;
};

struct EvalOptions {
    GameSpec game;
    std::string rom_path;
    std::string checkpoint_path;
    std::string output_path;
    std::string stop_path; // Optional enclosing experiment STOP file for nested evaluations.
    std::uint64_t seed = 1'000'100;
    int episodes = 20;
    int max_seconds = 900;
};

// The viewer uses a separate CPU model; all training inference is batched on CUDA.
class Policy {
public:
    explicit Policy(const std::string& checkpoint_path, const std::string& device = "cpu");
    ~Policy();
    Policy(Policy&&) noexcept;
    Policy& operator=(Policy&&) noexcept;
    Policy(const Policy&) = delete;
    Policy& operator=(const Policy&) = delete;
    int predict(const std::array<std::uint8_t, 4 * 84 * 84>& observation,
                bool deterministic = true);
    int action_count() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

nlohmann::json run_training(const TrainOptions& options);
nlohmann::json run_evaluation(const EvalOptions& options);
// Budgets, worker count, and seed may change; game/algorithm semantics must match.
void validate_resume_config(const nlohmann::json& source, const nlohmann::json& expected);
// Action counts alone do not identify a game (Pong and Space Invaders both use
// six actions). Evaluation must also match the adjacent training configuration.
void validate_evaluation_config(const nlohmann::json& source, const GameSpec& game,
                                const std::string& rom_sha256, int action_count);
void validate_transfer_config(const nlohmann::json& source, const GameSpec& source_game,
                              const nlohmann::json& target);
// Copy the four encoder modules after the caller validates source configuration.
// All eight tensors are checked before any copy; this never loads heads or Adam.
nlohmann::json copy_feature_parameters(torch::nn::Module& target,
                                       const std::string& checkpoint_path, int source_actions);
// Keep the initial transfer/cold-start identity when continuing a stopped run.
// Legacy continuations without origin metadata are identified as unknown.
nlohmann::json training_origin_metadata(const nlohmann::json& config);

struct AdvantageBatch { std::vector<float> advantages, returns; };
// Inputs use time-major [time, environment] layout. Time-limit bootstrap values
// must already be included in rewards; ended then prevents cross-episode GAE.
AdvantageBatch compute_gae(const std::vector<float>& rewards,
                           const std::vector<float>& values,
                           const std::vector<std::uint8_t>& ended,
                           const std::vector<float>& last_values,
                           int steps, int environments,
                           float gamma = .99f, float lambda = .95f);
bool evidence_passes(int completed_episodes, int wins, double mean_raw_return,
                     int truncated_episodes);

} // namespace atari
