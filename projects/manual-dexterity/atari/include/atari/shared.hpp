#pragma once

#include "atari/games.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace atari {

// The first shared-policy experiment is deliberately limited to Pong and
// Breakout. max_steps is the TOTAL new decisions, divided equally between games.
struct SharedTrainOptions {
    std::string run_dir;
    std::string rom_directory;
    std::string source_checkpoint_path;
    std::string source_evaluation_path; // Optional immutable source Pong evidence.
    std::vector<GameSpec> games;
    std::uint64_t seed = 41;
    std::int64_t max_steps = 2'000'000;
    int max_seconds = 3600;
    int evaluation_max_seconds = 3600; // Per game, separately for before/after; range 1..7200.
    int num_envs = 8;
    int rollout_steps = 128;
    int epochs = 4;
    int minibatch_size = 256;
    int torch_threads = 2;
    bool evaluate_at_end = true;
};

nlohmann::json run_shared_training(const SharedTrainOptions& options);

// Extend a completed Pong+Breakout joint experiment to Space Invaders. The
// source is its shared-final.pt, with adjacent config/progress/retention/manifest.
// Existing encoder and BOTH learned heads are restored; Adam and the new head
// start fresh. max_steps is total new decisions, split equally among three games.
struct SharedExtendOptions {
    std::string run_dir;
    std::string rom_directory;
    std::string source_checkpoint_path;
    std::vector<GameSpec> games;
    std::uint64_t seed = 51;
    std::int64_t max_steps = 3'000'000;
    int max_seconds = 3600;
    int evaluation_max_seconds = 3600; // Per game, separately for before/after; range 1..7200.
    int num_envs = 12;
    int rollout_steps = 128;
    int epochs = 4;
    int minibatch_size = 384;
    int torch_threads = 2;
    bool evaluate_at_end = true;
};
nlohmann::json run_shared_extension(const SharedExtendOptions& options);

// Pure indexing helper used by the trainer and CPU tests. Rows have time-major
// layout, with the first half of environments playing Pong and the second half
// playing Breakout. Each minibatch samples equal numbers from these partitions.
std::vector<std::int64_t> shared_game_indices(int steps, int environments, int game, int games = 2);

// Exercises real Torch modules on CPU: unequal action heads, balanced gradients,
// joint model + Adam serialization, and compatible single-game policy exports.
// The supplied directory must be new or empty; generated evidence is preserved.
nlohmann::json check_shared_checkpoint_roundtrip(const std::string& directory);

} // namespace atari
