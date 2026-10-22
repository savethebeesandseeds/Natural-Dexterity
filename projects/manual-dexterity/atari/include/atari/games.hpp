#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace atari {

struct EvaluationCriterion {
    int episodes = 20;
    int min_positive_games = 0;
    double mean_raw_return_threshold = 0;
    bool mean_strict = false;
    int max_truncated_episodes = 0;
    std::string description;
};

struct GameSpec {
    std::string id;
    std::string title;
    std::string rom_filename;
    std::string rom_sha256;
    int expected_actions = 0;
    std::int64_t training_decisions = 5'000'000;
    std::uint64_t final_eval_seed = 1'000'100;
    EvaluationCriterion criterion;
};

using GameRegistry = std::vector<GameSpec>;

// Filenames in this registry are basenames, never paths. Callers choose the ROM
// directory and verify the actual ALE action count when opening a game.
GameRegistry load_game_registry(const std::string& path);
const GameSpec& find_game(const GameRegistry& registry, const std::string& id);
nlohmann::json game_spec_json(const GameSpec& game);

// A result is accepted only for exactly the declared number of completed games.
// Positive-return games are Pong's win measure; other games use a score target.
bool evaluate_game_criterion(const GameSpec& game, int completed,
                             int positive_games, double mean_raw_return,
                             int truncated);

} // namespace atari
