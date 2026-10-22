#include "atari/games.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace atari {
namespace {
using json = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Invalid game registry: " + message);
}

bool safe_id(const std::string& value) {
    if (value.empty() || value.size() > 64 || value.front() < 'a' || value.front() > 'z') return false;
    for (char c : value) {
        if (!(c >= 'a' && c <= 'z') && !(c >= '0' && c <= '9') && c != '_') return false;
    }
    return true;
}

std::string string_field(const json& object, const char* name) {
    require(object.contains(name) && object.at(name).is_string(), std::string(name) + " must be a string");
    auto value = object.at(name).get<std::string>();
    require(!value.empty(), std::string(name) + " must not be empty");
    return value;
}

std::int64_t integer_field(const json& object, const char* name,
                           std::int64_t minimum, std::int64_t maximum) {
    require(object.contains(name) && object.at(name).is_number_integer(), std::string(name) + " must be an integer");
    const auto& value = object.at(name);
    if (value.is_number_unsigned()) {
        require(value.get<std::uint64_t>() <= static_cast<std::uint64_t>(maximum), std::string(name) + " is too large");
    }
    const auto result = value.get<std::int64_t>();
    require(result >= minimum && result <= maximum, std::string(name) + " is out of range");
    return result;
}

bool boolean_field(const json& object, const char* name) {
    require(object.contains(name) && object.at(name).is_boolean(), std::string(name) + " must be boolean");
    return object.at(name).get<bool>();
}
} // namespace

GameRegistry load_game_registry(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("Cannot open game registry: " + path);
    try {
        const json document = json::parse(input);
        require(document.is_object(), "root must be an object");
        require(integer_field(document, "schema_version", 1, 1) == 1, "unsupported schema");
        require(document.contains("games") && document.at("games").is_array(), "games must be an array");
        require(!document.at("games").empty(), "games must not be empty");
        GameRegistry result;
        std::set<std::string> ids;
        std::set<std::string> roms;
        for (const auto& record : document.at("games")) {
            require(record.is_object(), "game must be an object");
            GameSpec game;
            game.id = string_field(record, "id");
            require(safe_id(game.id), "unsafe game ID: " + game.id);
            require(ids.insert(game.id).second, "duplicate game ID: " + game.id);
            game.title = string_field(record, "title");
            game.rom_filename = string_field(record, "rom_filename");
            const auto& rom = game.rom_filename;
            require(rom.size() > 4 && rom.ends_with(".bin") && safe_id(rom.substr(0, rom.size() - 4)),
                    "ROM filename must be a safe .bin basename");
            require(roms.insert(rom).second, "duplicate ROM filename: " + rom);
            game.rom_sha256 = string_field(record, "rom_sha256");
            require(game.rom_sha256.size() == 64, "ROM SHA256 must have 64 lowercase hexadecimal characters");
            for (char c : game.rom_sha256)
                require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), "ROM SHA256 must be lowercase hexadecimal");
            game.expected_actions = static_cast<int>(integer_field(record, "expected_actions", 1, 18));
            game.training_decisions = integer_field(record, "training_decisions", 1, std::numeric_limits<std::int64_t>::max());
            // ALE consumes signed integer seeds. Reserve enough contiguous seeds
            // for the declared final evaluation without overflow or train overlap.
            game.final_eval_seed = static_cast<std::uint64_t>(integer_field(record, "final_eval_seed", 100'000, std::numeric_limits<int>::max()));
            require(record.contains("criterion") && record.at("criterion").is_object(), "criterion must be an object");
            const auto& criterion = record.at("criterion");
            auto& c = game.criterion;
            c.episodes = static_cast<int>(integer_field(criterion, "episodes", 1, 100'000));
            c.min_positive_games = static_cast<int>(integer_field(criterion, "min_positive_games", 0, c.episodes));
            require(criterion.contains("mean_raw_return_threshold") && criterion.at("mean_raw_return_threshold").is_number(), "mean threshold must be numeric");
            c.mean_raw_return_threshold = criterion.at("mean_raw_return_threshold").get<double>();
            require(std::isfinite(c.mean_raw_return_threshold), "mean threshold must be finite");
            c.mean_strict = boolean_field(criterion, "mean_strict");
            c.max_truncated_episodes = static_cast<int>(integer_field(criterion, "max_truncated_episodes", 0, 0));
            c.description = string_field(criterion, "description");
            require(game.final_eval_seed + static_cast<std::uint64_t>(c.episodes - 1) <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()), "final evaluation seed range overflows ALE");
            result.push_back(std::move(game));
        }
        return result;
    } catch (const json::exception& error) {
        throw std::invalid_argument("Invalid game registry " + path + ": " + error.what());
    }
}

const GameSpec& find_game(const GameRegistry& registry, const std::string& id) {
    for (const auto& game : registry) if (game.id == id) return game;
    throw std::invalid_argument("Unknown game ID: " + id);
}

nlohmann::json game_spec_json(const GameSpec& game) {
    const auto& c = game.criterion;
    return {{"id", game.id}, {"title", game.title}, {"rom_filename", game.rom_filename},
        {"rom_sha256", game.rom_sha256},
        {"expected_actions", game.expected_actions}, {"training_decisions", game.training_decisions},
        {"final_eval_seed", game.final_eval_seed},
        {"criterion", {{"episodes", c.episodes}, {"min_positive_games", c.min_positive_games},
            {"mean_raw_return_threshold", c.mean_raw_return_threshold}, {"mean_strict", c.mean_strict},
            {"max_truncated_episodes", c.max_truncated_episodes}, {"description", c.description}}}};
}

bool evaluate_game_criterion(const GameSpec& game, int completed, int positive_games,
                             double mean_raw_return, int truncated) {
    const auto& c = game.criterion;
    if (c.episodes <= 0 || c.min_positive_games < 0 || c.min_positive_games > c.episodes ||
        !std::isfinite(c.mean_raw_return_threshold) || c.max_truncated_episodes != 0 ||
        completed != c.episodes || positive_games < c.min_positive_games ||
        positive_games > completed || truncated != 0 || !std::isfinite(mean_raw_return)) return false;
    return c.mean_strict ? mean_raw_return > c.mean_raw_return_threshold
                         : mean_raw_return >= c.mean_raw_return_threshold;
}

} // namespace atari
