#include "atari/games.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using json = nlohmann::json;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("atari-games-test-" + std::to_string(timestamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("Cannot create test directory");
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

void expect_invalid(const std::filesystem::path& path, const json& document) {
    { std::ofstream output(path); output << document.dump(); }
    bool rejected = false;
    try { (void)atari::load_game_registry(path.string()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Malformed registry accepted");
}

void validate_parser(const std::string& source, const atari::GameRegistry& registry) {
    TemporaryDirectory temporary;
    const auto path = temporary.path / "games.json";
    std::ifstream input(source);
    const json valid = json::parse(input);

    // The serialized spec is also the immutable criterion snapshot used in
    // evidence files; it must retain every required registry field.
    json roundtrip{{"schema_version", 1}, {"games", json::array()}};
    for (const auto& game : registry) roundtrip["games"].push_back(atari::game_spec_json(game));
    { std::ofstream output(path); output << roundtrip.dump(); }
    auto loaded = atari::load_game_registry(path.string());
    require(loaded.size() == 3, "Roundtrip registry lost games");
    require(atari::game_spec_json(loaded[2]) == roundtrip["games"][2], "Spec snapshot changed on roundtrip");

    for (const std::string& id : std::vector<std::string>{"", "../pong", "pong/rom", "pong\\rom", "Pong", "_pong", "pong:rom", std::string("pong\0suffix", 11)}) {
        auto document = valid; document["games"][0]["id"] = id; expect_invalid(path, document);
    }
    for (const char* rom : {"../pong.bin", "dir/pong.bin", "dir\\pong.bin", "C:pong.bin", "/pong.bin", "pong..bin", "Pong.bin", "pong.zip", ".bin"}) {
        auto document = valid; document["games"][0]["rom_filename"] = rom; expect_invalid(path, document);
    }
    for (const char* field : {"id", "rom_filename", "rom_sha256", "expected_actions", "training_decisions", "final_eval_seed", "criterion"}) {
        auto document = valid; document["games"][0].erase(field); expect_invalid(path, document);
    }
    for (const json bad_hash : {json("abcdef"), json(std::string(64, 'g')), json(std::string(64, 'A')), json(nullptr)}) {
        auto document = valid; document["games"][0]["rom_sha256"] = bad_hash; expect_invalid(path, document);
    }
    for (const auto* duplicate : {"id", "rom_filename"}) {
        auto document = valid; document["games"][1][duplicate] = document["games"][0][duplicate]; expect_invalid(path, document);
    }
    auto document = valid; document["schema_version"] = 2; expect_invalid(path, document);
    document = valid; document["games"] = json::array(); expect_invalid(path, document);
    document = valid; document["games"] = json::object(); expect_invalid(path, document);
    expect_invalid(path, json::array());

    for (const json bad : {json(0), json(-1), json(19), json(6.0), json(true), json("6")}) {
        document = valid; document["games"][0]["expected_actions"] = bad; expect_invalid(path, document);
    }
    for (const json bad : {json(0), json(-1), json(std::numeric_limits<std::uint64_t>::max()), json(1.5)}) {
        document = valid; document["games"][0]["training_decisions"] = bad; expect_invalid(path, document);
    }
    for (const json bad : {json(0), json(99'999), json(std::numeric_limits<int>::max()), json(std::numeric_limits<std::uint64_t>::max())}) {
        document = valid; document["games"][0]["final_eval_seed"] = bad; expect_invalid(path, document);
    }
    for (const json bad : {json(0), json(-1), json(100'001), json(20.5)}) {
        document = valid; document["games"][0]["criterion"]["episodes"] = bad; expect_invalid(path, document);
    }
    for (const json bad : {json(-1), json(21), json("16")}) {
        document = valid; document["games"][0]["criterion"]["min_positive_games"] = bad; expect_invalid(path, document);
    }
    for (const json bad : {json(nullptr), json("100"), json(true)}) {
        document = valid; document["games"][0]["criterion"]["mean_raw_return_threshold"] = bad; expect_invalid(path, document);
    }
    document = valid; document["games"][0]["criterion"]["mean_strict"] = 1; expect_invalid(path, document);
    document = valid; document["games"][0]["criterion"]["max_truncated_episodes"] = 1; expect_invalid(path, document);
    document = valid; document["games"][0]["criterion"]["description"] = ""; expect_invalid(path, document);

    // Parsing must reject trailing data, and floating-point overflow rather than
    // silently turning an unbounded score target into a pass/fail calculation.
    for (const std::string malformed : {valid.dump() + " garbage", std::string("{\"schema_version\":1,\"games\":[1e309]}")}) {
        { std::ofstream output(path); output << malformed; }
        bool rejected = false;
        try { (void)atari::load_game_registry(path.string()); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Malformed JSON accepted");
    }
    bool missing_rejected = false;
    try { (void)atari::load_game_registry((temporary.path / "missing.json").string()); }
    catch (const std::invalid_argument&) { missing_rejected = true; }
    require(missing_rejected, "Missing registry accepted");
}

void validate_criteria(const atari::GameRegistry& registry) {
    const auto& pong = atari::find_game(registry, "pong");
    const auto& breakout = atari::find_game(registry, "breakout");
    const auto& invaders = atari::find_game(registry, "space_invaders");
    require(pong.expected_actions == 6 && breakout.expected_actions == 4 && invaders.expected_actions == 6, "Unexpected action counts");
    require(atari::evaluate_game_criterion(pong, 20, 16, .1, 0), "Valid Pong evidence rejected");
    require(!atari::evaluate_game_criterion(pong, 20, 16, 0, 0), "Pong zero mean accepted");
    require(!atari::evaluate_game_criterion(pong, 20, 15, 10, 0), "Pong insufficient wins accepted");
    require(atari::evaluate_game_criterion(breakout, 20, 20, 100, 0), "Breakout threshold rejected");
    require(!atari::evaluate_game_criterion(breakout, 20, 20, 99.9, 0), "Breakout below threshold accepted");
    require(atari::evaluate_game_criterion(invaders, 20, 20, 1000, 0), "Space Invaders threshold rejected");
    require(!atari::evaluate_game_criterion(invaders, 20, 20, 999.9, 0), "Space Invaders below threshold accepted");
    for (const auto& game : registry) {
        require(game.criterion.episodes == 20 && game.training_decisions == 5'000'000, "Unexpected default budget");
        require(!atari::evaluate_game_criterion(game, 19, 19, 5000, 0), "Incomplete evaluation accepted");
        require(!atari::evaluate_game_criterion(game, 21, 21, 5000, 0), "Different-sized evaluation accepted");
        require(!atari::evaluate_game_criterion(game, 20, 20, 5000, 1), "Truncated evaluation accepted");
        require(!atari::evaluate_game_criterion(game, 20, 21, 5000, 0), "Impossible positive-game count accepted");
        require(!atari::evaluate_game_criterion(game, 20, 20, std::numeric_limits<double>::infinity(), 0), "Infinite result accepted");
        require(!atari::evaluate_game_criterion(game, 20, 20, std::numeric_limits<double>::quiet_NaN(), 0), "NaN result accepted");
    }
    bool unknown_rejected = false;
    try { (void)atari::find_game(registry, "unknown"); }
    catch (const std::invalid_argument&) { unknown_rejected = true; }
    require(unknown_rejected, "Unknown game accepted");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: atari-games-test <games.json>");
        const auto registry = atari::load_game_registry(argv[1]);
        require(registry.size() == 3, "Expected three supported games");
        validate_parser(argv[1], registry);
        validate_criteria(registry);
        std::cout << "Game registry validation, safe ROM names, score boundaries and evidence criteria passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
