#include "atari/shared.hpp"
#include "atari/learning.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const auto pong = atari::shared_game_indices(3, 8, 0), breakout = atari::shared_game_indices(3, 8, 1);
        if (pong != std::vector<std::int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19} ||
            breakout != std::vector<std::int64_t>{4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23})
            throw std::runtime_error("Balanced time-major partition incorrect");
        auto combined = pong; combined.insert(combined.end(), breakout.begin(), breakout.end());
        std::sort(combined.begin(), combined.end());
        for (std::size_t i = 0; i < combined.size(); ++i) if (combined[i] != static_cast<std::int64_t>(i))
            throw std::runtime_error("Shared partitions duplicate or omit samples");
        std::vector<std::int64_t> three_game_rows;
        for (int game = 0; game < 3; ++game) {
            const auto rows = atari::shared_game_indices(5, 12, game, 3);
            if (rows.size() != 20) throw std::runtime_error("Three-game partition is unbalanced");
            for (const auto row : rows) if ((row % 12) / 4 != game)
                throw std::runtime_error("Three-game partition crossed game boundaries");
            three_game_rows.insert(three_game_rows.end(), rows.begin(), rows.end());
        }
        std::sort(three_game_rows.begin(), three_game_rows.end());
        for (std::size_t i = 0; i < three_game_rows.size(); ++i) if (three_game_rows[i] != static_cast<std::int64_t>(i))
            throw std::runtime_error("Three-game partitions duplicate or omit samples");
        for (const auto& arguments : {std::array<int, 3>{0, 8, 0}, {2, 7, 0}, {2, 8, 2}}) {
            bool rejected = false;
            try { atari::shared_game_indices(arguments[0], arguments[1], arguments[2]); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid balanced partition accepted");
        }
        // The interleaved games must bootstrap only their own environment stream.
        auto gae = atari::compute_gae({1, 10, 2, 20}, {0, 0, 0, 0}, {0, 1, 1, 0}, {0, 30}, 2, 2, 1, 1);
        if (gae.advantages != std::vector<float>{3, 10, 2, 50})
            throw std::runtime_error("Shared GAE crossed games or episode boundaries");
        auto three_gae = atari::compute_gae({1, 10, 100, 2, 20, 200}, {0, 0, 0, 0, 0, 0},
            {0, 1, 0, 1, 0, 1}, {0, 30, 300}, 2, 3, 1, 1);
        if (three_gae.advantages != std::vector<float>{3, 10, 300, 2, 50, 200})
            throw std::runtime_error("Three-game GAE crossed games or terminal boundaries");
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const auto directory = argc > 1 ? std::filesystem::path(argv[1]) :
            std::filesystem::temp_directory_path() / ("atari-shared-test-" + std::to_string(stamp));
        auto result = atari::check_shared_checkpoint_roundtrip(directory.string());
        result["balanced_partition_and_gae"] = true; result["evidence_directory"] = directory.string();
        std::cout << result.dump(2) << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
