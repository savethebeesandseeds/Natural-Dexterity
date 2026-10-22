#include "atari/server.hpp"
#include "atari/environment.hpp"
#include "atari/games.hpp"
#include "atari/learning.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <png.h>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <sys/file.h>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace atari {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
json read_json(const fs::path& path) {
    try { std::ifstream file(path); return json::parse(file); } catch (...) { return json::object(); }
}
json learner_liveness() {
    // Serialize HTTP probes so they cannot mistake each other's short lock for a learner.
    static std::mutex probe_mutex;
    std::lock_guard guard(probe_mutex);
    const auto failure = [](const char* operation, int error) {
        return json{{"active", nullptr}, {"error", std::string(operation) + ": " +
            std::error_code(error, std::generic_category()).message()}};
    };
    const int fd = ::open("/tmp/atari-training.lock", O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return failure("Cannot open learner lock", errno);
    const int acquired = ::flock(fd, LOCK_EX | LOCK_NB);
    const int lock_error = acquired == 0 ? 0 : errno;
    const int unlock_error = acquired == 0 && ::flock(fd, LOCK_UN) != 0 ? errno : 0;
    const int close_error = ::close(fd) != 0 ? errno : 0;
    if (unlock_error) return failure("Cannot release learner lock probe", unlock_error);
    if (close_error) return failure("Cannot close learner lock probe", close_error);
    if (acquired == 0) return {{"active", false}, {"error", nullptr}};
    if (lock_error == EWOULDBLOCK || lock_error == EAGAIN)
        return {{"active", true}, {"error", nullptr}};
    return failure("Cannot inspect learner lock", lock_error);
}
std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
fs::path current_run(const fs::path& root) {
    auto state = read_json(root / "runs/current.json");
    if (!state.contains("run_dir") || !state["run_dir"].is_string()) return {};
    auto run = fs::weakly_canonical(state["run_dir"].get<std::string>());
    auto relative = run.lexically_relative(fs::weakly_canonical(root / "runs"));
    if (relative.empty() || *relative.begin() == ".." || relative.is_absolute()) return {};
    return run;
}
const GameSpec& run_game(const json& config, const GameRegistry& registry, const std::string& fallback_rom) {
    if (config.contains("game_id")) return find_game(registry, config.at("game_id").get<std::string>());
    const auto filename = fs::path(config.value("rom", fallback_rom)).filename().string();
    for (const auto& game : registry) if (game.rom_filename == filename) return game;
    throw std::runtime_error("Run ROM is not in the game registry");
}
std::string encode_png(const std::vector<uint8_t>& rgb, int width, int height) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = static_cast<png_uint_32>(width);
    image.height = static_cast<png_uint_32>(height);
    image.format = PNG_FORMAT_RGB;
    png_alloc_size_t size = 0;
    if (!png_image_write_to_memory(&image, nullptr, &size, 0, rgb.data(), 0, nullptr))
        throw std::runtime_error(image.message);
    std::string bytes(size, '\0');
    if (!png_image_write_to_memory(&image, bytes.data(), &size, 0, rgb.data(), 0, nullptr))
        throw std::runtime_error(image.message);
    bytes.resize(size);
    png_image_free(&image);
    return bytes;
}
struct Playback {
    std::mutex mutex;
    std::string frame;
    json state = {{"policy", "Starting native emulator"}, {"score", 0}, {"episode", 1}, {"error", nullptr}};
};
void play(std::stop_token stop, Playback& view, const fs::path& root, const std::string& rom, const GameRegistry& registry) {
    try {
        AtariEnv env(rom, 900001);
        std::string active_rom = rom;
        std::unique_ptr<Policy> policy;
        fs::path previous_run, previous_checkpoint;
        fs::file_time_type previous_stamp{};
        auto next_check = std::chrono::steady_clock::now();
        std::mt19937 random(900001);
        int episode = 1;
        double score = 0;
        env.reset();
        while (!stop.stop_requested()) {
            auto tick = std::chrono::steady_clock::now();
            if (tick >= next_check) {
                next_check = tick + std::chrono::seconds(10);
                auto run = current_run(root);
                const auto config = run.empty() ? json::object() : read_json(run / "config.json");
                const auto& game = run_game(config, registry, rom);
                const auto next_rom = (fs::path("/opt/ale/roms") / game.rom_filename).string();
                if (config.value("rom_sha256", game.rom_sha256) != game.rom_sha256)
                    throw std::runtime_error("Run ROM identity differs from the registered game");
                if (run != previous_run || next_rom != active_rom) {
                    policy.reset(); previous_checkpoint.clear(); previous_run = run;
                    env = AtariEnv(next_rom, 900001); active_rom = next_rom;
                    score = 0; episode = 1;
                    std::lock_guard guard(view.mutex);
                    view.state["game_id"] = game.id; view.state["game_title"] = game.title;
                    view.state["last_score"] = nullptr;
                }
                auto checkpoint = run.empty() ? fs::path{} : run / "latest.pt";
                if (!run.empty() && fs::exists(run / "final.pt")) checkpoint = run / "final.pt";
                if (!checkpoint.empty() && fs::exists(checkpoint)) {
                    auto stamp = fs::last_write_time(checkpoint);
                    if (checkpoint != previous_checkpoint || stamp != previous_stamp) {
                        try {
                            auto candidate = std::make_unique<Policy>(checkpoint.string(), "cpu");
                            if (candidate->action_count() != env.actions()) throw std::runtime_error("Checkpoint action count mismatch");
                            policy = std::move(candidate); previous_checkpoint = checkpoint; previous_stamp = stamp;
                            env.reset(); score = 0;
                            auto progress = read_json(run / "progress.json");
                            auto steps = progress.value("checkpoint_steps", int64_t{0});
                            std::lock_guard guard(view.mutex);
                            view.state["policy"] = checkpoint.filename().string() + " · " + std::to_string(steps) + " training steps";
                            view.state["error"] = nullptr;
                        } catch (const std::exception& error) {
                            std::lock_guard guard(view.mutex); view.state["error"] = error.what();
                        }
                    }
                }
            }
            int action = policy ? policy->predict(env.observation()) : std::uniform_int_distribution<int>(0, env.actions() - 1)(random);
            auto transition = env.step(action);
            score += transition.reward;
            auto bytes = encode_png(env.rgb(), env.width(), env.height());
            {
                std::lock_guard guard(view.mutex);
                view.frame = std::move(bytes);
                view.state["score"] = score; view.state["episode"] = episode;
                if (!policy) view.state["policy"] = "Random baseline · waiting for first checkpoint";
                if (transition.done || transition.truncated) view.state["last_score"] = score;
            }
            if (transition.done || transition.truncated) { env.reset(); score = 0; ++episode; }
            std::this_thread::sleep_until(tick + std::chrono::microseconds(66667));
        }
    } catch (const std::exception& error) {
        std::lock_guard guard(view.mutex);
        view.state["policy"] = "Playback unavailable"; view.state["error"] = error.what();
    }
}
}
int serve(const std::string& project, const std::string& rom, const std::string& host, int port) {
    fs::path root = fs::canonical(project);
    const auto registry = load_game_registry((root / "games.json").string());
    Playback playback;
    std::jthread player([&](std::stop_token stop) { play(stop, playback, root, rom, registry); });
    httplib::Server server;
    server.set_default_headers({{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"},
        {"X-Robots-Tag", "noindex, nofollow"}, {"Content-Security-Policy", "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; style-src-elem 'self' 'unsafe-inline'; style-src-attr 'none'; frame-ancestors 'none'"}});
    server.Get("/healthz", [](const auto&, auto& response) { response.set_content("{\"status\":\"ok\",\"runtime\":\"C++\"}", "application/json"); });
    server.Get("/api/status", [&](const auto&, auto& response) {
        auto run = current_run(root);
        json state;
        { std::lock_guard guard(playback.mutex); state = playback.state; }
        json result = {{"run", run.empty() ? json(nullptr) : json(run.filename().string())}, {"playback", state}, {"runtime", "C++20 / ALE / LibTorch"}};
        for (const auto* key : {"progress", "evaluation", "config"}) result[key] = run.empty() ? json::object() : read_json(run / (std::string(key) + ".json"));
        const auto learner = learner_liveness();
        result["learner_active"] = learner["active"];
        result["learner_liveness_error"] = learner["error"];
        auto& progress = result["progress"];
        if (learner["active"] == false && progress.is_object() &&
            progress.contains("status") && progress["status"].is_string()) {
            const auto reported = progress["status"].get<std::string>();
            if (reported == "training" || reported == "evaluating" || reported == "evaluating_before" || reported == "initializing") {
                progress["reported_status"] = reported;
                progress["status"] = "interrupted";
            }
        }
        result["game"] = game_spec_json(run_game(result["config"], registry, rom));
        result["games"] = json::array();
        for (const auto& game : registry) result["games"].push_back(game_spec_json(game));
        result["proofs"] = read_json(root / "runs/proofs.json");
        result["experiments"] = read_json(root / "runs/experiments.json");
        result["comparisons"] = read_json(root / "runs/comparisons/index.json");
        result["shared_evaluation"] = run.empty() ? json::object() : read_json(run / "retention.json");
        response.set_content(result.dump(), "application/json");
    });
    server.Get("/frame.png", [&](const auto&, auto& response) {
        std::lock_guard guard(playback.mutex);
        if (playback.frame.empty()) { response.status = 503; response.set_content("Emulator starting", "text/plain"); }
        else response.set_content(playback.frame, "image/png");
    });
    for (auto item : {std::array<const char*, 3>{"/", "index.html", "text/html; charset=utf-8"},
                      std::array<const char*, 3>{"/app.js", "app.js", "text/javascript"},
                      std::array<const char*, 3>{"/style.css", "style.css", "text/css"}}) {
        server.Get(item[0], [&, filename = std::string(item[1]), mime = std::string(item[2])](const auto&, auto& response) {
            response.set_content(read_file(root / "web" / filename), mime);
        });
    }
    std::cout << "Native Atari viewer: " << host << ':' << port << std::endl;
    if (!server.listen(host, port)) throw std::runtime_error("Cannot bind viewer port");
    return 0;
}
}
