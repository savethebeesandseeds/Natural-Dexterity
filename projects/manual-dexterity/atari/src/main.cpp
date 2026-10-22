#include "atari/atomic_io.hpp"
#include "atari/environment.hpp"
#include "atari/games.hpp"
#include "atari/learning.hpp"
#include "atari/server.hpp"
#include "atari/shared.hpp"
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <thread>

namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
class Lease {
    int fd_;
public:
    Lease() : fd_(::open("/tmp/atari-training.lock", O_CREAT | O_RDWR | O_CLOEXEC, 0600)) {
        if (fd_ < 0) throw std::runtime_error("Learner lock is unavailable");
        // Allow a short read-only dashboard probe to release its transient lock.
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) return;
            if (errno != EWOULDBLOCK && errno != EAGAIN) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("An experiment is already active or its lock is unavailable");
    }
    ~Lease() { if (fd_ >= 0) ::close(fd_); }
};
void write_json(const fs::path& path, const json& value) {
    atari::write_text_atomically(path, value.dump(2) + '\n');
}
std::string run_name(const std::string& game, uint64_t seed) {
    auto now = std::time(nullptr);
    std::tm utc{}; gmtime_r(&now, &utc);
    char text[48]; std::strftime(text, sizeof(text), "%Y%m%dT%H%M%SZ", &utc);
    return game + "-" + text + "-seed" + std::to_string(seed) + "-" + std::to_string(getpid());
}
atari::GameSpec checkpoint_game(const atari::GameRegistry& registry, const std::string& checkpoint) {
    std::ifstream input(fs::path(checkpoint).parent_path() / "config.json");
    if (!input) throw std::runtime_error("Source checkpoint needs an adjacent config.json");
    const auto config = json::parse(input);
    if (config.contains("game_id")) return atari::find_game(registry, config.at("game_id").get<std::string>());
    const auto filename = fs::path(config.value("rom", "")).filename().string();
    for (const auto& game : registry) if (game.rom_filename == filename) return game;
    throw std::runtime_error("Source checkpoint game is not registered");
}
void run_experiment(const fs::path& root, const std::string& directory, const std::function<json()>& train) {
    Lease lease;
    const auto run = fs::absolute(directory);
    const auto relative = fs::weakly_canonical(run).lexically_relative(fs::weakly_canonical(root / "runs"));
    if (relative.empty() || relative == "." || *relative.begin() == ".." || relative.is_absolute())
        throw std::runtime_error("Run directory must be inside <root>/runs so the viewer can follow it");
    if (fs::exists(run) && !fs::is_empty(run)) throw std::runtime_error("Preserving nonempty run directory");
    auto log = run; log += ".log";
    if (fs::exists(log)) throw std::runtime_error("Preserving existing run log");
    fs::create_directories(run.parent_path());
    write_json(root / "runs/current.json", {{"run_dir", run.string()}});
    std::cout << "Native experiment: " << run << " | log: " << log << std::endl;
    if (!std::freopen(log.c_str(), "w", stdout) || ::dup2(fileno(stdout), fileno(stderr)) < 0)
        throw std::runtime_error("Cannot open experiment log");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try { std::cout << train().dump(2) << std::endl; }
    catch (const std::exception& error) {
        fs::create_directories(run);
        write_json(run / "progress.json", {{"status", "failed"}, {"error", error.what()}});
        throw;
    }
}
json diagnostics(const std::string& rom) {
    if (!torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable inside the container");
    torch::set_num_threads(1);
    auto input = torch::rand({2, 4, 84, 84}, torch::TensorOptions().device(torch::kCUDA).requires_grad(true));
    auto convolution = torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 32, 8).stride(4));
    convolution->to(torch::kCUDA);
    auto result = convolution->forward(input);
    result.square().mean().backward();
    torch::cuda::synchronize();
    if (!input.grad().defined() || !torch::isfinite(input.grad()).all().item<bool>()) throw std::runtime_error("CUDA gradient check failed");
    atari::AtariEnv env(rom, 777);
    std::mt19937 random(777);
    for (int i = 0; i < 64; ++i) { auto step = env.step(static_cast<int>(random() % env.actions())); if (step.done || step.truncated) env.reset(); }
    auto rgb = env.rgb();
    if (rgb.size() != static_cast<size_t>(env.width() * env.height() * 3)) throw std::runtime_error("RGB buffer size mismatch");
    return {{"ok", true}, {"runtime", "native C++"}, {"torch", TORCH_VERSION}, {"cuda_devices", torch::cuda::device_count()},
        {"gpu_forward_and_backward", true}, {"rom_steps", 64}, {"observation_shape", {4, 84, 84}},
        {"rgb_shape", {env.height(), env.width(), 3}}, {"actions", env.actions()}};
}
json benchmark(const std::string& rom, int count, int64_t steps) {
    std::vector<std::unique_ptr<atari::AtariEnv>> environments;
    for (int i = 0; i < count; ++i) environments.push_back(std::make_unique<atari::AtariEnv>(rom, 777 + i));
    std::vector<std::jthread> workers;
    std::vector<std::exception_ptr> errors(count);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) workers.emplace_back([&, i] {
        try {
            std::mt19937 random(777 + i);
            for (int64_t step = i; step < steps; step += count) {
                auto value = environments[i]->step(static_cast<int>(random() % environments[i]->actions()));
                if (value.done || value.truncated) environments[i]->reset();
            }
        } catch (...) { errors[i] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (auto& error : errors) if (error) std::rethrow_exception(error);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return {{"mode", "random-policy environment throughput; excludes GPU training"}, {"environments", count},
        {"steps", steps}, {"seconds", elapsed}, {"decisions_per_second", steps / elapsed}};
}
}
int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "help";
        if (mode == "help" || mode == "--help") {
            std::cout << "Native Atari Lab\nModes: games, serve, train, shared-train, shared-extend, evaluate, check, benchmark\n"
                "Common: --game pong --root /workspace [--registry PATH] [--rom PATH]\n"
                "Train: --steps 5000000 --hours 3 --envs 8 --seed 1 [--run-dir PATH] [--resume CHECKPOINT | --transfer-features CHECKPOINT]\n"
                "Resume retains weights and Adam; steps/hours are additional budgets in a new run.\n"
                "Shared-train: --source-checkpoint PATH [--source-evaluation PATH] --steps 2000000 --hours 1 --envs 8 --seed 41 [--evaluate true|false]\n"
                "Shared training balances Pong and Breakout; steps are total new decisions across both games.\n"
                "Shared evaluation: --evaluation-seconds 3600 per game/stage (1..7200); episode/frame caps stay fixed.\n"
                "Shared-extend: --source-checkpoint TWO_GAME_SHARED_FINAL --steps 3000000 --envs 12 --seed 51 [--hours 1] [--evaluate true|false]\n"
                "Evaluate: --checkpoint PATH --output PATH --episodes 20 --seed 100001 --seconds 900\n"
                "Benchmark: --steps 50000 --envs 8\n";
            return 0;
        }
        std::map<std::string, std::string> options;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 == argc || std::string(argv[i]).rfind("--", 0) != 0) throw std::runtime_error("Expected --option value");
            if (!options.emplace(argv[i], argv[i + 1]).second) throw std::runtime_error("Duplicate option");
        }
        auto get = [&](const std::string& key, const std::string& fallback) {
            auto found = options.find(key);
            if (found == options.end()) return fallback;
            auto value = found->second; options.erase(found); return value;
        };
        const auto root = fs::canonical(get("--root", "/workspace"));
        const auto registry = atari::load_game_registry(get("--registry", (root / "games.json").string()));
        const auto game = atari::find_game(registry, get("--game", "pong"));
        const auto rom = get("--rom", (fs::path("/opt/ale/roms") / game.rom_filename).string());
        auto validate = [&] { if (!options.empty()) throw std::runtime_error("Unknown option: " + options.begin()->first); };
        if (mode == "games") {
            validate(); json games = json::array();
            for (const auto& entry : registry) games.push_back(atari::game_spec_json(entry));
            std::cout << games.dump(2) << std::endl; return 0;
        }
        if (mode == "serve") {
            auto host = get("--host", "0.0.0.0"); int port = std::stoi(get("--port", "8080")); validate();
            if (port < 1 || port > 65535) throw std::runtime_error("Invalid port");
            torch::set_num_threads(1);
            return atari::serve(root.string(), rom, host, port);
        }
        if (mode == "check") { validate(); std::cout << diagnostics(rom).dump(2) << std::endl; return 0; }
        if (mode == "benchmark") {
            int envs = std::stoi(get("--envs", "8")); int64_t steps = std::stoll(get("--steps", "50000")); validate();
            if (envs < 1 || envs > 64 || steps < 1) throw std::runtime_error("Invalid benchmark budget");
            std::cout << benchmark(rom, envs, steps).dump(2) << std::endl; return 0;
        }
        if (mode == "evaluate") {
            atari::EvalOptions config;
            config.game = game;
            config.rom_path = rom; config.checkpoint_path = get("--checkpoint", ""); config.output_path = get("--output", "");
            config.episodes = std::stoi(get("--episodes", std::to_string(game.criterion.episodes))); config.seed = std::stoull(get("--seed", std::to_string(game.final_eval_seed)));
            config.max_seconds = std::stoi(get("--seconds", "900")); validate();
            if (config.checkpoint_path.empty() || config.output_path.empty()) throw std::runtime_error("Evaluation needs --checkpoint and --output");
            if (fs::exists(config.output_path)) throw std::runtime_error("Preserving existing evaluation file");
            std::cout << atari::run_evaluation(config).dump(2) << std::endl; return 0;
        }
        if (mode == "shared-extend") {
            if (game.id != "pong") throw std::runtime_error("Shared extension uses Pong, Breakout and Space Invaders");
            atari::SharedExtendOptions config;
            config.games = {atari::find_game(registry, "pong"), atari::find_game(registry, "breakout"), atari::find_game(registry, "space_invaders")};
            config.rom_directory = get("--rom-directory", fs::path(rom).parent_path().string());
            config.source_checkpoint_path = get("--source-checkpoint", "");
            const auto evaluation = get("--evaluate", "true");
            if (evaluation != "true" && evaluation != "false") throw std::runtime_error("--evaluate must be true or false");
            config.evaluate_at_end = evaluation == "true";
            config.evaluation_max_seconds = std::stoi(get("--evaluation-seconds", "3600"));
            config.max_steps = std::stoll(get("--steps", "3000000"));
            const double hours = std::stod(get("--hours", "1"));
            config.num_envs = std::stoi(get("--envs", "12"));
            config.seed = std::stoull(get("--seed", "51"));
            if (config.source_checkpoint_path.empty() || !(hours > 0 && hours <= 24) || config.max_steps < 1 ||
                config.num_envs < 3 || config.num_envs > 64 || config.num_envs % 3 || config.seed >= 100000 ||
                config.evaluation_max_seconds < 1 || config.evaluation_max_seconds > 7200)
                throw std::runtime_error("Shared extension needs a joint checkpoint, workers divisible by three and valid budget/seed");
            config.max_seconds = std::max(1, static_cast<int>(hours * 3600));
            config.run_dir = get("--run-dir", (root / "runs" / run_name("shared-pong-breakout-space-invaders", config.seed)).string());
            validate();
            run_experiment(root, config.run_dir, [&] { return atari::run_shared_extension(config); });
            return 0;
        }
        if (mode == "shared-train") {
            if (game.id != "pong") throw std::runtime_error("Shared training uses the declared Pong + Breakout mixture");
            atari::SharedTrainOptions config;
            config.games = {atari::find_game(registry, "pong"), atari::find_game(registry, "breakout")};
            config.rom_directory = get("--rom-directory", fs::path(rom).parent_path().string());
            config.source_checkpoint_path = get("--source-checkpoint", "");
            config.source_evaluation_path = get("--source-evaluation", "");
            const auto evaluation = get("--evaluate", "true");
            if (evaluation != "true" && evaluation != "false") throw std::runtime_error("--evaluate must be true or false");
            config.evaluate_at_end = evaluation == "true";
            config.evaluation_max_seconds = std::stoi(get("--evaluation-seconds", "3600"));
            config.max_steps = std::stoll(get("--steps", "2000000"));
            const double hours = std::stod(get("--hours", "1"));
            config.num_envs = std::stoi(get("--envs", "8"));
            config.seed = std::stoull(get("--seed", "41"));
            if (config.source_checkpoint_path.empty() || !(hours > 0 && hours <= 24) || config.max_steps < 1 ||
                config.num_envs < 2 || config.num_envs > 64 || config.num_envs % 2 || config.seed >= 100000 ||
                config.evaluation_max_seconds < 1 || config.evaluation_max_seconds > 7200)
                throw std::runtime_error("Shared training needs a source checkpoint, even worker count and valid budget/seed");
            config.max_seconds = std::max(1, static_cast<int>(hours * 3600));
            config.run_dir = get("--run-dir", (root / "runs" / run_name("shared-pong-breakout", config.seed)).string());
            validate();
            run_experiment(root, config.run_dir, [&] { return atari::run_shared_training(config); });
            return 0;
        }
        if (mode == "train") {
            atari::TrainOptions config;
            config.game = game;
            config.rom_path = rom; config.max_steps = std::stoll(get("--steps", std::to_string(game.training_decisions)));
            double hours = std::stod(get("--hours", "3"));
            config.num_envs = std::stoi(get("--envs", "8")); config.seed = std::stoull(get("--seed", "1"));
            if (!(hours > 0 && hours <= 24) || config.num_envs < 1 || config.num_envs > 64 || config.max_steps < 1 || config.seed >= 100000)
                throw std::runtime_error("Invalid training budget, workers, or seed");
            config.max_seconds = std::max(1, static_cast<int>(hours * 3600));
            config.run_dir = get("--run-dir", (root / "runs" / run_name(game.id, config.seed)).string());
            config.resume_checkpoint_path = get("--resume", "");
            config.transfer_checkpoint_path = get("--transfer-features", "");
            if (!config.transfer_checkpoint_path.empty()) config.transfer_game = checkpoint_game(registry, config.transfer_checkpoint_path);
            if (!config.resume_checkpoint_path.empty() && !config.transfer_checkpoint_path.empty())
                throw std::runtime_error("Choose resume or feature transfer, not both");
            validate();
            run_experiment(root, config.run_dir, [&] { return atari::run_training(config); });
            return 0;
        }
        throw std::runtime_error("Unknown mode: " + mode);
    } catch (const std::exception& error) {
        std::cerr << "Atari: " << error.what() << std::endl;
        return 1;
    }
}
