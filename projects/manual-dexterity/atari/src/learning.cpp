#include "atari/learning.hpp"
#include "atari/atomic_io.hpp"
#include "atari/environment.hpp"
#include <torch/torch.h>
#include <torch/version.h>
#include <openssl/evp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <tuple>

namespace atari {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
constexpr std::size_t pixels = 4 * 84 * 84;
constexpr const char* network_description = "conv32/8/4 conv64/4/2 conv64/3/1 fc512 actor+critic";
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
struct Signals {
    using Handler = void (*)(int);
    Handler previous_int, previous_term;
    Signals() : previous_int(std::signal(SIGINT, on_signal)),
                previous_term(std::signal(SIGTERM, on_signal)) {
        if (previous_int != on_signal) interrupted = 0;
    }
    ~Signals() { std::signal(SIGINT, previous_int); std::signal(SIGTERM, previous_term); }
};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read hash input " + path);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::array<char, 65536> buffer{};
    while (input) {
        input.read(buffer.data(), buffer.size());
        if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(input.gcount())) != 1)
            throw std::runtime_error("Cannot update SHA256");
    }
    if (!input.eof()) throw std::runtime_error("Cannot read complete hash input " + path);
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest, &size) != 1) throw std::runtime_error("Cannot finalize SHA256");
    constexpr char hex[] = "0123456789abcdef"; std::string result; result.reserve(size * 2);
    for (unsigned int i = 0; i < size; ++i) { result += hex[digest[i] >> 4]; result += hex[digest[i] & 15]; }
    return result;
}
std::string tensor_sha256(const torch::Tensor& tensor) {
    auto bytes = tensor.detach().to(torch::kCPU).contiguous();
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int size = 0;
    if (EVP_Digest(bytes.const_data_ptr(), bytes.nbytes(), digest, &size, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot fingerprint tensor");
    constexpr char hex[] = "0123456789abcdef"; std::string result; result.reserve(size * 2);
    for (unsigned int i = 0; i < size; ++i) { result += hex[digest[i] >> 4]; result += hex[digest[i] & 15]; }
    return result;
}
json tensor_fingerprint(const torch::Tensor& tensor) {
    return {{"shape", tensor.sizes().vec()}, {"dtype", c10::toString(tensor.scalar_type())},
            {"sha256", tensor_sha256(tensor)}};
}
json head_fingerprints(torch::nn::Module& model) {
    const auto parameters = model.named_parameters(); json result = json::object();
    for (const char* name : {"actor.weight", "actor.bias", "critic.weight", "critic.bias"}) {
        const auto* parameter = parameters.find(name);
        if (!parameter) throw std::invalid_argument(std::string("Missing target head parameter: ") + name);
        result[name] = tensor_fingerprint(*parameter);
    }
    return result;
}
void atomic_json(const fs::path& path, const json& value) {
    if (path.empty()) return;
    write_text_atomically(path, value.dump(2) + '\n');
}
json preprocessing() {
    return {{"observation", "4x84x84 grayscale, max of last two repeated frames"},
            {"frame_skip", 4}, {"sticky_action_probability", .25}, {"noop_max", 30},
            {"fire_reset", true}, {"max_episode_raw_frames", 108000},
            {"terminal_on_life_loss", false}, {"training_reward_clipping", true},
            {"evaluation_reward_clipping", false}, {"time_limit_bootstrap", true}};
}

struct NetworkImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear hidden{nullptr}, actor{nullptr}, critic{nullptr};
    explicit NetworkImpl(int actions, bool initialize = true) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 32, 8).stride(4)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)));
        hidden = register_module("hidden", torch::nn::Linear(64 * 7 * 7, 512));
        actor = register_module("actor", torch::nn::Linear(512, actions));
        critic = register_module("critic", torch::nn::Linear(512, 1));
        if (!initialize) return;
        torch::NoGradGuard guard;
        for (auto& parameter : named_parameters()) {
            if (parameter.key().find("bias") != std::string::npos) parameter.value().zero_();
            else torch::nn::init::orthogonal_(parameter.value(), std::sqrt(2.0));
        }
        torch::nn::init::orthogonal_(actor->weight, .01);
        torch::nn::init::orthogonal_(critic->weight, 1.0);
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        auto x = input.to(torch::kFloat32).div(255.0);
        x = torch::relu(conv1->forward(x)); x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = torch::relu(hidden->forward(x.flatten(1)));
        return {actor->forward(x), critic->forward(x).squeeze(1)};
    }
};
TORCH_MODULE(Network);
torch::Tensor observations(const std::uint8_t* bytes, std::int64_t count,
                           const torch::Device& device) {
    return torch::from_blob(const_cast<std::uint8_t*>(bytes), {count, 4, 84, 84},
                            torch::TensorOptions().dtype(torch::kUInt8)).to(device);
}
void save_checkpoint(Network& model, torch::optim::Adam& optimizer,
                     int actions, std::int64_t steps, const fs::path& path) {
    torch::serialize::OutputArchive archive, weights, state;
    archive.write("action_count", torch::tensor(actions, torch::kInt64));
    archive.write("steps", torch::tensor(steps, torch::kInt64));
    model->save(weights); optimizer.save(state);
    archive.write("model", weights); archive.write("optimizer", state);
    auto temporary = path.string() + ".tmp";
    archive.save_to(temporary); replace_file_atomically(temporary, path);
}

void validate_resumed_state(Network& model, const torch::optim::Adam& optimizer) {
    const auto parameters = model->parameters();
    if (optimizer.state().size() != parameters.size())
        throw std::runtime_error("Resume checkpoint lacks complete Adam state");
    for (const auto& group : optimizer.param_groups()) {
        const auto& settings = static_cast<const torch::optim::AdamOptions&>(group.options());
        if (settings.lr() != 2.5e-4 || settings.eps() != 1e-5 ||
            settings.betas() != std::make_tuple(.9, .999) || settings.weight_decay() != 0 || settings.amsgrad())
            throw std::runtime_error("Resume checkpoint has incompatible Adam settings");
    }
    for (const auto& parameter : parameters) {
        if (!parameter.is_cuda() || !torch::isfinite(parameter).all().item<bool>())
            throw std::runtime_error("Resume checkpoint has invalid model parameters");
        auto entry = optimizer.state().find(parameter.unsafeGetTensorImpl());
        if (entry == optimizer.state().end()) throw std::runtime_error("Resume Adam parameter mapping is incomplete");
        const auto* state = dynamic_cast<const torch::optim::AdamParamState*>(entry->second.get());
        if (!state || state->step() <= 0) throw std::runtime_error("Resume checkpoint has invalid Adam update count");
        auto valid_moment = [&](const torch::Tensor& moment) {
            return moment.defined() && moment.device() == parameter.device() &&
                moment.sizes() == parameter.sizes() && moment.scalar_type() == parameter.scalar_type() &&
                torch::isfinite(moment).all().item<bool>();
        };
        if (!valid_moment(state->exp_avg()) || !valid_moment(state->exp_avg_sq()) ||
            !state->exp_avg_sq().ge(0).all().item<bool>())
            throw std::runtime_error("Resume checkpoint has invalid Adam moments");
    }
}

// Each ALE instance stays on one persistent CPU worker. CUDA inference and
// gradient updates run only on the main thread, in batches across environments.
class Workers {
    std::mutex mutex_; std::condition_variable ready_, complete_;
    std::vector<std::thread> threads_; std::function<void(int)> job_;
    std::exception_ptr error_; std::size_t generation_ = 0, pending_ = 0;
    bool quitting_ = false;
public:
    explicit Workers(int count) {
        for (int index = 0; index < count; ++index) threads_.emplace_back([this, index] {
            std::size_t seen = 0;
            for (;;) {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return quitting_ || generation_ != seen; });
                if (quitting_) return;
                seen = generation_; auto work = job_; lock.unlock();
                try { work(index); } catch (...) {
                    std::lock_guard error_lock(mutex_); if (!error_) error_ = std::current_exception();
                }
                lock.lock(); if (--pending_ == 0) complete_.notify_one();
            }
        });
    }
    ~Workers() {
        { std::lock_guard lock(mutex_); quitting_ = true; }
        ready_.notify_all(); for (auto& thread : threads_) thread.join();
    }
    void run(std::function<void(int)> work) {
        std::unique_lock lock(mutex_); job_ = std::move(work); error_ = nullptr;
        pending_ = threads_.size(); ++generation_; ready_.notify_all();
        complete_.wait(lock, [&] { return pending_ == 0; });
        if (error_) std::rethrow_exception(error_);
    }
};
} // namespace

struct Policy::Impl {
    torch::Device device; Network model{nullptr}; int actions = 0;
    Impl(const std::string& checkpoint, const std::string& target) : device(target) {
        torch::serialize::InputArchive archive, weights;
        archive.load_from(checkpoint, device);
        torch::Tensor count; archive.read("action_count", count);
        actions = static_cast<int>(count.item<std::int64_t>());
        if (actions <= 0 || actions > 18) throw std::runtime_error("Invalid checkpoint action count");
        model = Network(actions, false); archive.read("model", weights); model->load(weights);
        model->to(device); model->eval();
    }
};
Policy::Policy(const std::string& path, const std::string& device)
    : impl_(std::make_unique<Impl>(path, device)) {}
Policy::~Policy() = default;
Policy::Policy(Policy&&) noexcept = default;
Policy& Policy::operator=(Policy&&) noexcept = default;
int Policy::action_count() const { return impl_->actions; }
int Policy::predict(const std::array<std::uint8_t, pixels>& observation, bool deterministic) {
    torch::NoGradGuard guard;
    auto [logits, values] = impl_->model->forward(observations(observation.data(), 1, impl_->device));
    return deterministic ? logits.argmax(1).item<int>()
                         : torch::multinomial(logits.softmax(1), 1).item<int>();
}

AdvantageBatch compute_gae(const std::vector<float>& rewards, const std::vector<float>& values,
                          const std::vector<std::uint8_t>& ended, const std::vector<float>& last_values,
                          int steps, int environments, float gamma, float lambda) {
    const auto size = static_cast<std::size_t>(steps) * environments;
    if (steps <= 0 || environments <= 0 || rewards.size() != size || values.size() != size ||
        ended.size() != size || last_values.size() != static_cast<std::size_t>(environments))
        throw std::invalid_argument("Invalid GAE dimensions");
    AdvantageBatch result{std::vector<float>(size), std::vector<float>(size)};
    std::vector<float> advantage(environments, 0);
    for (int t = steps - 1; t >= 0; --t) for (int e = 0; e < environments; ++e) {
        const auto i = static_cast<std::size_t>(t) * environments + e;
        const float continuation = ended[i] ? 0.f : 1.f;
        const float next = t == steps - 1 ? last_values[e] : values[i + environments];
        const float delta = rewards[i] + gamma * next * continuation - values[i];
        advantage[e] = delta + gamma * lambda * continuation * advantage[e];
        result.advantages[i] = advantage[e]; result.returns[i] = advantage[e] + values[i];
    }
    return result;
}
bool evidence_passes(int episodes, int wins, double mean, int truncated) {
    return episodes == 20 && wins >= 16 && wins <= episodes && mean > 0 &&
           std::isfinite(mean) && truncated == 0;
}

void validate_resume_config(const json& source, const json& expected) {
    if (!source.is_object() || !expected.is_object())
        throw std::invalid_argument("Resume configuration must be a JSON object");
    for (const char* key : {"algorithm", "language", "torch_version", "training_action_sampling",
             "rom_sha256", "action_count", "preprocessing", "network", "gamma", "gae_lambda",
             "clip_range", "entropy_coefficient", "learning_rate", "value_coefficient", "max_gradient_norm"}) {
        if (!source.contains(key) || !expected.contains(key) || source.at(key) != expected.at(key))
            throw std::invalid_argument(std::string("Incompatible resume configuration: ") + key);
    }
    if (expected.contains("game_id")) {
        if (source.contains("game_id") ? source.at("game_id") != expected.at("game_id")
                                       : expected.at("game_id") != "pong")
            throw std::invalid_argument("Incompatible resume configuration: game_id");
    }
}

void validate_evaluation_config(const json& source, const GameSpec& game,
                                const std::string& rom_hash, int action_count) {
    if (game.id.empty() || game.rom_sha256 != rom_hash || game.expected_actions != action_count)
        throw std::invalid_argument("Selected ROM does not match the registered game");
    if (!source.is_object() || !source.contains("rom_sha256") || source.at("rom_sha256") != rom_hash)
        throw std::invalid_argument("Evaluation ROM SHA256 differs from checkpoint training configuration");
    if (!source.contains("action_count") || !source.at("action_count").is_number_integer() ||
        source.at("action_count") != action_count)
        throw std::invalid_argument("Evaluation action count differs from checkpoint training configuration");
    if (!source.contains("network") || source.at("network") != network_description)
        throw std::invalid_argument("Evaluation network differs from checkpoint training configuration");
    if (!source.contains("preprocessing") || source.at("preprocessing") != preprocessing())
        throw std::invalid_argument("Evaluation preprocessing differs from checkpoint training configuration");
    // Existing Pong artifacts predate the game registry. Their ROM hash remains
    // authoritative; only those artifacts may omit an explicit game identity.
    if (source.contains("game_id") ? source.at("game_id") != game.id : game.id != "pong")
        throw std::invalid_argument("Evaluation game differs from checkpoint training configuration");
    if (source.contains("game_spec")) {
        const auto& recorded = source.at("game_spec");
        if (!recorded.is_object() || !recorded.contains("id") || recorded.at("id") != game.id ||
            !recorded.contains("rom_sha256") || recorded.at("rom_sha256") != rom_hash ||
            !recorded.contains("expected_actions") || recorded.at("expected_actions") != action_count)
            throw std::invalid_argument("Checkpoint game specification does not match the selected ROM");
    }
}

void validate_transfer_config(const json& source, const GameSpec& source_game, const json& target) {
    validate_evaluation_config(source, source_game, source_game.rom_sha256, source_game.expected_actions);
    if (!target.is_object()) throw std::invalid_argument("Transfer target configuration must be a JSON object");
    for (const char* key : {"algorithm", "language", "torch_version", "network", "preprocessing"}) {
        if (!source.contains(key) || !target.contains(key) || source.at(key) != target.at(key))
            throw std::invalid_argument(std::string("Incompatible feature-transfer configuration: ") + key);
    }
}

json training_origin_metadata(const json& config) {
    if (!config.is_object()) throw std::invalid_argument("Training configuration must be an object");
    if (config.contains("training_origin")) {
        const auto& origin = config.at("training_origin");
        if (!origin.is_object() || !origin.contains("initialization_mode") ||
            !origin.at("initialization_mode").is_string() || !origin.contains("transfer"))
            throw std::invalid_argument("Invalid training origin metadata");
        return origin;
    }
    const bool continued = config.value("previous_trained_steps", std::int64_t{0}) > 0;
    return {{"initialization_mode", continued ? json("legacy_origin_not_recorded")
            : config.value("initialization_mode", json("cold_start"))},
        {"seed", continued ? json(nullptr) : config.value("seed", json(nullptr))},
        {"initial_head_fingerprints", continued ? json(nullptr)
            : config.value("initial_head_fingerprints", json(nullptr))},
        {"transfer", config.value("transfer", json(nullptr))}};
}

nlohmann::json copy_feature_parameters(torch::nn::Module& target,
                                       const std::string& checkpoint_path, int source_actions) {
    const auto checkpoint_hash = sha256(checkpoint_path);
    const auto heads_before = head_fingerprints(target);
    torch::serialize::InputArchive archive, weights;
    archive.load_from(checkpoint_path, torch::Device(torch::kCPU));
    torch::Tensor actions, steps;
    archive.read("action_count", actions); archive.read("steps", steps);
    if (actions.numel() != 1 || steps.numel() != 1 || actions.scalar_type() != torch::kInt64 ||
        steps.scalar_type() != torch::kInt64 || source_actions <= 0 || source_actions > 18 ||
        actions.item<std::int64_t>() != source_actions || steps.item<std::int64_t>() <= 0)
        throw std::invalid_argument("Invalid feature-transfer checkpoint action/step metadata");
    archive.read("model", weights);
    struct Copy { std::string name; torch::Tensor source, target; };
    std::vector<Copy> copies;
    const auto parameters = target.named_parameters();
    // Read archives directly: constructing a source Network would consume RNG
    // and break the matched fresh-head/sampling initialization with scratch runs.
    for (const char* module_name : {"conv1", "conv2", "conv3", "hidden"}) {
        torch::serialize::InputArchive module;
        weights.read(module_name, module);
        for (const char* parameter_name : {"weight", "bias"}) {
            const std::string name = std::string(module_name) + "." + parameter_name;
            torch::Tensor source; module.read(parameter_name, source);
            const auto* destination = parameters.find(name);
            if (!destination || !source.defined() || source.scalar_type() != torch::kFloat32 ||
                destination->scalar_type() != source.scalar_type() || source.sizes() != destination->sizes() ||
                !torch::isfinite(source).all().item<bool>())
                throw std::invalid_argument("Invalid feature-transfer tensor: " + name);
            copies.push_back({name, source, *destination});
        }
    }
    json tensors = json::array();
    {
        torch::NoGradGuard guard;
        for (auto& copy : copies) {
            copy.target.copy_(copy.source);
            const auto source_fingerprint = tensor_fingerprint(copy.source);
            const auto target_fingerprint = tensor_fingerprint(copy.target);
            if (source_fingerprint != target_fingerprint)
                throw std::runtime_error("Feature-transfer copy verification failed: " + copy.name);
            tensors.push_back({{"name", copy.name}, {"source", source_fingerprint}, {"target", target_fingerprint}});
        }
    }
    if (head_fingerprints(target) != heads_before)
        throw std::runtime_error("Feature transfer unexpectedly changed a target head");
    if (sha256(checkpoint_path) != checkpoint_hash)
        throw std::runtime_error("Transfer checkpoint changed while copying; use an immutable source");
    return {{"source_checkpoint_sha256", checkpoint_hash},
        {"source_pretraining_decisions", steps.item<std::int64_t>()},
        {"tensor_count", copies.size()}, {"tensors", tensors}, {"heads_unchanged", true},
        {"head_fingerprints", heads_before}};
}

nlohmann::json run_evaluation(const EvalOptions& options) {
    if (options.episodes <= 0 || options.max_seconds <= 0 || options.game.id.empty() ||
        options.game.criterion.episodes <= 0)
        throw std::invalid_argument("Evaluation requires a registered game and positive episode/time budgets");
    if (!options.output_path.empty() && fs::exists(options.output_path))
        throw std::runtime_error("Evaluation output already exists; preserve prior evidence");
    Signals signals;
    torch::set_num_threads(1);
    const auto checkpoint_hash = sha256(options.checkpoint_path), rom_hash = sha256(options.rom_path);
    const auto config_path = fs::path(options.checkpoint_path).parent_path() / "config.json";
    const auto config_hash = sha256(config_path.string());
    std::ifstream config_stream(config_path);
    const auto source_config = json::parse(config_stream);
    validate_evaluation_config(source_config, options.game, rom_hash, options.game.expected_actions);
    Policy policy(options.checkpoint_path);
    if (sha256(options.checkpoint_path) != checkpoint_hash || sha256(config_path.string()) != config_hash)
        throw std::runtime_error("Checkpoint or configuration changed while loading; evaluate immutable artifacts");
    AtariEnv environment(options.rom_path, options.seed);
    if (policy.action_count() != environment.actions() || environment.actions() != options.game.expected_actions)
        throw std::runtime_error("Registered game/ROM/checkpoint action mismatch");
    auto start = Clock::now(); json episodes = json::array();
    auto save = [&] {
        int positive_games = 0, truncated = 0, primary_positive = 0, primary_truncated = 0;
        double sum = 0, primary_sum = 0;
        for (std::size_t i = 0; i < episodes.size(); ++i) {
            const auto& row = episodes[i]; const double score = row["raw_return"];
            sum += score; positive_games += row["positive_game"].get<bool>(); truncated += row["truncated"].get<bool>();
            if (i < static_cast<std::size_t>(options.game.criterion.episodes)) {
                          primary_sum += score; primary_positive += row["positive_game"].get<bool>();
                          primary_truncated += row["truncated"].get<bool>(); }
        }
        const int primary_episodes = std::min(options.game.criterion.episodes, static_cast<int>(episodes.size()));
        const double primary_mean = primary_episodes ? primary_sum / primary_episodes : 0;
        const bool complete = episodes.size() == static_cast<std::size_t>(options.episodes);
        const bool passed = evaluate_game_criterion(options.game, primary_episodes,
                                                    primary_positive, primary_mean, primary_truncated);
        json result = {{"status", complete ? "complete" : "incomplete"},
            {"game_id", options.game.id}, {"game_title", options.game.title}, {"game_spec", game_spec_json(options.game)},
            {"policy", "deterministic_ppo"}, {"checkpoint", options.checkpoint_path},
            {"checkpoint_sha256", checkpoint_hash}, {"rom", options.rom_path}, {"rom_sha256", rom_hash},
            {"checkpoint_config", config_path.string()}, {"checkpoint_config_sha256", config_hash},
            {"preprocessing", preprocessing()}, {"seed_start", options.seed},
            {"episodes_requested", options.episodes}, {"episodes_completed", episodes.size()},
            {"positive_games", positive_games}, {"wins", options.game.id == "pong" ? json(positive_games) : json(nullptr)},
            {"truncated_games", truncated}, {"passed", passed},
            {"verdict", passed ? "target_met" : primary_episodes == options.game.criterion.episodes ? "target_not_met" : "insufficient_evidence"},
            {"criterion", options.game.criterion.description},
            {"criterion_evidence", {{"episodes_required", options.game.criterion.episodes},
                {"episodes_completed", primary_episodes}, {"positive_games", primary_positive},
                {"mean_return", primary_episodes ? json(primary_mean) : json(nullptr)},
                {"truncated_games", primary_truncated}}},
            {"elapsed_seconds", elapsed(start)}, {"episodes", episodes}};
        result["mean_return"] = episodes.empty() ? json(nullptr) : json(sum / episodes.size());
        atomic_json(options.output_path, result); return result;
    };
    save();
    for (int game = 0; game < options.episodes; ++game) {
        environment.reset(options.seed + game); std::int64_t decisions = 0;
        for (;;) {
            const auto output_directory = fs::path(options.output_path).parent_path();
            if (elapsed(start) >= options.max_seconds || interrupted ||
                (!options.stop_path.empty() && fs::exists(options.stop_path)) ||
                (!output_directory.empty() && fs::exists(output_directory / "STOP"))) return save();
            const auto step = environment.step(policy.predict(environment.observation())); ++decisions;
            if (step.done || step.truncated) {
                const bool positive_game = step.done && !step.truncated && step.episode_return > 0;
                episodes.push_back({{"episode", game + 1}, {"seed", options.seed + game},
                    {"raw_return", step.episode_return}, {"raw_frames", step.episode_raw_frames},
                    {"agent_decisions", decisions}, {"terminated", step.done}, {"truncated", step.truncated},
                    {"positive_game", positive_game}, {"won", options.game.id == "pong" ? json(positive_game) : json(nullptr)}});
                save(); break;
            }
        }
    }
    return save();
}

nlohmann::json run_training(const TrainOptions& options) {
    if (!options.resume_checkpoint_path.empty() && !options.transfer_checkpoint_path.empty())
        throw std::invalid_argument("Resume and feature transfer are mutually exclusive");
    if (options.num_envs < 1 || options.num_envs > 64 || options.rollout_steps < 1 || options.epochs < 1 ||
        options.minibatch_size < 2 || options.max_steps < options.num_envs || options.max_seconds < 1 ||
        options.torch_threads < 1 || options.run_dir.empty() || options.game.id.empty() ||
        options.game.expected_actions <= 0 || options.game.criterion.episodes <= 0)
        throw std::invalid_argument("Invalid training configuration or missing registered game");
    const auto rom_hash = sha256(options.rom_path);
    if (rom_hash != options.game.rom_sha256)
        throw std::invalid_argument("Training ROM SHA256 differs from the registered game");
    if (!torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable; training requires the container GPU");
    const fs::path directory(options.run_dir);
    if (fs::exists(directory) && (!fs::is_directory(directory) || !fs::is_empty(directory)))
        throw std::runtime_error("Run directory must be new or empty; existing experiments are preserved");
    fs::create_directories(directory);
    Signals signals; torch::set_num_threads(options.torch_threads); torch::manual_seed(options.seed);
    const torch::Device device(torch::kCUDA);
    const int n = options.num_envs;
    Workers workers(n); std::vector<std::unique_ptr<AtariEnv>> environments(n);
    workers.run([&](int i) { environments[i] = std::make_unique<AtariEnv>(options.rom_path, options.seed + i); });
    const int action_count = environments.front()->actions();
    if (action_count != options.game.expected_actions)
        throw std::invalid_argument("Training ROM action count differs from the registered game");
    json config = {{"algorithm", "PPO"}, {"language", "C++20"}, {"device", "cuda"},
        {"game_id", options.game.id}, {"game_title", options.game.title}, {"game_spec", game_spec_json(options.game)},
        {"torch_version", TORCH_VERSION}, {"training_action_sampling", "categorical"},
        {"rom", options.rom_path}, {"rom_sha256", rom_hash},
        {"seed", options.seed}, {"max_agent_decisions", options.max_steps},
        {"max_seconds", options.max_seconds}, {"environments", n}, {"rollout_steps", options.rollout_steps},
        {"epochs", options.epochs}, {"minibatch_size", options.minibatch_size}, {"gamma", .99}, {"gae_lambda", .95},
        {"clip_range", .1}, {"entropy_coefficient", .01}, {"learning_rate", 2.5e-4},
        {"value_coefficient", .5}, {"max_gradient_norm", .5}, {"action_count", action_count},
        {"preprocessing", preprocessing()}, {"evaluation_seed", options.game.final_eval_seed},
        {"evaluation_episodes", options.game.criterion.episodes},
        {"network", network_description}};
    Network model(action_count, options.resume_checkpoint_path.empty()); model->to(device);
    torch::serialize::InputArchive resume_archive;
    std::int64_t previous_trained_steps = 0;
    std::string parent_checkpoint, parent_checkpoint_hash, parent_config_hash;
    json resume_config;
    if (!options.resume_checkpoint_path.empty()) {
        const auto checkpoint = fs::canonical(options.resume_checkpoint_path);
        if (!fs::is_regular_file(checkpoint)) throw std::invalid_argument("Resume checkpoint must be a regular file");
        const auto source_config_path = checkpoint.parent_path() / "config.json";
        parent_checkpoint = checkpoint.string();
        parent_checkpoint_hash = sha256(parent_checkpoint);
        parent_config_hash = sha256(source_config_path.string());
        std::ifstream source_stream(source_config_path);
        resume_config = json::parse(source_stream);
        validate_resume_config(resume_config, config);
        validate_evaluation_config(resume_config, options.game, rom_hash, action_count);
        resume_archive.load_from(parent_checkpoint, device);
        torch::Tensor source_actions, source_steps;
        resume_archive.read("action_count", source_actions); resume_archive.read("steps", source_steps);
        if (source_actions.numel() != 1 || source_steps.numel() != 1 ||
            source_actions.scalar_type() != torch::kInt64 || source_steps.scalar_type() != torch::kInt64 ||
            source_actions.item<std::int64_t>() != action_count)
            throw std::invalid_argument("Resume checkpoint has incompatible action count or step metadata");
        previous_trained_steps = source_steps.item<std::int64_t>();
        if (previous_trained_steps <= 0 ||
            previous_trained_steps > std::numeric_limits<std::int64_t>::max() - options.max_steps)
            throw std::invalid_argument("Resume checkpoint step count is invalid or cumulative budget would overflow");
        torch::serialize::InputArchive weights;
        resume_archive.read("model", weights); model->load(weights);
    }
    json transfer = nullptr;
    if (!options.transfer_checkpoint_path.empty()) {
        const auto checkpoint = fs::canonical(options.transfer_checkpoint_path);
        if (!fs::is_regular_file(checkpoint)) throw std::invalid_argument("Transfer checkpoint must be a regular file");
        const auto source_config_path = checkpoint.parent_path() / "config.json";
        const auto checkpoint_hash = sha256(checkpoint.string());
        const auto config_hash = sha256(source_config_path.string());
        std::ifstream source_stream(source_config_path);
        const auto source_config = json::parse(source_stream);
        validate_transfer_config(source_config, options.transfer_game, config);
        if (!source_config.contains("rom") || !source_config.at("rom").is_string())
            throw std::invalid_argument("Transfer source configuration lacks its ROM path");
        const auto source_rom = source_config.at("rom").get<std::string>();
        const auto source_rom_hash = sha256(source_rom);
        if (source_rom_hash != options.transfer_game.rom_sha256)
            throw std::invalid_argument("Transfer source ROM SHA256 differs from the registered game");
        transfer = copy_feature_parameters(*model, checkpoint.string(), options.transfer_game.expected_actions);
        if (transfer.at("source_checkpoint_sha256") != checkpoint_hash ||
            sha256(checkpoint.string()) != checkpoint_hash || sha256(source_config_path.string()) != config_hash ||
            sha256(source_rom) != source_rom_hash || sha256(options.rom_path) != rom_hash)
            throw std::runtime_error("Transfer source or ROM changed while copying; use immutable artifacts");
        transfer["source_checkpoint"] = checkpoint.string();
        transfer["source_config"] = source_config_path.string();
        transfer["source_config_sha256"] = config_hash;
        transfer["source_game_id"] = options.transfer_game.id;
        transfer["source_game_spec"] = game_spec_json(options.transfer_game);
        transfer["source_rom_sha256"] = source_rom_hash;
        transfer["target_step_count_at_initialization"] = 0;
        transfer["optimizer_restored"] = false;
    }
    // Construct Adam after loading model parameters so state maps to the actual
    // restored tensors, then restore its moments and update counters on CUDA.
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    if (!parent_checkpoint.empty()) {
        torch::serialize::InputArchive state;
        resume_archive.read("optimizer", state); optimizer.load(state);
        validate_resumed_state(model, optimizer);
        if (sha256(parent_checkpoint) != parent_checkpoint_hash ||
            sha256((fs::path(parent_checkpoint).parent_path() / "config.json").string()) != parent_config_hash)
            throw std::runtime_error("Resume source changed while loading; use an immutable checkpoint and configuration");
    }
    if (parent_checkpoint.empty() && !optimizer.state().empty())
        throw std::runtime_error("Fresh target Adam unexpectedly contains optimizer state");
    config["budget_scope"] = "additional_agent_decisions_in_this_run";
    config["checkpoint_steps_scope"] = "cumulative_agent_decisions";
    config["previous_trained_steps"] = previous_trained_steps;
    config["initialization_mode"] = !parent_checkpoint.empty() ? "model_and_adam_with_fresh_environments_and_rng"
        : transfer.is_null() ? "cold_start" : "feature_transfer_fresh_heads_and_adam";
    config["continuation"] = config["initialization_mode"];
    config["initial_head_fingerprints"] = head_fingerprints(*model);
    config["initial_adam_state_entries"] = optimizer.state().size();
    config["tensor_fingerprint_format"] = "SHA256 of contiguous CPU tensor bytes; shape and dtype recorded separately";
    config["transfer"] = transfer;
    config["parent_checkpoint"] = parent_checkpoint.empty() ? json(nullptr) : json(parent_checkpoint);
    config["parent_checkpoint_sha256"] = parent_checkpoint.empty() ? json(nullptr) : json(parent_checkpoint_hash);
    config["parent_config"] = parent_checkpoint.empty() ? json(nullptr)
        : json((fs::path(parent_checkpoint).parent_path() / "config.json").string());
    config["parent_config_sha256"] = parent_checkpoint.empty() ? json(nullptr) : json(parent_config_hash);
    config["training_origin"] = training_origin_metadata(parent_checkpoint.empty() ? config : resume_config);
    // A continuation restores trained heads and Adam. Its original feature-copy
    // evidence remains visible without claiming that this continuation reset them.
    if (!parent_checkpoint.empty()) config["transfer"] = config["training_origin"].at("transfer");
    atomic_json(directory / "config.json", config);
    std::ofstream episode_log(directory / "episodes.jsonl", std::ios::app);
    if (!episode_log) throw std::runtime_error("Cannot create episode log");
    const auto start = Clock::now(); auto last_save = start;
    std::int64_t total_steps = 0, completed_episodes = 0, updates = 0, checkpoint_steps = 0;
    double recent_sum = 0, training_seconds = -1; int recent_count = 0; json progress;
    std::string stop_reason;
    auto stopped = [&] { return interrupted || fs::exists(directory / "STOP"); };
    auto report = [&](const char* status, double loss) {
        const double total_seconds = elapsed(start);
        const double measured_training_seconds = training_seconds >= 0 ? training_seconds : total_seconds;
        if (std::string(status) == "stopped") stop_reason = interrupted ? "signal" : "STOP";
        progress = {{"status", status}, {"agent_decisions", total_steps}, {"steps", total_steps},
            {"game_id", options.game.id}, {"game_title", options.game.title},
            {"target_steps", options.max_steps},
            {"previous_trained_steps", previous_trained_steps},
            {"cumulative_agent_decisions", previous_trained_steps + total_steps},
            {"elapsed_seconds", measured_training_seconds}, {"training_elapsed_seconds", measured_training_seconds},
            {"total_elapsed_seconds", total_seconds},
            {"decisions_per_second", total_steps / std::max(.001, measured_training_seconds)},
            {"episodes", completed_episodes}, {"updates", updates}, {"loss", loss}, {"device", "cuda"},
            {"checkpoint_steps", previous_trained_steps + checkpoint_steps}, {"checkpoint_run_steps", checkpoint_steps},
            {"run_dir", directory.string()}, {"latest_checkpoint", (directory / "latest.pt").string()}};
        progress["recent_mean_raw_return"] = recent_count ? json(recent_sum / recent_count) : json(nullptr);
        progress["recent_mean_sample_count"] = recent_count;
        progress["fps"] = progress["decisions_per_second"];
        progress["recent_mean_reward"] = progress["recent_mean_raw_return"];
        progress["stop_reason"] = stop_reason.empty() ? json(nullptr) : json(stop_reason);
        atomic_json(directory / "progress.json", progress);
    };
    report("training", 0);
    try {
        while (options.max_steps - total_steps >= n && elapsed(start) < options.max_seconds && !stopped()) {
            const int horizon = static_cast<int>(std::min<std::int64_t>(options.rollout_steps, (options.max_steps - total_steps) / n));
            const int capacity = horizon * n;
            std::vector<std::uint8_t> frames(static_cast<std::size_t>(capacity) * pixels), ended(capacity);
            std::vector<float> rewards(capacity), values(capacity), old_logprob(capacity);
            std::vector<std::int64_t> actions(capacity);
            std::vector<Step> transitions(n); std::vector<Observation> time_limit_observations(n);
            int steps = 0;
            for (int t = 0; t < horizon; ++t) {
                auto* current = frames.data() + static_cast<std::size_t>(t * n) * pixels;
                for (int e = 0; e < n; ++e) std::memcpy(current + e * pixels, environments[e]->observation().data(), pixels);
                {
                    torch::NoGradGuard guard;
                    auto [logits, value] = model->forward(observations(current, n, device));
                    auto action = torch::multinomial(logits.softmax(1), 1);
                    auto logprob = logits.log_softmax(1).gather(1, action).squeeze(1).to(torch::kCPU).contiguous();
                    auto cpu_actions = action.squeeze(1).to(torch::kCPU).contiguous();
                    auto cpu_values = value.to(torch::kCPU).contiguous();
                    std::memcpy(actions.data() + t * n, cpu_actions.data_ptr<std::int64_t>(), n * sizeof(std::int64_t));
                    std::memcpy(values.data() + t * n, cpu_values.data_ptr<float>(), n * sizeof(float));
                    std::memcpy(old_logprob.data() + t * n, logprob.data_ptr<float>(), n * sizeof(float));
                }
                workers.run([&](int e) {
                    transitions[e] = environments[e]->step(static_cast<int>(actions[t * n + e]));
                    if (transitions[e].truncated && !transitions[e].done) time_limit_observations[e] = environments[e]->observation();
                    if (transitions[e].done || transitions[e].truncated) environments[e]->reset();
                });
                for (int e = 0; e < n; ++e) {
                    const auto& transition = transitions[e]; const int index = t * n + e;
                    rewards[index] = std::clamp(transition.reward, -1.f, 1.f);
                    ended[index] = transition.done || transition.truncated;
                    if (transition.truncated && !transition.done) {
                        torch::NoGradGuard guard;
                        rewards[index] += .99f * model->forward(observations(time_limit_observations[e].data(), 1, device)).second.item<float>();
                    }
                    if (ended[index]) {
                        ++completed_episodes; recent_sum += transition.episode_return; ++recent_count;
                        episode_log << json({{"episode", completed_episodes}, {"environment", e},
                            {"game_id", options.game.id},
                            {"agent_decisions", total_steps + n}, {"raw_return", transition.episode_return},
                            {"cumulative_agent_decisions", previous_trained_steps + total_steps + n},
                            {"raw_frames", transition.episode_raw_frames}, {"truncated", transition.truncated}}).dump() << '\n';
                    }
                }
                total_steps += n; ++steps;
                if (stopped() || elapsed(start) >= options.max_seconds) break;
            }
            const int batch = steps * n;
            rewards.resize(batch); values.resize(batch); ended.resize(batch);
            std::vector<std::uint8_t> last_frames(static_cast<std::size_t>(n) * pixels);
            for (int e = 0; e < n; ++e) std::memcpy(last_frames.data() + e * pixels, environments[e]->observation().data(), pixels);
            std::vector<float> final_values(n);
            { torch::NoGradGuard guard; auto value = model->forward(observations(last_frames.data(), n, device)).second.to(torch::kCPU).contiguous();
              std::memcpy(final_values.data(), value.data_ptr<float>(), n * sizeof(float)); }
            auto estimates = compute_gae(rewards, values, ended, final_values, steps, n);
            auto all_observations = observations(frames.data(), batch, device);
            auto float_tensor = [&](std::vector<float>& data) { return torch::from_blob(data.data(), {batch}, torch::kFloat32).to(device); };
            auto advantage = float_tensor(estimates.advantages);
            advantage = (advantage - advantage.mean()) / (advantage.std(false) + 1e-8);
            auto returns = float_tensor(estimates.returns), previous_logprob = float_tensor(old_logprob);
            auto all_actions = torch::from_blob(actions.data(), {batch}, torch::kInt64).to(device);
            double total_loss = 0; int minibatches = 0;
            for (int epoch = 0; epoch < options.epochs; ++epoch) {
                auto permutation = torch::randperm(batch, torch::TensorOptions().dtype(torch::kInt64).device(device));
                for (int offset = 0; offset < batch; offset += options.minibatch_size) {
                    auto indices = permutation.narrow(0, offset, std::min(options.minibatch_size, batch - offset));
                    auto [logits, value] = model->forward(all_observations.index_select(0, indices));
                    auto log_probs = logits.log_softmax(1);
                    auto selected = log_probs.gather(1, all_actions.index_select(0, indices).unsqueeze(1)).squeeze(1);
                    auto ratio = (selected - previous_logprob.index_select(0, indices)).exp();
                    auto advantages = advantage.index_select(0, indices);
                    auto policy_loss = -torch::minimum(ratio * advantages, ratio.clamp(.9, 1.1) * advantages).mean();
                    auto value_loss = (value - returns.index_select(0, indices)).pow(2).mean();
                    auto entropy = -(log_probs.exp() * log_probs).sum(1).mean();
                    auto loss = policy_loss + .5 * value_loss - .01 * entropy;
                    const double scalar_loss = loss.item<double>();
                    if (!std::isfinite(scalar_loss)) throw std::runtime_error("Non-finite PPO loss; training stopped");
                    optimizer.zero_grad(); loss.backward();
                    torch::nn::utils::clip_grad_norm_(model->parameters(), .5); optimizer.step();
                    total_loss += scalar_loss; ++minibatches;
                }
            }
            ++updates; episode_log.flush();
            if (updates == 1 || elapsed(last_save) >= 60 || stopped()) {
                save_checkpoint(model, optimizer, action_count, previous_trained_steps + total_steps, directory / "latest.pt");
                checkpoint_steps = total_steps; last_save = Clock::now();
            }
            report("training", total_loss / std::max(1, minibatches));
            std::cout << progress.dump() << std::endl;
            if (recent_count >= 100) { recent_sum = 0; recent_count = 0; }
        }
        training_seconds = elapsed(start);
        stop_reason = interrupted ? "signal" : fs::exists(directory / "STOP") ? "STOP"
            : options.max_steps - total_steps < n ? "target_steps" : "time_budget";
        save_checkpoint(model, optimizer, action_count, previous_trained_steps + total_steps, directory / "final.pt");
        save_checkpoint(model, optimizer, action_count, previous_trained_steps + total_steps, directory / "latest.pt");
        checkpoint_steps = total_steps;
        if (stopped()) { report("stopped", 0); return progress; }
        report("evaluating", 0);
        EvalOptions evaluation; evaluation.game = options.game; evaluation.rom_path = options.rom_path;
        evaluation.checkpoint_path = (directory / "final.pt").string();
        evaluation.output_path = (directory / "evaluation.json").string(); evaluation.seed = options.game.final_eval_seed;
        evaluation.episodes = options.game.criterion.episodes;
        auto evidence = run_evaluation(evaluation);
        report(stopped() ? "stopped" : evidence["status"] == "incomplete" ? "evaluation_incomplete" : "complete", 0);
        progress["evaluation"] = evidence;
        atomic_json(directory / "progress.json", progress); return progress;
    } catch (const std::exception& error) {
        progress["status"] = "failed"; progress["error"] = error.what();
        atomic_json(directory / "progress.json", progress); throw;
    }
}
} // namespace atari
