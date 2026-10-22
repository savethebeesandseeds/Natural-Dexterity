#include "atari/shared.hpp"
#include "atari/atomic_io.hpp"
#include "atari/environment.hpp"
#include "atari/learning.hpp"
#include <torch/torch.h>
#include <torch/version.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
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
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr std::size_t pixels = kObservationBytes;
constexpr std::array<std::uint64_t, 2> evaluation_seeds{4'000'100, 4'100'100};
constexpr std::array<std::uint64_t, 3> extension_evaluation_seeds{5'000'100, 5'100'100, 5'200'100};
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
struct Signals {
    using Handler = void (*)(int);
    Handler previous_int = std::signal(SIGINT, on_signal);
    Handler previous_term = std::signal(SIGTERM, on_signal);
    Signals() { interrupted = 0; }
    ~Signals() { std::signal(SIGINT, previous_int); std::signal(SIGTERM, previous_term); }
};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string hash_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read hash input " + path.string());
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::array<char, 65536> buffer{};
    while (input) {
        input.read(buffer.data(), buffer.size());
        if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(input.gcount())) != 1)
            throw std::runtime_error("Cannot update SHA256");
    }
    if (!input.eof()) throw std::runtime_error("Cannot read complete hash input");
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest, &length) != 1) throw std::runtime_error("Cannot finalize SHA256");
    constexpr char hex[] = "0123456789abcdef"; std::string result;
    for (unsigned int i = 0; i < length; ++i) { result += hex[digest[i] >> 4]; result += hex[digest[i] & 15]; }
    return result;
}
void atomic_json(const fs::path& path, const json& value) {
    write_text_atomically(path, value.dump(2) + '\n');
}
json read_json(const fs::path& path) {
    std::ifstream input(path); if (!input) throw std::runtime_error("Cannot read " + path.string());
    return json::parse(input);
}
json preprocessing() {
    return {{"observation", "4x84x84 grayscale, max of last two repeated frames"},
        {"frame_skip", 4}, {"sticky_action_probability", .25}, {"noop_max", 30},
        {"fire_reset", true}, {"max_episode_raw_frames", 108000},
        {"terminal_on_life_loss", false}, {"training_reward_clipping", true},
        {"evaluation_reward_clipping", false}, {"time_limit_bootstrap", true}};
}
json policy_config(const GameSpec& game, const fs::path& rom) {
    return {{"algorithm", "PPO"}, {"language", "C++20"}, {"device", "cuda"},
        {"torch_version", TORCH_VERSION}, {"game_id", game.id}, {"game_title", game.title},
        {"game_spec", game_spec_json(game)}, {"rom", rom.string()}, {"rom_sha256", game.rom_sha256},
        {"action_count", game.expected_actions}, {"training_action_sampling", "categorical"},
        {"preprocessing", preprocessing()}, {"network", "conv32/8/4 conv64/4/2 conv64/3/1 fc512 actor+critic"},
        {"gamma", .99}, {"gae_lambda", .95}, {"clip_range", .1}, {"entropy_coefficient", .01},
        {"learning_rate", 2.5e-4}, {"value_coefficient", .5}, {"max_gradient_norm", .5}};
}

struct SharedNetworkImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear hidden{nullptr}, actor{nullptr}, critic{nullptr};
    torch::nn::Linear breakout_actor{nullptr}, breakout_critic{nullptr};
    torch::nn::Linear invaders_actor{nullptr}, invaders_critic{nullptr};
    std::vector<int> actions;
    explicit SharedNetworkImpl(std::array<int, 2> counts, bool initialize = true)
        : SharedNetworkImpl(std::vector<int>(counts.begin(), counts.end()), initialize) {}
    explicit SharedNetworkImpl(std::array<int, 3> counts, bool initialize = true)
        : SharedNetworkImpl(std::vector<int>(counts.begin(), counts.end()), initialize) {}
    explicit SharedNetworkImpl(std::vector<int> counts, bool initialize = true) : actions(std::move(counts)) {
        if (actions.size() != 2 && actions.size() != 3) throw std::invalid_argument("Shared network supports two or three games");
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 32, 8).stride(4)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1)));
        hidden = register_module("hidden", torch::nn::Linear(64 * 7 * 7, 512));
        actor = register_module("actor", torch::nn::Linear(512, actions[0]));
        critic = register_module("critic", torch::nn::Linear(512, 1));
        breakout_actor = register_module("breakout_actor", torch::nn::Linear(512, actions[1]));
        breakout_critic = register_module("breakout_critic", torch::nn::Linear(512, 1));
        if (actions.size() == 3) {
            invaders_actor = register_module("invaders_actor", torch::nn::Linear(512, actions[2]));
            invaders_critic = register_module("invaders_critic", torch::nn::Linear(512, 1));
        }
        if (!initialize) return;
        torch::NoGradGuard guard;
        for (auto& parameter : named_parameters()) {
            if (parameter.key().find("bias") != std::string::npos) parameter.value().zero_();
            else torch::nn::init::orthogonal_(parameter.value(), std::sqrt(2.0));
        }
        torch::nn::init::orthogonal_(actor->weight, .01);
        torch::nn::init::orthogonal_(breakout_actor->weight, .01);
        torch::nn::init::orthogonal_(critic->weight, 1.0);
        torch::nn::init::orthogonal_(breakout_critic->weight, 1.0);
        if (actions.size() == 3) {
            torch::nn::init::orthogonal_(invaders_actor->weight, .01);
            torch::nn::init::orthogonal_(invaders_critic->weight, 1.0);
        }
    }
    torch::Tensor encode(torch::Tensor input) {
        auto x = input.to(torch::kFloat32).div(255.0);
        x = torch::relu(conv1->forward(x)); x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x)); return torch::relu(hidden->forward(x.flatten(1)));
    }
    std::pair<torch::Tensor, torch::Tensor> head(torch::Tensor features, int game) {
        if (game == 0) return {actor->forward(features), critic->forward(features).squeeze(1)};
        if (game == 1) return {breakout_actor->forward(features), breakout_critic->forward(features).squeeze(1)};
        if (game == 2 && actions.size() == 3) return {invaders_actor->forward(features), invaders_critic->forward(features).squeeze(1)};
        throw std::invalid_argument("Invalid shared game index");
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input, int game) {
        return head(encode(input), game);
    }
};
TORCH_MODULE(SharedNetwork);

torch::Tensor observations(const std::uint8_t* bytes, std::int64_t count, const torch::Device& device) {
    return torch::from_blob(const_cast<std::uint8_t*>(bytes), {count, 4, 84, 84},
                            torch::TensorOptions().dtype(torch::kUInt8)).to(device);
}
void save_archive(torch::serialize::OutputArchive& archive, const fs::path& path) {
    fs::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    archive.save_to(temporary); replace_file_atomically(temporary, path);
}
void save_joint(SharedNetwork& model, torch::optim::Adam& optimizer,
                std::int64_t steps, std::int64_t source_steps, const fs::path& path,
                const std::vector<std::int64_t>& previous_game_steps = {}) {
    torch::serialize::OutputArchive archive, weights, state;
    archive.write("schema_version", torch::tensor(1, torch::kInt64));
    archive.write("steps", torch::tensor(steps, torch::kInt64));
    const std::vector<std::int64_t> per_game(model->actions.size(), steps / model->actions.size());
    archive.write("per_game_steps", torch::tensor(per_game, torch::kInt64));
    archive.write("source_pretraining_steps", torch::tensor(source_steps, torch::kInt64));
    archive.write("action_counts", torch::tensor(model->actions, torch::kInt64));
    if (!previous_game_steps.empty()) {
        if (previous_game_steps.size() != model->actions.size()) throw std::invalid_argument("Shared prior-game metadata mismatch");
        auto cumulative = previous_game_steps;
        for (auto& value : cumulative) value += steps / model->actions.size();
        archive.write("previous_game_steps", torch::tensor(previous_game_steps, torch::kInt64));
        archive.write("cumulative_game_steps", torch::tensor(cumulative, torch::kInt64));
    }
    model->save(weights); optimizer.save(state);
    archive.write("model", weights); archive.write("optimizer", state); save_archive(archive, path);
}
void export_policy(SharedNetwork& model, int game, std::int64_t steps, const fs::path& path) {
    torch::serialize::OutputArchive archive, weights;
    auto module = [&](const char* name, torch::nn::Module& source) {
        torch::serialize::OutputArchive parameters; source.save(parameters); weights.write(name, parameters);
    };
    module("conv1", *model->conv1); module("conv2", *model->conv2); module("conv3", *model->conv3);
    module("hidden", *model->hidden);
    module("actor", game == 0 ? *model->actor : game == 1 ? *model->breakout_actor : *model->invaders_actor);
    module("critic", game == 0 ? *model->critic : game == 1 ? *model->breakout_critic : *model->invaders_critic);
    archive.write("action_count", torch::tensor(model->actions.at(game), torch::kInt64));
    archive.write("steps", torch::tensor(steps, torch::kInt64));
    archive.write("model", weights); save_archive(archive, path);
}
void load_export(SharedNetwork& model, int game, const fs::path& path) {
    torch::serialize::InputArchive archive, weights; archive.load_from(path.string(), torch::Device(torch::kCPU));
    archive.read("model", weights);
    auto module = [&](const char* name, torch::nn::Module& target) {
        torch::serialize::InputArchive parameters; weights.read(name, parameters); target.load(parameters);
    };
    module("conv1", *model->conv1); module("conv2", *model->conv2); module("conv3", *model->conv3);
    module("hidden", *model->hidden);
    module("actor", game == 0 ? *model->actor : game == 1 ? *model->breakout_actor : *model->invaders_actor);
    module("critic", game == 0 ? *model->critic : game == 1 ? *model->breakout_critic : *model->invaders_critic);
}
json load_pong_source(SharedNetwork& model, const fs::path& checkpoint) {
    auto transfer = copy_feature_parameters(*model, checkpoint.string(), model->actions[0]);
    torch::serialize::InputArchive archive, weights; archive.load_from(checkpoint.string(), torch::Device(torch::kCPU));
    archive.read("model", weights);
    struct Copy { torch::Tensor source, target; };
    std::vector<Copy> copies; const auto parameters = model->named_parameters();
    for (const char* name : {"actor", "critic"}) {
        torch::serialize::InputArchive head; weights.read(name, head);
        for (const char* part : {"weight", "bias"}) {
            const auto full_name = std::string(name) + "." + part;
            torch::Tensor source; head.read(part, source); const auto* target = parameters.find(full_name);
            if (!target || source.scalar_type() != target->scalar_type() || source.sizes() != target->sizes() ||
                !torch::isfinite(source).all().item<bool>()) throw std::invalid_argument("Invalid source Pong head " + full_name);
            copies.push_back({source, *target});
        }
    }
    torch::NoGradGuard guard; for (const auto& copy : copies) copy.target.copy_(copy.source);
    if (hash_file(checkpoint) != transfer.at("source_checkpoint_sha256"))
        throw std::runtime_error("Pong source checkpoint changed while loading");
    transfer["pong_heads_restored"] = true; transfer["breakout_heads_restored"] = false;
    transfer["optimizer_restored"] = false;
    // This field from the encoder-only copy refers to its intermediate state.
    transfer.erase("heads_unchanged"); transfer.erase("head_fingerprints");
    return transfer;
}

json restore_two_game_joint(SharedNetwork& target, const fs::path& checkpoint) {
    if (target->actions.size() != 3) throw std::invalid_argument("Joint extension target must have three games");
    const auto checkpoint_hash = hash_file(checkpoint);
    torch::serialize::InputArchive archive, weights, state;
    archive.load_from(checkpoint.string(), torch::Device(torch::kCPU));
    torch::Tensor count, steps, previous, per_game;
    archive.read("action_counts", count); archive.read("steps", steps);
    archive.read("source_pretraining_steps", previous); archive.read("per_game_steps", per_game);
    const auto counts = torch::tensor({target->actions[0], target->actions[1]}, torch::kInt64);
    if (count.scalar_type() != torch::kInt64 || !torch::equal(count, counts) ||
        steps.scalar_type() != torch::kInt64 || steps.numel() != 1 ||
        previous.scalar_type() != torch::kInt64 || previous.numel() != 1 ||
        per_game.scalar_type() != torch::kInt64 || per_game.sizes() != torch::IntArrayRef({2}))
        throw std::invalid_argument("Two-game joint checkpoint has invalid action/decision metadata");
    const auto joint_steps = steps.item<std::int64_t>(), original_pong_steps = previous.item<std::int64_t>();
    if (joint_steps <= 0 || joint_steps % 2 || original_pong_steps <= 0 ||
        original_pong_steps > std::numeric_limits<std::int64_t>::max() - joint_steps ||
        !torch::equal(per_game, torch::tensor({joint_steps / 2, joint_steps / 2}, torch::kInt64)))
        throw std::invalid_argument("Two-game joint checkpoint has invalid balanced step counts");
    SharedNetwork source(std::array<int, 2>{target->actions[0], target->actions[1]}, false);
    archive.read("model", weights); source->load(weights);
    torch::optim::Adam source_optimizer(source->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    archive.read("optimizer", state); source_optimizer.load(state);
    if (source_optimizer.state().size() != source->parameters().size())
        throw std::invalid_argument("Source joint Adam is incomplete");
    for (const auto& group : source_optimizer.param_groups()) {
        const auto& settings = static_cast<const torch::optim::AdamOptions&>(group.options());
        if (settings.lr() != 2.5e-4 || settings.eps() != 1e-5 || settings.betas() != std::make_tuple(.9, .999) ||
            settings.weight_decay() != 0 || settings.amsgrad()) throw std::invalid_argument("Source joint Adam settings differ");
    }
    struct Copy { std::string name; torch::Tensor source, target; };
    std::vector<Copy> copies; const auto target_parameters = target->named_parameters();
    for (const auto& item : source->named_parameters()) {
        const auto parameter = item.value(); const auto* destination = target_parameters.find(item.key());
        if (!destination || parameter.scalar_type() != torch::kFloat32 || parameter.sizes() != destination->sizes() ||
            !torch::isfinite(parameter).all().item<bool>()) throw std::invalid_argument("Invalid source joint parameter: " + item.key());
        const auto iterator = source_optimizer.state().find(parameter.unsafeGetTensorImpl());
        if (iterator == source_optimizer.state().end()) throw std::invalid_argument("Source joint Adam mapping is incomplete");
        const auto* moment = dynamic_cast<const torch::optim::AdamParamState*>(iterator->second.get());
        auto valid = [&](const torch::Tensor& tensor) { return tensor.defined() && tensor.sizes() == parameter.sizes() &&
            tensor.scalar_type() == parameter.scalar_type() && tensor.device() == parameter.device() && torch::isfinite(tensor).all().item<bool>(); };
        if (!moment || moment->step() <= 0 || !valid(moment->exp_avg()) || !valid(moment->exp_avg_sq()) ||
            !moment->exp_avg_sq().ge(0).all().item<bool>()) throw std::invalid_argument("Invalid source joint Adam moments: " + item.key());
        copies.push_back({item.key(), parameter, *destination});
    }
    auto fresh_actor = target->invaders_actor->weight.detach().clone(), fresh_critic = target->invaders_critic->weight.detach().clone();
    json copied = json::array();
    { torch::NoGradGuard guard;
      for (const auto& item : copies) { item.target.copy_(item.source); copied.push_back({{"name", item.name}, {"shape", item.source.sizes().vec()}}); } }
    if (!torch::equal(fresh_actor, target->invaders_actor->weight) || !torch::equal(fresh_critic, target->invaders_critic->weight))
        throw std::runtime_error("Joint transfer changed fresh Space Invaders heads");
    if (hash_file(checkpoint) != checkpoint_hash) throw std::runtime_error("Joint source changed while restoring");
    return {{"source_checkpoint_sha256", checkpoint_hash}, {"source_pretraining_decisions", original_pong_steps + joint_steps},
        {"original_pong_pretraining_decisions", original_pong_steps}, {"source_shared_new_decisions", joint_steps},
        {"previous_game_decisions", {{"pong", original_pong_steps + joint_steps / 2}, {"breakout", joint_steps / 2}, {"space_invaders", 0}}},
        {"source_game_ids", {"pong", "breakout"}}, {"restored_tensors", copied}, {"source_adam_validated", true},
        {"pong_heads_restored", true}, {"breakout_heads_restored", true}, {"space_invaders_heads_restored", false}, {"optimizer_restored", false}};
}

void validate_complete_evaluation(const json& evaluation, const GameSpec& game, std::uint64_t seed,
                                  const std::string& checkpoint_hash, const std::string& config_hash) {
    if (evaluation.at("status") != "complete" || evaluation.at("game_id") != game.id ||
        evaluation.at("checkpoint_sha256") != checkpoint_hash || evaluation.at("checkpoint_config_sha256") != config_hash ||
        evaluation.at("episodes_completed") != game.criterion.episodes || evaluation.at("seed_start") != seed ||
        evaluation.at("episodes").size() != static_cast<std::size_t>(game.criterion.episodes))
        throw std::invalid_argument("Source evaluation identity or episode count differs");
    double sum = 0; int truncated = 0, positive = 0;
    for (int index = 0; index < game.criterion.episodes; ++index) {
        const auto& row = evaluation.at("episodes").at(index);
        const double score = row.at("raw_return").get<double>();
        const bool terminated = row.at("terminated").get<bool>(), capped = row.at("truncated").get<bool>();
        if (row.at("seed") != seed + index || !std::isfinite(score) || (!terminated && !capped))
            throw std::invalid_argument("Source evaluation episode is missing, unfinalized or has a different seed");
        sum += score; truncated += capped; positive += terminated && !capped && score > 0;
    }
    const double mean = sum / game.criterion.episodes;
    if (evaluation.at("mean_return") != mean || evaluation.at("truncated_games") != truncated ||
        evaluation.at("positive_games") != positive || evaluation.at("passed") != evaluate_game_criterion(game, game.criterion.episodes, positive, mean, truncated))
        throw std::invalid_argument("Source evaluation summary disagrees with finalized episode records");
}

void validate_joint_lineage(SharedNetwork& restored, json& transfer, const fs::path& checkpoint, const std::vector<GameSpec>& games,
                            const json& source_config, const json& expected) {
    const auto directory = checkpoint.parent_path();
    if (checkpoint.filename() != "shared-final.pt" || fs::exists(directory / "STOP"))
        throw std::invalid_argument("Extension needs an unstopped final joint checkpoint");
    validate_resume_config(source_config, expected);
    if (source_config.at("training_mode") != "shared" || source_config.at("joint_checkpoint_game_order") != json({"pong", "breakout"}) ||
        source_config.at("shared_games").size() != 2 || source_config.at("minibatch_game_weights") != json({{"pong", .5}, {"breakout", .5}}) ||
        source_config.at("advantage_normalization") != "separate within each game and rollout" ||
        source_config.at("initialization_mode") != "source_encoder_and_pong_heads_fresh_breakout_heads_and_adam")
        throw std::invalid_argument("Extension source must be the declared two-game shared experiment");
    const auto n = source_config.at("environments").get<int>();
    if (n < 2 || n % 2 || source_config.at("environments_per_game") != n / 2 ||
        source_config.at("rollout_steps") != 128 || source_config.at("epochs") != 4 || source_config.at("minibatch_size") != 256)
        throw std::invalid_argument("Two-game source sampling or optimization settings differ");
    for (int game = 0; game < 2; ++game) if (source_config.at("shared_games").at(game) != game_spec_json(games[game]))
        throw std::invalid_argument("Two-game source game registry identity differs");
    json hashes = json::object();
    auto capture = [&](const fs::path& path) {
        const auto hash = hash_file(path); hashes[path.string()] = hash; return read_json(path);
    };
    const auto progress = capture(directory / "progress.json");
    const auto retention = capture(directory / "retention.json");
    const auto manifest = capture(directory / "checkpoint-manifest.json");
    const auto joint_steps = transfer.at("source_shared_new_decisions");
    if (progress.at("status") != "complete" || retention.at("status") != "complete" || retention.at("games").size() != 2 ||
        progress.at("agent_decisions") != joint_steps || source_config.at("max_agent_decisions") != joint_steps ||
        source_config.at("previous_trained_steps") != transfer.at("original_pong_pretraining_decisions") ||
        manifest.at("total_new_decisions") != joint_steps || retention.at("total_new_decisions") != joint_steps ||
        manifest.at("per_game_new_decisions") != joint_steps.get<std::int64_t>() / 2 ||
        manifest.at("source_pretraining_decisions") != transfer.at("original_pong_pretraining_decisions") ||
        manifest.at("game_order") != json({"pong", "breakout"}) || fs::canonical(manifest.at("shared_checkpoint").get<std::string>()) != checkpoint ||
        retention.at("shared_checkpoint_sha256") != transfer.at("source_checkpoint_sha256"))
        throw std::invalid_argument("Two-game source completion, decision counts or hash-bound joint manifest do not agree");
    if (manifest.contains("shared_checkpoint_sha256") && manifest.at("shared_checkpoint_sha256") != transfer.at("source_checkpoint_sha256"))
        throw std::invalid_argument("Source manifest joint hash differs");
    if (manifest.contains("config_sha256") && manifest.at("config_sha256") != hash_file(directory / "config.json"))
        throw std::invalid_argument("Source manifest configuration hash differs");
    for (int game = 0; game < 2; ++game) {
        const auto& id = games[game].id;
        if (!retention.at("games").at(id).at("complete").get<bool>()) throw std::invalid_argument("Source retention measurement is incomplete");
        for (const auto& stage : {std::string("before"), std::string("after")}) {
            const auto path = stage == "before" ? directory / "before" / id : directory / id;
            const auto policy_path = path / (stage == "before" ? "initial.pt" : "final.pt");
            const auto config = capture(path / "config.json");
            validate_evaluation_config(config, games[game], games[game].rom_sha256, games[game].expected_actions);
            const auto evaluation = capture(path / "evaluation.json");
            const auto export_hash = hash_file(policy_path); hashes[policy_path.string()] = export_hash;
            const auto& artifact = retention.at("artifacts").at(id).at(stage);
            if (artifact.at("checkpoint_sha256") != export_hash || artifact.at("config_sha256") != hashes.at((path / "config.json").string()) ||
                artifact.at("evaluation_sha256") != hashes.at((path / "evaluation.json").string()))
                throw std::invalid_argument("Source per-game evaluated artifacts do not match retained evidence");
            validate_complete_evaluation(evaluation, games[game], evaluation_seeds[game], export_hash, artifact.at("config_sha256").get<std::string>());
            if (retention.at("games").at(id).at(stage + "_mean_raw_return") != evaluation.at("mean_return") ||
                retention.at("games").at(id).at(stage + "_truncated_games") != evaluation.at("truncated_games") ||
                retention.at("games").at(id).at(stage + "_passed") != evaluation.at("passed"))
                throw std::invalid_argument("Source retention summary differs from its evaluated artifacts");
        }
        // Hashes bind the files; tensor equality also proves the evaluated
        // exports describe the restored joint encoder and each learned head.
        SharedNetwork exported(std::array<int, 2>{games[0].expected_actions, games[1].expected_actions}, false);
        load_export(exported, game, directory / id / "final.pt");
        const auto target_parameters = restored->named_parameters();
        const std::string actor = game == 0 ? "actor." : "breakout_actor.";
        const std::string critic = game == 0 ? "critic." : "breakout_critic.";
        for (const auto& parameter : exported->named_parameters()) {
            const auto& name = parameter.key();
            if (name.rfind("conv", 0) == 0 || name.rfind("hidden.", 0) == 0 || name.rfind(actor, 0) == 0 || name.rfind(critic, 0) == 0) {
                const auto* target = target_parameters.find(name);
                if (!target || !torch::equal(*target, parameter.value()))
                    throw std::invalid_argument("Evaluated export differs from source joint policy: " + id + "." + name);
            }
        }
    }
    for (const auto& item : hashes.items()) if (hash_file(item.key()) != item.value()) throw std::runtime_error("Source lineage changed while validating");
    transfer["source_evidence_hashes"] = hashes;
    transfer["source_retention"] = (directory / "retention.json").string();
    transfer["source_retention_sha256"] = hashes.at((directory / "retention.json").string());
    transfer["source_completion_validated"] = true;
}

class Workers {
    std::mutex mutex_; std::condition_variable ready_, complete_;
    std::vector<std::thread> threads_; std::function<void(int)> job_;
    std::exception_ptr error_; std::size_t generation_ = 0, pending_ = 0; bool quitting_ = false;
public:
    explicit Workers(int count) {
        for (int index = 0; index < count; ++index) threads_.emplace_back([this, index] {
            std::size_t seen = 0;
            for (;;) {
                std::unique_lock lock(mutex_); ready_.wait(lock, [&] { return quitting_ || generation_ != seen; });
                if (quitting_) return;
                seen = generation_; auto work = job_; lock.unlock();
                try { work(index); } catch (...) { std::lock_guard guard(mutex_); if (!error_) error_ = std::current_exception(); }
                lock.lock(); if (--pending_ == 0) complete_.notify_one();
            }
        });
    }
    ~Workers() { { std::lock_guard guard(mutex_); quitting_ = true; } ready_.notify_all(); for (auto& thread : threads_) thread.join(); }
    void run(std::function<void(int)> job) {
        std::unique_lock lock(mutex_); job_ = std::move(job); error_ = nullptr; pending_ = threads_.size();
        ++generation_; ready_.notify_all(); complete_.wait(lock, [&] { return pending_ == 0; });
        if (error_) std::rethrow_exception(error_);
    }
};

// Both games have the same number of samples. Normalize advantages within each
// game so reward-scale differences cannot determine that game's policy weight.
std::vector<torch::Tensor> balanced_partitions(int steps, int environments, const torch::Device& device, int games = 2) {
    std::vector<torch::Tensor> result(games);
    for (int game = 0; game < games; ++game) {
        auto indices = shared_game_indices(steps, environments, game, games);
        result[game] = torch::from_blob(indices.data(), {static_cast<std::int64_t>(indices.size())}, torch::kInt64).clone().to(device);
    }
    return result;
}
torch::Tensor ppo_loss(SharedNetwork& model, int game, const torch::Tensor& observations,
                      const torch::Tensor& actions, const torch::Tensor& previous_logprob,
                      const torch::Tensor& advantages, const torch::Tensor& returns) {
    auto [logits, value] = model->forward(observations, game);
    auto log_probs = logits.log_softmax(1);
    auto selected = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1);
    auto ratio = (selected - previous_logprob).exp();
    auto policy = -torch::minimum(ratio * advantages, ratio.clamp(.9, 1.1) * advantages).mean();
    auto value_loss = (value - returns).pow(2).mean();
    auto entropy = -(log_probs.exp() * log_probs).sum(1).mean();
    return policy + .5 * value_loss - .01 * entropy;
}
json paired_result(const json& before, const json& after, const GameSpec& game) {
    const bool complete = before.at("status") == "complete" && after.at("status") == "complete" &&
        before.at("episodes_completed") == game.criterion.episodes && after.at("episodes_completed") == game.criterion.episodes &&
        before.at("episodes").size() == static_cast<std::size_t>(game.criterion.episodes) &&
        after.at("episodes").size() == static_cast<std::size_t>(game.criterion.episodes);
    const bool has_truncations = before.at("truncated_games") != 0 || after.at("truncated_games") != 0;
    const bool full_games = complete && !has_truncations;
    if (before.at("seed_start") != after.at("seed_start")) throw std::runtime_error("Retention evaluation seeds differ");
    json differences = json::array(); double sum = 0;
    const auto count = std::min(before.at("episodes").size(), after.at("episodes").size());
    for (std::size_t i = 0; i < count; ++i) {
        if (before.at("episodes").at(i).at("seed") != after.at("episodes").at(i).at("seed"))
            throw std::runtime_error("Retention episode seed order differs");
        const auto difference = after.at("episodes").at(i).at("raw_return").get<double>() -
            before.at("episodes").at(i).at("raw_return").get<double>();
        differences.push_back({{"seed", before.at("episodes").at(i).at("seed")}, {"raw_return_delta", difference},
            {"before_truncated", before.at("episodes").at(i).at("truncated")},
            {"after_truncated", after.at("episodes").at(i).at("truncated")}}); sum += difference;
    }
    json result = {{"game_id", game.id}, {"complete", complete}, {"seed_start", before.at("seed_start")},
        {"before_mean_raw_return", before.at("mean_return")}, {"after_mean_raw_return", after.at("mean_return")},
        {"paired_episodes", count}, {"paired_mean_raw_return_delta", count ? json(sum / count) : json(nullptr)},
        {"before_passed", before.at("passed")}, {"after_passed", after.at("passed")},
        {"before_truncated_games", before.at("truncated_games")}, {"after_truncated_games", after.at("truncated_games")},
        {"full_game_comparison", full_games}, {"bounded_episode_comparison", complete},
        {"comparison_scope", !complete ? "incomplete_paired_episodes" : full_games ? "full_games" : "frame_capped_episodes"},
        {"limitations", json::array()}, {"paired_differences", differences}};
    if (has_truncations) result["limitations"].push_back(
        "Includes frame-limit-truncated episodes: returns and paired deltas are capped at the configured episode limit, not complete-game scores.");
    if (!complete) result["limitations"].push_back(
        "At least one evaluation lacks the declared number of finalized episodes; paired deltas cover only the available pairs.");
    result["criterion_retained"] = full_games && before.at("passed").get<bool>() && after.at("passed").get<bool>();
    result["criterion_retention_applicable"] = before.at("passed").get<bool>();
    if (game.id == "pong") {
        result["before_wins"] = before.at("wins"); result["after_wins"] = after.at("wins");
    }
    return result;
}
} // namespace

std::vector<std::int64_t> shared_game_indices(int steps, int environments, int game, int games) {
    if ((games != 2 && games != 3) || steps < 1 || environments < games || environments % games || game < 0 || game >= games)
        throw std::invalid_argument("Shared samples require two/three balanced games and positive dimensions");
    std::vector<std::int64_t> result; result.reserve(static_cast<std::size_t>(steps) * environments / games);
    for (int t = 0; t < steps; ++t) for (int e = game * environments / games; e < (game + 1) * environments / games; ++e)
        result.push_back(static_cast<std::int64_t>(t) * environments + e);
    return result;
}

static nlohmann::json run_shared_impl(const SharedTrainOptions& options, bool extension) {
    const int game_count = extension ? 3 : 2;
    const std::vector<std::uint64_t> paired_seeds = extension
        ? std::vector<std::uint64_t>(extension_evaluation_seeds.begin(), extension_evaluation_seeds.end())
        : std::vector<std::uint64_t>(evaluation_seeds.begin(), evaluation_seeds.end());
    if (options.games.size() != static_cast<std::size_t>(game_count) || options.num_envs < game_count || options.num_envs > 64 || options.num_envs % game_count ||
        options.rollout_steps < 1 || options.epochs < 1 || options.minibatch_size < game_count || options.minibatch_size % game_count ||
        options.max_steps < options.num_envs || options.max_seconds < 1 || options.torch_threads < 1 || options.run_dir.empty() ||
        options.evaluation_max_seconds < 1 || options.evaluation_max_seconds > 7200)
        throw std::invalid_argument("Shared training requires balanced games, divisible environments/minibatch, and positive budgets");
    std::vector<GameSpec> games(game_count);
    for (const auto& game : options.games) {
        if (game.id == "pong") games[0] = game;
        else if (game.id == "breakout") games[1] = game;
        else if (extension && game.id == "space_invaders") games[2] = game;
        else throw std::invalid_argument("Shared experiment received an unsupported game");
    }
    for (const auto& game : games) if (game.id.empty()) throw std::invalid_argument("Shared experiment needs each declared game exactly once");
    std::vector<fs::path> roms(game_count);
    for (int game = 0; game < game_count; ++game) {
        roms[game] = fs::path(options.rom_directory) / games[game].rom_filename;
        if (hash_file(roms[game]) != games[game].rom_sha256) throw std::invalid_argument("Shared ROM differs from registry: " + games[game].id);
    }
    if (!torch::cuda::is_available()) throw std::runtime_error("Shared training requires CUDA");
    const fs::path directory(options.run_dir);
    if (fs::exists(directory) && (!fs::is_directory(directory) || !fs::is_empty(directory)))
        throw std::runtime_error("Run directory must be new or empty; preserve existing shared experiments");
    fs::create_directories(directory);
    Signals signals; torch::set_num_threads(options.torch_threads); torch::manual_seed(options.seed);
    const torch::Device device(torch::kCUDA);
    const auto checkpoint = fs::canonical(options.source_checkpoint_path);
    const auto source_config_path = checkpoint.parent_path() / "config.json";
    const auto checkpoint_hash = hash_file(checkpoint), source_config_hash = hash_file(source_config_path);
    auto source_config = read_json(source_config_path);
    auto config = policy_config(games[0], roms[0]);
    validate_transfer_config(source_config, games[0], config);
    std::vector<int> action_counts; for (const auto& game : games) action_counts.push_back(game.expected_actions);
    SharedNetwork model(action_counts);
    auto transfer = extension ? restore_two_game_joint(model, checkpoint) : load_pong_source(model, checkpoint);
    if (extension) validate_joint_lineage(model, transfer, checkpoint, games, source_config, config);
    const auto source_steps = transfer.at("source_pretraining_decisions").get<std::int64_t>();
    std::vector<std::int64_t> previous_game_steps(game_count, 0);
    previous_game_steps[0] = source_steps;
    if (extension) for (int game = 0; game < game_count; ++game)
        previous_game_steps[game] = transfer.at("previous_game_decisions").at(games[game].id).get<std::int64_t>();
    if (source_steps > std::numeric_limits<std::int64_t>::max() - options.max_steps)
        throw std::invalid_argument("Cumulative shared checkpoint decisions overflow");
    if (checkpoint_hash != hash_file(checkpoint) || source_config_hash != hash_file(source_config_path))
        throw std::runtime_error("Source checkpoint/configuration changed while loading");
    transfer["source_checkpoint"] = checkpoint.string(); transfer["source_config"] = source_config_path.string();
    transfer["source_config_sha256"] = source_config_hash;
    if (!extension) transfer["source_game_id"] = "pong";
    transfer["source_rom_sha256"] = games[0].rom_sha256;
    if (!options.source_evaluation_path.empty()) {
        const auto proof_path = fs::canonical(options.source_evaluation_path); const auto proof_hash = hash_file(proof_path);
        auto proof = read_json(proof_path);
        const auto proof_begin = proof.at("seed_start").get<std::uint64_t>();
        const auto proof_count = proof.at("episodes_completed").get<std::uint64_t>();
        const bool overlapping = proof_begin < evaluation_seeds[0] + games[0].criterion.episodes &&
            proof_begin + proof_count > evaluation_seeds[0];
        if (proof.at("checkpoint_sha256") != checkpoint_hash || !proof.at("passed").get<bool>() ||
            proof.at("episodes_completed") != games[0].criterion.episodes || proof.at("truncated_games") != 0 ||
            overlapping)
            throw std::invalid_argument("Source proof does not identify the passed source policy or overlaps paired seeds");
        if (hash_file(proof_path) != proof_hash) throw std::runtime_error("Source proof changed while reading");
        transfer["source_evaluation"] = proof_path.string(); transfer["source_evaluation_sha256"] = proof_hash;
    }
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    if (!optimizer.state().empty()) throw std::runtime_error("Shared Adam did not start empty");
    config["training_mode"] = "shared"; config["shared_stage"] = extension ? "three_game_extension" : "two_game_pilot";
    config["shared_games"] = json::array(); for (const auto& game : games) config["shared_games"].push_back(game_spec_json(game));
    config["shared_network"] = "one conv32/64/64 fc512 encoder; independent actor and critic for each game";
    config["seed"] = options.seed; config["max_agent_decisions"] = options.max_steps;
    config["max_seconds"] = options.max_seconds; config["environments"] = options.num_envs;
    config["environments_per_game"] = options.num_envs / game_count; config["rollout_steps"] = options.rollout_steps;
    config["epochs"] = options.epochs; config["minibatch_size"] = options.minibatch_size;
    config["minibatch_game_weights"] = json::object(); for (const auto& game : games) config["minibatch_game_weights"][game.id] = 1.0 / game_count;
    config["advantage_normalization"] = "separate within each game and rollout";
    config["initialization_mode"] = extension ? "source_joint_encoder_pong_and_breakout_heads_fresh_space_invaders_heads_and_adam"
        : "source_encoder_and_pong_heads_fresh_breakout_heads_and_adam";
    config["initial_adam_state_entries"] = optimizer.state().size(); config["transfer"] = transfer;
    config["budget_scope"] = extension ? "total_new_decisions_across_three_games_equal_allocation" : "total_new_decisions_across_both_games_equal_allocation";
    config["checkpoint_steps_scope"] = "source_pong_decisions_plus_new_pong_decisions_for_pong_export";
    if (extension) config["checkpoint_steps_scope"] = "previous_pong_game_decisions_plus_new_pong_decisions_for_pong_export";
    config["previous_trained_steps"] = source_steps;
    config["single_game_exports"] = "inference_only; optimizer is preserved in shared checkpoints";
    config["joint_checkpoint_game_order"] = json::array(); config["evaluation_seeds"] = json::object();
    config["previous_game_decisions"] = json::object();
    for (int game = 0; game < game_count; ++game) {
        config["joint_checkpoint_game_order"].push_back(games[game].id);
        config["evaluation_seeds"][games[game].id] = paired_seeds[game];
        config["previous_game_decisions"][games[game].id] = previous_game_steps[game];
    }
    config["evaluation_episodes"] = games[0].criterion.episodes;
    config["evaluation_max_seconds_per_game"] = options.evaluation_max_seconds;
    config["evaluation_role"] = "paired_development_retention; same fresh seeds before and after shared training";
    config["sampling_rng"] = "reset to run seed after initial policy evaluation";
    atomic_json(directory / "config.json", config);
    for (int game = 0; game < game_count; ++game) {
        auto exported = policy_config(games[game], roms[game]);
        exported["training_mode"] = "shared_export"; exported["shared_parent_run"] = directory.string();
        exported["seed"] = options.seed; exported["transfer"] = transfer;
        exported["previous_trained_steps"] = previous_game_steps[game];
        exported["checkpoint_steps_scope"] = "source_game_decisions_plus_new_decisions_for_this_game; encoder also sees other game";
        exported["optimizer_available"] = false; exported["joint_checkpoint"] = (directory / "shared-final.pt").string();
        atomic_json(directory / games[game].id / "config.json", exported);
        atomic_json(directory / "before" / games[game].id / "config.json", exported);
        export_policy(model, game, previous_game_steps[game], directory / "before" / games[game].id / "initial.pt");
    }
    export_policy(model, 0, previous_game_steps[0], directory / "latest.pt");
    std::vector<json> before(game_count), after(game_count);
    const auto start = Clock::now(); auto training_start = start, last_save = start;
    std::int64_t total_steps = 0, updates = 0, checkpoint_steps = 0;
    std::vector<std::int64_t> completed(game_count, 0); std::vector<double> recent_sum(game_count, 0); std::vector<int> recent_count(game_count, 0);
    json progress; std::string stop_reason; double training_seconds = 0;
    auto stopped = [&] { return interrupted || fs::exists(directory / "STOP"); };
    auto report = [&](const char* status, double loss) {
        const double measured = std::string(status) == "training" ? elapsed(training_start) : training_seconds;
        progress = {{"status", status}, {"training_mode", "shared"}, {"game_id", "pong"}, {"game_title", extension ? "Pong + Breakout + Space Invaders" : "Pong + Breakout"},
            {"steps", total_steps}, {"agent_decisions", total_steps}, {"target_steps", options.max_steps},
            {"previous_trained_steps", source_steps}, {"cumulative_agent_decisions", source_steps + total_steps},
            {"training_elapsed_seconds", measured}, {"elapsed_seconds", measured}, {"total_elapsed_seconds", elapsed(start)},
            {"decisions_per_second", total_steps / std::max(.001, measured)}, {"fps", total_steps / std::max(.001, measured)},
            {"episodes", std::accumulate(completed.begin(), completed.end(), std::int64_t{0})}, {"updates", updates}, {"loss", loss}, {"device", "cuda"},
            {"checkpoint_steps", previous_game_steps[0] + checkpoint_steps / game_count}, {"checkpoint_run_steps", checkpoint_steps},
            {"run_dir", directory.string()}, {"latest_checkpoint", (directory / "latest.pt").string()},
            {"stop_reason", stop_reason.empty() ? json(nullptr) : json(stop_reason)}};
        progress["per_game"] = json::object();
        for (int game = 0; game < game_count; ++game) progress["per_game"][games[game].id] = {
            {"agent_decisions", total_steps / game_count}, {"target_steps", options.max_steps / game_count}, {"episodes", completed[game]},
            {"previous_game_decisions", previous_game_steps[game]}, {"cumulative_game_decisions", previous_game_steps[game] + total_steps / game_count},
            {"recent_mean_raw_return", recent_count[game] ? json(recent_sum[game] / recent_count[game]) : json(nullptr)},
            {"recent_mean_sample_count", recent_count[game]}};
        // The generic dashboard displays the Pong viewer's score; game-specific
        // values remain available in per_game and cannot be averaged together.
        progress["recent_mean_raw_return"] = progress["per_game"]["pong"]["recent_mean_raw_return"];
        progress["recent_mean_reward"] = progress["recent_mean_raw_return"];
        progress["recent_mean_sample_count"] = recent_count[0];
        atomic_json(directory / "progress.json", progress);
    };
    auto save = [&](bool final) {
        save_joint(model, optimizer, total_steps, source_steps, directory / "shared-latest.pt", previous_game_steps);
        for (int game = 0; game < game_count; ++game) {
            const auto steps = previous_game_steps[game] + total_steps / game_count;
            export_policy(model, game, steps, directory / games[game].id / "latest.pt");
            if (final) export_policy(model, game, steps, directory / games[game].id / "final.pt");
        }
        export_policy(model, 0, previous_game_steps[0] + total_steps / game_count, directory / "latest.pt");
        if (final) {
            save_joint(model, optimizer, total_steps, source_steps, directory / "shared-final.pt", previous_game_steps);
            export_policy(model, 0, previous_game_steps[0] + total_steps / game_count, directory / "final.pt");
        }
        checkpoint_steps = total_steps; last_save = Clock::now();
        atomic_json(directory / "checkpoint-manifest.json", {{"game_order", config.at("joint_checkpoint_game_order")},
            {"shared_checkpoint", (directory / (final ? "shared-final.pt" : "shared-latest.pt")).string()},
            {"shared_checkpoint_sha256", hash_file(directory / (final ? "shared-final.pt" : "shared-latest.pt"))},
            {"config_sha256", hash_file(directory / "config.json")}, {"previous_game_decisions", config.at("previous_game_decisions")},
            {"total_new_decisions", total_steps}, {"per_game_new_decisions", total_steps / game_count},
            {"source_pretraining_decisions", source_steps}, {"single_game_exports_include_optimizer", false}});
    };
    report("evaluating_before", 0);
    try {
        if (options.evaluate_at_end) {
            for (int game = 0; game < game_count && !stopped(); ++game) {
                EvalOptions evaluation; evaluation.game = games[game]; evaluation.rom_path = roms[game].string();
                evaluation.checkpoint_path = (directory / "before" / games[game].id / "initial.pt").string();
                evaluation.output_path = (directory / "before" / games[game].id / "evaluation.json").string();
                evaluation.stop_path = (directory / "STOP").string();
                evaluation.seed = paired_seeds[game]; evaluation.episodes = games[game].criterion.episodes;
                evaluation.max_seconds = options.evaluation_max_seconds;
                before[game] = run_evaluation(evaluation);
                if (before[game].at("status") != "complete") {
                    stop_reason = "baseline_evaluation_incomplete"; save(true); report("evaluation_incomplete", 0); return progress;
                }
            }
        }
        if (stopped()) { save(true); stop_reason = "STOP_or_signal"; report("stopped", 0); return progress; }
        torch::set_num_threads(options.torch_threads); torch::manual_seed(options.seed);
        const int n = options.num_envs, game_environments = n / game_count;
        Workers workers(n); std::vector<std::unique_ptr<AtariEnv>> environments(n);
        workers.run([&](int e) {
            const int game = e / game_environments;
            environments[e] = std::make_unique<AtariEnv>(roms[game].string(), options.seed + e);
            if (environments[e]->actions() != games[game].expected_actions) throw std::runtime_error("Shared ROM action mismatch");
        });
        std::ofstream episode_log(directory / "episodes.jsonl");
        if (!episode_log) throw std::runtime_error("Cannot write shared episode log");
        training_start = last_save = Clock::now(); report("training", 0);
        while (options.max_steps - total_steps >= n && elapsed(training_start) < options.max_seconds && !stopped()) {
            const int horizon = static_cast<int>(std::min<std::int64_t>(options.rollout_steps, (options.max_steps - total_steps) / n));
            const int capacity = horizon * n;
            std::vector<std::uint8_t> frames(static_cast<std::size_t>(capacity) * pixels), ended(capacity);
            std::vector<float> rewards(capacity), values(capacity), old_logprob(capacity);
            std::vector<std::int64_t> actions(capacity); std::vector<Step> transitions(n);
            std::vector<Observation> time_limit_observations(n); int steps = 0;
            for (int t = 0; t < horizon; ++t) {
                auto* current = frames.data() + static_cast<std::size_t>(t * n) * pixels;
                for (int e = 0; e < n; ++e) std::memcpy(current + e * pixels, environments[e]->observation().data(), pixels);
                {
                    torch::NoGradGuard guard; auto features = model->encode(observations(current, n, device));
                    for (int game = 0; game < game_count; ++game) {
                        auto [logits, value] = model->head(features.narrow(0, game * game_environments, game_environments), game);
                        auto action = torch::multinomial(logits.softmax(1), 1);
                        auto logprob = logits.log_softmax(1).gather(1, action).squeeze(1).to(torch::kCPU).contiguous();
                        auto cpu_actions = action.squeeze(1).to(torch::kCPU).contiguous(); auto cpu_values = value.to(torch::kCPU).contiguous();
                        const auto offset = t * n + game * game_environments;
                        std::memcpy(actions.data() + offset, cpu_actions.data_ptr<std::int64_t>(), game_environments * sizeof(std::int64_t));
                        std::memcpy(values.data() + offset, cpu_values.data_ptr<float>(), game_environments * sizeof(float));
                        std::memcpy(old_logprob.data() + offset, logprob.data_ptr<float>(), game_environments * sizeof(float));
                    }
                }
                workers.run([&](int e) {
                    transitions[e] = environments[e]->step(static_cast<int>(actions[t * n + e]));
                    if (transitions[e].truncated && !transitions[e].done) time_limit_observations[e] = environments[e]->observation();
                    if (transitions[e].done || transitions[e].truncated) environments[e]->reset();
                });
                for (int e = 0; e < n; ++e) {
                    const int game = e / game_environments, index = t * n + e; const auto& transition = transitions[e];
                    rewards[index] = std::clamp(transition.reward, -1.f, 1.f); ended[index] = transition.done || transition.truncated;
                    if (transition.truncated && !transition.done) {
                        torch::NoGradGuard guard;
                        rewards[index] += .99f * model->forward(observations(time_limit_observations[e].data(), 1, device), game).second.item<float>();
                    }
                    if (ended[index]) {
                        ++completed[game]; recent_sum[game] += transition.episode_return; ++recent_count[game];
                        episode_log << json({{"game_id", games[game].id}, {"episode", completed[game]}, {"environment", e},
                            {"agent_decisions", total_steps + n}, {"game_agent_decisions", (total_steps + n) / game_count},
                            {"raw_return", transition.episode_return}, {"raw_frames", transition.episode_raw_frames},
                            {"terminated", transition.done}, {"truncated", transition.truncated}}).dump() << '\n';
                    }
                }
                total_steps += n; ++steps;
                if (stopped() || elapsed(training_start) >= options.max_seconds) break;
            }
            const int batch = steps * n;
            rewards.resize(batch); values.resize(batch); ended.resize(batch);
            std::vector<std::uint8_t> last_frames(static_cast<std::size_t>(n) * pixels); std::vector<float> last_values(n);
            for (int e = 0; e < n; ++e) std::memcpy(last_frames.data() + e * pixels, environments[e]->observation().data(), pixels);
            {
                torch::NoGradGuard guard; auto features = model->encode(observations(last_frames.data(), n, device));
                for (int game = 0; game < game_count; ++game) {
                    auto value = model->head(features.narrow(0, game * game_environments, game_environments), game).second.to(torch::kCPU).contiguous();
                    std::memcpy(last_values.data() + game * game_environments, value.data_ptr<float>(), game_environments * sizeof(float));
                }
            }
            auto estimates = compute_gae(rewards, values, ended, last_values, steps, n);
            auto all_observations = observations(frames.data(), batch, device);
            auto floats = [&](std::vector<float>& data) { return torch::from_blob(data.data(), {batch}, torch::kFloat32).to(device); };
            auto advantage = floats(estimates.advantages), returns = floats(estimates.returns), previous_logprob = floats(old_logprob);
            auto all_actions = torch::from_blob(actions.data(), {batch}, torch::kInt64).to(device);
            const auto partitions = balanced_partitions(steps, n, device, game_count);
            for (int game = 0; game < game_count; ++game) {
                auto part = advantage.index_select(0, partitions[game]);
                advantage.index_copy_(0, partitions[game], (part - part.mean()) / (part.std(false) + 1e-8));
            }
            double total_loss = 0; int minibatches = 0;
            for (int epoch = 0; epoch < options.epochs; ++epoch) {
                std::vector<torch::Tensor> order(game_count);
                for (int game = 0; game < game_count; ++game) order[game] = partitions[game].index_select(0,
                    torch::randperm(batch / game_count, torch::TensorOptions().dtype(torch::kInt64).device(device)));
                for (int offset = 0; offset < batch / game_count; offset += options.minibatch_size / game_count) {
                    const int count = std::min(options.minibatch_size / game_count, batch / game_count - offset);
                    std::vector<torch::Tensor> loss(game_count);
                    for (int game = 0; game < game_count; ++game) {
                        auto indices = order[game].narrow(0, offset, count);
                        loss[game] = ppo_loss(model, game, all_observations.index_select(0, indices),
                            all_actions.index_select(0, indices), previous_logprob.index_select(0, indices),
                            advantage.index_select(0, indices), returns.index_select(0, indices));
                    }
                    // Preserve the original two-game arithmetic/order exactly.
                    auto balanced_loss = game_count == 2 ? .5 * (loss[0] + loss[1]) : (loss[0] + loss[1] + loss[2]) / 3.0;
                    const double scalar = balanced_loss.item<double>();
                    if (!std::isfinite(scalar)) throw std::runtime_error("Non-finite shared PPO loss");
                    optimizer.zero_grad(); balanced_loss.backward();
                    const auto norm = torch::nn::utils::clip_grad_norm_(model->parameters(), .5);
                    if (!std::isfinite(norm)) throw std::runtime_error("Non-finite shared PPO gradient");
                    optimizer.step(); total_loss += scalar; ++minibatches;
                }
            }
            ++updates; episode_log.flush();
            if (!episode_log) throw std::runtime_error("Cannot flush shared episode evidence");
            if (updates == 1 || elapsed(last_save) >= 60 || stopped()) save(false);
            report("training", total_loss / std::max(1, minibatches)); std::cout << progress.dump() << std::endl;
            for (int game = 0; game < game_count; ++game) if (recent_count[game] >= 100) { recent_count[game] = 0; recent_sum[game] = 0; }
        }
        training_seconds = elapsed(training_start);
        stop_reason = interrupted ? "signal" : fs::exists(directory / "STOP") ? "STOP"
            : options.max_steps - total_steps < n ? "target_steps" : "time_budget";
        save(true);
        if (stopped()) { report("stopped", 0); return progress; }
        if (!options.evaluate_at_end) { report("complete_without_evaluation", 0); return progress; }
        report("evaluating", 0);
        json retention = {{"status", "incomplete"}, {"training_seed", options.seed}, {"source_checkpoint_sha256", checkpoint_hash},
            {"total_new_decisions", total_steps}, {"per_game_new_decisions", total_steps / game_count}, {"games", json::object()},
            {"interpretation", "One training seed, paired evaluation seeds; descriptive retention measurement, not a generalization claim."}};
        for (int game = 0; game < game_count && !stopped(); ++game) {
            EvalOptions evaluation; evaluation.game = games[game]; evaluation.rom_path = roms[game].string();
            evaluation.checkpoint_path = (directory / games[game].id / "final.pt").string();
            evaluation.output_path = (directory / games[game].id / "evaluation.json").string();
            evaluation.stop_path = (directory / "STOP").string();
            evaluation.seed = paired_seeds[game]; evaluation.episodes = games[game].criterion.episodes;
            evaluation.max_seconds = options.evaluation_max_seconds;
            after[game] = run_evaluation(evaluation);
            retention["games"][games[game].id] = paired_result(before[game], after[game], games[game]);
            atomic_json(directory / "retention.json", retention);
            if (game == 0) atomic_json(directory / "evaluation.json", after[game]);
            if (after[game].at("status") != "complete") break;
        }
        bool complete = retention["games"].size() == static_cast<std::size_t>(game_count);
        bool full_games = complete;
        for (const auto& item : retention["games"].items()) {
            complete = complete && item.value().at("complete").get<bool>();
            full_games = full_games && item.value().at("full_game_comparison").get<bool>();
        }
        retention["status"] = complete ? "complete" : "incomplete";
        retention["bounded_episode_comparison"] = complete;
        retention["full_game_comparison"] = full_games;
        retention["shared_checkpoint_sha256"] = hash_file(directory / "shared-final.pt");
        retention["source_pretraining_decisions"] = source_steps;
        retention["previous_game_decisions"] = config.at("previous_game_decisions");
        retention["joint_checkpoint_game_order"] = config.at("joint_checkpoint_game_order");
        for (int game = 0; game < game_count; ++game) {
            auto& record = retention["artifacts"][games[game].id];
            for (const auto& stage : {std::string("before"), std::string("after")}) {
                const auto path = stage == "before" ? directory / "before" / games[game].id : directory / games[game].id;
                const auto checkpoint_path = path / (stage == "before" ? "initial.pt" : "final.pt");
                record[stage] = {{"checkpoint", checkpoint_path.string()}, {"checkpoint_sha256", hash_file(checkpoint_path)},
                    {"config_sha256", hash_file(path / "config.json")}};
                if (fs::exists(path / "evaluation.json")) record[stage]["evaluation_sha256"] = hash_file(path / "evaluation.json");
            }
        }
        atomic_json(directory / "retention.json", retention);
        report(stopped() ? "stopped" : complete ? "complete" : "evaluation_incomplete", 0);
        progress["retention"] = retention; atomic_json(directory / "progress.json", progress); return progress;
    } catch (const std::exception& error) {
        progress["status"] = "failed"; progress["error"] = error.what(); atomic_json(directory / "progress.json", progress); throw;
    }
}

nlohmann::json run_shared_training(const SharedTrainOptions& options) {
    return run_shared_impl(options, false);
}

nlohmann::json run_shared_extension(const SharedExtendOptions& extension) {
    SharedTrainOptions options;
    options.run_dir = extension.run_dir; options.rom_directory = extension.rom_directory;
    options.source_checkpoint_path = extension.source_checkpoint_path; options.games = extension.games;
    options.seed = extension.seed; options.max_steps = extension.max_steps; options.max_seconds = extension.max_seconds;
    options.num_envs = extension.num_envs; options.rollout_steps = extension.rollout_steps; options.epochs = extension.epochs;
    options.minibatch_size = extension.minibatch_size; options.torch_threads = extension.torch_threads;
    options.evaluate_at_end = extension.evaluate_at_end;
    options.evaluation_max_seconds = extension.evaluation_max_seconds;
    return run_shared_impl(options, true);
}

nlohmann::json check_shared_checkpoint_roundtrip(const std::string& output_directory) {
    const fs::path directory(output_directory);
    if (fs::exists(directory) && (!fs::is_directory(directory) || !fs::is_empty(directory)))
        throw std::invalid_argument("Shared selfcheck directory must be new or empty");
    fs::create_directories(directory); torch::set_num_threads(1); torch::manual_seed(71);
    // Completing all capped episodes is complete measurement even when it fails
    // the separate no-truncation criterion. Missing episode rows are incomplete.
    GameSpec comparison_game; comparison_game.id = "pong"; comparison_game.criterion.episodes = 20;
    json baseline = {{"status", "complete"}, {"episodes_completed", 20}, {"truncated_games", 0},
        {"seed_start", 100}, {"mean_return", 2.0}, {"wins", 20}, {"passed", true}, {"episodes", json::array()}};
    for (int i = 0; i < 20; ++i) baseline["episodes"].push_back({{"seed", 100 + i}, {"raw_return", 2.0}, {"terminated", true}, {"truncated", false}});
    auto natural = paired_result(baseline, baseline, comparison_game);
    if (!natural.at("complete").get<bool>() || !natural.at("full_game_comparison").get<bool>() ||
        !natural.at("criterion_retained").get<bool>()) throw std::runtime_error("Natural paired evaluation classification failed");
    auto capped = baseline; capped["truncated_games"] = 1; capped["passed"] = false;
    capped["episodes"][0]["truncated"] = true;
    auto measured = paired_result(baseline, capped, comparison_game);
    if (!measured.at("complete").get<bool>() || !measured.at("bounded_episode_comparison").get<bool>() ||
        measured.at("full_game_comparison").get<bool>() || measured.at("criterion_retained").get<bool>() ||
        measured.at("limitations").empty()) throw std::runtime_error("Capped evaluation was mistaken for incomplete measurement or a retained criterion");
    capped["episodes"].erase(capped["episodes"].end() - 1);
    auto missing = paired_result(baseline, capped, comparison_game);
    if (missing.at("complete").get<bool>() || missing.at("bounded_episode_comparison").get<bool>() ||
        missing.at("full_game_comparison").get<bool>()) throw std::runtime_error("Missing paired episode was accepted as complete measurement");
    SharedNetwork model(std::array<int, 2>{6, 4});
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    auto input = torch::randint(256, {4, 4, 84, 84}, torch::TensorOptions().dtype(torch::kUInt8));
    auto [pong_logits, pong_value] = model->forward(input, 0);
    auto [breakout_logits, breakout_value] = model->forward(input, 1);
    if (pong_logits.sizes() != torch::IntArrayRef({4, 6}) || breakout_logits.sizes() != torch::IntArrayRef({4, 4}))
        throw std::runtime_error("Shared head dimensions are incorrect");
    auto before_encoder = model->hidden->weight.detach().clone();
    auto before_pong = model->actor->weight.detach().clone(), before_breakout = model->breakout_actor->weight.detach().clone();
    const auto actions = torch::zeros({4}, torch::kInt64), advantages = torch::tensor({1.f, -.5f, .3f, -.8f});
    auto pong_old = pong_logits.detach().log_softmax(1).select(1, 0);
    auto breakout_old = breakout_logits.detach().log_softmax(1).select(1, 0);
    auto pong_loss = ppo_loss(model, 0, input, actions, pong_old, advantages, torch::ones({4}));
    auto breakout_loss = ppo_loss(model, 1, input, actions, breakout_old, advantages, -torch::ones({4}));
    optimizer.zero_grad(); pong_loss.backward();
    const auto pong_gradient = model->hidden->weight.grad().clone();
    if (model->breakout_actor->weight.grad().defined()) throw std::runtime_error("Pong loss reached Breakout head");
    optimizer.zero_grad(); breakout_loss.backward(); const auto breakout_gradient = model->hidden->weight.grad().clone();
    optimizer.zero_grad();
    auto balanced = .5 * (ppo_loss(model, 0, input, actions, pong_old, advantages, torch::ones({4})) +
                          ppo_loss(model, 1, input, actions, breakout_old, advantages, -torch::ones({4})));
    balanced.backward();
    if (!torch::allclose(model->hidden->weight.grad(), .5 * (pong_gradient + breakout_gradient), 1e-4, 1e-6))
        throw std::runtime_error("Shared encoder gradient does not give both games equal weight");
    if (!torch::isfinite(balanced).all().item<bool>()) throw std::runtime_error("Unequal shared heads produced nonfinite loss");
    optimizer.step();
    if (torch::equal(before_encoder, model->hidden->weight) || torch::equal(before_pong, model->actor->weight) ||
        torch::equal(before_breakout, model->breakout_actor->weight)) throw std::runtime_error("Shared update did not train both heads and encoder");
    save_joint(model, optimizer, 1024, 5000, directory / "shared.pt");
    SharedNetwork restored(std::array<int, 2>{6, 4}, false);
    torch::serialize::InputArchive archive, weights, state; archive.load_from((directory / "shared.pt").string(), torch::Device(torch::kCPU));
    archive.read("model", weights); restored->load(weights);
    torch::optim::Adam restored_optimizer(restored->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    archive.read("optimizer", state); restored_optimizer.load(state);
    torch::Tensor steps, counts; archive.read("steps", steps); archive.read("action_counts", counts);
    if (steps.item<std::int64_t>() != 1024 || !torch::equal(counts, torch::tensor({6, 4}, torch::kInt64)) ||
        restored_optimizer.state().size() != model->parameters().size()) throw std::runtime_error("Joint checkpoint metadata or Adam incomplete");
    for (std::size_t index = 0; index < model->parameters().size(); ++index) {
        const auto original = model->parameters()[index], copied = restored->parameters()[index];
        if (!torch::equal(original, copied)) throw std::runtime_error("Joint model roundtrip differs");
        const auto& a = static_cast<const torch::optim::AdamParamState&>(*optimizer.state().at(original.unsafeGetTensorImpl()));
        const auto& b = static_cast<const torch::optim::AdamParamState&>(*restored_optimizer.state().at(copied.unsafeGetTensorImpl()));
        if (a.step() != b.step() || !torch::equal(a.exp_avg(), b.exp_avg()) || !torch::equal(a.exp_avg_sq(), b.exp_avg_sq()))
            throw std::runtime_error("Joint Adam roundtrip differs");
    }
    json result = {{"joint_model_and_adam_roundtrip", true}, {"balanced_encoder_gradients", true},
        {"measurement_completion_distinct_from_full_game_criterion", true},
        {"both_heads_updated", true}, {"unequal_action_heads", {6, 4}}, {"exports", json::array()}};
    Observation first{}; std::memcpy(first.data(), input[0].contiguous().data_ptr<std::uint8_t>(), pixels);
    for (int game = 0; game < 2; ++game) {
        const auto path = directory / (game == 0 ? "pong.pt" : "breakout.pt");
        export_policy(model, game, 512, path); Policy policy(path.string());
        SharedNetwork exported(std::array<int, 2>{6, 4}, false); load_export(exported, game, path);
        torch::NoGradGuard guard;
        const auto expected = model->forward(input, game), actual = exported->forward(input, game);
        if (!torch::equal(expected.first, actual.first) || !torch::equal(expected.second, actual.second) ||
            policy.action_count() != model->actions[game] || policy.predict(first) != expected.first[0].argmax().item<int>())
            throw std::runtime_error("Single-game shared export differs from joint policy");
        result["exports"].push_back({{"game", game == 0 ? "pong" : "breakout"}, {"sha256", hash_file(path)}, {"exact_logits_and_values", true}});
    }
    SharedNetwork initialized(std::array<int, 2>{6, 4});
    const auto fresh_breakout_actor = initialized->breakout_actor->weight.detach().clone();
    const auto fresh_breakout_critic = initialized->breakout_critic->weight.detach().clone();
    load_pong_source(initialized, directory / "pong.pt");
    torch::optim::Adam fresh_optimizer(initialized->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    {
        torch::NoGradGuard guard;
        const auto expected = model->forward(input, 0), actual = initialized->forward(input, 0);
        if (!torch::equal(expected.first, actual.first) || !torch::equal(expected.second, actual.second) ||
            !torch::equal(fresh_breakout_actor, initialized->breakout_actor->weight) ||
            !torch::equal(fresh_breakout_critic, initialized->breakout_critic->weight) || !fresh_optimizer.state().empty())
            throw std::runtime_error("Source initialization did not preserve Pong and fresh Breakout/Adam");
    }
    result["source_pong_restored_breakout_and_adam_fresh"] = true;

    // Real two-to-three transfer: preserve both learned policies, initialize
    // only the new head, and verify equal thirds through the shared encoder.
    SharedNetwork three(std::array<int, 3>{6, 4, 6});
    std::array<torch::Tensor, 4> new_head_before{three->invaders_actor->weight.detach().clone(),
        three->invaders_actor->bias.detach().clone(), three->invaders_critic->weight.detach().clone(),
        three->invaders_critic->bias.detach().clone()};
    auto three_transfer = restore_two_game_joint(three, directory / "shared.pt");
    torch::optim::Adam three_optimizer(three->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    if (!three_optimizer.state().empty() || !torch::equal(new_head_before[0], three->invaders_actor->weight) ||
        !torch::equal(new_head_before[1], three->invaders_actor->bias) || !torch::equal(new_head_before[2], three->invaders_critic->weight) ||
        !torch::equal(new_head_before[3], three->invaders_critic->bias)) throw std::runtime_error("Three-game transfer changed new head or restored Adam");
    if (three_transfer.at("source_pretraining_decisions") != 6024 ||
        three_transfer.at("previous_game_decisions") != json({{"pong", 5512}, {"breakout", 512}, {"space_invaders", 0}}))
        throw std::runtime_error("Three-game transfer conflated total source cost and game decisions");
    for (int game = 0; game < 2; ++game) {
        torch::NoGradGuard guard; const auto expected = model->forward(input, game), actual = three->forward(input, game);
        if (!torch::equal(expected.first, actual.first) || !torch::equal(expected.second, actual.second))
            throw std::runtime_error("Two-to-three transfer failed exact learned-policy preservation");
    }
    // A synthetic, explicitly named lineage fixture exercises completion and
    // cryptographic linkage; these are CPU test data, never Atari game evidence.
    const auto fixture = directory / "synthetic-lineage-fixture";
    fs::create_directories(fixture);
    save_joint(model, optimizer, 1024, 5000, fixture / "shared-final.pt");
    std::vector<GameSpec> fixture_games(3);
    const std::array<std::string, 3> names{"pong", "breakout", "space_invaders"};
    for (int game = 0; game < 3; ++game) {
        fixture_games[game].id = names[game]; fixture_games[game].title = names[game];
        fixture_games[game].rom_filename = names[game] + ".bin";
        fixture_games[game].rom_sha256 = std::string(64, static_cast<char>('a' + game));
        fixture_games[game].expected_actions = game == 1 ? 4 : 6;
    }
    auto fixture_config = policy_config(fixture_games[0], fixture / "not-a-real-rom-pong.bin");
    fixture_config["training_mode"] = "shared"; fixture_config["joint_checkpoint_game_order"] = {"pong", "breakout"};
    fixture_config["shared_games"] = {game_spec_json(fixture_games[0]), game_spec_json(fixture_games[1])};
    fixture_config["minibatch_game_weights"] = {{"pong", .5}, {"breakout", .5}};
    fixture_config["advantage_normalization"] = "separate within each game and rollout";
    fixture_config["initialization_mode"] = "source_encoder_and_pong_heads_fresh_breakout_heads_and_adam";
    fixture_config["environments"] = 8; fixture_config["environments_per_game"] = 4;
    fixture_config["rollout_steps"] = 128; fixture_config["epochs"] = 4; fixture_config["minibatch_size"] = 256;
    fixture_config["max_agent_decisions"] = 1024; fixture_config["previous_trained_steps"] = 5000;
    atomic_json(fixture / "config.json", fixture_config);
    atomic_json(fixture / "progress.json", {{"status", "complete"}, {"agent_decisions", 1024}});
    atomic_json(fixture / "checkpoint-manifest.json", {{"game_order", {"pong", "breakout"}},
        {"shared_checkpoint", (fixture / "shared-final.pt").string()}, {"shared_checkpoint_sha256", hash_file(fixture / "shared-final.pt")},
        {"config_sha256", hash_file(fixture / "config.json")}, {"total_new_decisions", 1024},
        {"per_game_new_decisions", 512}, {"source_pretraining_decisions", 5000}});
    json fixture_retention = {{"status", "complete"}, {"total_new_decisions", 1024},
        {"shared_checkpoint_sha256", hash_file(fixture / "shared-final.pt")}, {"games", json::object()}, {"artifacts", json::object()}};
    for (int game = 0; game < 2; ++game) {
        fixture_retention["games"][names[game]] = {{"complete", true}, {"before_mean_raw_return", 2.0},
            {"after_mean_raw_return", 2.0}, {"before_truncated_games", 0}, {"after_truncated_games", 0}, {"before_passed", true}, {"after_passed", true}};
        for (const auto& stage : {std::string("before"), std::string("after")}) {
            const auto path = stage == "before" ? fixture / "before" / names[game] : fixture / names[game];
            const auto policy_path = path / (stage == "before" ? "initial.pt" : "final.pt");
            atomic_json(path / "config.json", policy_config(fixture_games[game], path / "not-a-real-rom.bin"));
            export_policy(model, game, game == 0 ? 5512 : 512, policy_path);
            auto evaluation = baseline; evaluation["game_id"] = names[game]; evaluation["positive_games"] = 20;
            evaluation["checkpoint_sha256"] = hash_file(policy_path);
            evaluation["checkpoint_config_sha256"] = hash_file(path / "config.json");
            evaluation["seed_start"] = evaluation_seeds[game];
            for (int row = 0; row < 20; ++row) evaluation["episodes"][row]["seed"] = evaluation_seeds[game] + row;
            atomic_json(path / "evaluation.json", evaluation);
            fixture_retention["artifacts"][names[game]][stage] = {
                {"checkpoint_sha256", hash_file(policy_path)}, {"config_sha256", hash_file(path / "config.json")},
                {"evaluation_sha256", hash_file(path / "evaluation.json")}};
        }
    }
    atomic_json(fixture / "retention.json", fixture_retention);
    three_transfer = restore_two_game_joint(three, fixture / "shared-final.pt");
    validate_joint_lineage(three, three_transfer, fs::canonical(fixture / "shared-final.pt"), fixture_games, fixture_config, fixture_config);
    atomic_json(fixture / "progress.json", {{"status", "evaluation_incomplete"}, {"agent_decisions", 1024}});
    bool rejected_incomplete_source = false;
    try { validate_joint_lineage(three, three_transfer, fs::canonical(fixture / "shared-final.pt"), fixture_games, fixture_config, fixture_config); }
    catch (const std::invalid_argument&) { rejected_incomplete_source = true; }
    atomic_json(fixture / "progress.json", {{"status", "complete"}, {"agent_decisions", 1024}});
    if (!rejected_incomplete_source) throw std::runtime_error("Extension accepted unfinished source measurements");
    auto wrong_hash = fixture_retention; wrong_hash["shared_checkpoint_sha256"] = std::string(64, '0');
    atomic_json(fixture / "retention.json", wrong_hash);
    bool rejected_wrong_hash = false;
    try { validate_joint_lineage(three, three_transfer, fs::canonical(fixture / "shared-final.pt"), fixture_games, fixture_config, fixture_config); }
    catch (const std::invalid_argument&) { rejected_wrong_hash = true; }
    atomic_json(fixture / "retention.json", fixture_retention);
    if (!rejected_wrong_hash) throw std::runtime_error("Extension accepted joint checkpoint evidence with a different hash");
    {
        torch::NoGradGuard guard; const auto old_bias = model->actor->bias.detach().clone();
        model->actor->bias[0].fill_(std::numeric_limits<float>::quiet_NaN());
        save_joint(model, optimizer, 1024, 5000, directory / "invalid-nonfinite-joint.pt");
        model->actor->bias.copy_(old_bias);
    }
    bool rejected_nonfinite = false;
    try { restore_two_game_joint(three, directory / "invalid-nonfinite-joint.pt"); }
    catch (const std::invalid_argument&) { rejected_nonfinite = true; }
    if (!rejected_nonfinite) throw std::runtime_error("Extension accepted nonfinite source weights");
    std::array<torch::Tensor, 3> three_old, three_gradients;
    std::array<torch::Tensor, 3> head_weights{three->actor->weight, three->breakout_actor->weight, three->invaders_actor->weight};
    std::array<torch::Tensor, 3> heads_before{head_weights[0].detach().clone(), head_weights[1].detach().clone(), head_weights[2].detach().clone()};
    for (int game = 0; game < 3; ++game) {
        three_old[game] = three->forward(input, game).first.detach().log_softmax(1).select(1, 0);
        three_optimizer.zero_grad();
        auto loss = ppo_loss(three, game, input, actions, three_old[game], advantages, torch::full({4}, game + 1.f));
        loss.backward(); three_gradients[game] = three->hidden->weight.grad().clone();
        for (int other = 0; other < 3; ++other) if (other != game && head_weights[other].grad().defined())
            throw std::runtime_error("Three-game loss reached another game's actor head");
    }
    three_optimizer.zero_grad();
    auto combined_loss = (ppo_loss(three, 0, input, actions, three_old[0], advantages, torch::ones({4})) +
        ppo_loss(three, 1, input, actions, three_old[1], advantages, torch::full({4}, 2.f)) +
        ppo_loss(three, 2, input, actions, three_old[2], advantages, torch::full({4}, 3.f))) / 3.0;
    combined_loss.backward();
    if (!torch::allclose(three->hidden->weight.grad(), (three_gradients[0] + three_gradients[1] + three_gradients[2]) / 3.0, 1e-4, 1e-6))
        throw std::runtime_error("Three-game encoder gradient is not balanced in thirds");
    three_optimizer.step();
    for (int game = 0; game < 3; ++game) if (torch::equal(heads_before[game], head_weights[game]))
        throw std::runtime_error("Three-game update missed a head");
    const std::vector<std::int64_t> prior_decisions{5512, 512, 0};
    save_joint(three, three_optimizer, 1536, 6024, directory / "three-game.pt", prior_decisions);
    SharedNetwork three_restored(std::array<int, 3>{6, 4, 6}, false);
    torch::serialize::InputArchive three_archive, three_weights, three_state;
    three_archive.load_from((directory / "three-game.pt").string(), torch::Device(torch::kCPU));
    three_archive.read("model", three_weights); three_restored->load(three_weights);
    torch::optim::Adam three_restored_optimizer(three_restored->parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    three_archive.read("optimizer", three_state); three_restored_optimizer.load(three_state);
    torch::Tensor recorded_prior, cumulative;
    three_archive.read("previous_game_steps", recorded_prior); three_archive.read("cumulative_game_steps", cumulative);
    if (!torch::equal(recorded_prior, torch::tensor(prior_decisions, torch::kInt64)) ||
        !torch::equal(cumulative, torch::tensor({6024, 1024, 512}, torch::kInt64)) ||
        three_restored_optimizer.state().size() != three->parameters().size()) throw std::runtime_error("Three-game metadata or Adam incomplete");
    for (std::size_t i = 0; i < three->parameters().size(); ++i) {
        const auto original = three->parameters()[i], copied = three_restored->parameters()[i];
        if (!torch::equal(original, copied)) throw std::runtime_error("Three-game model roundtrip differs");
        const auto& a = static_cast<const torch::optim::AdamParamState&>(*three_optimizer.state().at(original.unsafeGetTensorImpl()));
        const auto& b = static_cast<const torch::optim::AdamParamState&>(*three_restored_optimizer.state().at(copied.unsafeGetTensorImpl()));
        if (a.step() != b.step() || !torch::equal(a.exp_avg(), b.exp_avg()) || !torch::equal(a.exp_avg_sq(), b.exp_avg_sq()))
            throw std::runtime_error("Three-game Adam roundtrip differs");
    }
    result["three_game_extension"] = {{"both_learned_policies_exact", true}, {"new_head_and_adam_fresh", true},
        {"equal_third_encoder_gradients", true}, {"head_isolation", true}, {"all_heads_updated", true},
        {"joint_model_and_adam_roundtrip", true}, {"source_lineage_validated", true}, {"incomplete_source_rejected", true},
        {"wrong_evidence_hash_rejected", true}, {"nonfinite_source_rejected", true}, {"exports", json::array()}};
    for (int game = 0; game < 3; ++game) {
        const auto path = directory / ("three-" + names[game] + ".pt");
        export_policy(three, game, prior_decisions[game] + 512, path);
        SharedNetwork exported(std::array<int, 3>{6, 4, 6}, false); load_export(exported, game, path);
        torch::NoGradGuard guard; const auto expected = three->forward(input, game), actual = exported->forward(input, game);
        Policy policy(path.string());
        if (!torch::equal(expected.first, actual.first) || !torch::equal(expected.second, actual.second) ||
            policy.action_count() != three->actions[game] || policy.predict(first) != expected.first[0].argmax().item<int>())
            throw std::runtime_error("Three-game export differs from joint policy");
        result["three_game_extension"]["exports"].push_back({{"game_id", names[game]}, {"sha256", hash_file(path)}, {"exact_logits_and_values", true}});
    }
    comparison_game.id = "breakout";
    auto non_pong_natural = paired_result(baseline, baseline, comparison_game);
    auto capped_complete = baseline; capped_complete["truncated_games"] = 1; capped_complete["passed"] = false;
    capped_complete["episodes"][0]["truncated"] = true;
    auto non_pong_capped = paired_result(baseline, capped_complete, comparison_game);
    if (!non_pong_natural.at("criterion_retained").get<bool>() || !non_pong_capped.at("complete").get<bool>() ||
        !non_pong_capped.at("criterion_retention_applicable").get<bool>() || non_pong_capped.at("criterion_retained").get<bool>())
        throw std::runtime_error("Non-Pong capped criterion retention classification failed");
    result["three_game_extension"]["non_pong_capped_criterion_retention"] = true;
    atomic_json(directory / "result.json", result); return result;
}
} // namespace atari
