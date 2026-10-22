#include "atari/learning.hpp"
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void near(float actual, float expected) { require(std::abs(actual - expected) < 1e-5f, "GAE value mismatch"); }
nlohmann::json known_preprocessing() {
    // This is the protocol recorded by the preserved Pong proof and current runs.
    return {{"observation", "4x84x84 grayscale, max of last two repeated frames"},
        {"frame_skip", 4}, {"sticky_action_probability", .25}, {"noop_max", 30},
        {"fire_reset", true}, {"max_episode_raw_frames", 108000},
        {"terminal_on_life_loss", false}, {"training_reward_clipping", true},
        {"evaluation_reward_clipping", false}, {"time_limit_bootstrap", true}};
}

void resume_compatibility() {
    const nlohmann::json expected = {{"algorithm", "PPO"}, {"language", "C++20"},
        {"torch_version", "2.7.1"}, {"training_action_sampling", "categorical"},
        {"rom_sha256", std::string(64, 'a')}, {"action_count", 6},
        {"preprocessing", known_preprocessing()},
        {"network", "conv32/8/4 conv64/4/2 conv64/3/1 fc512 actor+critic"},
        {"gamma", .99}, {"gae_lambda", .95}, {"clip_range", .1},
        {"entropy_coefficient", .01}, {"learning_rate", 2.5e-4},
        {"value_coefficient", .5}, {"max_gradient_norm", .5}};
    // The original cold-start format has no parent/continuation metadata.
    // A fresh run may choose a new seed, budget, and number of workers.
    auto source = expected;
    source["seed"] = 1; source["max_agent_decisions"] = 100000; source["environments"] = 8;
    auto next = expected;
    next["seed"] = 9; next["max_agent_decisions"] = 5000000; next["environments"] = 16;
    atari::validate_resume_config(source, next);
    next["game_id"] = "pong";
    atari::validate_resume_config(source, next); // Legacy Pong configuration remains valid.
    auto rejected = [&](const nlohmann::json& candidate) {
        bool invalid = false;
        try { atari::validate_resume_config(candidate, next); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "Incompatible continuation source accepted");
    };
    for (const char* key : {"rom_sha256", "action_count", "network", "preprocessing", "learning_rate", "torch_version"}) {
        auto missing = source; missing.erase(key); rejected(missing);
        auto changed = source; changed[key] = nullptr; rejected(changed);
    }
    auto changed_game = source; changed_game["rom_sha256"] = std::string(64, 'b'); rejected(changed_game);
    auto wrong_identity = source; wrong_identity["game_id"] = "space_invaders"; rejected(wrong_identity);
    auto changed_frames = source; changed_frames["preprocessing"]["frame_skip"] = 2; rejected(changed_frames);
    rejected(nlohmann::json::array());
}

void evaluation_identity() {
    atari::GameSpec pong;
    pong.id = "pong"; pong.expected_actions = 6; pong.rom_sha256 = std::string(64, 'a');
    nlohmann::json legacy = {{"rom_sha256", pong.rom_sha256}, {"action_count", 6},
        {"network", "conv32/8/4 conv64/4/2 conv64/3/1 fc512 actor+critic"},
        {"preprocessing", known_preprocessing()}};
    atari::validate_evaluation_config(legacy, pong, pong.rom_sha256, 6);
    auto current = legacy;
    current["game_id"] = pong.id; current["game_spec"] = atari::game_spec_json(pong);
    atari::validate_evaluation_config(current, pong, pong.rom_sha256, 6);
    auto rejected = [&](const nlohmann::json& source, const atari::GameSpec& game,
                        const std::string& rom_hash, int actions) {
        bool invalid = false;
        try { atari::validate_evaluation_config(source, game, rom_hash, actions); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "Incompatible evaluation checkpoint accepted");
    };
    // Same action count is insufficient: Space Invaders must not load Pong.
    auto invaders = pong;
    invaders.id = "space_invaders"; invaders.rom_sha256 = std::string(64, 'b');
    rejected(legacy, invaders, invaders.rom_sha256, 6);
    rejected(current, invaders, invaders.rom_sha256, 6);
    rejected(current, pong, invaders.rom_sha256, 6);
    rejected(current, pong, pong.rom_sha256, 4);
    auto missing_hash = current; missing_hash.erase("rom_sha256");
    rejected(missing_hash, pong, pong.rom_sha256, 6);
    // Legacy Pong may omit game_id, but it must still name the actual model and
    // observation/episode protocol. Accepting a different one silently changes
    // the meaning of frozen-policy and matched-seed comparisons.
    for (const auto& valid : {legacy, current}) {
        for (const char* key : {"network", "preprocessing"}) {
            auto missing = valid; missing.erase(key);
            rejected(missing, pong, pong.rom_sha256, 6);
            auto malformed = valid; malformed[key] = nullptr;
            rejected(malformed, pong, pong.rom_sha256, 6);
        }
        auto wrong_network = valid; wrong_network["network"] = "conv32/8/4 conv64/4/2 conv64/3/1 fc256 actor+critic";
        rejected(wrong_network, pong, pong.rom_sha256, 6);
        auto missing_setting = valid; missing_setting["preprocessing"].erase("frame_skip");
        rejected(missing_setting, pong, pong.rom_sha256, 6);
        const nlohmann::json changed_settings = {
            {"frame_skip", 2}, {"sticky_action_probability", 0.0},
            {"terminal_on_life_loss", true}, {"evaluation_reward_clipping", true},
            {"observation", "1x84x84 grayscale"}};
        for (const auto& changed_setting : changed_settings.items()) {
            auto changed = valid;
            changed["preprocessing"][changed_setting.key()] = changed_setting.value();
            rejected(changed, pong, pong.rom_sha256, 6);
        }
    }
    auto wrong_identity = current; wrong_identity["game_id"] = invaders.id;
    rejected(wrong_identity, pong, pong.rom_sha256, 6);
    auto wrong_spec = current; wrong_spec["game_spec"]["id"] = invaders.id;
    rejected(wrong_spec, pong, pong.rom_sha256, 6);
    auto unidentified = legacy; unidentified["rom_sha256"] = invaders.rom_sha256;
    rejected(unidentified, invaders, invaders.rom_sha256, 6);
    unidentified["game_id"] = invaders.id;
    atari::validate_evaluation_config(unidentified, invaders, invaders.rom_sha256, 6);
    rejected(nlohmann::json::array(), pong, pong.rom_sha256, 6);
}

void transfer_compatibility() {
    atari::GameSpec pong;
    pong.id = "pong"; pong.expected_actions = 6; pong.rom_sha256 = std::string(64, 'a');
    nlohmann::json source = {{"rom_sha256", pong.rom_sha256}, {"action_count", 6},
        {"algorithm", "PPO"}, {"language", "C++20"}, {"torch_version", "2.7.1"},
        {"network", "conv32/8/4 conv64/4/2 conv64/3/1 fc512 actor+critic"},
        {"preprocessing", known_preprocessing()}};
    auto target = source;
    target["game_id"] = "breakout"; target["action_count"] = 4;
    target["rom_sha256"] = std::string(64, 'b');
    atari::validate_transfer_config(source, pong, target);
    auto rejected = [&](const nlohmann::json& candidate) {
        bool invalid = false;
        try { atari::validate_transfer_config(candidate, pong, target); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "Incompatible feature-transfer source accepted");
    };
    for (const char* key : {"algorithm", "language", "torch_version", "network", "preprocessing",
                            "rom_sha256", "action_count"}) {
        auto missing = source; missing.erase(key); rejected(missing);
        auto changed = source; changed[key] = nullptr; rejected(changed);
    }
    auto wrong_game = source; wrong_game["game_id"] = "space_invaders"; rejected(wrong_game);
}

// Small modules exercise the production archive-copy path without allocating
// the full CNN. The names and weight/bias nesting match real policy archives.
struct TransferFixture : torch::nn::Module {
    explicit TransferFixture(int actions) {
        for (const char* name : {"conv1", "conv2", "conv3", "hidden"})
            register_module(name, torch::nn::Linear(3, 3));
        register_module("actor", torch::nn::Linear(3, actions));
        register_module("critic", torch::nn::Linear(3, 1));
    }
};

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("atari-transfer-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() {
        require(std::filesystem::create_directory(path), "Could not create isolated transfer test directory");
    }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

std::vector<torch::Tensor> snapshot(torch::nn::Module& model) {
    std::vector<torch::Tensor> result;
    for (const auto& tensor : model.parameters()) result.push_back(tensor.detach().clone());
    return result;
}
void unchanged(torch::nn::Module& model, const std::vector<torch::Tensor>& before) {
    const auto parameters = model.parameters();
    require(parameters.size() == before.size(), "Parameter count changed");
    for (std::size_t i = 0; i < parameters.size(); ++i)
        require(torch::equal(parameters[i], before[i]), "Rejected transfer partially changed target parameters");
}

void feature_copy() {
    TemporaryDirectory directory;
    torch::manual_seed(31);
    TransferFixture source(6);
    torch::optim::Adam source_adam(source.parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    auto loss = torch::zeros({}, torch::kFloat32);
    for (const auto& parameter : source.parameters()) loss = loss + parameter.square().sum();
    loss.backward(); source_adam.step();
    require(!source_adam.state().empty(), "Source fixture must have trained Adam state");
    const auto source_parameters = source.named_parameters();
    auto save = [&](const std::string& name, const std::string& corruption = "") {
        const auto path = directory.path / name;
        torch::serialize::OutputArchive archive, weights, optimizer;
        archive.write("action_count", torch::tensor(corruption == "actions" ? 4 : 6, torch::kInt64));
        archive.write("steps", torch::tensor(corruption == "steps" ? 0 : 123456, torch::kInt64));
        for (const char* module_name : {"conv1", "conv2", "conv3", "hidden", "actor", "critic"}) {
            torch::serialize::OutputArchive module;
            for (const char* parameter_name : {"weight", "bias"}) {
                const std::string key = std::string(module_name) + "." + parameter_name;
                auto value = source_parameters[key].detach().clone();
                // Corrupt the last copied tensor: all previous tensors must
                // still remain untouched when validation rejects this source.
                if (key == "hidden.bias") {
                    if (corruption == "shape") value = torch::zeros({1});
                    if (corruption == "dtype") value = value.to(torch::kFloat64);
                    if (corruption == "nonfinite") value[0] = std::numeric_limits<float>::quiet_NaN();
                }
                module.write(parameter_name, value);
            }
            weights.write(module_name, module);
        }
        source_adam.save(optimizer);
        archive.write("model", weights); archive.write("optimizer", optimizer);
        archive.save_to(path.string()); return path;
    };
    const auto checkpoint = save("source.pt");
    torch::manual_seed(700);
    TransferFixture cold(4);
    const auto cold_parameters = snapshot(cold);
    const auto expected_random = torch::rand({32});
    torch::manual_seed(700);
    TransferFixture target(4);
    unchanged(target, cold_parameters);
    const auto copied = atari::copy_feature_parameters(target, checkpoint.string(), 6);
    require(torch::equal(torch::rand({32}), expected_random), "Feature loading consumed the target RNG stream");
    torch::optim::Adam fresh_adam(target.parameters(), torch::optim::AdamOptions(2.5e-4).eps(1e-5));
    require(fresh_adam.state().empty(), "Target unexpectedly restored source Adam state");
    require(copied.at("source_pretraining_decisions") == 123456, "Source pretraining steps lost");
    require(copied.at("tensor_count") == 8 && copied.at("tensors").size() == 8, "Wrong feature tensor count");
    require(copied.at("source_checkpoint_sha256").get<std::string>().size() == 64, "Source checkpoint hash missing");
    const auto target_parameters = target.named_parameters();
    const auto fresh_parameters = cold.named_parameters();
    for (const auto& parameter : target_parameters) {
        const auto& key = parameter.key();
        const bool head = key.rfind("actor.", 0) == 0 || key.rfind("critic.", 0) == 0;
        require(torch::equal(parameter.value(), head ? fresh_parameters[key] : source_parameters[key]),
                "Transfer did not preserve fresh heads and copy the exact source encoder");
    }
    for (const auto& tensor : copied.at("tensors"))
        require(tensor.at("source") == tensor.at("target"), "Copied feature fingerprints differ");
    for (const char* corruption : {"shape", "dtype", "nonfinite", "actions", "steps"}) {
        TransferFixture invalid_target(4);
        const auto before = snapshot(invalid_target);
        const auto bad_checkpoint = save(std::string(corruption) + ".pt", corruption);
        bool rejected = false;
        try { atari::copy_feature_parameters(invalid_target, bad_checkpoint.string(), 6); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Malformed feature checkpoint accepted");
        unchanged(invalid_target, before);
    }
}

void continuation_origin() {
    const nlohmann::json transferred = {{"initialization_mode", "feature_transfer_fresh_heads_and_adam"},
        {"seed", 21}, {"previous_trained_steps", 0},
        {"initial_head_fingerprints", {{"actor.weight", "fresh-head-sha"}}},
        {"transfer", {{"source_game_id", "pong"}, {"source_checkpoint_sha256", std::string(64, 'a')},
                      {"source_pretraining_decisions", 3031808}}}};
    const auto origin = atari::training_origin_metadata(transferred);
    auto continuation = transferred;
    continuation["previous_trained_steps"] = 100000;
    continuation["initialization_mode"] = "model_and_adam_with_fresh_environments_and_rng";
    continuation["seed"] = 99;
    continuation["initial_head_fingerprints"] = {{"actor.weight", "trained-head-sha"}};
    continuation["training_origin"] = origin;
    require(atari::training_origin_metadata(continuation) == origin,
            "Resume lost or relabeled original feature transfer, seed, or fresh heads");
    auto second = continuation; second["previous_trained_steps"] = 200000;
    second["training_origin"] = atari::training_origin_metadata(continuation);
    require(atari::training_origin_metadata(second) == origin, "Second continuation lost original lineage");
    nlohmann::json legacy = {{"previous_trained_steps", 513792}, {"seed", 9}};
    const auto unknown = atari::training_origin_metadata(legacy);
    require(unknown.at("initialization_mode") == "legacy_origin_not_recorded" && unknown.at("seed").is_null(),
            "Legacy continuation claimed an unrecorded original seed or initialization");
}
}
int main() {
    torch::set_num_threads(1);
    // Terminal reward does not bootstrap from the next episode's large value.
    auto terminal = atari::compute_gae({1, 2}, {.5f, 90}, {1, 0}, {100}, 2, 1, 1, 1);
    near(terminal.advantages[0], .5f); near(terminal.returns[0], 1);
    near(terminal.advantages[1], 12); near(terminal.returns[1], 102);
    // Interleaved environments must keep separate recursive traces.
    auto parallel = atari::compute_gae({1, 10, 2, 20}, {0, 0, 0, 0}, {0, 0, 1, 1}, {99, 99}, 2, 2, 1, 1);
    near(parallel.returns[0], 3); near(parallel.returns[1], 30);
    // A truncated transition bootstraps its final state, then cuts the trace.
    auto truncated = atari::compute_gae({1 + .9f * 5, 100}, {2, 0}, {1, 1}, {0}, 2, 1, .9f, .95f);
    near(truncated.returns[0], 5.5f);
    require(atari::evidence_passes(20, 16, .1, 0), "Valid criterion rejected");
    require(!atari::evidence_passes(19, 19, 10, 0), "Incomplete evidence accepted");
    require(!atari::evidence_passes(20, 15, 10, 0), "Insufficient wins accepted");
    require(!atari::evidence_passes(20, 16, 0, 0), "Nonpositive return accepted");
    require(!atari::evidence_passes(20, 20, 10, 1), "Truncated game accepted");
    bool rejected = false;
    try { atari::compute_gae({1}, {}, {1}, {0}, 1, 1); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Malformed GAE accepted");
    resume_compatibility();
    evaluation_identity();
    transfer_compatibility();
    feature_copy();
    continuation_origin();
    std::cout << "GAE, criteria, resume/evaluation identity, feature copy, fresh heads/RNG/Adam and lineage passed\n";
}
