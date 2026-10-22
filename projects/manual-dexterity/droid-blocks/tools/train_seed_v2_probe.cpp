#include "droid/environment.hpp"
#include "droid/learner.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

struct Weights {
    double bias{0.0};
    double previous_effort{0.0};
    double delta_effort{0.0};
    double delta_velocity{0.0};
};

struct Episode {
    std::uint64_t seed{0};
    bool positive_side{false};
    double reward_rate{0.0};
    double start_x{0.0};
    double end_x{0.0};
    double minimum_x{0.0};
    double maximum_x{0.0};
    double start_lux{0.0};
    double end_lux{0.0};
    double mean_abs_action{0.0};
    double final_mean_action{0.0};
    std::size_t direction_changes{0};
    std::size_t safety_vetoes{0};
    bool terminated{false};
    std::vector<Json> checkpoints;
};

struct Score {
    Weights weights;
    double negative_gain{0.0};
    double positive_gain{0.0};
    double balanced_gain{0.0};
    double worst_gain{0.0};
    double negative_displacement{0.0};
    double positive_displacement{0.0};
    double mean_abs_action{0.0};
    std::size_t safety_vetoes{0};
    std::size_t terminations{0};
};

[[nodiscard]] std::vector<std::string> motor_ids(const Json& spec) {
    return spec.at("action").at("motor_ids").get<std::vector<std::string>>();
}

[[nodiscard]] double mean_action(
    const std::map<std::string, double>& actions) {
    double sum = 0.0;
    for (const auto& [unused, effort] : actions) {
        (void)unused;
        sum += effort;
    }
    return sum / static_cast<double>(actions.size());
}

[[nodiscard]] Episode run_episode(
    const std::filesystem::path& model_path,
    const droid::PolicyArtifactData& template_artifact,
    const Weights& weights,
    std::uint64_t seed,
    std::size_t horizon,
    bool checkpoints) {
    droid::PolicyArtifactData artifact = template_artifact;
    artifact.parameters.fill(0.0);
    artifact.parameters[29] = weights.previous_effort;
    artifact.parameters[30] = weights.bias;
    artifact.parameters[31] = weights.delta_effort;
    artifact.parameters[32] = weights.delta_velocity;
    artifact.self_sha256.clear();

    droid::SharedLinearMemoryPolicy policy(std::move(artifact));
    droid::DroidEnvironment environment(model_path, false);
    const std::vector<std::string> ids = motor_ids(environment.spec());
    const Json reset = environment.reset(
        "demo_rover_v0", seed, false, droid::kLightSearch1dEpisodeProfile);
    policy.reset(ids);
    Json observation = reset.at("observation");
    const Json initial_state = environment.visualization_state();

    Episode result;
    result.seed = seed;
    result.positive_side = droid::light_search_seed_has_positive_side(seed);
    result.start_x = initial_state.at("robot").at("position").at(0).get<double>();
    result.minimum_x = result.start_x;
    result.maximum_x = result.start_x;
    result.start_lux = initial_state.at("sensors").at("ambient_light_lux").get<double>();

    double reward = 0.0;
    double absolute_action = 0.0;
    double previous_mean_action = 0.0;
    int previous_sign = 0;
    std::size_t executed = 0;
    static constexpr std::array<std::size_t, 12> checkpoint_steps{
        0, 1, 2, 4, 9, 19, 49, 99, 199, 399, 699, 999};
    for (std::size_t step = 0; step < horizon; ++step) {
        const auto actions = policy.act(observation, ids);
        const double action = mean_action(actions);
        absolute_action += std::abs(action);
        const int sign = action > 1e-9 ? 1 : (action < -1e-9 ? -1 : 0);
        if (previous_sign != 0 && sign != 0 && sign != previous_sign) {
            ++result.direction_changes;
        }
        if (sign != 0) {
            previous_sign = sign;
        }
        previous_mean_action = action;

        const Json transition = environment.step(actions, 0.02);
        ++executed;
        reward += transition.at("reward").get<double>();
        if (transition.at("info").contains("safety_veto") &&
            transition.at("info").at("safety_veto").get<bool>()) {
            ++result.safety_vetoes;
        }
        observation = transition.at("observation");
        const Json state = environment.visualization_state();
        const double x = state.at("robot").at("position").at(0).get<double>();
        result.minimum_x = std::min(result.minimum_x, x);
        result.maximum_x = std::max(result.maximum_x, x);
        if (checkpoints && std::ranges::find(checkpoint_steps, step) !=
                checkpoint_steps.end()) {
            result.checkpoints.push_back(Json{
                {"step", step + 1U},
                {"x_m", x},
                {"lux", state.at("sensors").at("ambient_light_lux")},
                {"mean_action", action},
                {"mean_velocity_rad_s",
                 [&state] {
                     const auto values = state.at("actuators")
                                             .at("shaft_velocity_rad_s")
                                             .get<std::vector<double>>();
                     double sum = 0.0;
                     for (double value : values) {
                         sum += value;
                     }
                     return sum / static_cast<double>(values.size());
                 }()},
            });
        }
        if (transition.at("terminated").get<bool>() ||
            transition.at("truncated").get<bool>()) {
            result.terminated = transition.at("terminated").get<bool>();
            break;
        }
    }
    const Json final_state = environment.visualization_state();
    result.end_x = final_state.at("robot").at("position").at(0).get<double>();
    result.end_lux = final_state.at("sensors").at("ambient_light_lux").get<double>();
    result.final_mean_action = previous_mean_action;
    result.mean_abs_action = executed == 0 ? 0.0 :
        absolute_action / static_cast<double>(executed);
    result.reward_rate = executed == 0 ? 0.0 :
        reward / (0.02 * static_cast<double>(executed));
    return result;
}

[[nodiscard]] Score score_candidate(
    const std::filesystem::path& model_path,
    const droid::PolicyArtifactData& artifact,
    const Weights& weights,
    std::span<const std::uint64_t> seeds,
    const std::map<std::uint64_t, double>& zero_rates,
    std::size_t horizon) {
    Score result;
    result.weights = weights;
    double negative_gain = 0.0;
    double positive_gain = 0.0;
    double negative_displacement = 0.0;
    double positive_displacement = 0.0;
    double absolute_action = 0.0;
    std::size_t negative_count = 0;
    std::size_t positive_count = 0;
    for (const std::uint64_t seed : seeds) {
        const Episode episode = run_episode(
            model_path, artifact, weights, seed, horizon, false);
        const double gain = episode.reward_rate - zero_rates.at(seed);
        if (episode.positive_side) {
            positive_gain += gain;
            positive_displacement += episode.end_x - episode.start_x;
            ++positive_count;
        } else {
            negative_gain += gain;
            negative_displacement += episode.end_x - episode.start_x;
            ++negative_count;
        }
        absolute_action += episode.mean_abs_action;
        result.safety_vetoes += episode.safety_vetoes;
        result.terminations += episode.terminated ? 1U : 0U;
    }
    result.negative_gain = negative_gain / static_cast<double>(negative_count);
    result.positive_gain = positive_gain / static_cast<double>(positive_count);
    result.balanced_gain = 0.5 * (result.negative_gain + result.positive_gain);
    result.worst_gain = std::min(result.negative_gain, result.positive_gain);
    result.negative_displacement =
        negative_displacement / static_cast<double>(negative_count);
    result.positive_displacement =
        positive_displacement / static_cast<double>(positive_count);
    result.mean_abs_action = absolute_action / static_cast<double>(seeds.size());
    return result;
}

[[nodiscard]] bool better(const Score& a, const Score& b) {
    if (a.terminations != b.terminations) {
        return a.terminations < b.terminations;
    }
    if (a.safety_vetoes != b.safety_vetoes) {
        return a.safety_vetoes < b.safety_vetoes;
    }
    if (a.worst_gain != b.worst_gain) {
        return a.worst_gain > b.worst_gain;
    }
    return a.balanced_gain > b.balanced_gain;
}

[[nodiscard]] Json score_json(const Score& score) {
    return Json{
        {"weights", {{"bias", score.weights.bias},
                     {"previous_effort", score.weights.previous_effort},
                     {"feature31_delta_x_effort", score.weights.delta_effort},
                     {"feature32_delta_x_velocity", score.weights.delta_velocity}}},
        {"negative_gain_vs_zero_reward_per_s", score.negative_gain},
        {"positive_gain_vs_zero_reward_per_s", score.positive_gain},
        {"balanced_gain_vs_zero_reward_per_s", score.balanced_gain},
        {"worst_side_gain_vs_zero_reward_per_s", score.worst_gain},
        {"negative_mean_displacement_m", score.negative_displacement},
        {"positive_mean_displacement_m", score.positive_displacement},
        {"mean_abs_action", score.mean_abs_action},
        {"safety_vetoes", score.safety_vetoes},
        {"terminations", score.terminations},
    };
}

[[nodiscard]] std::vector<std::uint64_t> representative_pair(
    std::span<const std::uint64_t> train) {
    std::vector<std::uint64_t> result;
    for (const bool side : {false, true}) {
        const auto found = std::ranges::find_if(train, [side](std::uint64_t seed) {
            return droid::light_search_seed_has_positive_side(seed) == side;
        });
        if (found == train.end()) {
            throw std::runtime_error("training split is not side-balanced");
        }
        result.push_back(*found);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 4) {
        throw std::invalid_argument(
            "usage: train_seed_v2_probe MODEL TEMPLATE_ARTIFACT WORKERS");
    }
    const std::filesystem::path model_path(argv[1]);
    const droid::PolicyArtifactData artifact =
        droid::load_policy_artifact(argv[2]);
    const std::size_t workers = std::stoull(argv[3]);
    if (workers == 0) {
        throw std::invalid_argument("WORKERS must be positive");
    }

    // Reconstruct the frozen split, then deliberately retain and execute only
    // its preregistered training member. Validation/test members are never
    // passed to an environment by this diagnostic.
    const auto split = droid::make_mirrored_seed_splits(0, 16, 8, 32);
    const std::vector<std::uint64_t> representatives =
        representative_pair(split.train);
    const Weights zero{};
    std::map<std::uint64_t, double> representative_zero_rates;
    for (const std::uint64_t seed : representatives) {
        representative_zero_rates.emplace(
            seed,
            run_episode(model_path, artifact, zero, seed, 1000, false)
                .reward_rate);
    }

    if (argc == 9 && std::string(argv[8]) == "all-train") {
        const Weights supplied{
            std::stod(argv[4]),
            std::stod(argv[5]),
            std::stod(argv[6]),
            std::stod(argv[7]),
        };
        struct SideStats {
            std::size_t count{0};
            std::size_t wins{0};
            double gain_sum{0.0};
            double minimum_gain{std::numeric_limits<double>::infinity()};
            double maximum_gain{-std::numeric_limits<double>::infinity()};
            double displacement_sum{0.0};
            double minimum_toward_displacement{
                std::numeric_limits<double>::infinity()};
        };
        std::array<SideStats, 2> sides;
        std::size_t safety_vetoes = 0;
        std::size_t terminations = 0;
        double mean_abs_action_sum = 0.0;
        for (const std::uint64_t seed : split.train) {
            const Episode zero_episode = run_episode(
                model_path, artifact, zero, seed, 1000, false);
            const Episode learned_episode = run_episode(
                model_path, artifact, supplied, seed, 1000, false);
            const double gain =
                learned_episode.reward_rate - zero_episode.reward_rate;
            SideStats& side = sides[learned_episode.positive_side ? 1U : 0U];
            ++side.count;
            side.wins += gain > 0.0 ? 1U : 0U;
            side.gain_sum += gain;
            side.minimum_gain = std::min(side.minimum_gain, gain);
            side.maximum_gain = std::max(side.maximum_gain, gain);
            const double displacement =
                learned_episode.end_x - learned_episode.start_x;
            const double toward_displacement = learned_episode.positive_side
                ? displacement
                : -displacement;
            side.displacement_sum += toward_displacement;
            side.minimum_toward_displacement = std::min(
                side.minimum_toward_displacement, toward_displacement);
            safety_vetoes += learned_episode.safety_vetoes;
            terminations += learned_episode.terminated ? 1U : 0U;
            mean_abs_action_sum += learned_episode.mean_abs_action;
        }
        const auto side_json = [](const SideStats& side) {
            return Json{
                {"scenario_count", side.count},
                {"wins_vs_zero", side.wins},
                {"mean_gain_vs_zero_reward_per_s",
                 side.gain_sum / static_cast<double>(side.count)},
                {"minimum_gain_vs_zero_reward_per_s", side.minimum_gain},
                {"maximum_gain_vs_zero_reward_per_s", side.maximum_gain},
                {"mean_toward_light_displacement_m",
                 side.displacement_sum / static_cast<double>(side.count)},
                {"minimum_toward_light_displacement_m",
                 side.minimum_toward_displacement},
            };
        };
        std::cout << std::setw(2) << Json{
            {"schema_version", "droid-blocks.train-seed-v2-fixed-controller.v1"},
            {"seed_scope", "all_train_only"},
            {"train_seed_count", split.train.size()},
            {"train_seed_sha256", droid::seed_list_sha256(split.train)},
            {"validation_scenarios_executed", 0},
            {"test_scenarios_executed", 0},
            {"weights", {{"bias", supplied.bias},
                         {"previous_effort", supplied.previous_effort},
                         {"feature31_delta_x_effort", supplied.delta_effort},
                         {"feature32_delta_x_velocity", supplied.delta_velocity}}},
            {"negative_x", side_json(sides[0])},
            {"positive_x", side_json(sides[1])},
            {"balanced_mean_gain_vs_zero_reward_per_s",
             0.5 * (sides[0].gain_sum / static_cast<double>(sides[0].count) +
                    sides[1].gain_sum / static_cast<double>(sides[1].count))},
            {"worst_side_mean_gain_vs_zero_reward_per_s",
             std::min(
                 sides[0].gain_sum / static_cast<double>(sides[0].count),
                 sides[1].gain_sum / static_cast<double>(sides[1].count))},
            {"total_wins_vs_zero", sides[0].wins + sides[1].wins},
            {"mean_abs_action",
             mean_abs_action_sum / static_cast<double>(split.train.size())},
            {"safety_vetoes", safety_vetoes},
            {"terminations", terminations},
        } << '\n';
        return EXIT_SUCCESS;
    }

    if (argc == 8) {
        const Weights supplied{
            std::stod(argv[4]),
            std::stod(argv[5]),
            std::stod(argv[6]),
            std::stod(argv[7]),
        };
        const Score score = score_candidate(
            model_path,
            artifact,
            supplied,
            representatives,
            representative_zero_rates,
            1000);
        Json traces = Json::array();
        for (const std::uint64_t seed : representatives) {
            const Episode episode = run_episode(
                model_path, artifact, supplied, seed, 1000, true);
            traces.push_back(Json{
                {"seed", episode.seed},
                {"side", episode.positive_side ? "positive_x" : "negative_x"},
                {"reward_rate", episode.reward_rate},
                {"zero_reward_rate", representative_zero_rates.at(seed)},
                {"gain_vs_zero_reward_per_s",
                 episode.reward_rate - representative_zero_rates.at(seed)},
                {"start_x_m", episode.start_x},
                {"end_x_m", episode.end_x},
                {"minimum_x_m", episode.minimum_x},
                {"maximum_x_m", episode.maximum_x},
                {"start_lux", episode.start_lux},
                {"end_lux", episode.end_lux},
                {"mean_abs_action", episode.mean_abs_action},
                {"final_mean_action", episode.final_mean_action},
                {"direction_changes", episode.direction_changes},
                {"safety_vetoes", episode.safety_vetoes},
                {"terminated", episode.terminated},
                {"checkpoints", episode.checkpoints},
            });
        }
        std::cout << std::setw(2) << Json{
            {"schema_version", "droid-blocks.train-seed-v2-quick-probe.v1"},
            {"seed_scope", "train_only_representative_pair"},
            {"validation_scenarios_executed", 0},
            {"test_scenarios_executed", 0},
            {"score", score_json(score)},
            {"trajectories", std::move(traces)},
        } << '\n';
        return EXIT_SUCCESS;
    }

    const std::array<double, 7> biases{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0};
    const std::array<double, 7> memories{-1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0};
    const std::array<double, 10> interactions{
        -32.0, -16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0, 32.0, 64.0};
    std::vector<Weights> grid;
    // First isolate each interaction, then include equal signed combinations.
    for (double bias : biases) {
        for (double memory : memories) {
            for (double interaction : interactions) {
                grid.push_back({bias, memory, interaction, 0.0});
                grid.push_back({bias, memory, 0.0, interaction});
                grid.push_back({bias, memory, interaction, interaction});
            }
        }
    }

    std::vector<Score> scores(grid.size());
    std::vector<std::exception_ptr> errors(grid.size());
    std::atomic<std::size_t> next{0};
    std::vector<std::jthread> threads;
    const std::size_t thread_count = std::min(workers, grid.size());
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        threads.emplace_back([&] {
            while (true) {
                const std::size_t index = next.fetch_add(1);
                if (index >= grid.size()) {
                    return;
                }
                try {
                    scores[index] = score_candidate(
                        model_path,
                        artifact,
                        grid[index],
                        representatives,
                        representative_zero_rates,
                        1000);
                } catch (...) {
                    errors[index] = std::current_exception();
                }
            }
        });
    }
    threads.clear();
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    std::stable_sort(scores.begin(), scores.end(), better);

    // Re-score the strongest distinct candidates on all 32 training worlds.
    std::vector<Weights> finalists;
    for (const Score& score : scores) {
        const auto same = [&score](const Weights& value) {
            return value.bias == score.weights.bias &&
                value.previous_effort == score.weights.previous_effort &&
                value.delta_effort == score.weights.delta_effort &&
                value.delta_velocity == score.weights.delta_velocity;
        };
        if (std::ranges::find_if(finalists, same) == finalists.end()) {
            finalists.push_back(score.weights);
        }
        if (finalists.size() == 24) {
            break;
        }
    }

    std::map<std::uint64_t, double> all_zero_rates;
    for (const std::uint64_t seed : split.train) {
        all_zero_rates.emplace(
            seed,
            run_episode(model_path, artifact, zero, seed, 1000, false)
                .reward_rate);
    }
    std::vector<Score> final_scores(finalists.size());
    next.store(0);
    errors.assign(finalists.size(), nullptr);
    threads.clear();
    const std::size_t final_thread_count = std::min(workers, finalists.size());
    for (std::size_t worker = 0; worker < final_thread_count; ++worker) {
        threads.emplace_back([&] {
            while (true) {
                const std::size_t index = next.fetch_add(1);
                if (index >= finalists.size()) {
                    return;
                }
                try {
                    final_scores[index] = score_candidate(
                        model_path,
                        artifact,
                        finalists[index],
                        split.train,
                        all_zero_rates,
                        1000);
                } catch (...) {
                    errors[index] = std::current_exception();
                }
            }
        });
    }
    threads.clear();
    for (const auto& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    std::stable_sort(final_scores.begin(), final_scores.end(), better);

    const Score& best = final_scores.front();
    Json traces = Json::array();
    for (const std::uint64_t seed : representatives) {
        const Episode episode = run_episode(
            model_path, artifact, best.weights, seed, 1000, true);
        traces.push_back(Json{
            {"seed", episode.seed},
            {"side", episode.positive_side ? "positive_x" : "negative_x"},
            {"reward_rate", episode.reward_rate},
            {"zero_reward_rate", all_zero_rates.at(seed)},
            {"gain_vs_zero_reward_per_s",
             episode.reward_rate - all_zero_rates.at(seed)},
            {"start_x_m", episode.start_x},
            {"end_x_m", episode.end_x},
            {"minimum_x_m", episode.minimum_x},
            {"maximum_x_m", episode.maximum_x},
            {"start_lux", episode.start_lux},
            {"end_lux", episode.end_lux},
            {"mean_abs_action", episode.mean_abs_action},
            {"final_mean_action", episode.final_mean_action},
            {"direction_changes", episode.direction_changes},
            {"safety_vetoes", episode.safety_vetoes},
            {"terminated", episode.terminated},
            {"checkpoints", episode.checkpoints},
        });
    }

    Json top = Json::array();
    for (std::size_t index = 0; index < std::min<std::size_t>(10, final_scores.size()); ++index) {
        top.push_back(score_json(final_scores[index]));
    }
    std::cout << std::setw(2) << Json{
        {"schema_version", "droid-blocks.train-seed-v2-capacity-probe.v1"},
        {"seed_scope", "train_only"},
        {"train_seed_count", split.train.size()},
        {"train_seed_sha256", droid::seed_list_sha256(split.train)},
        {"validation_scenarios_executed", 0},
        {"test_scenarios_executed", 0},
        {"coarse_candidate_count", grid.size()},
        {"full_train_finalist_count", finalists.size()},
        {"best", score_json(best)},
        {"top_full_train", std::move(top)},
        {"representative_train_trajectories", std::move(traces)},
    } << '\n';
    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << "train_seed_v2_probe: " << error.what() << '\n';
    return EXIT_FAILURE;
}
