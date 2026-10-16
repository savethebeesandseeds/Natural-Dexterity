#include "droid/construction.hpp"
#include "droid/light_learner.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// One persistent, isolated batch of native worlds over stdin/stdout. This
// executable neither connects to nor takes ownership of a workshop session.
namespace {
using Json = nlohmann::json;
constexpr std::array<double, 3> requests{-.15, 0., .15};
constexpr int native_steps = 5;
struct Slot {
    droid::ConstructionWorld world;
    Json config;
    double previous_effort{};
    bool stopped{};
    explicit Slot(const Json& value) : config(value) {
        if (!value.is_object() || value.size() != 2 ||
            !value.contains("assembly") || !value.contains("sun"))
            throw std::invalid_argument("case requires only assembly and sun");
        world.rebuild(value.at("assembly"));
        world.set_sun(value.at("sun"));
        // Acquire the initial queued sample under the configured source too.
        world.reset();
    }
    Json initial(bool record) const {
        Json result{{"observation", world.observation()}, {"reward", 0.0},
            {"previous_request", 0.0}, {"executed_effort", 0.0},
            {"terminated", false}, {"elapsed_s", 0.0}};
        if (record) result["physics"] = world.state();
        return result;
    }
    Json advance(int action, bool record) {
        double reward = 0., light_reward = 0., imu_reward = 0.;
        double max_current = 0., max_speed = 0., max_temperature = 0.;
        Json transition;
        Json commands = Json::array();
        Json native_transitions = Json::array();
        std::string audit;
        int count = 0;
        for (; count < native_steps && !stopped; ++count) {
            previous_effort = droid::LightLearner::bounded_effort(
                requests.at(action), previous_effort, world.observation());
            transition = world.step(previous_effort);
            if (record) {
                commands.push_back(previous_effort);
                native_transitions.push_back(transition);
                audit += Json{{"effort", previous_effort}, {"transition", transition}}.dump();
                audit += '\n';
            }
            reward += transition.at("reward").get<double>();
            for (const Json& component : transition.at("reward_components")) {
                if (component.at("family_id") == "ambient_light_v0")
                    light_reward += component.at("transition_reward").get<double>();
                else if (component.at("family_id") == "imu_6axis_v0")
                    imu_reward += component.at("transition_reward").get<double>();
            }
            const Json& feedback = transition.at("observation").at("actuator_feedback").at(0).at("feedback");
            max_current = std::max(max_current, std::abs(feedback.at("current_a").get<double>()));
            max_speed = std::max(max_speed, std::abs(feedback.at("velocity_rad_s").get<double>()));
            max_temperature = std::max(max_temperature, feedback.at("temperature_c").get<double>());
            stopped = transition.at("terminated").get<bool>() ||
                transition.at("safety").at("veto").get<bool>() || !feedback.at("fault_flags").empty();
        }
        const Json physical = world.state();
        Json result{{"observation", world.observation()}, {"reward", reward},
            {"light_reward", light_reward}, {"imu_reward", imu_reward},
            {"previous_request", requests.at(action)}, {"executed_effort", previous_effort},
            {"terminated", stopped}, {"elapsed_s", .02 * count},
            {"native_steps", count}, {"time_s", physical.at("elapsed_s")},
            {"energy_j", physical.at("diagnostics").at("electrical_energy_j")},
            {"max_current_a", max_current}, {"max_speed_rad_s", max_speed},
            {"max_temperature_c", max_temperature}, {"safety", physical.at("safety")}};
        if (record) {
            result["physics"] = physical;
            result["efforts"] = commands;
            result["native_transitions"] = native_transitions;
            result["transition_sha256"] = droid::sha256_hex(audit);
        }
        return result;
    }
};
}

int main() {
    std::vector<std::unique_ptr<Slot>> slots;
    bool poisoned = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        bool advancing = false;
        Json partial = Json::array();
        try {
            if (line.size() > 1024 * 1024) throw std::invalid_argument("request too large");
            const Json input = Json::parse(line);
            const auto op = input.at("op").get<std::string>();
            const bool record = input.value("record", false);
            Json result = Json::array();
            if (op == "configure") {
                const Json& cases = input.at("cases");
                if (!cases.is_array() || cases.empty() || cases.size() > 32)
                    throw std::invalid_argument("batch must contain 1 to 32 cases");
                std::vector<std::unique_ptr<Slot>> replacements;
                for (const Json& config : cases) replacements.push_back(std::make_unique<Slot>(config));
                slots = std::move(replacements);
                poisoned = false;
                for (const auto& slot : slots) result.push_back(slot->initial(record));
            } else if (op == "reset") {
                if (poisoned) throw std::invalid_argument("failed batch requires full configure");
                const Json& replacements = input.at("slots");
                if (!replacements.is_array()) throw std::invalid_argument("slots must be an array");
                std::vector<std::pair<std::size_t, std::unique_ptr<Slot>>> prepared;
                std::vector<bool> seen(slots.size());
                for (const Json& item : replacements) {
                    if (!item.at("index").is_number_unsigned() && !item.at("index").is_number_integer())
                        throw std::invalid_argument("invalid slot index");
                    const auto index = item.at("index").get<std::size_t>();
                    if (index >= slots.size() || seen[index]) throw std::invalid_argument("invalid or duplicate slot index");
                    seen[index] = true;
                    prepared.emplace_back(index, std::make_unique<Slot>(item.value("case", slots[index]->config)));
                }
                for (auto& [index, slot] : prepared) {
                    slots[index] = std::move(slot);
                    result.push_back(Json{{"index", index}, {"initial", slots[index]->initial(record)}});
                }
            } else if (op == "step") {
                if (poisoned) throw std::invalid_argument("failed batch requires full configure");
                const Json& actions = input.at("actions");
                if (slots.empty() || !actions.is_array() || actions.size() != slots.size())
                    throw std::invalid_argument("actions must match configured batch");
                for (std::size_t i = 0; i < actions.size(); ++i) {
                    if (!actions[i].is_number_integer() || actions[i] < 0 || actions[i] > 2)
                        throw std::invalid_argument("actions must be integer 0, 1, or 2");
                    if (slots[i]->stopped) throw std::invalid_argument("terminated slot requires reset");
                }
                advancing = true;
                for (std::size_t i = 0; i < slots.size(); ++i)
                    partial.push_back(slots[i]->advance(actions[i].get<int>(), record));
                result = std::move(partial);
                advancing = false;
            } else if (op == "spec") {
                result = Json{{"schema", "actuation_transfer_bridge_v1"}, {"native_dt_s", .02},
                    {"decision_dt_s", .1}, {"native_steps_per_decision", native_steps},
                    {"requests", requests}, {"batch_limit", 32},
                    {"observation", "unmodified construction_observation_v2"},
                    {"reward", "sum of five native transition rewards; no shaping or rescaling"},
                    {"envelope", "LightLearner::bounded_effort every native step"}};
            } else if (op == "close") {
                std::cout << Json{{"ok", true}}.dump() << std::endl;
                return 0;
            } else throw std::invalid_argument("unknown operation");
            std::cout << Json{{"ok", true}, {"result", result}}.dump() << std::endl;
        } catch (const std::exception& error) {
            if (advancing) poisoned = true;
            std::cout << Json{{"ok", false}, {"error", error.what()},
                {"batch_failed", advancing}, {"completed_slots_before_failure", partial},
                {"uncertain_current_slot_native_steps_upper_bound", advancing ? native_steps : 0}}.dump() << std::endl;
        }
    }
}
