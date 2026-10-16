#include "droid/light_learner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
void close(double actual, double expected, std::string_view message) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 1e-10, message);
}
template <typename Function>
void require_throws(Function&& function, std::string_view message) {
    try { std::invoke(std::forward<Function>(function)); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(std::string(message));
}

// Synthetic clocks match the actual delivery contract. Light changes only at a
// new acquisition; these fixtures isolate learning and validation, not physics.
Json frame(std::size_t step, bool rename = false, bool change_light = false) {
    const bool valid = step > 0;
    const double now = static_cast<double>(step) * 0.02;
    const double imu_time = now - 0.01;
    const std::size_t light_index = valid ? (step - 1) / 5 : 0;
    const double light_time = static_cast<double>(light_index) * 0.1;
    Json imu{{"module_id", rename ? "different-imu" : "opaque-imu"},
        {"family_id", "imu_6axis_v0"}, {"valid", valid},
        {"sequence", static_cast<std::uint64_t>(2 * step)},
        {"sample_time_s", valid ? Json(imu_time) : Json(nullptr)},
        {"delivered_time_s", valid ? Json(imu_time + 0.006) : Json(nullptr)},
        {"age_s", valid ? Json(0.01) : Json(nullptr)},
        {"sample_period_s", 0.01}, {"latency_s", 0.006},
        {"observations", Json{{"specific_force_m_s2", Json::array({0.0, 0.0, valid ? 9.81 : 0.0})},
                              {"angular_velocity_rad_s", Json::array({0.0, 0.0, 0.0})}}}};
    Json light{{"module_id", rename ? "different-eye" : "opaque-eye"},
        {"family_id", "ambient_light_v0"}, {"valid", valid},
        {"sequence", static_cast<std::uint64_t>(valid ? light_index + 1 : 0)},
        {"sample_time_s", valid ? Json(light_time) : Json(nullptr)},
        {"delivered_time_s", valid ? Json(light_time + 0.02) : Json(nullptr)},
        {"age_s", valid ? Json(now - light_time) : Json(nullptr)},
        {"sample_period_s", 0.1}, {"latency_s", 0.02},
        {"observations", Json{{"illuminance_lux", valid ? (change_light && light_index >= 20 ? 600.0 : 100.0) : 0.0},
                              {"saturated", false}}}};
    Json motor{{"module_id", rename ? "different-motor" : "opaque-motor"},
        {"sku_id", "rotary_dc_gearmotor_v0"}, {"valid", true},
        {"feedback", Json{{"position_rad", 0.0}, {"velocity_rad_s", 0.0}, {"current_a", 0.0},
            {"bus_voltage_v", 7.4}, {"temperature_c", 25.0}, {"output_torque_nm", 0.0},
            {"load_impedance_nm_s_per_rad", 0.0}, {"stuck_score", 0.0}, {"stuck", false},
            {"fault_flags", Json::array()}}}};
    return Json{{"schema_version", "construction_observation_v2"},
        {"reward_sensors", rename ? Json::array({light, imu}) : Json::array({imu, light})},
        {"actuator_feedback", Json::array({motor})}};
}

Json transition(Json observation, double imu_reward = 0.0, double light_reward = 0.0) {
    Json components = Json::array();
    for (const Json& sensor : observation.at("reward_sensors")) {
        const double reward = sensor.at("family_id") == "imu_6axis_v0" ? imu_reward : light_reward;
        components.push_back(Json{{"module_id", sensor.at("module_id")}, {"family_id", sensor.at("family_id")},
            {"transition_reward", reward}, {"reward_rate", reward / 0.02}, {"valid", sensor.at("valid")}});
    }
    return Json{{"observation", std::move(observation)}, {"reward", imu_reward + light_reward},
        {"reward_components", std::move(components)}, {"safety", Json{{"veto", false}, {"flags", Json::array()}}},
        {"terminated", false}, {"truncated", false},
        {"info", Json{{"elapsed_control_dt_s", 0.02}, {"physics_steps_executed", 10}}}};
}

void test_cadence_and_atomic_failure() {
    droid::LightLearner learner(31);
    const Json initial = learner.diagnostics();
    require_throws([&] { learner.observe(transition(frame(1))); }, "observe accepted without act");
    require(learner.diagnostics() == initial, "failed observe changed fresh state");
    close(learner.act(frame(0)), 0.0, "initial undelivered samples caused motion");
    const Json pending = learner.diagnostics();
    require_throws([&] { (void)learner.act(frame(0)); }, "duplicate act accepted");
    Json bad = transition(frame(1));
    bad["reward"] = 0.1;
    require_throws([&] { learner.observe(bad); }, "inconsistent reward sum accepted");
    require(learner.diagnostics() == pending, "rejected transition partially committed");
    learner.observe(transition(frame(1)));
    const Json observed = learner.diagnostics();
    require_throws([&] { learner.observe(transition(frame(1))); }, "duplicate observation accepted");
    require(learner.diagnostics() == observed, "duplicate observe changed state");
    learner.reset(31);
    require(learner.diagnostics() == initial, "reset retained learner state");
}

void test_option_timing_and_actual_summed_reward_update() {
    droid::LightLearner positive(9), negative(9), neutral(9);
    std::string selected;
    for (std::size_t step = 0; step < 25; ++step) {
        const double effort = positive.act(frame(step));
        close(negative.act(frame(step)), effort, "first option depended on unseen future reward");
        close(neutral.act(frame(step)), effort, "untrained first option differed");
        const Json acting = positive.diagnostics();
        if (step < 5) {
            close(effort, 0.0, "quiet interval moved");
            require(acting.at("decision_count") == 0, "decision began before quiet interval ended");
        } else {
            if (step == 5) selected = acting.at("action_id").get<std::string>();
            require(acting.at("action_id") == selected && acting.at("decision_count") == 1,
                    "option did not hold for exactly twenty transitions");
        }
        const bool active = step >= 5;
        // Identical light and motion observations, identical positive light vote.
        // Changing the IMU contribution reverses the total learning signal.
        positive.observe(transition(frame(step + 1), active ? 0.01 : 0.0, active ? 0.005 : 0.0));
        negative.observe(transition(frame(step + 1), active ? -0.02 : 0.0, active ? 0.005 : 0.0));
        neutral.observe(transition(frame(step + 1)));
        if (step < 24) require(positive.diagnostics().at("update_count") == 0, "partial option was learned as a full option");
    }
    const double discount_step = std::pow(0.92, 1.0 / 20.0);
    const double expected_return = 0.015 * (1.0 - 0.92) / (1.0 - discount_step);
    const Json plus = positive.diagnostics(), minus = negative.diagnostics(), zero = neutral.diagnostics();
    close(plus.at("last_td_error"), expected_return, "learned target omitted actual discounted option reward");
    close(minus.at("last_td_error"), -expected_return, "IMU vote was omitted from summed reward");
    require(plus.at("update_count") == 1 && plus.at("replay_update_count") == 4,
            "completed option did not update online and recent replay values");
    require(plus.at("parameter_l1").get<double>() > 0.0 && minus.at("parameter_l1").get<double>() > 0.0,
            "observed rewards did not change learned parameters");
    close(zero.at("parameter_l1"), 0.0, "constant lux invented a reward update");
    close(plus.at("option_remaining_s"), 0.0, "completed option retained remaining duration");
    (void)positive.act(frame(25));
    require(positive.diagnostics().at("decision_count") == 2, "next option did not start at its boundary");
}

void test_determinism_bounded_memory_and_sensory_change() {
    droid::LightLearner first(17), renamed(17);
    double previous = 0.0;
    std::set<std::string> initial_options;
    for (std::size_t step = 0; step < 700; ++step) {
        const double effort = first.act(frame(step, false, true));
        close(renamed.act(frame(step, true, true)), effort, "opaque IDs or sensor ordering changed control");
        require(std::abs(effort) <= 0.150000000001 && std::abs(effort - previous) <= 0.100000000001,
                "governed effort or slew exceeded declared envelope");
        if (step == 5 || step == 25 || step == 45) initial_options.insert(first.diagnostics().at("action_id"));
        const double light_reward = step < 100 ? 0.01 : -0.01;
        first.observe(transition(frame(step + 1, false, true), 0.005, light_reward));
        renamed.observe(transition(frame(step + 1, true, true), 0.005, light_reward));
        Json first_diagnostics = first.diagnostics(), renamed_diagnostics = renamed.diagnostics();
        require(first_diagnostics.at("parameter_fingerprint_sha256") == renamed_diagnostics.at("parameter_fingerprint_sha256"),
                "opaque identities changed learned parameter bits");
        // Both state fingerprints deliberately include bound identities,
        // because those identities are continuity-validation state.
        require(first_diagnostics.at("state_fingerprint_sha256") != renamed_diagnostics.at("state_fingerprint_sha256"),
                "complete state fingerprint omitted bound module identities");
        first_diagnostics.erase("state_fingerprint_sha256");
        renamed_diagnostics.erase("state_fingerprint_sha256");
        require(first_diagnostics.at("nonparameter_state_fingerprint_sha256") != renamed_diagnostics.at("nonparameter_state_fingerprint_sha256"),
                "nonparameter fingerprint omitted bound module identities");
        first_diagnostics.erase("nonparameter_state_fingerprint_sha256");
        renamed_diagnostics.erase("nonparameter_state_fingerprint_sha256");
        require(first_diagnostics == renamed_diagnostics, "seeded learning did not replay under ID renaming");
        previous = effort;
    }
    const Json result = first.diagnostics();
    require(initial_options.size() == 3, "initial shuffled exploration omitted an action");
    require(result.at("history_size") == 64 && result.at("replay_size") == 32, "finite memories violated capacities");
    require(result.at("unique_imu_samples") == 700 && result.at("unique_light_samples") == 140,
            "held light samples counted as fresh measurements");
    require(result.at("update_count") == 34 && result.at("replay_update_count") == 136,
            "continuous sensory change reset or skipped learning");
    require(result.at("recent_experience").size() == 8, "diagnostics did not expose bounded timestamped history");
    const Json last = result.at("recent_experience").back();
    close(last.at("time_s"), 14.0, "history lost control timestamp");
    close(last.at("illuminance_lux"), 600.0, "fresh changed illuminance was not retained");
    close(result.at("cumulative_reward"), -1.5, "actual reward history sum wrong");
    require(result.at("parameter_change_l1").get<double>() > 0.0, "long run never learned");
}

void test_measured_equal_initialization_and_td_residual() {
    droid::LightLearner steady(7), changed(7);
    const double discount_step = std::pow(0.92, 1.0 / 20.0);
    const double common_value = 0.015 / (1.0 - discount_step);
    for (std::size_t step = 0; step < 5; ++step) {
        close(steady.act(frame(step)), 0.0, "initialization moved the body");
        close(changed.act(frame(step)), 0.0, "initialization differs across learners");
        // Deliberately negative delivery transient must not bias the estimate.
        const Json observed = transition(frame(step + 1), step == 0 ? -0.02 : 0.01, step == 0 ? -0.02 : 0.005);
        steady.observe(observed);
        changed.observe(observed);
        if (step < 4) require(steady.diagnostics().at("values_initialized") == false,
                              "value initialization occurred before quiet data were complete");
    }
    const Json initialized = steady.diagnostics();
    require(initialized.at("values_initialized") == true && initialized.at("initial_reward_samples") == 4,
            "initialization did not use exactly the fully delivered quiet intervals");
    close(initialized.at("initial_reward_rate"), 0.75, "initial delivery transient entered reward estimate");
    close(initialized.at("initial_value"), common_value, "initial value used the wrong discount horizon");
    close(initialized.at("parameter_change_l1"), 0.0, "initialization was counted as learned TD updates");
    for (const Json& option : initialized.at("option_values")) {
        close(option.at("value"), common_value, "initialization preferred an untried action");
        require(option.at("visits") == 0, "measurement counted as an action trial");
    }
    for (std::size_t step = 5; step < 25; ++step) {
        close(steady.act(frame(step)), changed.act(frame(step)), "first choice anticipated later rewards");
        steady.observe(transition(frame(step + 1), 0.01, 0.005));
        changed.observe(transition(frame(step + 1), 0.0, 0.005));
    }
    close(steady.diagnostics().at("last_td_error"), 0.0, "constant background reward manufactured an advantage");
    const double expected_residual = -0.01 * (1.0 - 0.92) / (1.0 - discount_step);
    close(changed.diagnostics().at("last_td_error"), expected_residual,
          "later raw summed reward did not produce the exact TD residual against the measured initial value");
    require(changed.diagnostics().at("parameter_change_l1").get<double>() > 0.0,
            "new sensory reward evidence did not revise the initial estimate");
}

void test_freeze_boundaries_and_reset() {
    droid::LightLearner learner(43), untouched(43);
    const Json fresh = learner.diagnostics();
    require_throws([&] { learner.freeze_learning(); }, "uninitialized values were frozen");
    require(learner.diagnostics() == fresh, "failed initial freeze changed state");
    for (std::size_t step = 0; step < 5; ++step) {
        close(learner.act(frame(step)), untouched.act(frame(step)), "rejected freeze changed quiet action");
        learner.observe(transition(frame(step + 1), 0.005, 0.005));
        untouched.observe(transition(frame(step + 1), 0.005, 0.005));
        if (step < 4) {
            const Json before = learner.diagnostics();
            require_throws([&] { learner.freeze_learning(); }, "partial quiet initialization was frozen");
            require(learner.diagnostics() == before, "rejected quiet freeze changed state");
        }
    }
    close(learner.act(frame(5)), untouched.act(frame(5)), "rejected freeze changed first option");
    const Json pending = learner.diagnostics();
    require_throws([&] { learner.freeze_learning(); }, "freeze accepted with an unfinished action");
    require(learner.diagnostics() == pending, "pending freeze rejection partially changed learner");
    learner.observe(transition(frame(6), 0.005, 0.005));
    untouched.observe(transition(frame(6), 0.005, 0.005));
    require(learner.diagnostics() == untouched.diagnostics(), "rejected freeze altered subsequent complete state");
    learner.freeze_learning();
    const Json frozen = learner.diagnostics();
    require(frozen.at("learning_frozen") == true && frozen.at("learning_enabled") == false,
            "freeze did not declare its learning state");
    learner.freeze_learning();
    require(learner.diagnostics() == frozen, "second freeze was not idempotent");
    (void)learner.act(frame(6));
    const Json frozen_pending = learner.diagnostics();
    require_throws([&] { learner.freeze_learning(); }, "idempotent freeze bypassed pending-action precondition");
    require(learner.diagnostics() == frozen_pending, "rejected repeated freeze changed pending state");
    learner.reset(43);
    require(learner.diagnostics() == fresh, "reset retained the freeze or its bookkeeping");
}

void test_freeze_mid_option_retains_sensing_and_experience() {
    droid::LightLearner positive(19), negative(19), continuing(19);
    constexpr std::size_t freeze_step = 32; // Seven transitions into option two.
    for (std::size_t step = 0; step < freeze_step; ++step) {
        const Json observation = frame(step, false, true);
        const double effort = positive.act(observation);
        close(negative.act(observation), effort, "paired prefix action differs");
        close(continuing.act(observation), effort, "continuing prefix action differs");
        const Json observed = transition(frame(step + 1, false, true), step < 5 ? 0.002 : 0.01, 0.005);
        positive.observe(observed); negative.observe(observed); continuing.observe(observed);
    }
    const Json before = positive.diagnostics();
    require(before == negative.diagnostics() && before == continuing.diagnostics(), "paired prefix internal states differ");
    require(before.at("update_count") == 1 && before.at("option_remaining_s") == 0.26,
            "test did not freeze learned values in the middle of an option");
    require(before.at("parameter_change_l1").get<double>() > 0.0, "test froze only initialized values");
    const Json parameter_hash = before.at("parameter_fingerprint_sha256");
    positive.freeze_learning(); negative.freeze_learning();
    Json expected_frozen = before;
    expected_frozen["learning_frozen"] = true;
    expected_frozen["learning_enabled"] = false;
    Json immediately_frozen = positive.diagnostics();
    expected_frozen.erase("state_fingerprint_sha256");
    immediately_frozen.erase("state_fingerprint_sha256");
    expected_frozen.erase("nonparameter_state_fingerprint_sha256");
    immediately_frozen.erase("nonparameter_state_fingerprint_sha256");
    require(immediately_frozen == expected_frozen, "freeze changed more than its one-way flag");
    for (std::size_t step = freeze_step; step < 165; ++step) {
        const Json observation = frame(step, false, true);
        (void)positive.act(observation); (void)negative.act(observation); (void)continuing.act(observation);
        positive.observe(transition(frame(step + 1, false, true), 0.01, 0.005));
        negative.observe(transition(frame(step + 1, false, true), -0.01, -0.005));
        continuing.observe(transition(frame(step + 1, false, true), 0.01, 0.005));
        require(positive.diagnostics().at("parameter_fingerprint_sha256") == parameter_hash &&
                negative.diagnostics().at("parameter_fingerprint_sha256") == parameter_hash,
                "frozen online or replay processing changed exact parameter bits");
    }
    const Json plus = positive.diagnostics(), minus = negative.diagnostics(), active = continuing.diagnostics();
    require(active.at("parameter_fingerprint_sha256") != parameter_hash,
            "continuing comparison did not keep updating parameters");
    require(plus.at("step") == 165 && plus.at("decision_count") == 8 && plus.at("history_size") == 64 &&
            plus.at("unique_light_samples") == 33 && plus.at("unique_imu_samples") == 165,
            "freeze stopped sensing, decisions or bounded history");
    require(plus.at("update_count") == 8 && plus.at("replay_update_count") == 32 && plus.at("replay_size") == 8 &&
            plus.at("suppressed_update_count") == 7 && plus.at("suppressed_replay_update_count") == 28,
            "freeze skipped option or replay bookkeeping");
    require(plus.at("decision_count") == active.at("decision_count") &&
            plus.at("exploration_rate") == active.at("exploration_rate") &&
            plus.at("exploratory_decisions") == active.at("exploratory_decisions"),
            "freeze changed the exploration schedule or action RNG progression");
    require(plus.at("parameter_change_l1") == before.at("parameter_change_l1") &&
            minus.at("parameter_change_l1") == before.at("parameter_change_l1"), "frozen parameter-change accounting increased");
    require(plus.at("cumulative_reward") != minus.at("cumulative_reward") &&
            plus.at("recent_experience").back().at("reward") != minus.at("recent_experience").back().at("reward") &&
            plus.at("state_fingerprint_sha256") != minus.at("state_fingerprint_sha256"),
            "frozen controller discarded different received reward histories");
    require(plus.at("recent_experience").back().at("illuminance_lux") == 600.0,
            "frozen controller discarded new light samples");
}

void test_value_checkpoint_boundaries_and_identity() {
    using Checkpoint = droid::LightLearner::ValueCheckpoint;
    static_assert(!std::is_default_constructible_v<Checkpoint>);
    static_assert(std::is_copy_constructible_v<Checkpoint> && std::is_copy_assignable_v<Checkpoint>);
    droid::LightLearner donor(61), receiver(61), renamed(61);
    for (std::size_t step = 0; step < 5; ++step) {
        (void)donor.act(frame(step));
        donor.observe(transition(frame(step + 1), 0.005, 0.005));
    }
    const Json donor_before = donor.diagnostics();
    const droid::LightLearner& read_only = donor;
    const auto checkpoint = read_only.capture_values();
    auto copied = checkpoint;
    copied = checkpoint;
    require(donor.diagnostics() == donor_before, "const checkpoint capture changed donor state");
    const Json fresh = receiver.diagnostics();
    require_throws([&] { (void)receiver.capture_values(); }, "uninitialized value capture accepted");
    require_throws([&] { receiver.restore_values(copied); }, "uninitialized value restore accepted");
    require(receiver.diagnostics() == fresh, "rejected checkpoint access changed fresh state");
    for (std::size_t step = 0; step < 5; ++step) {
        (void)receiver.act(frame(step));
        receiver.observe(transition(frame(step + 1), 0.005, 0.005));
        (void)renamed.act(frame(step, true));
        renamed.observe(transition(frame(step + 1, true), 0.005, 0.005));
    }
    (void)receiver.act(frame(5));
    const Json pending = receiver.diagnostics();
    require_throws([&] { (void)receiver.capture_values(); }, "capture accepted during pending action");
    require_throws([&] { receiver.restore_values(copied); }, "restore accepted during pending action");
    require(receiver.diagnostics() == pending, "rejected pending checkpoint access changed state");
    receiver.observe(transition(frame(6), 0.005, 0.005));
    const Json renamed_before = renamed.diagnostics();
    require_throws([&] { renamed.restore_values(copied); }, "checkpoint ignored incompatible module identities");
    require(renamed.diagnostics() == renamed_before, "incompatible checkpoint partially changed state");

    droid::LightLearner oversized;
    const auto large_frame = [](std::size_t step) {
        Json result = frame(step);
        result["actuator_feedback"][0]["module_id"] = std::string(129, 'm');
        return result;
    };
    for (std::size_t step = 0; step < 5; ++step) {
        (void)oversized.act(large_frame(step));
        oversized.observe(transition(large_frame(step + 1)));
    }
    const Json oversized_before = oversized.diagnostics();
    require_throws([&] { (void)oversized.capture_values(); }, "checkpoint captured unbounded identity metadata");
    require(oversized.diagnostics() == oversized_before, "bounded checkpoint rejection changed ordinary control");
}

void test_value_checkpoint_restores_only_exact_weights() {
    droid::LightLearner recalled(53), current(53);
    for (std::size_t step = 0; step < 32; ++step) {
        const double effort = recalled.act(frame(step, false, true));
        close(current.act(frame(step, false, true)), effort, "checkpoint donor prefix differed");
        const Json observed = transition(frame(step + 1, false, true), step < 5 ? 0.002 : 0.01, 0.005);
        recalled.observe(observed); current.observe(observed);
    }
    const Json saved_diagnostics = recalled.diagnostics();
    const auto saved = recalled.capture_values();
    auto saved_copy = saved;
    require(recalled.diagnostics() == saved_diagnostics && current.diagnostics() == saved_diagnostics,
            "capture changed complete state or paired prefix");
    for (std::size_t step = 32; step < 72; ++step) {
        const double effort = recalled.act(frame(step, false, true));
        close(current.act(frame(step, false, true)), effort, "capture affected later choices or RNG");
        const Json observed = transition(frame(step + 1, false, true), -0.01, -0.005);
        recalled.observe(observed); current.observe(observed);
    }
    const Json before = recalled.diagnostics();
    require(before == current.diagnostics(), "capture changed subsequent complete state");
    require(before.at("parameter_fingerprint_sha256") != saved_diagnostics.at("parameter_fingerprint_sha256"),
            "test did not revise values between capture and recall");
    close(before.at("option_remaining_s"), 0.26, "recall test missed the middle of its current option");
    const auto later = current.capture_values();
    current.restore_values(later);
    require(current.diagnostics() == before, "restoring current values was not an exact no-op");
    recalled.restore_values(saved_copy);
    const Json restored = recalled.diagnostics();
    require(restored.at("parameter_fingerprint_sha256") == saved_diagnostics.at("parameter_fingerprint_sha256"),
            "checkpoint did not restore exact original weight bits");
    require(restored.at("nonparameter_state_fingerprint_sha256") == before.at("nonparameter_state_fingerprint_sha256"),
            "restore altered history, traces, replay, option, RNG, counters or freeze state");
    require(restored.at("state_fingerprint_sha256") != before.at("state_fingerprint_sha256"),
            "full state fingerprint missed changed weights");
    require(restored.at("recent_experience") == before.at("recent_experience") &&
            restored.at("parameter_change_l1") == before.at("parameter_change_l1"),
            "restore rewrote sensed experience or learning-write accounting");
    recalled.freeze_learning(); current.freeze_learning();
    require(recalled.diagnostics().at("nonparameter_state_fingerprint_sha256") ==
            current.diagnostics().at("nonparameter_state_fingerprint_sha256"),
            "the recalled/current branches differed beyond weights at freezing");
    const Json frozen = recalled.diagnostics();
    recalled.restore_values(saved);
    require(recalled.diagnostics() == frozen, "idempotent recall changed frozen state");
    for (std::size_t step = 72; step < 105; ++step) {
        const double remembered_effort = recalled.act(frame(step, false, true));
        const double current_effort = current.act(frame(step, false, true));
        if (step < 85) close(remembered_effort, current_effort, "recall restarted or replaced the pending option");
        const Json observed = transition(frame(step + 1, false, true), 0.01, 0.005);
        recalled.observe(observed); current.observe(observed);
        require(recalled.diagnostics().at("parameter_fingerprint_sha256") == saved_diagnostics.at("parameter_fingerprint_sha256"),
                "recalled frozen parameters changed");
        require(current.diagnostics().at("parameter_fingerprint_sha256") == before.at("parameter_fingerprint_sha256"),
                "current frozen parameters changed");
    }
    recalled.reset(53);
    droid::LightLearner fresh(53);
    require(recalled.diagnostics() == fresh.diagnostics(), "reset retained checkpoint-related state");
    require_throws([&] { recalled.restore_values(saved); }, "reset bypassed restore initialization requirement");
    for (std::size_t step = 0; step < 5; ++step) {
        (void)recalled.act(frame(step));
        recalled.observe(transition(frame(step + 1)));
    }
    const Json reinitialized = recalled.diagnostics();
    recalled.restore_values(saved);
    require(recalled.diagnostics().at("parameter_fingerprint_sha256") == saved_diagnostics.at("parameter_fingerprint_sha256") &&
            recalled.diagnostics().at("nonparameter_state_fingerprint_sha256") == reinitialized.at("nonparameter_state_fingerprint_sha256"),
            "immutable checkpoint aliased donor reset or changed fresh nonparameter state");
}

void test_observation_boundary_and_held_samples() {
    droid::LightLearner learner;
    (void)learner.act(frame(0));
    learner.observe(transition(frame(1)));
    const auto reject = [&](const std::function<void(Json&)>& mutation) {
        Json bad = frame(1);
        mutation(bad);
        const Json before = learner.diagnostics();
        require_throws([&] { (void)learner.act(bad); }, "malformed or privileged observation accepted");
        require(learner.diagnostics() == before, "bad observation partially changed learner");
    };
    reject([](Json& v) { v["sun_position"] = Json::array({1, 0, 1}); });
    reject([](Json& v) { v["assembly"] = "default"; });
    reject([](Json& v) { v["reward"] = 1.0; });
    reject([](Json& v) { v["reward_sensors"][0]["observations"]["world_pose"] = 0.0; });
    reject([](Json& v) { v["reward_sensors"].erase(0); });
    reject([](Json& v) { v["reward_sensors"][1]["family_id"] = "oracle_light_v0"; });
    reject([](Json& v) { v["reward_sensors"][1]["module_id"] = "changed-eye"; });
    reject([](Json& v) { v["actuator_feedback"][0]["module_id"] = "changed-motor"; });
    reject([](Json& v) { v["reward_sensors"][1]["valid"] = false; });
    reject([](Json& v) { v["actuator_feedback"][0]["valid"] = false; });
    reject([](Json& v) { v["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"voltage_limited"}); });
    reject([](Json& v) { v["actuator_feedback"][0]["feedback"]["velocity_rad_s"] = std::numeric_limits<double>::quiet_NaN(); });
    reject([](Json& v) { v["reward_sensors"][0]["observations"]["specific_force_m_s2"][0] = std::numeric_limits<double>::infinity(); });
    reject([](Json& v) { v["reward_sensors"][1]["observations"]["illuminance_lux"] = -1.0; });
    reject([](Json& v) { v["reward_sensors"][0]["age_s"] = 0.1; });
    reject([](Json& v) { v["reward_sensors"][1]["age_s"] = 0.6; });
    reject([](Json& v) { v["reward_sensors"][1]["latency_s"] = 0.0; });
    reject([](Json& v) { v["reward_sensors"][1]["sequence"] = -1; });
    reject([](Json& v) { v["reward_sensors"][1]["sequence"] = 2; });
    reject([](Json& v) { v["reward_sensors"][1]["observations"]["illuminance_lux"] = 101.0; });
    for (const std::size_t sensor_index : {std::size_t{0}, std::size_t{1}}) {
        droid::LightLearner fresh;
        Json duplicate = frame(0);
        duplicate["actuator_feedback"][0]["module_id"] = duplicate["reward_sensors"][sensor_index]["module_id"];
        require_throws([&] { (void)fresh.act(duplicate); }, "motor and sensor identity collision accepted");
    }
    (void)learner.act(frame(1));
    Json held = frame(2);
    held["reward_sensors"][1]["observations"]["illuminance_lux"] = 101.0;
    require_throws([&] { learner.observe(transition(held)); }, "held light changed during physics transition");
    learner.observe(transition(frame(2)));
    require(learner.diagnostics().at("unique_light_samples") == 1, "rejected held sample changed freshness count");
}

void test_learning_boundary_and_common_governor() {
    droid::LightLearner learner;
    (void)learner.act(frame(0));
    const auto reject = [&](const std::function<void(Json&)>& mutation) {
        Json bad = transition(frame(1));
        mutation(bad);
        const Json before = learner.diagnostics();
        require_throws([&] { learner.observe(bad); }, "malformed learning feedback accepted");
        require(learner.diagnostics() == before, "rejected learning feedback changed memory");
    };
    reject([](Json& v) { v["info"]["sun_moved"] = true; });
    reject([](Json& v) { v["safety"]["veto"] = true; });
    reject([](Json& v) { v["safety"]["flags"] = Json::array({"imu_gyro_limit"}); });
    reject([](Json& v) { v["terminated"] = true; });
    reject([](Json& v) { v["truncated"] = true; });
    reject([](Json& v) { v["info"]["elapsed_control_dt_s"] = 0.04; });
    reject([](Json& v) { v["reward"] = std::numeric_limits<double>::infinity(); });
    reject([](Json& v) { v["reward_components"][0]["module_id"] = "opaque-motor"; });
    reject([](Json& v) { v["reward_components"][0]["valid"] = false; });
    reject([](Json& v) { v["reward_components"][0]["transition_reward"] = 0.1; v["reward"] = 0.1; });
    reject([](Json& v) { v["reward_components"][0]["reward_rate"] = 2.0; });
    learner.observe(transition(frame(1)));

    close(droid::LightLearner::bounded_effort(0.15, 0.0, frame(0)), 0.0, "undelivered sensor did not coast");
    Json current = frame(1);
    close(droid::LightLearner::bounded_effort(0.15, 0.0, current), 0.1, "baseline initial slew differs");
    current["actuator_feedback"][0]["feedback"]["velocity_rad_s"] = 3.0;
    close(droid::LightLearner::bounded_effort(0.15, 0.0, current), 0.0, "governor drove into positive speed target");
    close(droid::LightLearner::bounded_effort(-0.15, 0.0, current), -0.1, "governor blocked counter-motion");
    close(droid::LightLearner::bounded_effort(0.15, 0.15, current), 0.05, "governor bypassed declared slew");
    current["actuator_feedback"][0]["feedback"]["velocity_rad_s"] = -3.0;
    close(droid::LightLearner::bounded_effort(-0.15, 0.0, current), 0.0, "negative speed governor asymmetric");
    require_throws([&] { (void)droid::LightLearner::bounded_effort(std::numeric_limits<double>::quiet_NaN(), 0.0, current); }, "nonfinite request accepted");
    require_throws([&] { (void)droid::LightLearner::bounded_effort(0.0, 0.4, current); }, "invalid previous effort accepted");
    current["actuator_feedback"][0]["feedback"]["fault_flags"] = Json::array({"overspeed"});
    require_throws([&] { (void)droid::LightLearner::bounded_effort(0.0, 0.0, current); }, "baseline helper ignored motor fault");
    const Json spec = droid::LightLearner::specification();
    require(spec.at("option_duration_s") == 0.4 && spec.at("shaft_target_rad_s") == 3.0,
            "published protocol differs from tested constants");
}
} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"cadence and atomic failure", test_cadence_and_atomic_failure},
        {"option timing and actual summed-reward learning", test_option_timing_and_actual_summed_reward_update},
        {"measured equal initialization and TD residual", test_measured_equal_initialization_and_td_residual},
        {"determinism, bounded memory and sensory change", test_determinism_bounded_memory_and_sensory_change},
        {"freeze boundaries and reset", test_freeze_boundaries_and_reset},
        {"mid-option freeze retains sensing and experience", test_freeze_mid_option_retains_sensing_and_experience},
        {"typed value checkpoint boundaries and identity", test_value_checkpoint_boundaries_and_identity},
        {"value checkpoint restores only exact weights", test_value_checkpoint_restores_only_exact_weights},
        {"physical boundary and held samples", test_observation_boundary_and_held_samples},
        {"learning boundary and common governor", test_learning_boundary_and_common_governor}};
    for (const auto& [name, test] : tests) {
        try { test(); }
        catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            return 1;
        }
        std::cout << "PASS " << name << '\n';
    }
    std::cout << tests.size() << " native light learner tests passed\n";
    return 0;
}
