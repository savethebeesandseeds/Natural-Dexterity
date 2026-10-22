#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace humanoid::walk {

// Screen-only measurements: this collector neither computes a reward nor
// changes the simulator's geometric-contact walking acceptance criteria.
class SwingDiagnostics {
 public:
  void reset(double time, std::array<bool, 2> supports,
             std::array<double, 2> sole_heights) {
    validate_values(time, sole_heights);
    initialized_ = true;
    reset_time_ = last_time_ = time;
    supports_ = supports;
    last_qualifying_ = -1;
    completed_.clear();
    for (int side = 0; side < 2; ++side) {
      started_airborne_[side] = !supports[side];
      active_[side] = {};
      if (!supports[side]) active_[side] = {true, true, time, sole_heights[side], 1};
    }
  }

  void observe(double time, std::array<bool, 2> supports,
               std::array<double, 2> sole_heights) {
    if (!initialized_) throw std::logic_error("SwingDiagnostics requires reset before observe");
    validate_values(time, sole_heights);
    // Missing samples could hide additional contacts or overstate airtime.
    if (std::abs((time - last_time_) - kInterval) > kTimeTolerance)
      throw std::invalid_argument("SwingDiagnostics requires consecutive nominal 20ms samples");
    std::array<std::size_t, 2> landed{};
    std::size_t landing_count = 0;
    for (int side = 0; side < 2; ++side) {
      auto& swing = active_[side];
      if (!supports[side]) {
        if (supports_[side]) swing = {true, false, time, sole_heights[side], 1};
        else {
          swing.peak_clearance = std::max(swing.peak_clearance, sole_heights[side]);
          ++swing.unsupported_samples;
        }
      } else if (!supports_[side]) {
        const double duration = time - swing.loss_time;
        const bool qualifies = !swing.initial && duration + kTimeTolerance >= kMinimumAir
            && swing.peak_clearance >= kMinimumClearance;
        landed[landing_count++] = completed_.size();
        completed_.push_back({side, swing.loss_time, time, duration, swing.peak_clearance,
                              swing.unsupported_samples, swing.initial, qualifies,
                              time - reset_time_ > kLateAfter + kTimeTolerance, false, false});
        swing = {};
      }
    }
    // A simultaneous landing cannot create an ordering between the feet.
    // A nonqualifying landing also breaks the chain rather than allowing
    // several loading/unloading events to masquerade as a clean alternation.
    if (landing_count == 2) last_qualifying_ = -1;
    else if (landing_count == 1) {
      auto& current = completed_[landed[0]];
      if (!current.qualifies) last_qualifying_ = -1;
      else {
        if (last_qualifying_ >= 0) {
          const auto& previous = completed_[static_cast<std::size_t>(last_qualifying_)];
          current.alternation = previous.side != current.side
              && current.touchdown_time - previous.touchdown_time + kTimeTolerance >= kMinimumAlternation;
          current.late_alternation = current.alternation && previous.late && current.late;
        }
        last_qualifying_ = static_cast<std::ptrdiff_t>(landed[0]);
      }
    }
    supports_ = supports;
    last_time_ = time;
  }

  nlohmann::json report() const {
    if (!initialized_) throw std::logic_error("SwingDiagnostics requires reset before report");
    using Json = nlohmann::json;
    Json events = Json::array(), incomplete = Json::array(), initial = Json::array();
    std::array<std::size_t, 2> counts{}, late_counts{}, lifts{}, late_lifts{};
    std::size_t alternations = 0, late_alternations = 0;
    for (const auto& event : completed_) {
      Json entry = {{"foot", name(event.side)}, {"loss_time", event.loss_time},
        {"touchdown_time", event.touchdown_time}, {"observed_air_seconds", event.air_seconds},
        {"peak_sole_clearance_m", event.peak_clearance}, {"unsupported_samples", event.unsupported_samples},
        {"started_airborne", event.initial}, {"complete", true}, {"qualifies_lift", event.qualifies},
        {"late", event.late}, {"qualifying_alternation", event.alternation},
        {"late_qualifying_alternation", event.late_alternation}};
      ++counts[event.side];
      late_counts[event.side] += event.late;
      lifts[event.side] += event.qualifies;
      late_lifts[event.side] += event.qualifies && event.late;
      alternations += event.alternation;
      late_alternations += event.late_alternation;
      if (event.initial) initial.push_back(entry);
      events.push_back(std::move(entry));
    }
    for (int side = 0; side < 2; ++side) {
      const auto& swing = active_[side];
      if (!swing.open) continue;
      Json entry = {{"foot", name(side)}, {"loss_time", swing.loss_time}, {"touchdown_time", nullptr},
        {"observed_until", last_time_}, {"observed_air_seconds", last_time_ - swing.loss_time},
        {"peak_sole_clearance_m", swing.peak_clearance}, {"unsupported_samples", swing.unsupported_samples},
        {"started_airborne", swing.initial}, {"complete", false}, {"qualifies_lift", false}};
      if (swing.initial) initial.push_back(entry);
      incomplete.push_back(std::move(entry));
    }
    const auto feet = [](const auto& values) {
      return Json{{"left", values[0]}, {"right", values[1]}, {"total", values[0] + values[1]}};
    };
    return {{"schema", 1}, {"diagnostic_only", true}, {"reset_time", reset_time_},
      {"observed_until", last_time_},
      {"started_airborne", {{"left", started_airborne_[0]}, {"right", started_airborne_[1]}}},
      {"summary", {{"completed_touchdowns", feet(counts)}, {"late_touchdowns", feet(late_counts)},
        {"qualifying_lifts", feet(lifts)}, {"late_qualifying_lifts", feet(late_lifts)},
        {"qualifying_alternations", alternations}, {"late_qualifying_alternations", late_alternations},
        {"initial_airborne_intervals", initial.size()}, {"incomplete_swings", incomplete.size()}}},
      {"completed_swings", std::move(events)}, {"initial_airborne_intervals", std::move(initial)},
      {"incomplete_swings", std::move(incomplete)},
      {"definitions", {
        {"sampling", "Consecutive nominal 20ms control endpoints (50Hz), after actual physics; timestamps are supplied control-interval times, not accumulated MuJoCo clock readings."},
        {"support", "Simulator force-support flag: touching foot-floor contact with an active constraint and normal force >1e-6 N."},
        {"air_seconds", "Touchdown endpoint minus first unsupported endpoint. Peak clearance uses unsupported samples only; transition times between endpoints are unknown."},
        {"sole_clearance", "Lowest sole collision-sphere surface above the z=0 floor, metres; not foot-body-origin height."},
        {"qualifying_lift", "Completed noninitial support-loss interval with observed air time >=0.06s and peak sampled sole clearance >=0.01m. Initial-airborne and incomplete intervals never qualify."},
        {"late", "Touchdown strictly more than 5 seconds after reset; a late alternation requires both participating touchdowns to be late."},
        {"alternation", "Successive qualifying opposite-foot touchdowns separated by >=0.16s; simultaneous or nonqualifying landings break the chain."},
        {"initial_airborne", "Support was absent at reset; actual takeoff time is unknown. Reported loss_time equals reset_time and duration is only the observed portion."},
        {"scope", "The 1cm threshold is diagnostic only, never a reward gate. Force-support touchdowns differ from the legacy goal's geometric-contact touchdowns; these diagnostics do not replace its acceptance criteria."}
      }},
      {"thresholds", {{"sample_interval_seconds", kInterval}, {"minimum_air_seconds", kMinimumAir},
        {"minimum_peak_clearance_m", kMinimumClearance}, {"late_after_seconds", kLateAfter},
        {"minimum_alternation_seconds", kMinimumAlternation}}}};
  }

 private:
  static constexpr double kInterval = .02, kMinimumAir = .06, kMinimumClearance = .01;
  static constexpr double kLateAfter = 5., kMinimumAlternation = .16, kTimeTolerance = 1e-9;
  struct Swing {
    bool open = false, initial = false;
    double loss_time = 0, peak_clearance = 0;
    std::size_t unsupported_samples = 0;
  };
  struct Touchdown {
    int side;
    double loss_time, touchdown_time, air_seconds, peak_clearance;
    std::size_t unsupported_samples;
    bool initial, qualifies, late, alternation, late_alternation;
  };
  static const char* name(int side) { return side == 0 ? "left" : "right"; }
  static void validate_values(double time, const std::array<double, 2>& heights) {
    if (!std::isfinite(time) || !std::isfinite(heights[0]) || !std::isfinite(heights[1]))
      throw std::invalid_argument("SwingDiagnostics requires finite times and sole heights");
  }
  bool initialized_ = false;
  double reset_time_ = 0, last_time_ = 0;
  std::array<bool, 2> supports_{}, started_airborne_{};
  std::array<Swing, 2> active_{};
  std::vector<Touchdown> completed_;
  std::ptrdiff_t last_qualifying_ = -1;
};

}  // namespace humanoid::walk
