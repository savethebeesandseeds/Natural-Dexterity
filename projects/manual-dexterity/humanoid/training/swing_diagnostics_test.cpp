#include "swing_diagnostics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Trace {
  humanoid::walk::SwingDiagnostics diagnostics;
  int step = 0;
  Trace(std::array<bool, 2> supports = {true, true},
        std::array<double, 2> heights = {0, 0}) { diagnostics.reset(0, supports, heights); }
  void sample(std::array<bool, 2> supports, std::array<double, 2> heights = {0, 0}) {
    diagnostics.observe(++step * .02, supports, heights);
  }
  void stand(int intervals) { for (int i = 0; i < intervals; ++i) sample({true, true}); }
  void lift(int side, double peak = .02, int intervals = 3) {
    for (int i = 0; i < intervals; ++i) {
      std::array<bool, 2> supports{true, true}; supports[side] = false;
      std::array<double, 2> heights{}; heights[side] = peak;
      sample(supports, heights);
    }
    sample({true, true});
  }
};
void standing_and_chatter() {
  Trace standing;
  standing.stand(1000);
  require(standing.diagnostics.report().at("completed_swings").empty(), "Standing fabricated a swing");
  Trace chatter;
  chatter.lift(0, .004);
  chatter.stand(4);
  chatter.lift(1, .006);
  const auto report = chatter.diagnostics.report();
  require(report["summary"]["completed_touchdowns"]["total"] == 2, "Loading transitions were not recorded");
  require(report["summary"]["qualifying_lifts"]["total"] == 0, "Low-clearance chatter counted as lifted feet");
  require(report["summary"]["qualifying_alternations"] == 0, "Chatter created valid alternation");
}
void real_lifts_and_late_boundary() {
  Trace trace;
  trace.stand(246);
  trace.lift(0, .01);  // 60ms unsupported, lands exactly at 5.00s.
  trace.stand(4);
  trace.lift(1, .01);  // Lands 160ms later; only this touchdown is late.
  trace.stand(4);
  trace.lift(0, .015); // Both endpoints of this alternation are late.
  const auto report = trace.diagnostics.report();
  require(report["summary"]["qualifying_lifts"]["total"] == 3, "Boundary-height/airtime swings were rejected");
  require(report["summary"]["late_qualifying_lifts"]["total"] == 2, "5s late boundary was misclassified");
  require(report["summary"]["qualifying_alternations"] == 2, "Real opposite-foot swings did not alternate");
  require(report["summary"]["late_qualifying_alternations"] == 1, "Early touchdown leaked into late alternation");
  require(std::abs(report["completed_swings"][0]["observed_air_seconds"].get<double>() - .06) < 1e-9,
          "Airtime counted an extra control interval");
}
void simultaneous_landing_breaks_chain() {
  Trace trace;
  trace.lift(0);
  trace.stand(4);
  for (int i = 0; i < 3; ++i) trace.sample({false, false}, {.02, .02});
  trace.sample({true, true});
  trace.stand(4);
  trace.lift(1);
  const auto report = trace.diagnostics.report();
  require(report["summary"]["qualifying_lifts"]["total"] == 4, "Simultaneous real lifts were discarded");
  require(report["summary"]["qualifying_alternations"] == 0, "Simultaneous landing invented a foot order");
}
void initial_airborne_and_truncated_swings() {
  Trace trace({false, true}, {.025, 0});
  for (int i = 0; i < 3; ++i) trace.sample({false, true}, {.025, 0});
  trace.sample({true, true});
  trace.sample({true, false}, {0, .03});
  trace.sample({true, false}, {0, .04});
  const auto report = trace.diagnostics.report();
  require(report["initial_airborne_intervals"].size() == 1, "Initial airborne interval missing");
  require(report["initial_airborne_intervals"][0]["complete"] == true, "Initial landing was not recorded");
  require(report["incomplete_swings"].size() == 1, "Truncated swing was lost");
  require(report["incomplete_swings"][0]["touchdown_time"].is_null(), "Truncation invented touchdown time");
  require(report["summary"]["qualifying_lifts"]["total"] == 0, "Initial or incomplete swing qualified");
  require(report == trace.diagnostics.report(), "Reporting changed collector state");
  trace.diagnostics.reset(0, {true, true}, {0, 0});
  require(trace.diagnostics.report()["completed_swings"].empty(), "Reset retained previous episode swings");
}
void missing_samples_are_rejected() {
  Trace trace;
  trace.sample({false, true}, {.02, 0});
  bool rejected = false;
  try { trace.diagnostics.observe(.08, {true, true}, {0, 0}); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "Missing samples fabricated a 60ms flight interval");
  require(trace.diagnostics.report()["completed_swings"].empty(), "Rejected timestamp mutated touchdown history");
}
}  // namespace

int main() {
  try {
    standing_and_chatter();
    real_lifts_and_late_boundary();
    simultaneous_landing_breaks_chain();
    initial_airborne_and_truncated_swings();
    missing_samples_are_rejected();
    std::cout << "Swing diagnostics: clearance, timing, simultaneous landings, initial/truncated intervals passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
