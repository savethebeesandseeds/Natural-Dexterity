#pragma once

#include "rotor_model.hpp"
#include "entropy_controller.hpp"
#include <casadi/casadi.hpp>
#include <cmath>
#include <utility>
#include <vector>

namespace helicopter {

struct OptimizerStage {
  casadi::Function function;
  std::vector<double> lower,upper;
};

// One compact SX kernel for every k>0 prediction stage. The caller supplies
// landing=(time+k*H>=68), terminal=(k==N), and approach=(time+k*H>=65).
// Terminal-only velocity, body-rate, and heading bounds remain in the trajectory
// assembly. Approach bounds prepare the aircraft for explicit PID support;
// this kernel still models free flight, not skid contact.
inline OptimizerStage make_optimizer_stage(const rotor::Params& p) {
  using casadi::SX;
  const auto sx=SX::sym("state",rotor::NX);
  const auto su=SX::sym("command",rotor::NU);
  const auto sw=SX::sym("wind",3);
  const auto reference=SX::sym("reference",6);
  const auto landing=SX::sym("landing");
  const auto terminal=SX::sym("terminal");
  const auto approach=SX::sym("approach");
  rotor::State<SX> state;
  rotor::Input<SX> input;
  rotor::Vec3<SX> wind;
  for(int j=0;j<rotor::NX;++j) state[j]=sx(j);
  for(int j=0;j<rotor::NU;++j) input[j]=su(j);
  for(int j=0;j<3;++j) wind[j]=sw(j);
  const auto force=rotor::evaluate(state,input,wind,p);

  std::vector<SX> constraints;
  std::vector<double> lower,upper;
  constraints.reserve(32);lower.reserve(32);upper.reserve(32);
  auto bound=[&](const SX& value,double lo,double hi) {
    constraints.push_back(value);lower.push_back(lo);upper.push_back(hi);
  };

  // Planner tubes are deliberately tighter than the independently audited
  // mission corridors (.6 m horizontally and .35 m vertically).
  for(int j=0;j<3;++j) {
    const SX tube=SX::if_else(landing,j==2?.025:.10,
                            SX::if_else(terminal,.10,j==2?.20:.4));
    bound((state[j]-reference(j))/tube,-1,1);
    bound(state[10+j]/p.body_rate_max,-1,1);
    bound(SX::if_else(landing,(state[3+j]-reference(3+j))/.10,0),-1,1);
    bound(SX::if_else(landing,state[10+j]/.20,0),-1,1);
  }
  bound((state[2]-.2)/1.0,0,6);
  bound(1-2*(state[7]*state[7]+state[8]*state[8])-std::cos(EntropyController::planning_tilt_max_rad),0,2);
  bound(SX::if_else(landing,
                   1-2*(state[7]*state[7]+state[8]*state[8])-std::cos(.0872664626),1),0,2);
  SX airspeed=0;
  for(int j=0;j<3;++j) airspeed+=(state[3+j]-wind[j])*(state[3+j]-wind[j]);
  bound(airspeed/(EntropyController::planning_airspeed_max_m_s*EntropyController::planning_airspeed_max_m_s),0,1);
  for(int j=21;j<23;++j)
    bound((state[j]-p.ambient_temperature)/(p.temperature_max-p.ambient_temperature),0,1);
  bound(force.electrical_power/p.electrical_power_max,0,1);
  bound(force.main_shaft_power/p.main_shaft_power_max,-1,1);
  bound(force.tail_shaft_power/p.tail_shaft_power_max,-1,1);
  bound(force.main_current/p.main_current_max,-1,1);
  bound(force.tail_current/p.tail_current_max,-1,1);
  bound((force.main_axial_airspeed+p.max_descent_inflow_ratio*state[17])/p.airspeed_max,0,2);

  for(int j=0;j<2;++j) {
    bound(SX::if_else(approach,(state[j]-reference(j))/.05,0),-1,1);
    bound(SX::if_else(approach,(state[3+j]-reference(3+j))/.05,0),-1,1);
  }
  for(int j=0;j<3;++j)bound(SX::if_else(approach,state[10+j]/.10,0),-1,1);
  bound(SX::if_else(approach,
                   1-2*(state[7]*state[7]+state[8]*state[8])-std::cos(.0872664626),1),0,2);

  return {casadi::Function("optimizer_stage",{sx,su,sw,reference,landing,terminal,approach},
                          {SX::vertcat(constraints)},casadi::Dict{{"max_num_dir",4}}),
          std::move(lower),std::move(upper)};
}

} // namespace helicopter
