#pragma once

#include "rotor_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace helicopter::rotor {

struct PhysicalViolation {
  // Zero is feasible. Positive values are dimensionless excess over a bound.
  // Non-finite inputs/derived physics return infinity and finite=false.
  double maximum=0;
  std::string worst="none";
  bool finite=true;
};

inline constexpr double minimum_flight_altitude_m=.2;
inline constexpr double minimum_contact_altitude_m=.18;
inline constexpr double maximum_flight_altitude_m=6.2;

// Independent numeric checks for both re-integrated candidate trajectories and
// executed plant states. This deliberately contains no reference/tracking cost
// or corridor. Command slew is a two-sample property checked by the caller.
// airborne=false allows the compliant contact floor down to 0.18 m and exempts
// only the free-flight descent envelope. Every other vehicle limit stays active.
inline PhysicalViolation physical_violation(const State<double>& x,
                                             const Input<double>& command,
                                             const Vec3<double>& wind,
                                             const Params& p,
                                             bool airborne=true) {
  PhysicalViolation result;
  auto invalid=[&](const std::string& name) {
    result.maximum=std::numeric_limits<double>::infinity();
    result.worst=name;result.finite=false;
  };
  for(int i=0;i<NX;++i)if(!std::isfinite(x[i])) {
    invalid("nonfinite_state_"+std::to_string(i));return result;
  }
  for(int i=0;i<NU;++i)if(!std::isfinite(command[i])) {
    invalid("nonfinite_command_"+std::to_string(i));return result;
  }
  for(int i=0;i<3;++i)if(!std::isfinite(wind[i])) {
    invalid("nonfinite_wind_"+std::to_string(i));return result;
  }
  auto consider=[&](double violation,const std::string& name) {
    if(!std::isfinite(violation)) { invalid("nonfinite_"+name);return; }
    if(violation>result.maximum) { result.maximum=violation;result.worst=name; }
  };
  // Bounds share one explicit scale, making lower and upper excess comparable.
  auto range=[&](double value,double lower,double upper,double scale,const std::string& name) {
    if(!std::isfinite(value)||!std::isfinite(lower)||!std::isfinite(upper)||
       !std::isfinite(scale)||!(scale>0)||upper<lower) {
      invalid("invalid_"+name);return;
    }
    consider((lower-value)/scale,name+"_lower");
    consider((value-upper)/scale,name+"_upper");
  };

  const double temperature_range=p.temperature_max-p.ambient_temperature;
  range(x[21],p.ambient_temperature,p.temperature_max,temperature_range,"main_temperature");
  range(x[22],p.ambient_temperature,p.temperature_max,temperature_range,"tail_temperature");
  for(int i=0;i<NU;++i) {
    const double width=p.input_max[i]-p.input_min[i];
    range(command[i],p.input_min[i],p.input_max[i],width,"command_pitch_"+std::to_string(i));
    range(x[13+i],p.input_min[i],p.input_max[i],width,"actual_pitch_"+std::to_string(i));
  }
  for(int i=0;i<3;++i)
    range(x[10+i],-p.body_rate_max,p.body_rate_max,p.body_rate_max,"body_rate_"+std::to_string(i));
  if(!result.finite)return result;

  double quaternion_norm_squared=0;
  for(int i=6;i<10;++i)quaternion_norm_squared+=x[i]*x[i];
  if(!std::isfinite(quaternion_norm_squared)||quaternion_norm_squared<1e-16) {
    invalid("invalid_quaternion");return result;
  }
  // Normalize only for the geometric measurement, never modify the plant state.
  const double body_vertical_world_z=1.0-2.0*(x[7]*x[7]+x[8]*x[8])/quaternion_norm_squared;
  const double tilt=std::acos(std::clamp(body_vertical_world_z,-1.0,1.0));
  range(tilt,0,p.tilt_max_rad,p.tilt_max_rad,"body_tilt");
  const double airspeed=std::hypot(x[3]-wind[0],x[4]-wind[1],x[5]-wind[2]);
  range(airspeed,0,p.airspeed_max,p.airspeed_max,"airspeed");
  // Metres divided by one metre, matching the altitude scaling in the NLP.
  const double minimum_altitude=airborne?minimum_flight_altitude_m:minimum_contact_altitude_m;
  consider((minimum_altitude-x[2])/1.0,"altitude_lower");
  consider((x[2]-maximum_flight_altitude_m)/1.0,"altitude_upper");
  if(!result.finite)return result;

  const auto force=evaluate(x,command,wind,p);
  range(force.electrical_power,0,p.electrical_power_max,p.electrical_power_max,"electrical_power");
  range(force.main_shaft_power,-p.main_shaft_power_max,p.main_shaft_power_max,p.main_shaft_power_max,"main_shaft_power");
  range(force.tail_shaft_power,-p.tail_shaft_power_max,p.tail_shaft_power_max,p.tail_shaft_power_max,"tail_shaft_power");
  range(force.main_current,-p.main_current_max,p.main_current_max,p.main_current_max,"main_current");
  range(force.tail_current,-p.tail_current_max,p.tail_current_max,p.tail_current_max,"tail_current");
  // Nonnegative induced flow is part of the model's stated thermodynamic domain.
  consider(-x[17]/p.airspeed_max,"main_inflow_lower");
  consider(-x[18]/p.airspeed_max,"tail_inflow_lower");
  if(airborne)
    consider(-(force.main_axial_airspeed+p.max_descent_inflow_ratio*x[17])/p.airspeed_max,
             "main_descent_envelope");
  if(!std::isfinite(force.entropy_rate)||!std::isfinite(force.first_law_residual))
    invalid("nonfinite_thermodynamics");
  return result;
}

} // namespace helicopter::rotor
