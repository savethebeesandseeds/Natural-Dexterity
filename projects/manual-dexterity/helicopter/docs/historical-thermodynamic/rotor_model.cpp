#include "rotor_model.hpp"

#include <algorithm>
#include <stdexcept>

namespace helicopter::rotor {
namespace {
constexpr double pi=3.14159265358979323846;
constexpr Vec3<double> position_kp{1.45,1.45,6.0};
constexpr Vec3<double> position_kd{2.35,2.35,4.2};
constexpr Vec3<double> position_ki{.3,.3,2.0};
constexpr Vec3<double> attitude_kp{1.30,1.80,.95};
constexpr Vec3<double> attitude_kd{.40,.58,.40};
Vec3<double> unit(const Vec3<double>& a) {
  return detail::scale(a,1/std::max(1e-12,std::sqrt(detail::dot(a,a))));
}
double inverse_collective(double thrust,double inflow,double axial_speed,
                          double radius,double omega,double solidity,const Params& p) {
  const double tip=radius*omega,area=pi*radius*radius;
  const double gain=p.air_density*area*tip*tip*solidity*p.lift_slope*.5;
  return 3*(thrust/gain+(inflow+axial_speed)/(2*tip));
}
}

State<double> initial_state(const Params& p) {
  // Mechanical hover trim in still air, with already governed/spooled rotors.
  // Motors start at the declared ambient temperature, not thermal equilibrium.
  // Unknowns are main/tail thrust and the two steady flapping angles. Pitch and
  // inflow follow the blade-element/momentum equations, and body attitude is
  // subsequently chosen to align the total rotor force with world vertical.
  auto build=[&](const std::array<double,4>& trim) {
    State<double> state{};
    state[2]=.22;state[6]=1;
    state[21]=p.ambient_temperature;state[22]=p.ambient_temperature;
    state[17]=std::sqrt(trim[0]/(2*p.air_density*pi*p.main_radius*p.main_radius));
    state[18]=std::sqrt(trim[1]/(2*p.air_density*pi*p.tail_radius*p.tail_radius));
    state[13]=inverse_collective(trim[0],state[17],0,p.main_radius,p.main_omega,p.main_solidity,p);
    state[16]=inverse_collective(trim[1],state[18],0,p.tail_radius,p.tail_omega,p.tail_solidity,p);
    state[14]=state[19]=trim[2];state[15]=state[20]=trim[3];
    return state;
  };
  auto residual=[&](const std::array<double,4>& trim) {
    const auto state=build(trim);
    const Input<double> command{state[13],state[14],state[15],state[16]};
    const auto force=evaluate(state,command,Vec3<double>{},p);
    return std::array<double,4>{std::sqrt(detail::dot(force.force_body,force.force_body))-p.mass*p.gravity,
      force.torque_body[0],force.torque_body[1],force.torque_body[2]};
  };
  auto norm=[](const std::array<double,4>& r) {
    double result=0;for(double v:r)result=std::max(result,std::abs(v));return result;
  };
  std::array<double,4> trim{p.mass*p.gravity,1.0,0,0};
  bool converged=false;
  for(int iteration=0;iteration<25;++iteration) {
    const auto error=residual(trim);
    if(norm(error)<1e-10) {converged=true;break;}
    std::array<std::array<double,5>,4> matrix{};
    for(int j=0;j<4;++j) {
      auto shifted=trim;
      const double h=1e-5*std::max(1.0,std::abs(trim[j]));
      shifted[j]+=h;
      const auto perturbed=residual(shifted);
      for(int i=0;i<4;++i) matrix[i][j]=(perturbed[i]-error[i])/h;
    }
    for(int i=0;i<4;++i)matrix[i][4]=-error[i];
    // Tiny pivoted dense solve is only used during initialization, never online.
    for(int j=0;j<4;++j) {
      int pivot=j;
      for(int i=j+1;i<4;++i)if(std::abs(matrix[i][j])>std::abs(matrix[pivot][j]))pivot=i;
      if(std::abs(matrix[pivot][j])<1e-12)throw std::runtime_error("Hover trim Jacobian is singular");
      std::swap(matrix[j],matrix[pivot]);
      const double divisor=matrix[j][j];
      for(int k=j;k<5;++k)matrix[j][k]/=divisor;
      for(int i=0;i<4;++i)if(i!=j) {
        const double factor=matrix[i][j];
        for(int k=j;k<5;++k)matrix[i][k]-=factor*matrix[j][k];
      }
    }
    bool accepted=false;
    for(double alpha=1;alpha>=1.0/1024;alpha*=.5) {
      auto trial=trim;
      for(int i=0;i<4;++i)trial[i]+=alpha*matrix[i][4];
      if(trial[0]<=0||trial[1]<=0)continue;
      if(norm(residual(trial))<norm(error)) {trim=trial;accepted=true;break;}
    }
    if(!accepted)throw std::runtime_error("Hover trim failed to reduce force/moment residual");
  }
  if(!converged&&norm(residual(trim))>=1e-10)throw std::runtime_error("Hover trim did not converge");
  State<double> x=build(trim);
  const Input<double> command{x[13],x[14],x[15],x[16]};
  for(int i=0;i<NU;++i)
    if(command[i]<p.input_min[i]||command[i]>p.input_max[i])
      throw std::runtime_error("Hover trim is outside declared pitch limits");
  const auto force=evaluate(x,command,Vec3<double>{},p);
  const double roll=std::atan2(force.force_body[1],force.force_body[2]);
  const double pitch=-std::atan2(force.force_body[0],std::hypot(force.force_body[1],force.force_body[2]));
  const double cr=std::cos(roll*.5),sr=std::sin(roll*.5),cp=std::cos(pitch*.5),sp=std::sin(pitch*.5);
  x[6]=cp*cr;x[7]=cp*sr;x[8]=sp*cr;x[9]=-sp*sr;
  return x;
}

void BaselineController::reset() { integral_={};previous_={}; }
void BaselineController::reset(const Input<double>& previous) { integral_={};previous_=previous; }

Input<double> BaselineController::command(const State<double>& x,const Reference& ref,
                                          const Vec3<double>& wind,const Params& p,double dt) {
  using namespace detail;
  if(!(dt>0)||!std::isfinite(dt)) throw std::invalid_argument("Baseline controller dt must be positive");
  const Input<double> actual{x[13],x[14],x[15],x[16]};
  const auto f=evaluate(x,actual,wind,p);
  const auto r=rotation(x);
  const Vec3<double> position{x[0],x[1],x[2]},velocity{x[3],x[4],x[5]},omega{x[10],x[11],x[12]};
  const auto error=subtract(ref.position,position),velocity_error=subtract(ref.velocity,velocity);
  Vec3<double> acceleration{};
  for(int i=0;i<3;++i) {
    integral_[i]=std::clamp(integral_[i]+dt*error[i],-2.0,2.0);
    acceleration[i]=ref.acceleration[i]+position_kp[i]*error[i]+position_kd[i]*velocity_error[i]+position_ki[i]*integral_[i];
  }
  const double horizontal=std::hypot(acceleration[0],acceleration[1]);
  if(horizontal>3.5) { acceleration[0]*=3.5/horizontal;acceleration[1]*=3.5/horizontal; }
  acceleration[2]=std::clamp(acceleration[2],-4.0,5.0);
  auto required=subtract(scale(add(acceleration,Vec3<double>{0,0,p.gravity}),p.mass),f.drag_world);
  // Explicit wind-drag and actual tail-force feedforward is shared fairly by
  // both the comparison PID and the NMPC warm-start controller.
  required=subtract(required,rotate(r,scale(f.tail_disc_axis,f.tail_thrust)));
  const auto desired_z=unit(required),desired_y=unit(cross(desired_z,Vec3<double>{1,0,0}));
  const auto desired_x=cross(desired_y,desired_z);
  const Vec3<double> current_x{r[0],r[3],r[6]},current_y{r[1],r[4],r[7]},current_z{r[2],r[5],r[8]};
  auto attitude_error=inverse_rotate(r,scale(add(add(cross(current_x,desired_x),cross(current_y,desired_y)),
                                                    cross(current_z,desired_z)),.5));
  const Vec3<double> angular_momentum{p.inertia[0]*omega[0],p.inertia[1]*omega[1],p.inertia[2]*omega[2]};
  const auto gyro=cross(omega,angular_momentum);
  Vec3<double> desired_torque{};
  for(int i=0;i<3;++i)
    desired_torque[i]=attitude_kp[i]*attitude_error[i]-attitude_kd[i]*omega[i]-f.damping_body[i]+gyro[i];
  double main_thrust=std::max(0.0,dot(required,rotate(r,f.main_disc_axis)));
  // The final reference is a landing surface, not a hover setpoint. Transfer a
  // small fraction of weight to the skids as that descending reference reaches
  // the surface. Retaining full hover thrust would make contact intermittent as
  // roll changes the lowest skid height. This changes force, never the reference.
  const double landing_load_fraction=ref.velocity[2]<=0&&x[2]<.30?
    .08*std::clamp((.22-ref.position[2])/.02,0.0,1.0):0.0;
  main_thrust*=1-landing_load_fraction;
  const double tail_lever=std::max(.05,-p.tail_hub[0]);
  const double tail_thrust=std::max(0.0,(desired_torque[2]+f.main_aero_torque*f.main_disc_axis[2])/tail_lever);
  const double hT=p.main_hub[2]*std::max(main_thrust,1.0),q=f.main_aero_torque;
  const double roll_rhs=desired_torque[0]-p.tail_hub[2]*tail_thrust;
  const double pitch_rhs=desired_torque[1]-f.tail_aero_torque;
  const double denominator=std::max(1e-8,q*q+hT*hT);
  const double nx=(-q*roll_rhs+hT*pitch_rhs)/denominator;
  const double ny=(-hT*roll_rhs-q*pitch_rhs)/denominator;
  const double lateral=std::asin(std::clamp(-ny,-.3,.3));
  const double longitudinal=std::asin(std::clamp(nx/std::cos(lateral),-.3,.3));
  Input<double> result{
    inverse_collective(main_thrust,x[17],f.main_axial_airspeed,p.main_radius,p.main_omega,p.main_solidity,p),
    longitudinal,lateral,
    inverse_collective(tail_thrust,x[18],f.tail_axial_airspeed,p.tail_radius,p.tail_omega,p.tail_solidity,p)};
  for(int i=0;i<NU;++i) {
    result[i]=std::clamp(result[i],p.input_min[i],p.input_max[i]);
    result[i]=std::clamp(result[i],previous_[i]-p.input_rate_max[i]*dt,previous_[i]+p.input_rate_max[i]*dt);
  }
  previous_=result;
  return result;
}

Input<double> normalized_input(const Input<double>& physical,const Params& p) {
  Input<double> result{};
  for(int i=0;i<NU;++i) {
    if(i==1||i==2) result[i]=physical[i]/p.input_max[i];
    else result[i]=(physical[i]-p.input_min[i])/(p.input_max[i]-p.input_min[i]);
  }
  return result;
}

nlohmann::json model_metadata(const Params& p) {
  return {
    {"name","Illustrative electric main/tail-rotor helicopter with governed rotor speed"},
    {"calibration","Documented simulated reference model; no measured aircraft identification or hardware validation"},
    {"initialization",{
      {"mode","Already spooled and mechanically trimmed for still-air hover; no motor-start or spool-up model"},
      {"position_m",{0,0,.22}},{"motor_temperature_k",p.ambient_temperature},
      {"conditions","Zero translation/body rates, governed rotor speeds, steady pitch/inflow/flap states, rotor force balances gravity and rotor body moments cancel including tail force and both reaction torques. Body roll/pitch compensate lateral force; yaw is zero."},
      {"energy_boundary","Mission accounting starts after this common prepared initial condition. Initial wake energy is nonzero and reported; ambient-temperature motors are an assumed initial condition, not a simulated spool-up history."}}},
    {"state_dimension",NX},{"input_dimension",NU},
    {"state",{
      {"position",{{"indices",{0,1,2}},{"unit","m"},{"frame","world"}}},
      {"velocity",{{"indices",{3,4,5}},{"unit","m/s"},{"frame","world"}}},
      {"quaternion",{{"indices",{6,7,8,9}},{"order","wxyz"},{"meaning","body-to-world rotation"}}},
      {"body_rates",{{"indices",{10,11,12}},{"unit","rad/s"}}},
      {"actual_pitch",{{"indices",{13,14,15,16}},{"unit","rad"},{"order",{"main collective","longitudinal cyclic","lateral cyclic","tail collective"}}}},
      {"inflow",{{"indices",{17,18}},{"unit","m/s"},{"order",{"main","tail"}}}},
      {"flap",{{"indices",{19,20}},{"unit","rad"},{"order",{"longitudinal","lateral"}}}},
      {"motor_temperature",{{"indices",{21,22}},{"unit","K"},{"order",{"main","tail"}}}}}},
    {"coordinates","Right handed: world z up; body x forward, y left, z up. Main positive collective thrusts along disc normal. Positive longitudinal cyclic tips disc +x; positive lateral cyclic tips disc -y. Positive tail collective thrusts -y at negative x, producing positive yaw."},
    {"airframe",{{"mass_kg",p.mass},{"inertia_kg_m2",p.inertia},{"gravity_m_s2",p.gravity},
                  {"linear_drag_n_per_m_s",p.linear_drag},{"angular_damping_nm_per_rad_s",p.angular_damping}}},
    {"environment",{{"air_density_kg_m3",p.air_density},{"ambient_temperature_k",p.ambient_temperature}}},
    {"rotors",{
      {"main",{{"radius_m",p.main_radius},{"omega_rad_s",p.main_omega},{"solidity",p.main_solidity},
               {"profile_cd",p.main_profile_cd},{"induced_factor",p.main_induced_factor},{"hub_m",p.main_hub},
               {"inflow_tau_s",p.main_inflow_tau},{"effective_inflow_mass_kg",p.main_inflow_mass}}},
      {"tail",{{"radius_m",p.tail_radius},{"omega_rad_s",p.tail_omega},{"solidity",p.tail_solidity},
               {"profile_cd",p.tail_profile_cd},{"induced_factor",p.tail_induced_factor},{"hub_m",p.tail_hub},
               {"inflow_tau_s",p.tail_inflow_tau},{"effective_inflow_mass_kg",p.tail_inflow_mass}}},
      {"lift_slope_per_rad",p.lift_slope},{"actuator_tau_s",p.actuator_tau},{"flap_tau_s",p.flap_tau}}},
    {"motors",{
      {"main",{{"torque_constant_nm_a",p.main_motor_kt},{"resistance_ohm",p.main_resistance},
               {"iron_loss_w",p.main_iron_loss},{"thermal_capacity_j_k",p.main_thermal_capacity},
               {"heat_transfer_w_k",p.main_heat_transfer}}},
      {"tail",{{"torque_constant_nm_a",p.tail_motor_kt},{"resistance_ohm",p.tail_resistance},
               {"iron_loss_w",p.tail_iron_loss},{"thermal_capacity_j_k",p.tail_thermal_capacity},
               {"heat_transfer_w_k",p.tail_heat_transfer}}},
      {"resistance_alpha_per_k",p.resistance_alpha},{"resistance_reference_temperature_k",p.resistance_reference_temperature},
      {"drivetrain_loss_fraction",p.drivetrain_loss_fraction},
      {"drive","Non-regenerative governed-speed drive; negative shaft work becomes explicit braking heat"}}},
    {"limits",{{"input_min_rad",p.input_min},{"input_max_rad",p.input_max},{"input_rate_max_rad_s",p.input_rate_max},
               {"temperature_max_k",p.temperature_max},{"electrical_power_max_w",p.electrical_power_max},
               {"main_shaft_power_max_w",p.main_shaft_power_max},{"tail_shaft_power_max_w",p.tail_shaft_power_max},
               {"main_current_max_a",p.main_current_max},{"tail_current_max_a",p.tail_current_max},
               {"tilt_max_rad",p.tilt_max_rad},{"body_rate_max_rad_s",p.body_rate_max},{"airspeed_max_m_s",p.airspeed_max},
               {"max_descent_inflow_ratio",p.max_descent_inflow_ratio}}},
    {"baseline",{{"position_kp",position_kp},{"position_kd",position_kd},{"position_ki",position_ki},
                  {"attitude_kp",attitude_kp},{"attitude_kd",attitude_kd},
                  {"integral_limit_m_s",2.0},{"horizontal_acceleration_max_m_s2",3.5},
                  {"vertical_acceleration_bounds_m_s2",{-4.0,5.0}},
                  {"landing_thrust_unload_fraction",.08},{"landing_reference_height_interval_m",{.22,.20}},
                  {"landing_policy","During the final descending reference below 0.22 m, smoothly transfer up to 8 percent of commanded main-rotor support to the skids; reference and constraints remain unchanged"},
                  {"description","Privileged-state PID/PD with wind drag, tail force, gyro and damping feedforward, inverse rotor allocation, and identical pitch/rate limits"}}},
    {"equations",{
      {"blade_element","T=positive[rho A (Omega R)^2 solidity a/2 (theta/3 - (vi+Vaxial)/(2 Omega R))]"},
      {"inflow","vi_eq=sqrt(Vaxial^2/4+T/(2 rho A))-Vaxial/2; dvi=(vi_eq-vi)/tau_i"},
      {"flap","d beta=(actual_cyclic-beta)/tau_flap; n=[sin(beta_long)cos(beta_lat),-sin(beta_lat),cos(beta_long)cos(beta_lat)]"},
      {"wake_storage","E_i=0.5 M_i vi^2; dE_i=M_i vi dvi"},
      {"aerodynamic_power","Paero=T Vhub_air + kappa T vi + rho A (Omega R)^3 solidity Cd/8 + dE_i/dt"},
      {"reaction_and_shaft","Q=Paero/(Omega+omega_body dot n); Pshaft=Q Omega; tau_rotor=r_hub cross (T n)-Q n"},
      {"motor","gear=fraction*positive(Pshaft); I=(Pshaft+gear)/(Omega Kt); R(T)=Rref[1+alpha(T-Tref)]; copper=I^2 R(T); braking=positive(-Pshaft)"},
      {"electrical_power","Pelec=positive(Pshaft)+copper+iron+gear; motor_loss=copper+iron+gear+braking"},
      {"thermal","C dT/dt=motor_loss-h(T-T0)"},
      {"entropy","sigma=sum(motor_loss/T+h(T-T0)(1/T0-1/T))+(wake_dissipation+drag_dissipation+angular_dissipation)/T0"},
      {"wind_port","Pwind=(R Frotor+Fdrag) dot wind"},
      {"first_law","Pelec+Pwind=dE_mechanical/dt+dE_inflow/dt+dE_thermal/dt+wake+drag+angular_dissipation+heat_out"},
      {"thermal_exergy","Bth=sum C[(T-T0)-T0 log(T/T0)]"},
      {"terminal_commitment","S_commit=(Bth+E_inflow)/T0; eventual passive thermal cooling plus complete dissipation of remaining modeled wake kinetic energy"}}},
    {"boundary","Aircraft mechanical and motor/drive thermal storage plus effective induced-flow kinetic storage and eventual wake relaxation at ambient temperature. Electrical input and imposed uniform-wind work are external ports."},
    {"smoothing",{{"positive","positive(z)=(z+sqrt(z^2+epsilon^2))/2"},
                  {"thrust_epsilon_n",p.thrust_smoothing},{"power_epsilon_w",p.power_smoothing},
                  {"inflow_sqrt_epsilon_m2_s2",1e-12},{"quaternion_norm_epsilon",1e-24}}},
    {"validity","Near-hover, nonnegative rotor thrust/inflow, positive absolute blade speed, temperatures at or above ambient. Main axial descent is limited to a declared fraction of induced speed; edgewise airspeed, body tilt and rates are bounded. No stall, vortex-ring, ground-effect, compressibility, detailed blade elasticity, finite governor transient, battery chemistry or hardware calibration."},
    {"contact","Shared derivative is free flight. Ground-contact work is outside this derivative and must be separately reported by the plant; only free-flight windows establish this first-law residual."},
    {"power_units","All powers and loss rates W; force N; torque N m; current A; energy J; temperature K; entropy rate W/K; terminal commitment J/K"}
  };
}
} // namespace helicopter::rotor
