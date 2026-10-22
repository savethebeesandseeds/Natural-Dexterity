#pragma once

#include <array>
#include <cmath>
#include <nlohmann/json.hpp>

namespace helicopter::rotor {
constexpr int NX=23, NU=4;
template<class S> using State=std::array<S,NX>;
template<class S> using Input=std::array<S,NU>;
template<class S> using Vec3=std::array<S,3>;

// p[0:3], v_world[3:6], q_wxyz[6:10], omega_body[10:13],
// actual pitches[13:17], inflow speeds[17:19], flap angles[19:21], T_motor[21:23].
// An illustrative electric 2 kg helicopter; no parameter is identified from hardware.
struct Params {
  double mass=2.0;
  Vec3<double> inertia{.060,.095,.090};
  double gravity=9.81, air_density=1.225, ambient_temperature=293.15;
  double main_radius=.55, tail_radius=.11;
  double main_omega=180.0, tail_omega=650.0;
  double main_solidity=.045, tail_solidity=.13, lift_slope=5.7;
  double main_profile_cd=.011, tail_profile_cd=.012;
  double main_induced_factor=1.15, tail_induced_factor=1.15;
  Vec3<double> main_hub{0,0,.18}, tail_hub{-.70,0,.05};
  double actuator_tau=.08, flap_tau=.06, main_inflow_tau=.12, tail_inflow_tau=.06;
  double main_inflow_mass=.60, tail_inflow_mass=.010;
  double linear_drag=.65, angular_damping=.035;
  double main_motor_kt=.080, tail_motor_kt=.025;
  double main_resistance=.10, tail_resistance=.30, resistance_alpha=.00393;
  double resistance_reference_temperature=293.15;
  double main_iron_loss=3.0, tail_iron_loss=.8, drivetrain_loss_fraction=.035;
  double main_thermal_capacity=120.0, tail_thermal_capacity=30.0;
  double main_heat_transfer=1.8, tail_heat_transfer=.50;
  double thrust_smoothing=1e-6, power_smoothing=1e-6;
  Input<double> input_min{0,-.18,-.18,0};
  Input<double> input_max{.22,.18,.18,.35};
  Input<double> input_rate_max{.65,1.2,1.2,1.2};
  double temperature_max=353.15, electrical_power_max=650.0;
  double main_shaft_power_max=550.0, tail_shaft_power_max=90.0;
  double main_current_max=35.0, tail_current_max=15.0;
  double tilt_max_rad=.52, body_rate_max=2.5, airspeed_max=5.0;
  double max_descent_inflow_ratio=.50;
};

template<class S> struct Forces {
  Vec3<S> force_body{},torque_body{},drag_world{},damping_body{};
  Vec3<S> main_disc_axis{},tail_disc_axis{};
  S main_thrust{},tail_thrust{},main_aero_torque{},tail_aero_torque{};
  S main_aero_power{},tail_aero_power{},main_axial_airspeed{},tail_axial_airspeed{};
  S main_inflow_dot{},tail_inflow_dot{};
  S main_inflow_equilibrium{},tail_inflow_equilibrium{};
  S main_shaft_power{},tail_shaft_power{},shaft_power{},electrical_power{};
  S main_electrical_power{},tail_electrical_power{};
  S main_induced_power{},tail_induced_power{},main_profile_power{},tail_profile_power{};
  S main_copper_loss{},tail_copper_loss{},main_iron_loss{},tail_iron_loss{};
  S main_drivetrain_loss{},tail_drivetrain_loss{},main_braking_loss{},tail_braking_loss{};
  S main_motor_loss{},tail_motor_loss{},main_current{},tail_current{};
  S main_heat_out{},tail_heat_out{},main_temperature_dot{},tail_temperature_dot{};
  S stored_wake_energy{},stored_wake_energy_dot{},stored_thermal_energy{},stored_thermal_energy_dot{};
  S main_stored_wake_energy{},tail_stored_wake_energy{},main_stored_wake_energy_dot{},tail_stored_wake_energy_dot{};
  S wake_dissipation{},drag_dissipation{},angular_dissipation{},mechanical_power{},wind_power{};
  S entropy_motor{},entropy_heat_transfer{},entropy_wake{},entropy_drag{},entropy_rate{};
  S first_law_residual{};
};

template<class S> Forces<S> evaluate(const State<S>& x,const Input<S>& u,
                                    const Vec3<S>& wind,const Params& p);
template<class S> State<S> derivative(const State<S>& x,const Input<S>& u,
                                     const Vec3<S>& wind,const Params& p);
template<class S> S thermal_exergy(const State<S>& x,const Params& p);
template<class S> S stored_wake_energy(const State<S>& x,const Params& p);
template<class S> S terminal_entropy_commitment(const State<S>& x,const Params& p);
template<class S> State<S> integrate_rk4(const State<S>& x,const Input<S>& u,
                                       const Vec3<S>& wind,const Params& p,double dt,int substeps=1);
State<double> initial_state(const Params& p);

struct Reference { Vec3<double> position{},velocity{},acceleration{}; };
class BaselineController {
 public:
  void reset();
  void reset(const Input<double>& previous);
  Input<double> command(const State<double>& x,const Reference& reference,
                        const Vec3<double>& wind,const Params& p,double dt);
  [[nodiscard]] Vec3<double> integral() const { return integral_; }
  [[nodiscard]] bool limited() const { return limited_; }
 private:
  Vec3<double> integral_{};
  Input<double> previous_{};
  bool pending_initialization_=true;
  bool limited_=false;
};
Input<double> normalized_input(const Input<double>& physical,const Params& p);
nlohmann::json model_metadata(const Params& p);
} // namespace helicopter::rotor

// Header-only scalar algebra deliberately supports both double and CasADi SX
// through argument-dependent lookup, without requiring CasADi in every client.
namespace helicopter::rotor::detail {
template<class S> S root(const S& a) { using std::sqrt; return sqrt(a); }
template<class S> S sine(const S& a) { using std::sin; return sin(a); }
template<class S> S cosine(const S& a) { using std::cos; return cos(a); }
template<class S> S logarithm(const S& a) { using std::log; return log(a); }
template<class S> S positive(const S& a,double epsilon) { return (a+root(a*a+S(epsilon*epsilon)))*S(.5); }
template<class S> Vec3<S> add(const Vec3<S>& a,const Vec3<S>& b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
template<class S> Vec3<S> subtract(const Vec3<S>& a,const Vec3<S>& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
template<class S> Vec3<S> scale(const Vec3<S>& a,const S& b) { return {a[0]*b,a[1]*b,a[2]*b}; }
template<class S> S dot(const Vec3<S>& a,const Vec3<S>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
template<class S> Vec3<S> cross(const Vec3<S>& a,const Vec3<S>& b) {
  return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
template<class S> Vec3<S> cast(const Vec3<double>& a) { return {S(a[0]),S(a[1]),S(a[2])}; }
template<class S> std::array<S,9> rotation(const State<S>& x) {
  const S qnorm=root(x[6]*x[6]+x[7]*x[7]+x[8]*x[8]+x[9]*x[9]+S(1e-24));
  const S w=x[6]/qnorm,a=x[7]/qnorm,b=x[8]/qnorm,c=x[9]/qnorm;
  return {S(1)-S(2)*(b*b+c*c),S(2)*(a*b-w*c),S(2)*(a*c+w*b),
          S(2)*(a*b+w*c),S(1)-S(2)*(a*a+c*c),S(2)*(b*c-w*a),
          S(2)*(a*c-w*b),S(2)*(b*c+w*a),S(1)-S(2)*(a*a+b*b)};
}
template<class S> Vec3<S> rotate(const std::array<S,9>& r,const Vec3<S>& v) {
  return {r[0]*v[0]+r[1]*v[1]+r[2]*v[2],r[3]*v[0]+r[4]*v[1]+r[5]*v[2],r[6]*v[0]+r[7]*v[1]+r[8]*v[2]};
}
template<class S> Vec3<S> inverse_rotate(const std::array<S,9>& r,const Vec3<S>& v) {
  return {r[0]*v[0]+r[3]*v[1]+r[6]*v[2],r[1]*v[0]+r[4]*v[1]+r[7]*v[2],r[2]*v[0]+r[5]*v[1]+r[8]*v[2]};
}
} // namespace helicopter::rotor::detail

namespace helicopter::rotor {
template<class S> Forces<S> evaluate(const State<S>& x,const Input<S>&,
                                    const Vec3<S>& wind,const Params& p) {
  using namespace detail;
  constexpr double pi=3.14159265358979323846;
  Forces<S> f;
  const auto r=rotation(x);
  const Vec3<S> velocity{x[3],x[4],x[5]},omega{x[10],x[11],x[12]};
  const Vec3<S> relative_world=subtract(velocity,wind);
  const Vec3<S> relative_body=inverse_rotate(r,relative_world);
  const Vec3<S> main_hub=cast<S>(p.main_hub),tail_hub=cast<S>(p.tail_hub);
  f.main_disc_axis={sine(x[19])*cosine(x[20]),-sine(x[20]),cosine(x[19])*cosine(x[20])};
  f.tail_disc_axis={S(0),S(-1),S(0)};
  f.main_axial_airspeed=dot(add(relative_body,cross(omega,main_hub)),f.main_disc_axis);
  f.tail_axial_airspeed=dot(add(relative_body,cross(omega,tail_hub)),f.tail_disc_axis);
  const double main_area=pi*p.main_radius*p.main_radius,tail_area=pi*p.tail_radius*p.tail_radius;
  const double main_tip=p.main_omega*p.main_radius,tail_tip=p.tail_omega*p.tail_radius;
  const double main_gain=p.air_density*main_area*main_tip*main_tip*p.main_solidity*p.lift_slope*.5;
  const double tail_gain=p.air_density*tail_area*tail_tip*tail_tip*p.tail_solidity*p.lift_slope*.5;
  f.main_thrust=positive(S(main_gain)*(x[13]/S(3)-(x[17]+f.main_axial_airspeed)/S(2*main_tip)),p.thrust_smoothing);
  f.tail_thrust=positive(S(tail_gain)*(x[16]/S(3)-(x[18]+f.tail_axial_airspeed)/S(2*tail_tip)),p.thrust_smoothing);
  f.main_inflow_equilibrium=root(f.main_axial_airspeed*f.main_axial_airspeed*S(.25)+
                                      f.main_thrust/S(2*p.air_density*main_area)+S(1e-12))-f.main_axial_airspeed*S(.5);
  f.tail_inflow_equilibrium=root(f.tail_axial_airspeed*f.tail_axial_airspeed*S(.25)+
                                      f.tail_thrust/S(2*p.air_density*tail_area)+S(1e-12))-f.tail_axial_airspeed*S(.5);
  f.main_inflow_dot=(f.main_inflow_equilibrium-x[17])/S(p.main_inflow_tau);
  f.tail_inflow_dot=(f.tail_inflow_equilibrium-x[18])/S(p.tail_inflow_tau);
  f.main_stored_wake_energy=S(.5*p.main_inflow_mass)*x[17]*x[17];
  f.tail_stored_wake_energy=S(.5*p.tail_inflow_mass)*x[18]*x[18];
  f.main_stored_wake_energy_dot=S(p.main_inflow_mass)*x[17]*f.main_inflow_dot;
  f.tail_stored_wake_energy_dot=S(p.tail_inflow_mass)*x[18]*f.tail_inflow_dot;
  f.stored_wake_energy=f.main_stored_wake_energy+f.tail_stored_wake_energy;
  f.stored_wake_energy_dot=f.main_stored_wake_energy_dot+f.tail_stored_wake_energy_dot;
  f.main_induced_power=S(p.main_induced_factor)*f.main_thrust*x[17];
  f.tail_induced_power=S(p.tail_induced_factor)*f.tail_thrust*x[18];
  f.main_profile_power=S(p.air_density*main_area*main_tip*main_tip*main_tip*p.main_solidity*p.main_profile_cd/8.0);
  f.tail_profile_power=S(p.air_density*tail_area*tail_tip*tail_tip*tail_tip*p.tail_solidity*p.tail_profile_cd/8.0);
  f.main_aero_power=f.main_thrust*f.main_axial_airspeed+f.main_induced_power+f.main_profile_power+f.main_stored_wake_energy_dot;
  f.tail_aero_power=f.tail_thrust*f.tail_axial_airspeed+f.tail_induced_power+f.tail_profile_power+f.tail_stored_wake_energy_dot;
  // Absolute blade angular speed includes body spin. This denominator and the
  // reaction moment below make shaft power exactly consistent with body work.
  f.main_aero_torque=f.main_aero_power/(S(p.main_omega)+dot(omega,f.main_disc_axis));
  f.tail_aero_torque=f.tail_aero_power/(S(p.tail_omega)+dot(omega,f.tail_disc_axis));
  f.main_shaft_power=f.main_aero_torque*S(p.main_omega);
  f.tail_shaft_power=f.tail_aero_torque*S(p.tail_omega);
  f.shaft_power=f.main_shaft_power+f.tail_shaft_power;
  const Vec3<S> main_force=scale(f.main_disc_axis,f.main_thrust),tail_force=scale(f.tail_disc_axis,f.tail_thrust);
  f.force_body=add(main_force,tail_force);
  f.torque_body=subtract(add(cross(main_hub,main_force),cross(tail_hub,tail_force)),
                        add(scale(f.main_disc_axis,f.main_aero_torque),scale(f.tail_disc_axis,f.tail_aero_torque)));
  f.drag_world=scale(relative_world,S(-p.linear_drag));
  f.damping_body=scale(omega,S(-p.angular_damping));
  const S main_positive=positive(f.main_shaft_power,p.power_smoothing),tail_positive=positive(f.tail_shaft_power,p.power_smoothing);
  f.main_drivetrain_loss=S(p.drivetrain_loss_fraction)*main_positive;
  f.tail_drivetrain_loss=S(p.drivetrain_loss_fraction)*tail_positive;
  f.main_current=(f.main_shaft_power+f.main_drivetrain_loss)/S(p.main_omega*p.main_motor_kt);
  f.tail_current=(f.tail_shaft_power+f.tail_drivetrain_loss)/S(p.tail_omega*p.tail_motor_kt);
  const S main_resistance=S(p.main_resistance)*(S(1)+S(p.resistance_alpha)*(x[21]-S(p.resistance_reference_temperature)));
  const S tail_resistance=S(p.tail_resistance)*(S(1)+S(p.resistance_alpha)*(x[22]-S(p.resistance_reference_temperature)));
  f.main_copper_loss=f.main_current*f.main_current*main_resistance;
  f.tail_copper_loss=f.tail_current*f.tail_current*tail_resistance;
  f.main_iron_loss=S(p.main_iron_loss);f.tail_iron_loss=S(p.tail_iron_loss);
  f.main_braking_loss=positive(-f.main_shaft_power,p.power_smoothing);
  f.tail_braking_loss=positive(-f.tail_shaft_power,p.power_smoothing);
  f.main_motor_loss=f.main_copper_loss+f.main_iron_loss+f.main_drivetrain_loss+f.main_braking_loss;
  f.tail_motor_loss=f.tail_copper_loss+f.tail_iron_loss+f.tail_drivetrain_loss+f.tail_braking_loss;
  f.main_electrical_power=main_positive+f.main_copper_loss+f.main_iron_loss+f.main_drivetrain_loss;
  f.tail_electrical_power=tail_positive+f.tail_copper_loss+f.tail_iron_loss+f.tail_drivetrain_loss;
  f.electrical_power=f.main_electrical_power+f.tail_electrical_power;
  f.main_heat_out=S(p.main_heat_transfer)*(x[21]-S(p.ambient_temperature));
  f.tail_heat_out=S(p.tail_heat_transfer)*(x[22]-S(p.ambient_temperature));
  f.main_temperature_dot=(f.main_motor_loss-f.main_heat_out)/S(p.main_thermal_capacity);
  f.tail_temperature_dot=(f.tail_motor_loss-f.tail_heat_out)/S(p.tail_thermal_capacity);
  f.stored_thermal_energy=S(p.main_thermal_capacity)*(x[21]-S(p.ambient_temperature))+
                          S(p.tail_thermal_capacity)*(x[22]-S(p.ambient_temperature));
  f.stored_thermal_energy_dot=f.main_motor_loss+f.tail_motor_loss-f.main_heat_out-f.tail_heat_out;
  f.wake_dissipation=f.main_induced_power+f.tail_induced_power+f.main_profile_power+f.tail_profile_power;
  f.drag_dissipation=S(p.linear_drag)*dot(relative_world,relative_world);
  f.angular_dissipation=S(p.angular_damping)*dot(omega,omega);
  const Vec3<S> rotor_world=rotate(r,f.force_body);
  f.mechanical_power=dot(add(rotor_world,f.drag_world),velocity)+dot(add(f.torque_body,f.damping_body),omega);
  f.wind_power=dot(add(rotor_world,f.drag_world),wind);
  f.entropy_motor=f.main_motor_loss/x[21]+f.tail_motor_loss/x[22];
  f.entropy_heat_transfer=f.main_heat_out*(S(1/p.ambient_temperature)-S(1)/x[21])+
                          f.tail_heat_out*(S(1/p.ambient_temperature)-S(1)/x[22]);
  f.entropy_wake=f.wake_dissipation/S(p.ambient_temperature);
  f.entropy_drag=(f.drag_dissipation+f.angular_dissipation)/S(p.ambient_temperature);
  f.entropy_rate=f.entropy_motor+f.entropy_heat_transfer+f.entropy_wake+f.entropy_drag;
  f.first_law_residual=f.electrical_power+f.wind_power-f.mechanical_power-f.stored_wake_energy_dot-
    f.stored_thermal_energy_dot-f.wake_dissipation-f.drag_dissipation-f.angular_dissipation-f.main_heat_out-f.tail_heat_out;
  return f;
}

template<class S> State<S> derivative(const State<S>& x,const Input<S>& u,
                                     const Vec3<S>& wind,const Params& p) {
  using namespace detail;
  const auto f=evaluate(x,u,wind,p);
  const auto r=rotation(x);
  const auto force=add(rotate(r,f.force_body),f.drag_world);
  const Vec3<S> omega{x[10],x[11],x[12]};
  const Vec3<S> angular_momentum{S(p.inertia[0])*x[10],S(p.inertia[1])*x[11],S(p.inertia[2])*x[12]};
  const auto torque=subtract(add(f.torque_body,f.damping_body),cross(omega,angular_momentum));
  State<S> dx;
  for(int i=0;i<NX;++i) dx[i]=S(0);
  for(int i=0;i<3;++i) { dx[i]=x[3+i];dx[3+i]=force[i]/S(p.mass);dx[10+i]=torque[i]/S(p.inertia[i]); }
  dx[5]-=S(p.gravity);
  dx[6]=-S(.5)*(x[7]*x[10]+x[8]*x[11]+x[9]*x[12]);
  dx[7]=S(.5)*(x[6]*x[10]+x[8]*x[12]-x[9]*x[11]);
  dx[8]=S(.5)*(x[6]*x[11]+x[9]*x[10]-x[7]*x[12]);
  dx[9]=S(.5)*(x[6]*x[12]+x[7]*x[11]-x[8]*x[10]);
  for(int i=0;i<NU;++i) dx[13+i]=(u[i]-x[13+i])/S(p.actuator_tau);
  dx[17]=f.main_inflow_dot;dx[18]=f.tail_inflow_dot;
  dx[19]=(x[14]-x[19])/S(p.flap_tau);dx[20]=(x[15]-x[20])/S(p.flap_tau);
  dx[21]=f.main_temperature_dot;dx[22]=f.tail_temperature_dot;
  return dx;
}

template<class S> S thermal_exergy(const State<S>& x,const Params& p) {
  using namespace detail;
  return S(p.main_thermal_capacity)*(x[21]-S(p.ambient_temperature)-S(p.ambient_temperature)*logarithm(x[21]/S(p.ambient_temperature)))+
         S(p.tail_thermal_capacity)*(x[22]-S(p.ambient_temperature)-S(p.ambient_temperature)*logarithm(x[22]/S(p.ambient_temperature)));
}
template<class S> S stored_wake_energy(const State<S>& x,const Params& p) {
  return S(.5*p.main_inflow_mass)*x[17]*x[17]+S(.5*p.tail_inflow_mass)*x[18]*x[18];
}
template<class S> S terminal_entropy_commitment(const State<S>& x,const Params& p) {
  return (thermal_exergy(x,p)+stored_wake_energy(x,p))/S(p.ambient_temperature);
}
template<class S> State<S> integrate_rk4(const State<S>& initial,const Input<S>& u,
                                       const Vec3<S>& wind,const Params& p,double dt,int substeps) {
  State<S> x=initial;
  const double h=dt/substeps;
  for(int step=0;step<substeps;++step) {
    const auto k1=derivative(x,u,wind,p);
    State<S> work;
    for(int i=0;i<NX;++i) work[i]=x[i]+S(h*.5)*k1[i];
    const auto k2=derivative(work,u,wind,p);
    for(int i=0;i<NX;++i) work[i]=x[i]+S(h*.5)*k2[i];
    const auto k3=derivative(work,u,wind,p);
    for(int i=0;i<NX;++i) work[i]=x[i]+S(h)*k3[i];
    const auto k4=derivative(work,u,wind,p);
    for(int i=0;i<NX;++i) x[i]+=S(h/6)*(k1[i]+S(2)*k2[i]+S(2)*k3[i]+k4[i]);
    const S qnorm=detail::root(x[6]*x[6]+x[7]*x[7]+x[8]*x[8]+x[9]*x[9]+S(1e-24));
    for(int i=6;i<10;++i) x[i]/=qnorm;
  }
  return x;
}
} // namespace helicopter::rotor
