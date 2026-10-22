#include "helicopter.hpp"
#include "hinf_controller.hpp"
#include "rotor_model.hpp"
#include "flight_constraints.hpp"
#include "mass_estimator.hpp"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <stdexcept>
#include <string>

namespace helicopter {
namespace {
using Json=nlohmann::json; using V3=rotor::Vec3<double>;
using rotor::detail::add; using rotor::detail::subtract; using rotor::detail::scale;
using rotor::detail::dot; using rotor::detail::rotate;
constexpr double dt=.01,duration=70,pi=3.14159265358979323846;
double norm(const V3& v) { return std::sqrt(dot(v,v)); }
struct Reference { rotor::Reference values; std::string phase; };
Reference interpolate(double t,double begin,double end,V3 from,V3 to,const std::string& phase) {
  const double length=end-begin,u=std::clamp((t-begin)/length,0.0,1.0);
  const double s=u*u*u*(10+u*(-15+6*u)),ds=30*u*u*(1-u)*(1-u)/length;
  const double dds=60*u*(1-u)*(1-2*u)/(length*length); auto delta=subtract(to,from);
  return {{add(from,scale(delta,s)),scale(delta,ds),scale(delta,dds)},phase};
}
Reference reference(double t) {
  if(t<8) return interpolate(t,0,8,{0,0,.22},{0,0,2.5},"takeoff");
  if(t<14) return {{{0,0,2.5},{},{}},"hover"};
  if(t<24) return interpolate(t,14,24,{0,0,2.5},{3,0,2.8},"waypoint 1");
  if(t<31) return {{{3,0,2.8},{},{}},"hover at waypoint 1"};
  if(t<41) return interpolate(t,31,41,{3,0,2.8},{3,3,2.8},"waypoint 2 / wind recovery");
  if(t<46) return {{{3,3,2.8},{},{}},"hover at waypoint 2"};
  if(t<58) return interpolate(t,46,58,{3,3,2.8},{0,0,2.5},"return home");
  if(t<68) return interpolate(t,58,68,{0,0,2.5},{0,0,.2},"landing");
  return {{{0,0,.2},{},{}},"touchdown"};
}
const char* xml=R"mjcf(
<mujoco model="electric reference helicopter">
 <compiler angle="radian"/>
 <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
 <default><geom friction="0.9 0.02 0.01" condim="3"/></default>
 <worldbody><geom name="ground" type="plane" size="30 30 .1"/>
 <body name="helicopter" pos="0 0 .22"><freejoint/>
 <inertial pos="0 0 0" mass="2" diaginertia=".060 .095 .090"/>
 <geom name="fuselage" type="ellipsoid" size=".32 .13 .14" contype="0" conaffinity="0"/>
 <geom name="left skid" type="box" pos="0 .16 -.175" size=".28 .025 .025"/>
 <geom name="right skid" type="box" pos="0 -.16 -.175" size=".28 .025 .025"/>
 </body></worldbody>
</mujoco>)mjcf";
double mechanical_energy(const rotor::State<double>& x,const rotor::Params& p) {
  double energy=p.mass*p.gravity*x[2];
  for(int i=0;i<3;++i) energy+=.5*p.mass*x[3+i]*x[3+i]+.5*p.inertia[i]*x[10+i]*x[10+i];
  return energy;
}
Json channels(const rotor::Input<double>& input) {
  return {{"collective",input[0]},{"cyclic_long",input[1]},{"cyclic_lat",input[2]},{"tail",input[3]}};
}
void finite_or_null(Json& j) {
  if(j.is_number_float()&&!std::isfinite(j.get<double>())) j=nullptr;
  else if(j.is_structured()) for(auto& item:j) finite_or_null(item);
}
}

struct Simulation::Impl {
  mjModel* model=nullptr; mjData* data=nullptr; int body=1;
  rotor::Params plant,nominal;
  rotor::State<double> x{};
  rotor::Input<double> command{},baseline_command{},optimized_command{};
  rotor::BaselineController baseline;
  MassEstimator mass_estimator;
  std::unique_ptr<HinfController> controller;
  ControllerMode mode=ControllerMode::Hinf;
  unsigned seed=0,ticks=0; double mass_scale=1,manual_gust=-100,noise_scale=1;
  bool enabled=true,crashed=false,completed=false,touchdown=false;
  bool landing_support=false,pid_active_last=false;
  V3 wind{}; Json solution,hinf_info,landing_handover;
  rotor::State<double> measured{},measurement_truth{};
  unsigned applied_decisions=0,landing_decisions=0,hold_decisions=0,fallback_decisions=0,saturation_steps=0;
  double update_total_ms=0,update_max_ms=0; unsigned deadline_misses=0;
  rotor::Forces<double> last{};
  double entropy=0,energy=0,wind_work=0,contact_work=0,heat=0,air_loss=0,initial_energy=0;
  double initial_commitment=0,ledger_residual=0,max_ledger_residual=0;
  double squared_error=0,max_error=0,max_tilt=0,max_altitude=.22,max_speed=0;
  double max_temperature=293.15,max_power=0,max_physical_violation=0;
  unsigned saturated=0,physical_violations=0;
  Impl(unsigned s,double m,bool a,ControllerMode c,double n) { reset(s,m,a,c,n); }
  ~Impl(){if(data)mj_deleteData(data);if(model)mj_deleteModel(model);}
  bool finished() const {return crashed||data->time>=duration-1e-8;}
  void sync() {
    for(int i=0;i<3;++i){x[i]=data->qpos[i];x[3+i]=data->qvel[i];}
    for(int i=0;i<4;++i)x[6+i]=data->qpos[3+i];
    mjtNum spatial[6]{};mj_objectVelocity(model,data,mjOBJ_BODY,body,spatial,1);
    for(int i=0;i<3;++i)x[10+i]=spatial[i];
  }
  void reset(unsigned s,double m,bool a,ControllerMode c,double n) {
    if(!std::isfinite(m)||m<.5||m>1.5) throw std::invalid_argument("mass_scale outside [0.5,1.5]");
    if(!std::isfinite(n)||n<0||n>10) throw std::invalid_argument("sensor_noise_scale outside [0,10]");
    if(data){mj_deleteData(data);data=nullptr;}if(model){mj_deleteModel(model);model=nullptr;}
    mjVFS vfs;mj_defaultVFS(&vfs);mj_addBufferVFS(&vfs,"reference.xml",xml,std::strlen(xml));
    char error[1024]{};model=mj_loadXML("reference.xml",&vfs,error,sizeof(error));mj_deleteVFS(&vfs);
    if(!model)throw std::runtime_error(error);
    plant=nominal;plant.mass*=m;for(auto& inertia:plant.inertia)inertia*=m;
    body=mj_name2id(model,mjOBJ_BODY,"helicopter");model->body_mass[body]=plant.mass;
    for(int i=0;i<3;++i)model->body_inertia[3*body+i]=plant.inertia[i];
    data=mj_makeData(model);if(!data)throw std::runtime_error("MuJoCo state allocation failed");
    mj_setConst(model,data);mj_resetData(model,data);mj_forward(model,data);
    x=rotor::initial_state(plant);
    for(int i=0;i<3;++i)data->qpos[i]=x[i];
    for(int i=0;i<4;++i)data->qpos[3+i]=x[6+i];
    mj_forward(model,data);sync();seed=s;mass_scale=m;enabled=a;mode=c;noise_scale=n;
    ticks=0;manual_gust=-100;crashed=completed=touchdown=false;
    landing_support=false;pid_active_last=mode==ControllerMode::Pid;
    landing_handover={{"active",false},{"time",nullptr},{"settled_at_handover",nullptr}};
    for(int i=0;i<4;++i)command[i]=x[13+i];
    baseline_command=optimized_command=command;baseline.reset(command);mass_estimator.reset();
    entropy=energy=wind_work=contact_work=heat=air_loss=ledger_residual=max_ledger_residual=0;
    squared_error=max_error=max_tilt=max_speed=max_power=max_physical_violation=0;
    max_altitude=.22;max_temperature=plant.ambient_temperature;saturated=physical_violations=0;
    wind={};last=rotor::evaluate(x,command,wind,plant);
    initial_energy=mechanical_energy(x,plant)+last.stored_thermal_energy+last.stored_wake_energy;
    initial_commitment=rotor::terminal_entropy_commitment(x,plant);
    solution={{"available",false},{"active",false},{"time",0},{"actual_thrust_n",0},{"commanded_thrust_n",0}};
    applied_decisions=landing_decisions=hold_decisions=fallback_decisions=saturation_steps=deadline_misses=0;
    update_total_ms=update_max_ms=0;measured=measurement_truth=x;
    hinf_info={{"available",false},{"status","initialized"},{"applied_controller",enabled?"trim_observation_hold":"disabled"},
      {"control_decisions",0},{"applied_decisions",0},{"landing_support_decisions",0},{"observation_hold_decisions",0},
      {"fallback_decisions",0},{"saturation_steps",0},{"update_ms",nullptr},{"control_interval_ms",10.0}};
    if(enabled&&mode==ControllerMode::Hinf) {
      if(!controller)controller=std::make_unique<HinfController>(nominal);else controller->reset();
      controller->reset(command);
    }
  }
  double noise(unsigned channel) const {
    // Stateless integer hash: paired modes see the same sensor sequence at a
    // given seed/tick. No platform-dependent standard-library RNG distribution.
    std::uint32_t v=seed*747796405u+ticks*2891336453u+(channel+1)*277803737u;
    v^=v>>16;v*=2246822519u;v^=v>>13;v*=3266489917u;v^=v>>16;
    return noise_scale*std::sqrt(3.0)*(2.0*double(v)/4294967295.0-1.0);
  }
  rotor::State<double> observe() const {
    auto result=x;
    for(int j=0;j<3;++j){result[j]+=.005*noise(j);result[3+j]+=.02*noise(3+j);result[10+j]+=.01*noise(9+j);}
    const double ax=.002*noise(6),ay=.002*noise(7),az=.002*noise(8);
    const double a=std::sqrt(ax*ax+ay*ay+az*az),s=a>1e-12?std::sin(a*.5)/a:.5;
    const double w=std::cos(a*.5),qx=ax*s,qy=ay*s,qz=az*s;
    result[6]=x[6]*w-x[7]*qx-x[8]*qy-x[9]*qz;
    result[7]=x[6]*qx+x[7]*w+x[8]*qz-x[9]*qy;
    result[8]=x[6]*qy-x[7]*qz+x[8]*w+x[9]*qx;
    result[9]=x[6]*qz+x[7]*qy-x[8]*qx+x[9]*w;
    // Extra rotor sensors are used by the model-based PID inversion. The H∞
    // output-feedback controller uses only the first twelve kinematic channels.
    for(int j=13;j<21;++j)result[j]+=(j==17||j==18?.02:.0002)*noise(12+j-13);
    return result;
  }
  V3 get_wind(double t) const {
    const double phase=(seed%997)*.041;
    V3 result{.35*std::sin(t*.31+phase),.25*std::cos(t*.23+phase),.03*std::sin(t*.7)};
    for(double start:{36.0,manual_gust}) {
      const double u=(t-start)/5;
      if(u>=0&&u<1)result=add(result,scale(V3{3.5,-2.5,.25},std::pow(std::sin(pi*u),2)));
    }
    return result;
  }
  void step() {
    if(finished())return;
    const double t=data->time;const auto ref=reference(t);wind=get_wind(t);
    const auto prior=command;measurement_truth=x;measured=observe();
    const bool observation_hold=enabled&&ticks<10;
    if(enabled&&mode==ControllerMode::Hinf&&!landing_support&&t>=58) {
      const auto rotation=rotor::detail::rotation(x);
      const double clearance=x[2]-.175*rotation[8]-.025*std::abs(rotation[8])-
        .28*std::abs(rotation[6])-.185*std::abs(rotation[7]);
      const double reference_clearance=ref.values.position[2]-.2;
      const double margin=.15+.5*std::max(0.0,-x[5]);
      if(std::min(clearance,reference_clearance)<=margin) {
        const double tilt=std::acos(std::clamp(rotation[8],-1.0,1.0));
        bool settled=tilt<=5*pi/180;
        for(int j=0;j<2;++j)settled=settled&&std::abs(x[j]-ref.values.position[j])<=.05&&std::abs(x[3+j])<=.05;
        for(int j=0;j<3;++j)settled=settled&&std::abs(x[10+j])<=.1;
        landing_support=true;
        landing_handover={{"active",true},{"time",t},{"settled_at_handover",settled},
          {"skid_clearance_m",clearance},{"reference_clearance_m",reference_clearance},{"guard_margin_m",margin},
          {"position",{x[0],x[1],x[2]}},{"velocity",{x[3],x[4],x[5]}},{"tilt_deg",tilt*180/pi},
          {"body_rates",{x[10],x[11],x[12]}},
          {"reason","The hover-linearized H-infinity guarantee excludes ground contact; explicit PID landing support"}};
      }
    }
    bool use_pid=enabled&&(mode==ControllerMode::Pid||landing_support);
    std::string applied=!enabled?"disabled":observation_hold?"trim_observation_hold":use_pid?
      (mode==ControllerMode::Pid?"pid_baseline":"pid_landing_support"):"minimum_entropy_hinf";
    hinf_info={{"available",false},{"status",applied},{"applied_controller",applied},
      {"update_ms",nullptr},{"control_interval_ms",10.0},{"saturated",false}};
    if(observation_hold)++hold_decisions;
    else if(enabled&&mode==ControllerMode::Hinf&&!landing_support) {
      const auto started=std::chrono::steady_clock::now();
      try {
        auto result=controller->update(measured,ref.values,dt);
        command=result.command;hinf_info=std::move(result.diagnostics);
        for(double u:command)if(!std::isfinite(u))throw std::runtime_error("Nonfinite H-infinity command");
        ++applied_decisions;
      } catch(const std::exception& error) {
        use_pid=true;applied="pid_fallback";++fallback_decisions;
        hinf_info={{"available",false},{"status","controller_failure"},{"reason",error.what()}};
      }
      const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      hinf_info["update_ms"]=elapsed;update_total_ms+=elapsed;update_max_ms=std::max(update_max_ms,elapsed);
      if(elapsed>dt*1000)++deadline_misses;
    }
    if(!observation_hold) {
      if(use_pid) {
        if(!pid_active_last)baseline.reset(command);
        // Both controllers use the same noisy kinematic observations. No true
        // wind or true mass is provided to either feedback law.
        baseline_command=baseline.command(measured,ref.values,V3{},nominal,dt);
        command=baseline_command;
        hinf_info["saturated"]=baseline.limited();
        if(mode==ControllerMode::Hinf&&landing_support)++landing_decisions;
      }
      if(!enabled)command={};
      pid_active_last=use_pid;
    }
    bool limited=hinf_info.value("saturated",false);
    for(int j=0;j<4;++j) {
      const double requested=command[j];
      command[j]=std::clamp(requested,std::max(plant.input_min[j],prior[j]-plant.input_rate_max[j]*dt),
        std::min(plant.input_max[j],prior[j]+plant.input_rate_max[j]*dt));
      limited=limited||std::abs(command[j]-requested)>1e-12;
    }
    if(limited)++saturation_steps;
    hinf_info.update({{"applied_controller",applied},{"time",t},{"control_interval_ms",10.0},
      {"applied_command",command},{"saturated",limited},{"control_decisions",ticks+1},
      {"applied_decisions",applied_decisions},{"landing_support_decisions",landing_decisions},
      {"observation_hold_decisions",hold_decisions},{"fallback_decisions",fallback_decisions},
      {"saturation_steps",saturation_steps},{"deadline_misses",deadline_misses},
      {"update_mean_ms",applied_decisions+fallback_decisions?update_total_ms/(applied_decisions+fallback_decisions):0},
      {"update_max_ms",update_max_ms}});
    finite_or_null(hinf_info);
    last=rotor::evaluate(x,command,wind,plant);
    const auto r=rotor::detail::rotation(x);
    const auto force=add(rotate(r,last.force_body),last.drag_world);
    const auto torque=rotate(r,add(last.torque_body,last.damping_body));
    // The observation model is evaluated with nominal constitutive parameters,
    // separately from the actual plant. Rotor/drag force does not depend on mass.
    const auto observed_load=rotor::evaluate(x,command,wind,nominal);
    const auto observed_force=add(rotate(r,observed_load.force_body),observed_load.drag_world);
    const V3 observed_velocity_before{x[3],x[4],x[5]};
    const bool contact_before=data->ncon>0;
    auto commanded=x;for(int j=0;j<4;++j)commanded[13+j]=command[j];
    const auto cf=rotor::evaluate(commanded,command,wind,plant);
    const V3 position{x[0],x[1],x[2]},velocity{x[3],x[4],x[5]};
    solution={{"available",true},{"active",enabled},{"time",t},
      {"position_error",subtract(ref.values.position,position)},{"velocity_error",subtract(ref.values.velocity,velocity)},
      {"position_integral",baseline.integral()},{"desired_acceleration",nullptr},{"desired_force_world",nullptr},
      {"attitude_error_body",nullptr},{"desired_torque_body",nullptr},{"projected_thrust_n",nullptr},
      {"commanded_thrust_n",cf.main_thrust},{"actual_thrust_n",last.main_thrust},
      {"force_body",last.force_body},{"force_world",force},{"torque_body",last.torque_body},{"torque_world",torque},
      {"drag_force_world",last.drag_world},{"damping_torque_world",rotate(r,last.damping_body)}};
    auto next=rotor::integrate_rk4(x,command,wind,plant,dt,2);
    std::array<double,6> previous_velocity{};for(int j=0;j<6;++j)previous_velocity[j]=data->qvel[j];
    mju_zero(data->qfrc_applied,model->nv);
    mj_applyFT(model,data,force.data(),torque.data(),data->xipos+3*body,body,data->qfrc_applied);
    mj_step(model,data);
    double contact_power=0;
    for(int j=0;j<6;++j)contact_power+=data->qfrc_constraint[j]*(previous_velocity[j]+data->qvel[j])*.5;
    mj_forward(model,data);
    for(int j=13;j<rotor::NX;++j)x[j]=next[j];
    sync();
    mass_estimator.observe(observed_force,observed_velocity_before,V3{x[3],x[4],x[5]},dt,
                           contact_before||data->ncon>0);
    entropy+=last.entropy_rate*dt;energy+=last.electrical_power*dt;wind_work+=last.wind_power*dt;
    contact_work+=contact_power*dt;heat+=(last.main_heat_out+last.tail_heat_out)*dt;
    air_loss+=(last.wake_dissipation+last.drag_dissipation+last.angular_dissipation)*dt;
    const auto now=rotor::evaluate(x,command,wind,plant);
    const double stored=mechanical_energy(x,plant)+now.stored_thermal_energy+now.stored_wake_energy;
    ledger_residual=energy+wind_work+contact_work-(stored-initial_energy)-heat-air_loss;
    max_ledger_residual=std::max(max_ledger_residual,std::abs(ledger_residual));
    const auto current=reference(data->time);const V3 pos{x[0],x[1],x[2]},vel{x[3],x[4],x[5]};
    const double error=norm(subtract(current.values.position,pos));
    const auto rotation=rotor::detail::rotation(x);
    const double tilt=std::acos(std::clamp(rotation[8],-1.0,1.0))*180/pi;
    squared_error+=error*error;max_error=std::max(max_error,error);max_tilt=std::max(max_tilt,tilt);
    max_altitude=std::max(max_altitude,x[2]);max_speed=std::max(max_speed,norm(vel));++ticks;
    const auto normalized=rotor::normalized_input(command,plant);
    if(normalized[0]>.995||std::abs(normalized[1])>.995||std::abs(normalized[2])>.995||normalized[3]>.995)++saturated;
    max_temperature=std::max({max_temperature,x[21],x[22]});max_power=std::max(max_power,last.electrical_power);
    const double violation=rotor::physical_violation(x,command,wind,plant,data->ncon==0).maximum;
    max_physical_violation=std::max(max_physical_violation,violation);if(violation>1e-4)++physical_violations;
    crashed=!std::isfinite(error)||!std::isfinite(tilt)||!std::isfinite(last.electrical_power)||norm(pos)>50||x[2]<-.1||(tilt>75&&x[2]<.6);
    touchdown=data->time>65&&data->ncon>0&&x[2]<.25&&norm(vel)<.25;
    if(data->time>=duration-1e-8)completed=enabled&&!crashed&&touchdown&&error<.25&&norm(vel)<.25;
  }
  Json state() const {
    const auto ref=reference(data->time);const auto r=rotor::detail::rotation(x);
    const V3 position{x[0],x[1],x[2]},velocity{x[3],x[4],x[5]};
    const V3 attitude{std::atan2(r[7],r[8])*180/pi,std::asin(std::clamp(-r[6],-1.0,1.0))*180/pi,std::atan2(r[3],r[0])*180/pi};
    const rotor::Input<double> actual{x[13],x[14],x[15],x[16]};
    const auto f=rotor::evaluate(x,command,wind,plant);
    const double commitment=rotor::terminal_entropy_commitment(x,plant);
    Json thermo={{"model","Uncalibrated simulated electric helicopter; explicit ambient wake boundary"},
      {"entropy_rate_w_per_k",f.entropy_rate},{"entropy_motor_w_per_k",f.entropy_motor},
      {"entropy_heat_transfer_w_per_k",f.entropy_heat_transfer},{"entropy_wake_w_per_k",f.entropy_wake},{"entropy_drag_w_per_k",f.entropy_drag},
      {"cumulative_entropy_j_per_k",entropy},{"electrical_power_w",f.electrical_power},{"electrical_energy_j",energy},
      {"motor_temperature_k",{x[21],x[22]}},{"motor_current_a",{f.main_current,f.tail_current}},
      {"air_dissipation_w",f.wake_dissipation+f.drag_dissipation+f.angular_dissipation},
      {"motor_loss_w",f.main_motor_loss+f.tail_motor_loss},{"heat_rejection_w",f.main_heat_out+f.tail_heat_out},
      {"stored_thermal_energy_j",f.stored_thermal_energy},{"stored_wake_energy_j",f.stored_wake_energy},
      {"terminal_cooling_entropy_j_per_k",rotor::thermal_exergy(x,plant)/plant.ambient_temperature},
      {"terminal_entropy_commitment_j_per_k",commitment},{"entropy_with_terminal_commitment_j_per_k",entropy+commitment-initial_commitment},
      {"energy_balance_residual_w",f.first_law_residual},{"integrated_energy_residual_j",ledger_residual},
      {"max_integrated_energy_residual_j",max_ledger_residual},{"wind_work_j",wind_work},{"contact_work_j",contact_work},
      {"mechanical_energy_j",mechanical_energy(x,plant)},{"contact_active",data->ncon>0}};
    return {{"time",data->time},{"completed",completed},{"crashed",crashed},{"phase",crashed?"crashed":completed?"landed":ref.phase},
      {"position",position},{"velocity",velocity},{"quaternion",{x[6],x[7],x[8],x[9]}},{"attitude_deg",attitude},{"body_rates",{x[10],x[11],x[12]}},
      {"target",ref.values.position},{"reference",{{"position",ref.values.position},{"velocity",ref.values.velocity},{"acceleration",ref.values.acceleration}}},
      {"controls",channels(rotor::normalized_input(command,plant))},{"actuators",channels(rotor::normalized_input(actual,plant))},
      {"input_rad",command},{"actuator_pitch_rad",actual},{"inflow_m_s",{x[17],x[18]}},{"flap_rad",{x[19],x[20]}},
      {"model_state",x},{"wind",wind},{"autopilot",enabled},{"seed",seed},{"mass_scale",mass_scale},{"mass_kg",plant.mass},{"inertia_body_kg_m2",plant.inertia},
      {"touchdown",touchdown},{"controller_mode",mode==ControllerMode::Hinf?"minimum_entropy_hinf":"pid_baseline"},
      {"controller",mode==ControllerMode::Hinf?"Central minimum-entropy H-infinity output feedback with explicit PID landing support":"PID baseline with matched noisy measurements and nominal parameters"},
      {"landing_handover",landing_handover},
      {"initial_observation_hold",{{"active",enabled&&ticks<10},{"duration_s",.1},
        {"applied_controller",enabled&&ticks<10?"trim_observation_hold":"finished"}}},
      {"parameter_estimation",mass_estimator.metadata()},
      {"solution",solution},{"hinf",hinf_info},{"thermodynamics",thermo},{"sensor_noise_scale",noise_scale},{"measured_state",measured},
      {"measurement_truth",measurement_truth},{"measurement_time",ticks?(ticks-1)*dt:0},
      {"metrics",{{"rms_error",ticks?std::sqrt(squared_error/ticks):0},{"max_error",max_error},{"endpoint_error",norm(subtract(ref.values.position,position))},
        {"saturation_fraction",ticks?double(saturated)/ticks:0},{"max_tilt_deg",max_tilt},{"max_altitude",max_altitude},{"max_speed",max_speed},{"final_speed",norm(velocity)},
        {"max_temperature_k",max_temperature},{"max_electrical_power_w",max_power},{"max_physical_constraint_violation",max_physical_violation},
        {"physical_constraint_violation_steps",physical_violations}}},
      {"mission",{{"name","Takeoff, hover, two waypoints, gust recovery, return and land"},{"duration",duration},{"step_seconds",dt}}}};
  }
};

Simulation::Simulation(unsigned s,double m,bool a,ControllerMode c,double n):impl_(std::make_unique<Impl>(s,m,a,c,n)){}
Simulation::~Simulation()=default;
void Simulation::reset(unsigned s,double m,bool a,ControllerMode c,double n){impl_->reset(s,m,a,c,n);}
void Simulation::step(){impl_->step();}
void Simulation::gust(){impl_->manual_gust=impl_->data->time;}
Json Simulation::state() const{return impl_->state();}
bool Simulation::finished() const{return impl_->finished();}
Json Simulation::model() {
  return {{"schema_version",4},{"name","Minimum-entropy H-infinity helicopter laboratory"},
    {"simulator","MuJoCo nonlinear plant with native rotor/electrical/thermal dynamics"},
    {"simulator_version",mj_versionString()},
    {"provenance",{{"model_source_sha256",HELICOPTER_MODEL_SOURCE_SHA256},{"evaluator_source_sha256",HELICOPTER_EVALUATOR_SOURCE_SHA256}}},
    {"plant",rotor::model_metadata(rotor::Params{})},{"controller",HinfController::description()},
    {"hinf",HinfController::design()},
    {"sensors",{{"method","Deterministic bounded independent-channel simulated measurement noise, shared seed and step sequence in paired runs"},
      {"rms_position_m",.005},{"rms_velocity_m_s",.02},{"rms_attitude_rad",.002},{"rms_body_rate_rad_s",.01},
      {"hinf_channels","Position, velocity, attitude-log, and body rate only; no actual mass or wind input"},
      {"pid_extra_channels","Noisy rotor pitch, inflow and flap sensors for the existing model-based PID inversion"}}},
    {"parameter_estimation",{{"role","Diagnostic only; neither controller receives this estimate"},
      {"method","Causal scalar least squares using simulated forces and velocity increments"},
      {"limitations","Uses exact simulated observations; not a validated sensor estimator"}}},
    {"initial_observation_hold",{{"duration_s",.1},
      {"input","Prepared initial plant trim held for ten 0.01 s steps in both enabled modes"},
      {"accounting","Included in all 7000 mission decisions; not counted as H-infinity feedback"}}},
    {"landing_support",{{"controller","PID baseline with integral reset at takeover"},{"descent_from_s",58},
      {"clearance_margin_m",.15},{"closing_time_allowance_s",.5},
      {"guard","Latch when min(actual lowest-skid clearance, reference height minus 0.20 m) <= 0.15 m + 0.5 s * max(0,-vertical speed)"},
      {"participation","Every landing-support interval is counted as PID in the full-mission controller participation criterion"}}},
    {"physics",{{"nominal_mass_kg",2.0},{"nominal_inertia_body_kg_m2",{.060,.095,.090}},
      {"timestep_s",dt},{"integrator","MuJoCo implicitfast + RK4 rotor/thermal states"}}},
    {"coordinates",{{"world","Right handed, z up; SI units"},{"quaternion","wxyz body-to-world"},
      {"input","Physical pitch radians; viewer also shows normalized channels"}}},
    {"telemetry",{{"timing","Physical state is at state.time; measured_state, solution and hinf describe the preceding 100 Hz decision. Thermal measurements are instantaneous at current physical state."}}},
    {"limitations",{"Explicitly parameterized simulated vehicle; no measured calibration or hardware validation.",
      "Minimum entropy means the normalized closed-loop log-determinant H-infinity functional, not thermodynamic entropy in J/K.",
      "Linear certificates apply to the disclosed nominal hover generalized plant and unconstrained controller only.",
      "Reference feedforward, command/slew clipping, nonlinear motion and contact are outside that certificate.",
      "PID handles ground approach/contact; any failure fallback and every support interval are counted explicitly.",
      "Near-hover governed-speed rotors omit stall, detailed ground effect, governor transients and battery electrochemistry.",
      "Mass changes and wind trials are empirical stress tests, not a certified structured uncertainty set."}}};
}
}
