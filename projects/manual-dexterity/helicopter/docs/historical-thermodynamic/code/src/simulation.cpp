#include "helicopter.hpp"
#include "entropy_controller.hpp"
#include "rotor_model.hpp"
#include "flight_constraints.hpp"
#include "mass_estimator.hpp"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
  std::unique_ptr<EntropyController> optimizer;
  ControllerMode mode=ControllerMode::Entropy;
  unsigned seed=0,ticks=0; double mass_scale=1,manual_gust=-100;
  bool enabled=true,crashed=false,completed=false,touchdown=false;
  bool landing_support=false,pid_active_last=false;
  V3 wind{}; Json solution,optimizer_info,landing_handover;
  rotor::Forces<double> last{};
  double entropy=0,energy=0,wind_work=0,contact_work=0,heat=0,air_loss=0,initial_energy=0;
  double initial_commitment=0,ledger_residual=0,max_ledger_residual=0;
  double squared_error=0,max_error=0,max_tilt=0,max_altitude=.22,max_speed=0;
  double max_temperature=293.15,max_power=0,max_physical_violation=0;
  unsigned saturated=0,physical_violations=0;
  Impl(unsigned s,double m,bool a,ControllerMode c) { reset(s,m,a,c); }
  ~Impl(){if(data)mj_deleteData(data);if(model)mj_deleteModel(model);}
  bool finished() const {return crashed||data->time>=duration-1e-8;}
  void sync() {
    for(int i=0;i<3;++i){x[i]=data->qpos[i];x[3+i]=data->qvel[i];}
    for(int i=0;i<4;++i)x[6+i]=data->qpos[3+i];
    mjtNum spatial[6]{};mj_objectVelocity(model,data,mjOBJ_BODY,body,spatial,1);
    for(int i=0;i<3;++i)x[10+i]=spatial[i];
  }
  void reset(unsigned s,double m,bool a,ControllerMode c) {
    if(!std::isfinite(m)||m<.5||m>1.5) throw std::invalid_argument("mass_scale outside [0.5,1.5]");
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
    mj_forward(model,data);sync();seed=s;mass_scale=m;enabled=a;mode=c;
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
    optimizer_info={{"available",false},{"status",mode==ControllerMode::Entropy?"waiting_for_first_solve":"pid_baseline"},
                    {"accepted",false},{"fallback",false},{"solves",0},{"accepted_solves",0},{"fallback_solves",0},
                    {"control_decisions",0},{"landing_support_decisions",0},{"observation_hold_decisions",0},{"optimizer_attempted",false},
                    {"applied_controller",mode==ControllerMode::Entropy?"waiting":"pid_baseline"}};
    if(enabled&&mode==ControllerMode::Entropy) {
      if(!optimizer)optimizer=std::make_unique<EntropyController>(nominal);else optimizer->reset();
    }
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
    const auto prior=command;
    // The first measured interval preserves the explicitly prepared trim in
    // both enabled modes. Its flight time, energy, entropy and decision count
    // remain inside the mission; no true mass is passed to either controller.
    const bool observation_hold=enabled&&ticks<10;
    if(enabled&&mode==ControllerMode::Pid) {
      optimizer_info["applied_controller"]=observation_hold?"trim_observation_hold":"pid_baseline";
      optimizer_info["status"]=observation_hold?"observation_hold":"pid_baseline";
    }
    // Both controllers receive the same estimate based only on earlier observed
    // velocity changes and known modeled force. The evaluator's plant mass is
    // never supplied to this estimator or to either controller.
    auto estimated=nominal;
    estimated.mass*=mass_estimator.mass_scale();
    for(auto& inertia:estimated.inertia)inertia*=mass_estimator.mass_scale();
    if(enabled&&mode==ControllerMode::Entropy) {
      if(ticks%10==0) {
        if(!landing_support&&t>=58) {
          const auto rotation=rotor::detail::rotation(x);
          // Exact minimum of the two box skids in the MuJoCo geometry. Ground
          // support is outside the free-flight optimizer's model domain.
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
              {"reason","Free-flight predictor hands approaching ground contact to explicit PID landing support"}};
          }
        }
        EntropyResult result;
        if(observation_hold)result=optimizer->observation_hold(t);
        else if(landing_support)result=optimizer->landing_support(t);
        else {
          std::vector<rotor::Reference> refs;
          for(int k=0;k<=EntropyController::horizon_steps;++k)refs.push_back(reference(t+k*EntropyController::interval).values);
          result=optimizer->solve(x,command,wind,refs,t,mass_estimator.mass_scale());
        }
        optimizer_info=std::move(result.diagnostics);finite_or_null(optimizer_info);
        if(result.accepted)optimized_command=result.command;
      }
    }
    const bool use_pid=!enabled||mode==ControllerMode::Pid||optimizer_info.value("fallback",true);
    // The shadow controller must not carry integral error accumulated while
    // another controller was flying. Start every PID takeover from actual input.
    if(!observation_hold) {
      if(enabled&&mode==ControllerMode::Entropy&&use_pid&&!pid_active_last)baseline.reset(command);
      baseline_command=baseline.command(x,ref.values,wind,estimated,dt);
      pid_active_last=use_pid;
      command=!enabled?rotor::Input<double>{}:use_pid?baseline_command:optimized_command;
    }
    // NMPC targets are held for exactly the prediction interval. Their sampled
    // slew bound was checked in the optimizer; re-ramping targets here would
    // introduce unmodeled input dynamics. Physical actuator lag is in x[13:17].
    if(mode==ControllerMode::Pid||!enabled||optimizer_info.value("fallback",true))
      for(int j=0;j<4;++j)command[j]=std::clamp(command[j],
        std::max(plant.input_min[j],prior[j]-plant.input_rate_max[j]*dt),
        std::min(plant.input_max[j],prior[j]+plant.input_rate_max[j]*dt));
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
      {"touchdown",touchdown},{"controller_mode",mode==ControllerMode::Entropy?"entropy_nmpc":"pid_baseline"},
      {"controller",mode==ControllerMode::Entropy?"Airborne entropy NMPC with counted PID fallback and landing support":"PID baseline on the same thermodynamic plant"},
      {"landing_handover",landing_handover},
      {"initial_observation_hold",{{"active",enabled&&ticks<10},{"duration_s",.1},
        {"applied_controller",enabled&&ticks<10?"trim_observation_hold":"finished"}}},
      {"parameter_estimation",mass_estimator.metadata()},
      {"solution",solution},{"optimizer",optimizer_info},{"thermodynamics",thermo},
      {"metrics",{{"rms_error",ticks?std::sqrt(squared_error/ticks):0},{"max_error",max_error},{"endpoint_error",norm(subtract(ref.values.position,position))},
        {"saturation_fraction",ticks?double(saturated)/ticks:0},{"max_tilt_deg",max_tilt},{"max_altitude",max_altitude},{"max_speed",max_speed},{"final_speed",norm(velocity)},
        {"max_temperature_k",max_temperature},{"max_electrical_power_w",max_power},{"max_physical_constraint_violation",max_physical_violation},
        {"physical_constraint_violation_steps",physical_violations}}},
      {"mission",{{"name","Takeoff, hover, two waypoints, gust recovery, return and land"},{"duration",duration},{"step_seconds",dt}}}};
  }
};

Simulation::Simulation(unsigned s,double m,bool a,ControllerMode c):impl_(std::make_unique<Impl>(s,m,a,c)){}
Simulation::~Simulation()=default;
void Simulation::reset(unsigned s,double m,bool a,ControllerMode c){impl_->reset(s,m,a,c);}
void Simulation::step(){impl_->step();}
void Simulation::gust(){impl_->manual_gust=impl_->data->time;}
Json Simulation::state() const{return impl_->state();}
bool Simulation::finished() const{return impl_->finished();}
Json Simulation::model() {
  auto physical=rotor::model_metadata(rotor::Params{});
  return {{"schema_version",3},{"name","Thermodynamic electric reference helicopter"},{"simulator","MuJoCo plant with native shared rotor/thermal model"},
    {"simulator_version",mj_versionString()},{"provenance",{{"model_source_sha256",HELICOPTER_MODEL_SOURCE_SHA256},{"evaluator_source_sha256",HELICOPTER_EVALUATOR_SOURCE_SHA256}}},
    {"plant",physical},{"controller",EntropyController::description()},
    {"parameter_estimation",{{"method","Causal scalar least squares: F = mass * (velocity change / dt + gravity)"},
      {"source","Previous airborne velocity increments and rotor/drag force evaluated from observed state; no true plant mass input"},
      {"prior_mass_kg",2.0},{"mass_scale_range",{.5,1.5}},
      {"sharing","The same estimate is supplied to the PID baseline, PID fallback/landing support, and NMPC predictor/seeds"},
      {"inertia_assumption","All three principal inertias scale with estimated mass; this is an explicit scenario assumption, not independent inertia identification"},
      {"limitations","Exact simulated velocity and assumed force map; not a noisy sensor estimator or aerodynamic parameter identification"}}},
    {"initial_observation_hold",{{"duration_s",.1},{"input","Prepared initial trim held unchanged for ten 0.01 s steps in both enabled modes"},
      {"accounting","Included in mission time, energy, entropy, and all-decision NMPC participation; no hidden preflight observation interval"}}},
    {"landing_support",{{"controller","PID baseline with integral reset at takeover"},{"descent_from_s",58},
      {"clearance_margin_m",.15},{"closing_time_allowance_s",.5},
      {"guard","Latch when min(actual lowest-skid clearance, reference height minus 0.20 m) <= 0.15 m + 0.5 s * max(0,-vertical speed)"},
      {"participation","Every landing-support interval counts as PID in the full-mission NMPC participation criterion"}}},
    {"physics",{{"nominal_mass_kg",2.0},{"nominal_inertia_body_kg_m2",{.060,.095,.090}},{"timestep_s",dt},{"integrator","MuJoCo implicitfast + RK4 rotor/thermal states"}}},
    {"coordinates",{{"world","Right handed, z up; SI units"},{"quaternion","wxyz body-to-world"},{"input","Physical pitch radians; viewer also shows normalized channels"}}},
    {"telemetry",{{"timing","State at state.time; applied solution is preceding integration step; optimizer diagnostics timestamp the last 10 Hz control decision. Power/entropy in thermodynamics are instantaneous at the current state."}}},
    {"limitations",{"Explicitly parameterized simulated vehicle; no measured calibration or hardware validation.",
       "Near-hover governed-speed rotors; no detailed stall, ground effect, governor transient or battery electrochemistry.",
       "Privileged simulator state and current-wind persistence forecast; no state estimator.",
       "NMPC optimizes airborne free flight. Ground approach/contact uses explicit PID landing support; this is not contact-optimal control. Solver rejection also causes counted PID fallback.",
       "Finite-budget local optimization; feasibility, convergence, and demonstrated savings are separate claims."}}};
}
}
