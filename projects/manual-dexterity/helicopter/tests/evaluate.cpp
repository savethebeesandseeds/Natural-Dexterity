#include "helicopter.hpp"
#include "hinf_controller.hpp"
#include "rotor_model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Json=nlohmann::json;
namespace rotor=helicopter::rotor;
using State=rotor::State<double>;
using Input=rotor::Input<double>;
using Vec3=rotor::Vec3<double>;
using Mode=helicopter::ControllerMode;
constexpr double dt=.01, constraint_tolerance=1e-4;
struct Scenario {std::string id,name;unsigned seed;double mass_scale;bool autopilot;std::vector<double> extra_gusts;double noise_scale=1;};
const std::vector<Scenario> scenarios{
 {"nominal","nominal",0,1,true,{}},
 {"alternate-wind","alternate wind phase",71,1,true,{}},
 {"extra-gusts","extra gusts",19,1,true,{18,44}},
 {"mass-minus-20","mass minus 20 percent",3,.8,true,{}},
 {"mass-plus-20","mass plus 20 percent with gust",31,1.2,true,{20}},
 {"noise-times-three","threefold sensor noise",11,1,true,{},3},
 {"noise-and-gust","threefold noise with gusts",19,1,true,{18,44},3},
 {"disabled","controller disabled negative control",0,1,false,{}}
};
bool finite_json(const Json& value) {
 if(value.is_number())return std::isfinite(value.get<double>());
 if(value.is_structured())for(const auto& child:value)if(!finite_json(child))return false;
 return true;
}
double dot(const Vec3& a,const Vec3& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(const Vec3& a){return std::sqrt(dot(a,a));}
double mechanical_energy(const State& x,const rotor::Params& p) {
    double value=p.mass*p.gravity*x[2];
    for(int i=0;i<3;++i)value+=.5*p.mass*x[i+3]*x[i+3]+.5*p.inertia[i]*x[i+10]*x[i+10];
    return value;
}
double wake_energy(const State& x,const rotor::Params& p){return .5*p.main_inflow_mass*x[17]*x[17]+.5*p.tail_inflow_mass*x[18]*x[18];}
double thermal_energy(const State& x,const rotor::Params& p) {
    return p.main_thermal_capacity*(x[21]-p.ambient_temperature)+p.tail_thermal_capacity*(x[22]-p.ambient_temperature);
}
double stored_energy(const State& x,const rotor::Params& p){return mechanical_energy(x,p)+wake_energy(x,p)+thermal_energy(x,p);}
double completion_entropy(const State& x,const rotor::Params& p) {
    const double t0=p.ambient_temperature;
    return (wake_energy(x,p)+p.main_thermal_capacity*(x[21]-t0-t0*std::log(x[21]/t0))+
            p.tail_thermal_capacity*(x[22]-t0-t0*std::log(x[22]/t0)))/t0;
}

struct LedgerAudit {
 State initial;rotor::Params p;double electrical=0,generated=0,heat=0,air=0,wind_work=0;
 double max_energy=0,max_entropy=0,max_motor=0,max_instant=0,min_entropy=0;
 double energy_residual=0,entropy_residual=0;std::array<double,2> motor_loss{},motor_heat{};
 bool reported=true;
 void inspect(const State& before,const State& next,const Input& u,const Vec3& wind,const Json& state) {
  const auto f=rotor::evaluate(before,u,wind,p);const auto dx=rotor::derivative(before,u,wind,p);
  electrical+=f.electrical_power*dt;generated+=f.entropy_rate*dt;wind_work+=f.wind_power*dt;
  heat+=(f.main_heat_out+f.tail_heat_out)*dt;air+=(f.wake_dissipation+f.drag_dissipation+f.angular_dissipation)*dt;
  motor_loss[0]+=f.main_motor_loss*dt;motor_loss[1]+=f.tail_motor_loss*dt;
  motor_heat[0]+=f.main_heat_out*dt;motor_heat[1]+=f.tail_heat_out*dt;
  for(int j=0;j<2;++j){const double capacity=j?p.tail_thermal_capacity:p.main_thermal_capacity;
   max_motor=std::max(max_motor,std::max(0.0,std::abs(motor_loss[j]-motor_heat[j]-capacity*(next[21+j]-initial[21+j]))-.5)/std::max(1.0,motor_loss[j]));}
  double derivative_energy=p.mass*p.gravity*dx[2];
  for(int j=0;j<3;++j)derivative_energy+=p.mass*before[3+j]*dx[3+j]+p.inertia[j]*before[10+j]*dx[10+j];
  derivative_energy+=p.main_inflow_mass*before[17]*dx[17]+p.tail_inflow_mass*before[18]*dx[18]+p.main_thermal_capacity*dx[21]+p.tail_thermal_capacity*dx[22];
  max_instant=std::max(max_instant,std::abs(f.electrical_power+f.wind_power-derivative_energy-f.wake_dissipation-f.drag_dissipation-f.angular_dissipation-f.main_heat_out-f.tail_heat_out));
  min_entropy=std::min(min_entropy,f.entropy_rate);
  const auto& t=state.at("thermodynamics");
  energy_residual=electrical+wind_work+t.at("contact_work_j").get<double>()-(stored_energy(next,p)-stored_energy(initial,p))-heat-air;
  const double entropy_storage=p.main_thermal_capacity*std::log(next[21]/initial[21])+p.tail_thermal_capacity*std::log(next[22]/initial[22]);
  entropy_residual=generated-entropy_storage-(heat+air)/p.ambient_temperature;
  max_energy=std::max(max_energy,std::max(0.0,std::abs(energy_residual)-.5)/std::max(1.0,electrical));
  max_entropy=std::max(max_entropy,std::max(0.0,std::abs(entropy_residual)-.002)/std::max(1.0,generated));
  reported=reported&&std::abs(t.at("electrical_energy_j").get<double>()-electrical)<1e-6&&
   std::abs(t.at("cumulative_entropy_j_per_k").get<double>()-generated)<1e-8&&
   std::abs(t.at("integrated_energy_residual_j").get<double>()-energy_residual)<1e-6&&
   std::abs(t.at("entropy_with_terminal_commitment_j_per_k").get<double>()-generated-completion_entropy(next,p)+completion_entropy(initial,p))<1e-7;
 }
 bool passed()const{return reported&&max_energy<=.002&&max_entropy<=.002&&max_motor<=.002&&max_instant<1e-6&&min_entropy>=-1e-10;}
 Json report()const{return {{"passed",passed()},{"reported_integrals_verified",reported},{"max_normalized_energy_residual",max_energy},
  {"max_normalized_entropy_residual",max_entropy},{"max_normalized_motor_residual",max_motor},{"max_instant_energy_residual_w",max_instant},
  {"energy_residual_j",energy_residual},{"entropy_residual_j_per_k",entropy_residual},{"minimum_entropy_rate_w_per_k",min_entropy},
  {"relative_tolerance",.002},{"absolute_energy_allowance_j",.5},{"absolute_entropy_allowance_j_per_k",.002}};}
};
struct ConstraintAudit {
    std::map<std::string,double> maximum;
    std::map<std::string,double> maximum_time;
    unsigned steps=0;
    void inspect(const State& x,const Input& u,const Vec3& wind,const rotor::Forces<double>& f,
                 const Json& sample,const rotor::Params& p,bool controlled) {
        double worst=0;
        const auto record=[&](const std::string& name,double violation) {
            if(!std::isfinite(violation))violation=1e30;
            if(violation>maximum[name]){maximum[name]=violation;maximum_time[name]=sample.at("time").get<double>();}
            worst=std::max(worst,violation);
        };
        for(int j=0;j<4;++j) {
            const double span=p.input_max[j]-p.input_min[j];
            record("command_bounds",std::max(p.input_min[j]-u[j],u[j]-p.input_max[j])/span);
            record("actual_pitch_bounds",std::max(p.input_min[j]-x[13+j],x[13+j]-p.input_max[j])/span);
        }
        record("electrical_power",f.electrical_power/p.electrical_power_max-1);
        record("main_shaft_power",std::abs(f.main_shaft_power)/p.main_shaft_power_max-1);
        record("tail_shaft_power",std::abs(f.tail_shaft_power)/p.tail_shaft_power_max-1);
        record("main_current",std::abs(f.main_current)/p.main_current_max-1);record("tail_current",std::abs(f.tail_current)/p.tail_current_max-1);
        for(int j=21;j<23;++j) {
            record("temperature_upper",(x[j]-p.temperature_max)/(p.temperature_max-p.ambient_temperature));
            record("temperature_lower",(p.ambient_temperature-x[j])/(p.temperature_max-p.ambient_temperature));
        }
        for(int j=0;j<3;++j)record("body_rates",std::abs(x[10+j])/p.body_rate_max-1);
        const double tilt=std::acos(std::clamp(1-2*(x[7]*x[7]+x[8]*x[8]),-1.0,1.0));
        record("tilt",tilt/p.tilt_max_rad-1);
        record("airspeed",norm({x[3]-wind[0],x[4]-wind[1],x[5]-wind[2]})/p.airspeed_max-1);
        record("inflow_nonnegative",-std::min(x[17],x[18])/5);
        record("positive_main_blade_speed",-(p.main_omega+dot({x[10],x[11],x[12]},f.main_disc_axis))/p.main_omega);
        record("positive_tail_blade_speed",-(p.tail_omega+dot({x[10],x[11],x[12]},f.tail_disc_axis))/p.tail_omega);
        const bool contact=sample.at("thermodynamics").value("contact_active",false);
        // MuJoCo contact is outside the free-flight predictor. Its compliant
        // floor receives an explicit 2 cm center-height allowance.
        record("clearance",(contact?.18:.2)-x[2]);record("altitude_upper",(x[2]-6.2)/6);
        if(!contact)record("main_descent_envelope",-(f.main_axial_airspeed+p.max_descent_inflow_ratio*x[17])/p.airspeed_max);
        if(controlled)for(int j=0;j<3;++j)record("tracking_tube",std::abs(x[j]-sample.at("target")[j].get<double>())/(j==2?.35:.6)-1);
        if(controlled&&sample.at("time").get<double>()>=68-1e-8) {
            for(int j=0;j<3;++j) {
                record("landing_position",std::abs(x[j]-sample.at("target")[j].get<double>())/(j==2?.025:.10)-1);
                record("landing_velocity",std::abs(x[3+j]-sample.at("reference").at("velocity")[j].get<double>())/.10-1);
                record("landing_body_rates",std::abs(x[10+j])/.20-1);
            }
            record("landing_tilt",tilt/(5*std::acos(-1.0)/180)-1);
        }
        if(worst>constraint_tolerance)++steps;
    }
    double worst() const {double value=0;for(const auto& item:maximum)value=std::max(value,item.second);return value;}
};

Json reproducible_snapshot(Json state) {
 for(const char* key:{"update_ms","update_mean_ms","update_max_ms","deadline_misses"})state["hinf"].erase(key);
 return state;
}
bool replay_check(Mode mode) {
 helicopter::Simulation simulation(41,1,true,mode,1);
 std::vector<Json> first;
 for(int i=0;i<350;++i){simulation.step();if(i%37==0)first.push_back(reproducible_snapshot(simulation.state()));}
 simulation.reset(41,1,true,mode,1);std::size_t sample=0;
 for(int i=0;i<350;++i){simulation.step();if(i%37==0&&reproducible_snapshot(simulation.state())!=first.at(sample++))return false;}
 return true;
}
Json run(const Scenario& scenario,Mode mode) {
 const auto start=std::chrono::steady_clock::now();
 helicopter::Simulation simulation(scenario.seed,scenario.mass_scale,scenario.autopilot,mode,scenario.noise_scale);
 rotor::Params plant;plant.mass*=scenario.mass_scale;for(auto& inertia:plant.inertia)inertia*=scenario.mass_scale;
 auto prior=simulation.state().at("input_rad").get<Input>();
 ConstraintAudit audit; LedgerAudit ledger{simulation.state().at("model_state").get<State>(),plant};
 unsigned steps=0,local_region_exits=0;double slew_violation=0,max_noise=0,squared_error=0,peak_error=0;std::vector<double> update_times;
 std::map<std::string,unsigned> decisions;
 std::size_t gust=0;Json trace=Json::array();
 while(!simulation.finished()&&steps<7000) {
  if(gust<scenario.extra_gusts.size()&&steps*dt>=scenario.extra_gusts[gust]-1e-8){simulation.gust();++gust;}
  simulation.step();++steps;const auto state=simulation.state();
  const auto x=state.at("model_state").get<State>();const auto u=state.at("input_rad").get<Input>();
  const auto wind=state.at("wind").get<Vec3>();const auto forces=rotor::evaluate(x,u,wind,plant);
  audit.inspect(x,u,wind,forces,state,plant,scenario.autopilot);
  ledger.inspect(state.at("measurement_truth").get<State>(),x,u,wind,state);
  const auto target=state.at("target").get<Vec3>();const double error=norm({x[0]-target[0],x[1]-target[1],x[2]-target[2]});squared_error+=error*error;peak_error=std::max(peak_error,error);
  for(int j=0;j<4;++j)slew_violation=std::max(slew_violation,std::abs(u[j]-prior[j])-plant.input_rate_max[j]*dt);
  prior=u;
  if(!finite_json(state))throw std::runtime_error("Nonfinite state in "+scenario.id);
  const auto& control=state.at("hinf");++decisions[control.at("applied_controller").get<std::string>()];
  if(control.at("update_ms").is_number())update_times.push_back(control.at("update_ms").get<double>());
  if(control.contains("local_scope")&&!control.at("local_scope").value("within_declared_region",true))++local_region_exits;
  for(int j=0;j<6;++j)max_noise=std::max(max_noise,std::abs(state.at("measured_state")[j].get<double>()-state.at("measurement_truth")[j].get<double>()));
  if(steps%100==0||simulation.finished())trace.push_back({{"time",state.at("time")},{"position",state.at("position")},
    {"target",state.at("target")},{"velocity",state.at("velocity")},{"input_rad",u},
    {"applied_controller",control.at("applied_controller")},{"local_scope",control.value("local_scope",Json(nullptr))}});
 }
 const auto end=simulation.state();const auto& h=end.at("hinf");const auto& m=end.at("metrics");
 const bool active_hinf=scenario.autopilot&&mode==Mode::Hinf;
 const unsigned applied=decisions["minimum_entropy_hinf"],landing=decisions["pid_landing_support"],hold=decisions["trim_observation_hold"],fallback=decisions["pid_fallback"];
 const double participation=steps?double(applied)/steps:0;
 const bool accounting=h.at("control_decisions")==steps&&h.at("applied_decisions")==applied&&h.at("landing_support_decisions")==landing&&
  h.at("observation_hold_decisions")==hold&&h.at("fallback_decisions")==fallback;
 const bool controller_pass=accounting&&slew_violation<1e-12&&std::abs(std::sqrt(squared_error/steps)-m.at("rms_error").get<double>())<1e-10&&std::abs(peak_error-m.at("max_error").get<double>())<1e-10&&(!scenario.autopilot||hold==10)&&
  (!active_hinf||(applied+landing+hold+fallback==steps&&participation>=.90&&fallback==0));
 const bool flight_pass=end.at("completed").get<bool>()&&!end.at("crashed").get<bool>()&&end.at("touchdown").get<bool>()&&steps==7000&&
  m.at("rms_error").get<double>()<.55&&m.at("max_error").get<double>()<1.4&&m.at("endpoint_error").get<double>()<.25&&m.at("final_speed").get<double>()<.25;
 const bool physical_pass=audit.steps==0&&m.at("physical_constraint_violation_steps")==0;
 const bool negative_pass=!end.at("completed").get<bool>()&&m.at("rms_error").get<double>()>2&&m.at("max_altitude").get<double>()<.30;
 const bool passed=scenario.autopilot?(flight_pass&&physical_pass&&controller_pass&&ledger.passed()):(negative_pass&&controller_pass&&ledger.passed());
 const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::sort(update_times.begin(),update_times.end());
 const auto percentile=[&](double q){return update_times.empty()?0:update_times[std::min(update_times.size()-1,static_cast<std::size_t>(std::ceil(q*update_times.size())-1))];};
 Json checks={{"update_p95_ms",percentile(.95)},{"update_p99_ms",percentile(.99)},{"passed",controller_pass},{"control_decisions",steps},{"applied_decisions",applied},{"applied_fraction",participation},
  {"landing_support_decisions",landing},{"observation_hold_decisions",hold},{"fallback_decisions",fallback},
  {"decision_counts",decisions},{"decision_accounting_passed",accounting},{"max_slew_violation_rad",slew_violation},
  {"saturation_steps",h.at("saturation_steps")},{"local_region_exit_steps",local_region_exits},
  {"deadline_misses",h.at("deadline_misses")},{"update_mean_ms",h.at("update_mean_ms")},{"update_max_ms",h.at("update_max_ms")},
  {"timing_scope","Measured CPU controller updates including diagnostics; operating-system outliers reported, not a hard real-time guarantee"},
  {"sensor_noise_scale",scenario.noise_scale},{"max_sensor_error_position_velocity",max_noise}};
 std::cout<<scenario.id<<" / "<<(active_hinf?"hinf":scenario.autopilot?"pid":"disabled")
  <<": "<<(passed?"PASS":"FAIL")<<" rms="<<m.at("rms_error")<<" audited_steps="<<audit.steps<<" hinf_fraction="<<participation<<std::endl;
 return {{"scenario_id",scenario.id},{"name",scenario.name},{"mode",!scenario.autopilot?"disabled":active_hinf?"hinf":"pid"},
  {"controller_mode",end.at("controller_mode")},{"autopilot",scenario.autopilot},{"seed",scenario.seed},{"mass_scale",scenario.mass_scale},
  {"sensor_noise_scale",scenario.noise_scale},{"extra_gusts",scenario.extra_gusts},{"completed",end.at("completed")},
  {"crashed",end.at("crashed")},{"touchdown",end.at("touchdown")},{"steps",steps},{"metrics",m},
  {"controller_checks",checks},{"physical_constraints_passed",physical_pass},{"flight_criteria_passed",flight_pass},
  {"independent_constraint_audit",{{"violation_steps",audit.steps},{"maximum",audit.maximum},{"maximum_time",audit.maximum_time},
    {"maximum_violation",audit.worst()},{"tolerance",constraint_tolerance}}},
  {"thermodynamics",end.at("thermodynamics")},{"thermodynamic_audit",ledger.report()},{"landing_handover",end.at("landing_handover")},
  {"wall_seconds",wall},{"trace",trace},{"passed",passed}};
}
}
int main(int argc,char** argv) {
 try {
  std::string output,filter,mode_filter;
  for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--output"&&i+1<argc)output=argv[++i];
   else if(arg=="--scenario"&&i+1<argc)filter=argv[++i];else if(arg=="--mode"&&i+1<argc)mode_filter=argv[++i];
   else throw std::invalid_argument("Usage: helicopter-evaluate [--output file] [--scenario ID] [--mode hinf|pid]");}
  if(!mode_filter.empty()&&mode_filter!="hinf"&&mode_filter!="pid")throw std::invalid_argument("Unknown mode");
  Json trials=Json::array(),comparisons=Json::array();
  const bool replay=replay_check(Mode::Hinf)&&replay_check(Mode::Pid);
  bool passed=replay&&helicopter::HinfController::design().at("passed").get<bool>();
  for(const auto& scenario:scenarios) {
   if(!filter.empty()&&filter!=scenario.id)continue;
   if(!scenario.autopilot){auto trial=run(scenario,Mode::Pid);passed=passed&&trial.at("passed").get<bool>();trials.push_back(std::move(trial));continue;}
   Json pair=Json::object();
   for(Mode mode:{Mode::Pid,Mode::Hinf}) {
    const std::string name=mode==Mode::Hinf?"hinf":"pid";if(!mode_filter.empty()&&mode_filter!=name)continue;
    auto trial=run(scenario,mode);passed=passed&&trial.at("passed").get<bool>();pair[name]=trial;trials.push_back(std::move(trial));
   }
   if(pair.size()==2)comparisons.push_back({{"scenario",scenario.id},{"pid_rms_error_m",pair["pid"]["metrics"]["rms_error"]},
    {"hinf_rms_error_m",pair["hinf"]["metrics"]["rms_error"]},{"pid_max_error_m",pair["pid"]["metrics"]["max_error"]},
    {"hinf_max_error_m",pair["hinf"]["metrics"]["max_error"]},{"both_passed",pair["pid"]["passed"].get<bool>()&&pair["hinf"]["passed"].get<bool>()}});
  }
  if(trials.empty())throw std::invalid_argument("Unknown or empty scenario selection");
  bool hinf_passed=true,pid_passed=true;unsigned hinf_count=0,pid_count=0;
  for(const auto& trial:trials) {
   if(trial.at("mode")=="hinf"){++hinf_count;hinf_passed=hinf_passed&&trial.at("passed").get<bool>();}
   if(trial.at("mode")=="pid"){++pid_count;pid_passed=pid_passed&&trial.at("passed").get<bool>();}
  }
  Json report={{"schema_version",4},{"method","minimum_entropy_hinf"},{"complete_evaluation",filter.empty()&&mode_filter.empty()},
   {"passed",passed},{"design_passed",helicopter::HinfController::design().at("passed")},
   {"hinf_trials_passed",hinf_count>0&&hinf_passed},{"pid_trials_passed",pid_count>0&&pid_passed},
   {"hinf_trial_count",hinf_count},{"pid_trial_count",pid_count},
   {"reset_replay_identical",replay},{"model",helicopter::Simulation::model()},
   {"design",helicopter::HinfController::design()},{"scenarios",trials},{"comparisons",comparisons},
   {"acceptance",{{"full_mission_steps",7000},{"minimum_hinf_applied_fraction",.90},{"rms_error_m",.55},{"peak_error_m",1.4},
    {"physical_constraint_tolerance",constraint_tolerance},{"comparison","Common deterministic noise sequences, mission, physical limits and nominal feedback parameters; PID has auxiliary noisy rotor sensors"}}}};
  if(!output.empty()){const auto path=std::filesystem::path(output);if(path.has_parent_path())std::filesystem::create_directories(path.parent_path());std::ofstream file(path);file<<report.dump(2)<<'\n';if(!file)throw std::runtime_error("Failed to save evaluation");}
  else std::cout<<report.dump(2)<<'\n';
  return passed?0:1;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
