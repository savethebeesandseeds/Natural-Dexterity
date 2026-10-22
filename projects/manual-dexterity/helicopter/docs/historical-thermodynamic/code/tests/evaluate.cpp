#include "helicopter.hpp"
#include "entropy_controller.hpp"
#include "rotor_model.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Json=nlohmann::json;
namespace rotor=helicopter::rotor;
using State=rotor::State<double>;
using Input=rotor::Input<double>;
using Vec3=rotor::Vec3<double>;
using Mode=helicopter::ControllerMode;
constexpr double dt=.01, constraint_tolerance=1e-4;
constexpr double accepted_fraction_min=.90, changed_action_min=1e-5;
constexpr double balance_relative=.002, energy_balance_absolute=.5, entropy_balance_absolute=.002;
struct Scenario {std::string id,name;unsigned seed;double mass_scale;bool autopilot;std::vector<double> extra_gusts;};
const std::vector<Scenario> scenarios{
    {"nominal","nominal",0,1,true,{}},
    {"alternate-wind","alternate wind phase",71,1,true,{}},
    {"extra-gusts","extra gusts",19,1,true,{18,44}},
    {"mass-minus-20","mass minus 20 percent",3,.8,true,{}},
    {"mass-plus-20","mass plus 20 percent with gust",31,1.2,true,{20}},
    {"disabled","controller disabled baseline",0,1,false,{}}
};
bool finite_json(const Json& value) {
    if(value.is_number())return std::isfinite(value.get<double>());
    if(value.is_structured())for(const auto& child:value)if(!finite_json(child))return false;
    return true;
}
bool near(double a,double b,double tolerance=1e-8){return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance;}
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

struct SolveAudit {
    unsigned decisions=0,solves=0,accepted=0,converged=0,fallbacks=0,landing_support=0,observation_hold=0,deadline_misses=0,changed=0,improved=0;
    double sum_ms=0,max_ms=0,max_constraint=0,max_dynamics=0,max_objective_error=0,max_intermediate_physical=0,max_intermediate_planning=0;
    double landing_start_time=0;
    bool counters_consistent=true,accepted_feasible=true,model_parameter_consistent=true;
    std::map<std::string,unsigned> status_counts;
    Json records=Json::array();
    bool observe(const Json& d) {
        if(!d.value("available",false))return false;
        const unsigned decision=d.at("control_decisions").get<unsigned>();
        if(decision==decisions)return false;
        if(decision!=decisions+1)counters_consistent=false;
        ++decisions;
        counters_consistent=counters_consistent&&near(d.at("time").get<double>(),(decisions-1)*helicopter::EntropyController::update_period,1e-6);
        const bool attempted=d.at("optimizer_attempted").get<bool>();
        const bool ok=d.at("accepted").get<bool>(),conv=d.at("converged").get<bool>(),fallback=d.at("fallback").get<bool>();
        const std::string applied=d.at("applied_controller").get<std::string>();
        const unsigned iterations=d.at("iterations").get<unsigned>();
        const double elapsed=d.at("solve_ms").get<double>();
        const std::string status=d.value("status","missing_status");++status_counts[status];
        const double difference=attempted?d.at("action_difference_norm").get<double>():0.0;
        const bool prediction_empty=d.contains("prediction")?d.at("prediction").empty():d.at("prediction_empty").get<bool>();
        if(attempted) {
            // Planned landing support is a latched handover, not an opportunity
            // to hide intermittently skipped optimizer attempts.
            counters_consistent=counters_consistent&&landing_support==0&&observation_hold==1&&decision>1;
            ++solves;accepted+=ok;converged+=conv;fallbacks+=!ok;
            const double model_scale=d.at("model_mass_scale").get<double>();
            const double observed_scale=d.at("observed_mass_scale").get<double>();
            model_parameter_consistent=model_parameter_consistent&&model_scale>=.5&&model_scale<=1.5&&
                near(model_scale,observed_scale,1e-9);
            sum_ms+=elapsed;max_ms=std::max(max_ms,elapsed);
            deadline_misses+=elapsed>1000*helicopter::EntropyController::update_period;
            changed+=ok&&difference>changed_action_min;
            counters_consistent=counters_consistent&&std::isfinite(elapsed)&&elapsed>=0&&fallback!=ok&&
                applied==(ok?"entropy_nmpc":"pid_fallback")&&status!="landing_support"&&status!="observation_hold"&&
                conv==(status=="Solve_Succeeded"||status=="Solved_To_Acceptable_Level")&&prediction_empty!=ok;
            if(ok) {
                const double seed_violation=d.at("seed_constraint_violation").get<double>();
                const double constraint=d.at("max_constraint_violation").get<double>();
                const double dynamics=d.at("dynamics_residual").get<double>();
                const double objective_error=d.at("objective_check_error").get<double>();
                const double intermediate_physical=d.at("intermediate_physical_violation").get<double>();
                const double intermediate_planning=d.at("intermediate_planning_violation").get<double>();
                max_constraint=std::max(max_constraint,constraint);max_dynamics=std::max(max_dynamics,dynamics);
                max_objective_error=std::max(max_objective_error,objective_error);
                max_intermediate_physical=std::max(max_intermediate_physical,intermediate_physical);
                max_intermediate_planning=std::max(max_intermediate_planning,intermediate_planning);
                const double seed=d.at("seed_objective_j_per_k").get<double>(),value=d.at("objective_j_per_k").get<double>();
                accepted_feasible=accepted_feasible&&std::isfinite(value)&&constraint<=constraint_tolerance&&
                    dynamics<=constraint_tolerance&&objective_error<=1e-4&&
                    intermediate_physical<=constraint_tolerance&&intermediate_planning<=constraint_tolerance&&
                    (seed_violation>constraint_tolerance||value<=seed+1e-7);
                improved+=seed_violation<=constraint_tolerance&&seed-value>std::max(1e-8,std::abs(seed)*1e-6);
            }
        } else {
            if(status=="observation_hold") {
                counters_consistent=counters_consistent&&decision==1&&observation_hold==0&&solves==0&&landing_support==0&&
                    near(d.at("time").get<double>(),0,1e-9)&&applied=="trim_observation_hold";
                ++observation_hold;
            } else if(status=="landing_support") {
                if(landing_support==0)landing_start_time=d.at("time").get<double>();
                ++landing_support;
                counters_consistent=counters_consistent&&observation_hold==1&&decision>1&&applied=="pid_landing_support";
            } else counters_consistent=false;
            counters_consistent=counters_consistent&&!ok&&!conv&&fallback&&prediction_empty&&elapsed==0&&iterations==0;
            for(const char* key:{"seed_constraint_violation","seed_objective_j_per_k","objective_j_per_k",
                "objective_reduction_j_per_k","max_constraint_violation","dynamics_residual","objective_check_error","action_difference_norm",
                "intermediate_physical_violation","intermediate_planning_violation"})
                counters_consistent=counters_consistent&&d.at(key).is_null();
        }
        counters_consistent=counters_consistent&&d.at("solves").get<unsigned>()==solves&&
            d.at("accepted_solves").get<unsigned>()==accepted&&d.at("fallback_solves").get<unsigned>()==fallbacks&&
            d.at("landing_support_decisions").get<unsigned>()==landing_support&&
            d.at("observation_hold_decisions").get<unsigned>()==observation_hold&&decision==solves+landing_support+observation_hold;
        records.push_back({{"decision",decision},{"control_decisions",decision},{"solve",d.at("solves")},{"solves",d.at("solves")},
            {"optimizer_attempted",attempted},{"applied_controller",applied},{"fallback",fallback},{"prediction_empty",prediction_empty},
            {"model_mass_scale",attempted?d.at("model_mass_scale"):Json()},
            {"observed_mass_scale",attempted?d.at("observed_mass_scale"):Json()},
            {"time",d.value("time",0.0)},{"status",status},{"accepted",ok},{"converged",conv},{"iterations",iterations},
            {"solve_ms",elapsed},{"action_difference_norm",d.at("action_difference_norm")},{"seed_constraint_violation",d.value("seed_constraint_violation",Json())},
            {"seed_objective_j_per_k",d.value("seed_objective_j_per_k",Json())},
            {"objective_j_per_k",d.value("objective_j_per_k",Json())},
            {"objective_reduction_j_per_k",d.value("objective_reduction_j_per_k",Json())},
            {"dynamics_residual",d.value("dynamics_residual",Json())},
            {"objective_check_error",d.value("objective_check_error",Json())},
            {"intermediate_physical_violation",d.value("intermediate_physical_violation",Json())},
            {"intermediate_planning_violation",d.value("intermediate_planning_violation",Json())},
            {"accepted_solves",d.at("accepted_solves")},{"fallback_solves",d.at("fallback_solves")},
            {"landing_support_decisions",d.at("landing_support_decisions")},
            {"observation_hold_decisions",d.at("observation_hold_decisions")},
            {"max_constraint_violation",d.value("max_constraint_violation",Json())}});
        return true;
    }
    Json summary(bool required) const {
        const double fraction=decisions?double(accepted)/decisions:0;
        return {{"control_decisions",decisions},{"solves",solves},{"accepted_solves",accepted},{"converged_solves",converged},{"fallback_solves",fallbacks},
            {"landing_support_decisions",landing_support},{"observation_hold_decisions",observation_hold},
            {"fallback_decisions",fallbacks+landing_support+observation_hold},{"pid_decisions",fallbacks+landing_support},
            {"landing_support_start_time_s",landing_support?Json(landing_start_time):Json()},
            {"accepted_fraction",fraction},{"fallback_fraction",decisions?double(fallbacks+landing_support+observation_hold)/decisions:0},
            {"pid_fraction",decisions?double(fallbacks+landing_support)/decisions:0},
            {"landing_support_fraction",decisions?double(landing_support)/decisions:0},
            {"observation_hold_fraction",decisions?double(observation_hold)/decisions:0},
            {"attempt_acceptance_fraction",solves?double(accepted)/solves:0},{"attempt_rejection_fraction",solves?double(fallbacks)/solves:0},
            {"converged_fraction",solves?double(converged)/solves:0},{"deadline_misses",deadline_misses},
            {"max_solve_ms",max_ms},{"mean_solve_ms",solves?sum_ms/solves:0},{"changed_action_solves",changed},
            {"improved_accepted_solves",improved},{"participation_passed",!required||(decisions>0&&fraction>=accepted_fraction_min)},
            {"participation_denominator","All control decisions, including initial trim observation, planned PID landing support and rejected solves"},
            {"counters_consistent",counters_consistent},{"accepted_candidates_feasible",accepted_feasible},{"status_counts",status_counts},
            {"model_parameter_consistent",model_parameter_consistent},
            {"max_accepted_constraint_violation",max_constraint},{"max_accepted_dynamics_residual",max_dynamics},
            {"max_accepted_intermediate_physical_violation",max_intermediate_physical},
            {"max_accepted_intermediate_planning_violation",max_intermediate_planning},
            {"max_objective_check_error_j_per_k",max_objective_error},{"objective_optimization_demonstrated",changed>0&&improved>0},
            {"deadline_s",helicopter::EntropyController::update_period},{"unique_solves",records},
            {"unique_solves_scope","Historical field name: one record per control decision; optimizer_attempted distinguishes actual solves"}};
    }
    Json final_summary(bool required,unsigned steps) const {
        auto result=summary(required);
        const unsigned expected=required?static_cast<unsigned>(std::ceil(steps*dt/helicopter::EntropyController::update_period-1e-8)):0;
        const unsigned expected_attempts=expected-std::min(expected,landing_support+observation_hold);
        result["expected_control_decisions"]=expected;result["expected_solves"]=expected_attempts;
        result["decision_schedule_verified"]=decisions==expected&&counters_consistent;
        result["solve_schedule_verified"]=landing_support+observation_hold<=expected&&solves==expected_attempts&&counters_consistent;
        result["observation_schedule_verified"]=required?(expected>0&&observation_hold==1&&counters_consistent):observation_hold==0;
        result["objective_optimization_required"]=required;
        result["objective_optimization_passed"]=!required||result.at("objective_optimization_demonstrated").get<bool>();
        return result;
    }
};

struct ParameterAudit {
    unsigned observations=0,accepted=0,rejected=0,contact_rejections=0;
    double information=0,cross_information=0,mass=rotor::Params{}.mass,last_reported_mass=rotor::Params{}.mass;
    double maximum_mass_discrepancy=0,maximum_scale_discrepancy=0,maximum_force_discrepancy=0;
    bool counters_consistent=true,metadata_consistent=true,initial_prior_verified=true;
    Json checkpoints=Json::array();

    void inspect_metadata(const Json& metadata) {
        const double reported_mass=metadata.at("mass_kg").get<double>();
        last_reported_mass=reported_mass;
        const double reported_scale=metadata.at("mass_scale").get<double>();
        maximum_mass_discrepancy=std::max(maximum_mass_discrepancy,std::abs(reported_mass-mass));
        maximum_scale_discrepancy=std::max(maximum_scale_discrepancy,std::abs(reported_scale-mass/rotor::Params{}.mass));
        counters_consistent=counters_consistent&&metadata.at("accepted_samples").get<unsigned>()==accepted&&
            metadata.at("rejected_samples").get<unsigned>()==rejected;
        metadata_consistent=metadata_consistent&&near(reported_mass,mass,1e-9)&&near(reported_scale,mass/rotor::Params{}.mass,1e-9)&&
            near(metadata.at("information").get<double>(),information,1e-6)&&
            metadata.at("estimate_available").get<bool>()==(accepted>0)&&metadata.at("using_nominal_prior").get<bool>()==(accepted==0);
    }
    void checkpoint(double time,const Json& metadata) {
        checkpoints.push_back({time,cross_information,information,accepted,rejected,metadata.at("mass_kg")});
    }
    void initialize(const Json& metadata) {
        inspect_metadata(metadata);
        initial_prior_verified=metadata_consistent&&counters_consistent;
        checkpoint(0,metadata);
    }
    void inspect(const State& before,const State& after,const Input& input,const Vec3& wind,
                 bool contact_before,bool contact_after,const Vec3& applied_force,const Json& metadata) {
        // Reconstruct the force with nominal aerodynamic parameters. No true
        // plant mass/inertia enters identification, and the estimator class is
        // deliberately not reused by this audit.
        const rotor::Params nominal;
        const auto f=rotor::evaluate(before,input,wind,nominal);
        const double w=before[6],x=before[7],y=before[8],z=before[9];
        const Vec3 force{
            (1-2*y*y-2*z*z)*f.force_body[0]+(2*x*y-2*w*z)*f.force_body[1]+(2*x*z+2*w*y)*f.force_body[2]+f.drag_world[0],
            (2*x*y+2*w*z)*f.force_body[0]+(1-2*x*x-2*z*z)*f.force_body[1]+(2*y*z-2*w*x)*f.force_body[2]+f.drag_world[1],
            (2*x*z-2*w*y)*f.force_body[0]+(2*y*z+2*w*x)*f.force_body[1]+(1-2*x*x-2*y*y)*f.force_body[2]+f.drag_world[2]};
        Vec3 q{};
        for(int j=0;j<3;++j) {
            q[j]=(after[3+j]-before[3+j])/dt+(j==2?nominal.gravity:0);
            maximum_force_discrepancy=std::max(maximum_force_discrepancy,std::abs(force[j]-applied_force[j]));
        }
        const double sample_information=dot(q,q),sample_cross=dot(q,force);
        const double observed_mass=sample_information>1e-8?sample_cross/sample_information:0;
        const bool contact=contact_before||contact_after;
        bool usable=!contact&&std::isfinite(sample_information)&&std::isfinite(sample_cross)&&sample_information>1e-8&&
            observed_mass>=.5*nominal.mass&&observed_mass<=1.5*nominal.mass;
        const double next_information=information+sample_information,next_cross=cross_information+sample_cross;
        const double next_mass=usable?next_cross/next_information:mass;
        usable=usable&&std::isfinite(next_information)&&std::isfinite(next_cross)&&std::isfinite(next_mass)&&
            next_mass>=.5*nominal.mass&&next_mass<=1.5*nominal.mass;
        ++observations;
        if(usable){++accepted;information=next_information;cross_information=next_cross;mass=next_mass;}
        else {++rejected;contact_rejections+=contact;}
        inspect_metadata(metadata);
        if(observations%10==0)checkpoint(observations*dt,metadata);
    }
    bool passed() const {
        return initial_prior_verified&&metadata_consistent&&counters_consistent&&maximum_mass_discrepancy<=1e-9&&
            maximum_scale_discrepancy<=1e-9&&maximum_force_discrepancy<=1e-7;
    }
    Json summary(double true_mass_for_audit) const {
        Json saved_checkpoints=checkpoints;
        if(observations%10!=0)saved_checkpoints.push_back({observations*dt,cross_information,information,accepted,rejected,last_reported_mass});
        return {{"passed",passed()},{"observation_steps",observations},{"accepted_samples",accepted},{"rejected_samples",rejected},
            {"contact_rejections",contact_rejections},{"nominal_prior_verified",initial_prior_verified},
            {"metadata_consistent",metadata_consistent},{"counters_consistent",counters_consistent},
            {"max_mass_discrepancy_kg",maximum_mass_discrepancy},{"max_scale_discrepancy",maximum_scale_discrepancy},
            {"max_nominal_force_discrepancy_n",maximum_force_discrepancy},
            {"sum_q_squared",information},{"sum_q_dot_force",cross_information},
            {"final_mass_kg",mass},{"final_mass_scale",mass/rotor::Params{}.mass},
            {"final_absolute_mass_error_kg",std::abs(mass-true_mass_for_audit)},
            {"final_relative_mass_error",std::abs(mass/true_mass_for_audit-1)},
            {"truth_use","True plant mass is used only for these final error diagnostics, never for identification or its causal audit"},
            {"checkpoint_columns",{"time_s","sum_q_dot_force","sum_q_squared","accepted_samples","rejected_samples","reported_mass_kg"}},
            {"checkpoints",saved_checkpoints},
            {"audit_scope","Independent 100 Hz force/velocity/contact/counter audit. The merger verifies 10 Hz cumulative least-squares checkpoints and decision-time parameters, not raw flight observations."}};
    }
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

Json run(const Scenario& scenario,Mode mode) {
    const std::string short_mode=!scenario.autopilot?"disabled":mode==Mode::Entropy?"entropy":"pid";
    const std::string display=scenario.name+" / "+short_mode;
    Json result{{"name",display},{"scenario_name",scenario.name},{"scenario_id",scenario.id},{"mode",short_mode},
        {"controller_mode",mode==Mode::Entropy?"entropy_nmpc":"pid_baseline"},{"autopilot",scenario.autopilot},
        {"seed",scenario.seed},{"mass_scale",scenario.mass_scale},{"extra_gusts_seconds",scenario.extra_gusts},
        {"passed",false},{"completed",false},{"metrics",Json::object()}};
    const auto started=std::chrono::steady_clock::now();
    try {
        helicopter::Simulation simulation(scenario.seed,scenario.mass_scale,scenario.autopilot,mode);
        rotor::Params p;p.mass*=scenario.mass_scale;for(auto& value:p.inertia)value*=scenario.mass_scale;
        Json before=simulation.state();
        const State initial=before.at("model_state").get<State>();
        const Input initial_input=before.at("input_rad").get<Input>();
        const double initial_time=before.at("time").get<double>();
        const double initial_stored=stored_energy(initial,p),initial_commitment=completion_entropy(initial,p);
        std::size_t gust=0;unsigned steps=0;
        bool finite=true,bounded=true,telemetry=true,initial_dynamics=true,slew=true,reported=true;
        double electrical=0,generated=0,heat=0,air=0,wind_work=0,trap_electrical=0,trap_entropy=0,max_instant_energy_residual=0;
        double squared_error=0,max_error=0,max_tilt=0,max_altitude=initial[2],max_speed=0;
        double max_contact_speed=0,max_ledger_relative=0,max_entropy_relative=0;
        double final_energy_residual=0,final_entropy_residual=0,max_power=0,max_temperature=p.ambient_temperature;
        unsigned observation_hold_steps=0;
        double observation_hold_end=initial_time,observation_hold_input_error=0,observation_hold_energy=0,observation_hold_entropy=0;
        unsigned observation_hold_samples=0;
        std::array<double,2> motor_loss{},motor_heat{},motor_residual{},peak_current{},peak_shaft{};
        double max_motor_balance_relative=0,minimum_entropy_rate=1e30;
        SolveAudit solver;ConstraintAudit constraints;ParameterAudit parameters;
        parameters.initialize(before.at("parameter_estimation"));
        bool prior_contact=before.at("thermodynamics").value("contact_active",false);
        while(!simulation.finished()&&steps<7100) {
            if(gust<scenario.extra_gusts.size()&&before.at("time").get<double>()>=scenario.extra_gusts[gust]-1e-9){simulation.gust();++gust;}
            simulation.step();++steps;
            const Json after=simulation.state();finite=finite&&finite_json(after);
            const State x=before.at("model_state").get<State>(),next=after.at("model_state").get<State>();
            const Input u=after.at("input_rad").get<Input>(),previous=before.at("input_rad").get<Input>();
            const Vec3 wind=after.at("wind").get<Vec3>();
            const auto f=rotor::evaluate(x,u,wind,p),fn=rotor::evaluate(next,u,wind,p);
            if(scenario.autopilot&&steps<=10) {
                ++observation_hold_steps;observation_hold_end=after.at("time").get<double>();
                for(int j=0;j<4;++j)observation_hold_input_error=std::max(observation_hold_input_error,std::abs(u[j]-initial_input[j]));
                observation_hold_energy+=f.electrical_power*dt;observation_hold_entropy+=f.entropy_rate*dt;
                observation_hold_samples=after.at("parameter_estimation").at("accepted_samples").get<unsigned>();
            }
            const auto dx=rotor::derivative(x,u,wind,p);
            const auto& solution=after.at("solution");
            telemetry=telemetry&&solution.value("available",false)&&near(solution.at("time"),before.at("time"))&&
                near(after.at("time").get<double>()-before.at("time").get<double>(),dt)&&after.at("target")==after.at("reference").at("position");
            double qnorm=0;for(int j=6;j<10;++j)qnorm+=next[j]*next[j];telemetry=telemetry&&near(qnorm,1,1e-8);
            for(int j=0;j<3;++j)telemetry=telemetry&&near(next[j],after.at("position")[j])&&near(next[3+j],after.at("velocity")[j])&&
                near(next[10+j],after.at("body_rates")[j])&&near(solution.at("position_error")[j],before.at("target")[j].get<double>()-x[j])&&
                near(solution.at("force_body")[j],f.force_body[j],1e-7)&&near(solution.at("torque_body")[j],f.torque_body[j],1e-7);
            telemetry=telemetry&&near(solution.at("actual_thrust_n"),f.main_thrust,1e-7);
            for(int j=0;j<4;++j) {
                const double exact=u[j]+(x[13+j]-u[j])*std::exp(-dt/p.actuator_tau);
                telemetry=telemetry&&near(next[13+j],exact,3e-8);
                bounded=bounded&&u[j]>=p.input_min[j]-1e-9&&u[j]<=p.input_max[j]+1e-9&&next[13+j]>=p.input_min[j]-1e-8&&next[13+j]<=p.input_max[j]+1e-8;
            }
            Json optimizer_diagnostic=after.at("optimizer");
            optimizer_diagnostic["observed_mass_scale"]=before.at("parameter_estimation").at("mass_scale");
            const bool new_decision=solver.observe(optimizer_diagnostic);
            const bool optimized=scenario.autopilot&&mode==Mode::Entropy&&after.at("optimizer").value("accepted",false);
            for(int j=0;j<4;++j) {
                const double allowed=p.input_rate_max[j]*(optimized&&new_decision?helicopter::EntropyController::update_period:dt);
                slew=slew&&std::abs(u[j]-previous[j])<=allowed+1e-8;
                if(optimized&&!new_decision)slew=slew&&near(u[j],previous[j],1e-9);
            }
            if(steps==1)for(int j=0;j<3;++j)initial_dynamics=initial_dynamics&&
                near((next[3+j]-x[3+j])/dt,solution.at("force_world")[j].get<double>()/p.mass-(j==2?p.gravity:0),1e-6)&&
                near(after.at("inertia_body_kg_m2")[j],p.inertia[j]);
            constraints.inspect(next,u,wind,fn,after,p,scenario.autopilot);
            electrical+=f.electrical_power*dt;generated+=f.entropy_rate*dt;
            motor_loss[0]+=f.main_motor_loss*dt;motor_loss[1]+=f.tail_motor_loss*dt;
            motor_heat[0]+=f.main_heat_out*dt;motor_heat[1]+=f.tail_heat_out*dt;
            const std::array<double,2> capacity{p.main_thermal_capacity,p.tail_thermal_capacity};
            for(int j=0;j<2;++j) {
                motor_residual[j]=motor_loss[j]-motor_heat[j]-capacity[j]*(next[21+j]-initial[21+j]);
                max_motor_balance_relative=std::max(max_motor_balance_relative,std::max(0.0,std::abs(motor_residual[j])-energy_balance_absolute)/std::max(1.0,motor_loss[j]));
            }
            peak_current[0]=std::max({peak_current[0],std::abs(f.main_current),std::abs(fn.main_current)});
            peak_current[1]=std::max({peak_current[1],std::abs(f.tail_current),std::abs(fn.tail_current)});
            peak_shaft[0]=std::max({peak_shaft[0],std::abs(f.main_shaft_power),std::abs(fn.main_shaft_power)});
            peak_shaft[1]=std::max({peak_shaft[1],std::abs(f.tail_shaft_power),std::abs(fn.tail_shaft_power)});
            minimum_entropy_rate=std::min({minimum_entropy_rate,f.entropy_rate,fn.entropy_rate});
            heat+=(f.main_heat_out+f.tail_heat_out)*dt;air+=(f.wake_dissipation+f.drag_dissipation+f.angular_dissipation)*dt;wind_work+=f.wind_power*dt;
            trap_electrical+=(f.electrical_power+fn.electrical_power)*dt/2;trap_entropy+=(f.entropy_rate+fn.entropy_rate)*dt/2;
            double derivative_energy=p.mass*p.gravity*dx[2];
            for(int j=0;j<3;++j)derivative_energy+=p.mass*x[3+j]*dx[3+j]+p.inertia[j]*x[10+j]*dx[10+j];
            derivative_energy+=p.main_inflow_mass*x[17]*dx[17]+p.tail_inflow_mass*x[18]*dx[18]+p.main_thermal_capacity*dx[21]+p.tail_thermal_capacity*dx[22];
            const double instant=f.electrical_power+f.wind_power-derivative_energy-f.wake_dissipation-f.drag_dissipation-f.angular_dissipation-f.main_heat_out-f.tail_heat_out;
            max_instant_energy_residual=std::max(max_instant_energy_residual,std::abs(instant));
            const auto& thermo=after.at("thermodynamics");const double contact=thermo.at("contact_work_j").get<double>();
            final_energy_residual=electrical+wind_work+contact-(stored_energy(next,p)-initial_stored)-heat-air;
            const double entropy_storage=p.main_thermal_capacity*std::log(next[21]/initial[21])+p.tail_thermal_capacity*std::log(next[22]/initial[22]);
            final_entropy_residual=generated-entropy_storage-(heat+air)/p.ambient_temperature;
            max_ledger_relative=std::max(max_ledger_relative,std::max(0.0,std::abs(final_energy_residual)-energy_balance_absolute)/std::max(1.0,electrical));
            max_entropy_relative=std::max(max_entropy_relative,std::max(0.0,std::abs(final_entropy_residual)-entropy_balance_absolute)/std::max(1.0,generated));
            const double completed_entropy=generated+completion_entropy(next,p)-initial_commitment;
            reported=reported&&near(thermo.at("electrical_energy_j"),electrical,1e-6)&&near(thermo.at("cumulative_entropy_j_per_k"),generated,1e-8)&&
                near(thermo.at("entropy_with_terminal_commitment_j_per_k"),completed_entropy,1e-7)&&near(thermo.at("integrated_energy_residual_j"),final_energy_residual,1e-6);
            const bool contact_now=thermo.value("contact_active",false);
            parameters.inspect(x,next,u,wind,prior_contact,contact_now,solution.at("force_world").get<Vec3>(),after.at("parameter_estimation"));
            if(contact_now&&!prior_contact)max_contact_speed=std::max(max_contact_speed,norm({x[3],x[4],x[5]}));
            prior_contact=contact_now;
            const Vec3 target=after.at("target").get<Vec3>();const double error=norm({next[0]-target[0],next[1]-target[1],next[2]-target[2]});
            squared_error+=error*error;max_error=std::max(max_error,error);
            max_tilt=std::max(max_tilt,std::acos(std::clamp(1-2*(next[7]*next[7]+next[8]*next[8]),-1.0,1.0))*180/std::acos(-1.0));
            max_altitude=std::max(max_altitude,next[2]);max_speed=std::max(max_speed,norm({next[3],next[4],next[5]}));
            max_power=std::max({max_power,f.electrical_power,fn.electrical_power});max_temperature=std::max({max_temperature,next[21],next[22]});
            if(steps%1000==0)std::cerr<<display<<": "<<steps*dt<<"/70 s, NMPC applied "<<solver.accepted<<'/'<<solver.decisions
                <<" decisions, "<<solver.solves<<" attempts, "<<solver.landing_support<<" landing support, "<<solver.observation_hold
                <<" observation hold, error "<<error<<" m\n";
            before=after;
        }
        const State final=before.at("model_state").get<State>();const Vec3 target=before.at("target").get<Vec3>();
        const double rms=steps?std::sqrt(squared_error/steps):0,endpoint=norm({final[0]-target[0],final[1]-target[1],final[2]-target[2]});
        const double final_speed=norm({final[3],final[4],final[5]});
        const bool flight=scenario.autopilot?before.at("completed").get<bool>()&&before.at("touchdown").get<bool>()&&
            rms<.55&&max_error<1.4&&endpoint<.25&&final_speed<.25:!before.at("completed").get<bool>()&&rms>2&&max_altitude<.30;
        const bool physically_valid=constraints.worst()<=constraint_tolerance;
        const bool thermo_ok=reported&&max_instant_energy_residual<1e-6&&max_ledger_relative<=balance_relative&&
            max_entropy_relative<=balance_relative&&max_motor_balance_relative<=balance_relative&&minimum_entropy_rate>=-1e-10;
        const bool requires_optimizer=scenario.autopilot&&mode==Mode::Entropy;
        Json solver_summary=solver.final_summary(requires_optimizer,steps);
        // Flying on accepted, effectively unchanged PID seeds is insufficient:
        // each enabled entropy trial must demonstrate optimizer contribution.
        const bool optimizer_ok=solver_summary.at("participation_passed").get<bool>()&&
            solver_summary.at("objective_optimization_passed").get<bool>()&&solver_summary.at("decision_schedule_verified").get<bool>()&&
            solver_summary.at("solve_schedule_verified").get<bool>()&&solver_summary.at("observation_schedule_verified").get<bool>()&&
            solver.accepted_feasible&&solver.model_parameter_consistent;
        const bool observation_hold_ok=near(initial_time,0,1e-9)&&observation_hold_input_error<=1e-12&&
            observation_hold_steps==(scenario.autopilot?std::min(steps,10U):0)&&near(observation_hold_end,observation_hold_steps*dt,1e-8);
        const bool passed=finite&&bounded&&telemetry&&initial_dynamics&&slew&&thermo_ok&&physically_valid&&flight&&optimizer_ok&&
            parameters.passed()&&observation_hold_ok&&steps==7000&&!before.at("crashed").get<bool>();
        Json metrics=before.at("metrics");metrics["rms_error"]=rms;metrics["max_error"]=max_error;metrics["endpoint_error"]=endpoint;
        metrics["final_speed"]=final_speed;metrics["max_altitude"]=max_altitude;metrics["max_speed"]=max_speed;metrics["max_tilt_deg"]=max_tilt;
        metrics["max_temperature_k"]=max_temperature;metrics["max_electrical_power_w"]=max_power;metrics["max_contact_entry_speed_m_s"]=max_contact_speed;
        const auto handover=before.at("landing_handover");
        const bool unsettled_handover=handover.at("active").get<bool>()&&!handover.at("settled_at_handover").get<bool>();
        const double commitment=completion_entropy(final,p);
        Json thermo{{"electrical_energy_j",electrical},{"generated_entropy_j_per_k",generated},{"committed_entropy_j_per_k",generated+commitment-initial_commitment},
            {"terminal_commitment_j_per_k",commitment},{"initial_commitment_j_per_k",initial_commitment},{"wind_work_j",wind_work},
            {"contact_work_j",before.at("thermodynamics").at("contact_work_j")},{"mechanical_energy_j",mechanical_energy(final,p)},
            {"initial_mechanical_energy_j",mechanical_energy(initial,p)},{"stored_wake_energy_j",wake_energy(final,p)},
            {"stored_thermal_energy_j",thermal_energy(final,p)},{"motor_temperature_k",{final[21],final[22]}},
            {"motor_loss_energy_j",motor_loss},{"motor_heat_rejected_j",motor_heat},{"motor_energy_balance_residual_j",motor_residual},
            {"max_normalized_motor_energy_residual",max_motor_balance_relative},{"peak_motor_current_a",peak_current},
            {"peak_absolute_shaft_power_w",peak_shaft},{"minimum_entropy_generation_rate_w_per_k",minimum_entropy_rate},
            {"energy_balance_residual_j",final_energy_residual},{"normalized_energy_balance_residual",max_ledger_relative},
            {"entropy_balance_residual_j_per_k",final_entropy_residual},{"normalized_entropy_balance_residual",max_entropy_relative},
            {"max_instantaneous_energy_residual_w",max_instant_energy_residual},{"trapezoid_electrical_energy_j",trap_electrical},
            {"trapezoid_entropy_j_per_k",trap_entropy},{"energy_accounting_uncertainty_j",std::abs(final_energy_residual)+std::abs(electrical-trap_electrical)},
            {"entropy_accounting_uncertainty_j_per_k",std::abs(final_entropy_residual)+std::abs(generated-trap_entropy)},
            {"reported_integrals_verified",reported},{"passed",thermo_ok},
            {"integration_note","Independent 100 Hz left sums verify reported integrals; trapezoid resampling and first/second-law residuals quantify discretization uncertainty. Contact is an external mechanical port."}};
        result.update({{"steps",steps},{"finite",finite},{"controls_bounded",bounded},{"telemetry_consistent",telemetry},
            {"initial_dynamics_consistent",initial_dynamics},{"input_slew_consistent",slew},{"passed",passed},{"completed",before.at("completed")},
            {"crashed",before.at("crashed")},{"flight_criteria_passed",flight},{"physical_constraints_passed",physically_valid},
            {"optimizer",solver_summary},{"parameter_estimation",parameters.summary(p.mass)},
            {"initial_trim_observation",{{"required",scenario.autopilot},{"duration_s",.1},{"checked_steps",observation_hold_steps},
                {"start_time_s",initial_time},{"end_time_s",observation_hold_end},{"initial_input_rad",initial_input},
                {"max_command_deviation_rad",observation_hold_input_error},{"electrical_energy_j",observation_hold_energy},
                {"generated_entropy_j_per_k",observation_hold_entropy},{"accepted_identification_samples",observation_hold_samples},
                {"accounting_scope","The hold is inside the 70-second mission clock and its energy/entropy integrals"},{"passed",observation_hold_ok}}},
            {"thermodynamics",thermo},{"metrics",metrics},{"final_position",before.at("position")},
            {"final_velocity",before.at("velocity")},{"final_quaternion",before.at("quaternion")},{"final_body_rates",before.at("body_rates")},
            {"final_contact_active",before.at("thermodynamics").at("contact_active")},{"touchdown",before.at("touchdown")},
            {"landing_handover",handover},{"unsettled_landing_handover",unsettled_handover},
            {"physical_constraints",{{"max_normalized_violation",constraints.worst()},{"violating_steps",constraints.steps},
                {"max_by_constraint",constraints.maximum},{"worst_time_by_constraint_s",constraints.maximum_time},{"contact_center_height_min_m",.18},
                {"main_descent_envelope_scope","Airborne states; contact lies outside the free-flight predictor"}}}});
    } catch(const std::exception& error){result["error"]=error.what();std::cerr<<display<<": evaluation error: "<<error.what()<<'\n';}
    result["wall_time_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return result;
}

Json comparisons(const Json& results) {
    Json output=Json::array();
    for(const auto& scenario:scenarios) {
        if(!scenario.autopilot)continue;
        const Json *pid=nullptr,*entropy=nullptr;
        for(const auto& result:results)if(result.at("scenario_id")==scenario.id) {
            if(result.at("mode")=="pid")pid=&result;
            if(result.at("mode")=="entropy")entropy=&result;
        }
        if(!pid||!entropy)continue;
        const bool has_data=pid->contains("thermodynamics")&&entropy->contains("thermodynamics");
        const bool comparable=has_data&&pid->at("passed").get<bool>()&&entropy->at("passed").get<bool>();
        Json row{{"scenario",scenario.name},{"pid_name",pid->at("name")},{"entropy_name",entropy->at("name")},{"comparable",comparable},
            {"reason",comparable?"Both controllers satisfy the same mission, constraint, accounting, and participation criteria":"At least one paired trial failed or lacks data; savings are not claimed"},
            {"electrical_energy_saving_j",nullptr},{"electrical_energy_saving_percent",nullptr},{"entropy_saving_j_per_k",nullptr},
            {"entropy_saving_percent",nullptr},{"savings_demonstrated",false}};
        if(has_data) {
            const auto& a=pid->at("thermodynamics");const auto& b=entropy->at("thermodynamics");
            const double ea=a.at("electrical_energy_j"),eb=b.at("electrical_energy_j"),sa=a.at("committed_entropy_j_per_k"),sb=b.at("committed_entropy_j_per_k");
            const double endpoint_gap=std::abs(a.at("mechanical_energy_j").get<double>()-b.at("mechanical_energy_j").get<double>());
            const double contact_gap=std::abs(a.at("contact_work_j").get<double>()-b.at("contact_work_j").get<double>());
            const double energy_uncertainty=a.at("energy_accounting_uncertainty_j").get<double>()+b.at("energy_accounting_uncertainty_j").get<double>()+endpoint_gap+contact_gap;
            const double entropy_uncertainty=a.at("entropy_accounting_uncertainty_j_per_k").get<double>()+b.at("entropy_accounting_uncertainty_j_per_k").get<double>()+
                (endpoint_gap+contact_gap)/rotor::Params{}.ambient_temperature;
            row["pid"]={{"electrical_energy_j",ea},{"committed_entropy_j_per_k",sa}};row["entropy"]={{"electrical_energy_j",eb},{"committed_entropy_j_per_k",sb}};
            row["mechanical_endpoint_gap_j"]=endpoint_gap;row["contact_work_gap_j"]=contact_gap;
            row["energy_uncertainty_bound_j"]=energy_uncertainty;row["entropy_uncertainty_bound_j_per_k"]=entropy_uncertainty;
            if(comparable) {
                row["electrical_energy_saving_j"]=ea-eb;row["electrical_energy_saving_percent"]=100*(ea-eb)/ea;
                row["entropy_saving_j_per_k"]=sa-sb;row["entropy_saving_percent"]=100*(sa-sb)/sa;
                row["savings_demonstrated"]=(ea-eb)>energy_uncertainty&&(sa-sb)>entropy_uncertainty;
            }
        }
        output.push_back(std::move(row));
    }
    return output;
}

using TrialSpec=std::pair<const Scenario*,Mode>;
std::string short_mode(const Scenario& scenario,Mode mode) {
    return !scenario.autopilot?"disabled":mode==Mode::Entropy?"entropy":"pid";
}
std::string trial_key(const Scenario& scenario,Mode mode) {return scenario.id+"/"+short_mode(scenario,mode);}
std::vector<TrialSpec> selected_trials(const std::string& filter,const std::string& mode_filter) {
    if(!mode_filter.empty()&&mode_filter!="pid"&&mode_filter!="entropy")throw std::invalid_argument("--mode must be pid or entropy");
    bool found=filter.empty();for(const auto& scenario:scenarios)found=found||filter==scenario.id||filter==scenario.name;
    if(!found)throw std::invalid_argument("Unknown scenario: "+filter);
    std::vector<TrialSpec> result;
    for(const auto& scenario:scenarios) {
        if(!filter.empty()&&filter!=scenario.id&&filter!=scenario.name)continue;
        for(const auto mode:{Mode::Pid,Mode::Entropy}) {
            if(!scenario.autopilot&&mode==Mode::Entropy)continue;
            if(mode_filter=="pid"&&mode!=Mode::Pid)continue;
            if(mode_filter=="entropy"&&mode!=Mode::Entropy)continue;
            result.emplace_back(&scenario,mode);
        }
    }
    if(result.empty())throw std::invalid_argument("No trial matches the requested scenario/mode");
    return result;
}
Json report_header(const Json& model,const std::string& filter={},const std::string& mode_filter={}) {
    return {{"schema_version",3},{"model",model},{"simulator","MuJoCo plant with explicit electric rotor, thermal, and inflow-energy model"},
        {"controller","Paired airborne entropy NMPC with explicit PID landing support and PID baseline; exact simulated state; no hardware validation"},
        {"complete_evaluation",filter.empty()&&mode_filter.empty()},{"scenario_filter",filter},{"mode_filter",mode_filter},
        {"thresholds",{{"rms_error_m_max",.55},{"max_error_m_max",1.4},{"endpoint_error_m_max",.25},{"final_speed_m_s_max",.25},
            {"baseline_max_altitude_m_max",.30},{"baseline_rms_error_m_min",2},{"optimizer_accepted_fraction_min",accepted_fraction_min},
            {"optimizer_participation_denominator","All control decisions including initial trim observation and planned PID landing support"},
            {"optimizer_action_difference_rad_min",changed_action_min},{"physical_constraint_violation_max",constraint_tolerance},
            {"landing_from_s",68},{"landing_position_xy_m_max",.10},{"landing_position_z_m_max",.025},
            {"landing_velocity_component_m_s_max",.10},{"landing_body_rate_rad_s_max",.20},{"landing_tilt_deg_max",5},
            {"integrated_energy_balance_relative_max",balance_relative},{"integrated_energy_balance_absolute_j_max",energy_balance_absolute},
            {"integrated_entropy_balance_relative_max",balance_relative},{"integrated_entropy_balance_absolute_j_per_k_max",entropy_balance_absolute}}},
        {"reset_replay_scope","PID only; optimizer wall-time and finite-budget iterates are not assumed byte-identical"},
        {"savings_are_separate_from_test_pass",true},
        {"contact_boundary","Ground contact is an external mechanical port. Pairwise contact-work and mechanical endpoint differences enter conservative savings uncertainty; no ground-contact entropy is claimed."},
        {"timing_note","All control decisions include initial trim observation and planned PID landing support in participation. Solve, convergence and timing statistics count actual optimizer attempts only. The first 0.1 s hold is inside each enabled mission's clock and accounting. Deadline misses are separate; a passing simulated mission does not establish real-time operation."},
        {"scenarios",Json::array()}};
}
bool reset_replay() {
    helicopter::Simulation replay(47,1,true,Mode::Pid);for(int i=0;i<250;++i)replay.step();
    const Json first=replay.state();replay.reset(47,1,true,Mode::Pid);for(int i=0;i<250;++i)replay.step();
    return first==replay.state();
}
double number(const Json& object,const std::string& key) {
    const auto& value=object.at(key);
    if(!value.is_number()||!std::isfinite(value.get<double>()))throw std::invalid_argument("Missing finite numeric field: "+key);
    return value.get<double>();
}
unsigned count(const Json& object,const std::string& key,unsigned maximum=1000000) {
    const auto& value=object.at(key);
    if(!value.is_number_integer()||value.get<double>()<0||value.get<double>()>maximum)
        throw std::invalid_argument("Invalid counter: "+key);
    return value.get<unsigned>();
}
bool flag(const Json& object,const std::string& key) {
    if(!object.at(key).is_boolean())throw std::invalid_argument("Invalid boolean field: "+key);
    return object.at(key).get<bool>();
}
void validate_identity(const Json& row,const Scenario& scenario,Mode mode) {
    const std::string label=short_mode(scenario,mode);
    const Json identity{{"scenario_id",scenario.id},{"scenario_name",scenario.name},{"name",scenario.name+" / "+label},
        {"mode",label},{"controller_mode",mode==Mode::Entropy?"entropy_nmpc":"pid_baseline"},
        {"autopilot",scenario.autopilot},{"seed",scenario.seed},{"mass_scale",scenario.mass_scale},
        {"extra_gusts_seconds",scenario.extra_gusts}};
    for(const auto& item:identity.items())if(!row.contains(item.key())||row.at(item.key())!=item.value())
        throw std::invalid_argument("Scenario specification mismatch in "+trial_key(scenario,mode)+": "+item.key());
}

bool reaudit_parameters(Json& evidence,unsigned steps,double scenario_mass_scale,const Json& decisions) {
    const rotor::Params nominal;
    const auto policy=ParameterAudit{}.summary(nominal.mass*scenario_mass_scale);
    for(const char* key:{"truth_use","checkpoint_columns","audit_scope"})
        if(evidence.at(key)!=policy.at(key))throw std::invalid_argument(std::string("Parameter-estimation policy mismatch: ")+key);
    if(count(evidence,"observation_steps",7100)!=steps)throw std::invalid_argument("Parameter-estimation observation schedule mismatch");
    const unsigned accepted=count(evidence,"accepted_samples",steps),rejected=count(evidence,"rejected_samples",steps);
    if(accepted+rejected!=steps||count(evidence,"contact_rejections",steps)>rejected)
        throw std::invalid_argument("Parameter-estimation accepted/rejected counters disagree");
    const auto& checkpoints=evidence.at("checkpoints");
    const std::size_t expected=1+steps/10+(steps%10!=0);
    if(!checkpoints.is_array()||checkpoints.size()!=expected)throw std::invalid_argument("Missing parameter-estimation checkpoints");
    std::map<unsigned,double> mass_at_step;
    double prior_information=0,prior_cross=0,final_mass=nominal.mass,maximum_checkpoint_error=0;
    unsigned prior_accepted=0,prior_rejected=0,prior_step=0;
    for(std::size_t i=0;i<checkpoints.size();++i) {
        const auto& checkpoint=checkpoints[i];
        if(!checkpoint.is_array()||checkpoint.size()!=6)throw std::invalid_argument("Invalid parameter-estimation checkpoint shape");
        for(const auto& value:checkpoint)if(!value.is_number()||!std::isfinite(value.get<double>()))
            throw std::invalid_argument("Nonfinite parameter-estimation checkpoint");
        const unsigned step=std::min(static_cast<unsigned>(i)*10,steps);
        const double time=checkpoint[0].get<double>(),cross=checkpoint[1].get<double>(),information=checkpoint[2].get<double>();
        const double n=checkpoint[3].get<double>(),r=checkpoint[4].get<double>(),reported=checkpoint[5].get<double>();
        if(!near(time,step*dt,1e-8)||n<0||r<0||std::floor(n)!=n||std::floor(r)!=r||n+r!=step||
           n<prior_accepted||r<prior_rejected||information<prior_information||cross<prior_cross)
            throw std::invalid_argument("Noncausal parameter-estimation checkpoint counters or sufficient statistics");
        const unsigned new_accepted=static_cast<unsigned>(n)-prior_accepted;
        const double delta_information=information-prior_information,delta_cross=cross-prior_cross;
        if(n>step||r>step||new_accepted>step-prior_step||information<0||cross<0)
            throw std::invalid_argument("Invalid parameter-estimation checkpoint information");
        if(new_accepted==0) {
            if(delta_information!=0||delta_cross!=0)throw std::invalid_argument("Rejected samples changed parameter-estimation statistics");
        } else if(delta_information<=new_accepted*1e-8||
                  delta_cross/delta_information<.5*nominal.mass-1e-8||delta_cross/delta_information>1.5*nominal.mass+1e-8)
            throw std::invalid_argument("Parameter-estimation checkpoint contradicts sample admissibility");
        final_mass=n>0?cross/information:nominal.mass;
        if(!std::isfinite(final_mass)||final_mass<.5*nominal.mass-1e-9||final_mass>1.5*nominal.mass+1e-9)
            throw std::invalid_argument("Parameter-estimation checkpoint has an inadmissible estimate");
        maximum_checkpoint_error=std::max(maximum_checkpoint_error,std::abs(reported-final_mass));
        if(i==0&&(information!=0||cross!=0||reported!=nominal.mass))
            throw std::invalid_argument("Parameter estimation did not begin with the nominal prior");
        mass_at_step.emplace(step,reported);
        prior_accepted=static_cast<unsigned>(n);prior_rejected=static_cast<unsigned>(r);prior_step=step;
        prior_information=information;prior_cross=cross;
    }
    if(prior_accepted!=accepted||prior_rejected!=rejected||
       !near(number(evidence,"sum_q_squared"),prior_information,1e-6)||
       !near(number(evidence,"sum_q_dot_force"),prior_cross,1e-6)||
       !near(number(evidence,"final_mass_kg"),final_mass,1e-9)||
       !near(number(evidence,"final_mass_scale"),final_mass/nominal.mass,1e-9)||
       !near(number(evidence,"final_absolute_mass_error_kg"),std::abs(final_mass-nominal.mass*scenario_mass_scale),1e-9)||
       !near(number(evidence,"final_relative_mass_error"),std::abs(final_mass/(nominal.mass*scenario_mass_scale)-1),1e-9))
        throw std::invalid_argument("Parameter-estimation summary disagrees with checkpoints");
    for(const auto& decision:decisions)if(flag(decision,"optimizer_attempted")) {
        const double time=number(decision,"time");
        const unsigned step=static_cast<unsigned>(std::llround(time/dt));
        if(!near(time,step*dt,1e-8)||!mass_at_step.contains(step)||
           !near(number(decision,"observed_mass_scale"),mass_at_step.at(step)/nominal.mass,1e-9))
            throw std::invalid_argument("Optimizer decision used an estimate unavailable at its decision time");
    }
    for(const char* key:{"max_mass_discrepancy_kg","max_scale_discrepancy","max_nominal_force_discrepancy_n"})
        if(number(evidence,key)<0)throw std::invalid_argument(std::string("Negative parameter-estimation discrepancy: ")+key);
    if(maximum_checkpoint_error>number(evidence,"max_mass_discrepancy_kg")+1e-9)
        throw std::invalid_argument("Parameter-estimation maximum omits a checkpoint discrepancy");
    const bool passed=flag(evidence,"nominal_prior_verified")&&flag(evidence,"metadata_consistent")&&flag(evidence,"counters_consistent")&&
        number(evidence,"max_mass_discrepancy_kg")<=1e-9&&number(evidence,"max_scale_discrepancy")<=1e-9&&
        number(evidence,"max_nominal_force_discrepancy_n")<=1e-7&&maximum_checkpoint_error<=1e-9;
    evidence["passed"]=passed;
    return passed;
}

// Recompute gates from the recorded audit evidence, rather than accepting the
// input report's passed flag or its optimizer aggregate/counter claims. This is
// a report audit, not a new integration of the underlying flight trajectory.
void reaudit_trial(Json& row,const Scenario& scenario,Mode mode) {
    const unsigned steps=count(row,"steps",7100);
    for(const char* key:{"passed","finite","controls_bounded","telemetry_consistent","initial_dynamics_consistent",
        "input_slew_consistent","completed","touchdown","crashed","flight_criteria_passed","physical_constraints_passed","final_contact_active"})flag(row,key);
    for(const auto& [key,size]:std::vector<std::pair<std::string,std::size_t>>{
        {"final_position",3},{"final_velocity",3},{"final_quaternion",4},{"final_body_rates",3}}) {
        const auto& values=row.at(key);
        if(!values.is_array()||values.size()!=size)throw std::invalid_argument("Invalid terminal-state array: "+key);
        for(const auto& value:values)if(!value.is_number()||!std::isfinite(value.get<double>()))
            throw std::invalid_argument("Invalid terminal-state value: "+key);
    }
    const bool requires_optimizer=scenario.autopilot&&mode==Mode::Entropy;
    const auto old_solver=row.at("optimizer");
    const auto& records=old_solver.at("unique_solves");
    if(!records.is_array())throw std::invalid_argument("Optimizer unique_solves must be an array");
    SolveAudit audit;
    for(const auto& record:records) {
        Json diagnostic=record;
        diagnostic["available"]=true;
        if(count(record,"decision")!=count(record,"control_decisions")||count(record,"solve")!=count(record,"solves"))
            throw std::invalid_argument("Decision/solve index aliases disagree");
        count(record,"accepted_solves");count(record,"fallback_solves");count(record,"landing_support_decisions");
        count(record,"observation_hold_decisions");count(record,"iterations");
        const bool attempted=flag(record,"optimizer_attempted");
        flag(record,"fallback");flag(record,"prediction_empty");record.at("applied_controller").get<std::string>();
        const bool accepted=flag(record,"accepted"),converged=flag(record,"converged");
        const std::string status=record.at("status").get<std::string>();
        if(converged!=(status=="Solve_Succeeded"||status=="Solved_To_Acceptable_Level"))
            throw std::invalid_argument("Convergence flag disagrees with solver termination");
        for(const char* key:{"time","solve_ms"})
            if(number(record,key)<0)throw std::invalid_argument(std::string("Negative optimizer diagnostic: ")+key);
        if(attempted)for(const char* key:{"action_difference_norm","seed_constraint_violation","model_mass_scale","observed_mass_scale"})
            if(number(record,key)<0)throw std::invalid_argument(std::string("Negative optimizer diagnostic: ")+key);
        if(accepted) {
            for(const char* key:{"max_constraint_violation","dynamics_residual","objective_check_error",
                "intermediate_physical_violation","intermediate_planning_violation"})
                if(number(record,key)<0)throw std::invalid_argument(std::string("Negative accepted residual: ")+key);
            const double seed=number(record,"seed_objective_j_per_k"),value=number(record,"objective_j_per_k");
            if(!near(number(record,"objective_reduction_j_per_k"),seed-value,1e-8))
                throw std::invalid_argument("Objective reduction disagrees with recorded objectives");
        }
        if(!audit.observe(diagnostic))throw std::invalid_argument("Duplicate control-decision record");
    }
    if(!audit.counters_consistent)throw std::invalid_argument("Inconsistent optimizer/landing counters, applied controller, or decision timestamps");
    Json solver=audit.final_summary(requires_optimizer,steps);
    // All aggregates can be recomputed from the enriched unique-solve records.
    // Reject mismatches instead of silently concealing damaged input evidence.
    for(const auto& item:solver.items())if(item.key()!="unique_solves") {
        if(!old_solver.contains(item.key())||old_solver.at(item.key())!=item.value())
            throw std::invalid_argument("Optimizer aggregate mismatch: "+item.key());
    }
    row["optimizer"]=solver;
    const bool optimizer_ok=flag(solver,"participation_passed")&&flag(solver,"objective_optimization_passed")&&
        flag(solver,"decision_schedule_verified")&&flag(solver,"solve_schedule_verified")&&flag(solver,"observation_schedule_verified")&&
        audit.accepted_feasible&&audit.model_parameter_consistent;
    const bool parameters_ok=reaudit_parameters(row.at("parameter_estimation"),steps,scenario.mass_scale,records);

    const auto& handover=row.at("landing_handover");
    const bool active=flag(handover,"active");
    if(active!=(audit.landing_support>0))throw std::invalid_argument("Landing handover disagrees with decision records");
    bool unsettled=false;
    if(active) {
        const double time=number(handover,"time");
        const Vec3 position=handover.at("position").get<Vec3>(),velocity=handover.at("velocity").get<Vec3>(),rates=handover.at("body_rates").get<Vec3>();
        const double clearance=number(handover,"skid_clearance_m"),reference_clearance=number(handover,"reference_clearance_m");
        const double margin=number(handover,"guard_margin_m"),tilt=number(handover,"tilt_deg");
        if(time<58||!near(time,audit.landing_start_time,1e-8)||
           !near(margin,.15+.5*std::max(0.0,-velocity[2]),1e-8)||std::min(clearance,reference_clearance)>margin+1e-8)
            throw std::invalid_argument("Landing handover guard/time is inconsistent");
        bool settled=tilt<=5;
        for(int j=0;j<2;++j)settled=settled&&std::abs(position[j])<=.05&&std::abs(velocity[j])<=.05;
        for(int j=0;j<3;++j)settled=settled&&std::abs(rates[j])<=.1;
        if(flag(handover,"settled_at_handover")!=settled)throw std::invalid_argument("Landing settled diagnostic disagrees with handover state");
        unsettled=!settled;
    } else if(!handover.at("time").is_null()||!handover.at("settled_at_handover").is_null())
        throw std::invalid_argument("Inactive landing handover must have null time and settled status");
    if(flag(row,"unsettled_landing_handover")!=unsettled)throw std::invalid_argument("Unsettled handover flag mismatch");

    const auto& constraints=row.at("physical_constraints");
    const auto& residuals=constraints.at("max_by_constraint");
    if(!residuals.is_object()||residuals.empty())throw std::invalid_argument("Missing physical constraint residuals");
    double maximum=0;
    for(const auto& residual:residuals.items()) {
        const double value=number(residuals,residual.key());
        if(value<0)throw std::invalid_argument("Negative constraint maximum");
        maximum=std::max(maximum,value);
    }
    if(!near(maximum,number(constraints,"max_normalized_violation"),1e-10))
        throw std::invalid_argument("Physical constraint maximum disagrees with individual residuals");
    const unsigned violating_steps=count(constraints,"violating_steps",steps);
    if((maximum<=constraint_tolerance)!=(violating_steps==0))throw std::invalid_argument("Physical violation counter disagrees with residuals");
    if(number(constraints,"contact_center_height_min_m")!=.18||
       constraints.at("main_descent_envelope_scope")!="Airborne states; contact lies outside the free-flight predictor")
        throw std::invalid_argument("Physical contact policy mismatch");
    const bool physical=maximum<=constraint_tolerance;

    auto& thermo=row.at("thermodynamics");
    for(const char* key:{"electrical_energy_j","generated_entropy_j_per_k","committed_entropy_j_per_k",
        "terminal_commitment_j_per_k","initial_commitment_j_per_k","wind_work_j","contact_work_j","mechanical_energy_j",
        "initial_mechanical_energy_j","stored_wake_energy_j","stored_thermal_energy_j","energy_balance_residual_j",
        "entropy_balance_residual_j_per_k","trapezoid_electrical_energy_j","trapezoid_entropy_j_per_k"})number(thermo,key);
    for(const char* key:{"normalized_energy_balance_residual","normalized_entropy_balance_residual","max_normalized_motor_energy_residual",
        "max_instantaneous_energy_residual_w","energy_accounting_uncertainty_j","entropy_accounting_uncertainty_j_per_k"})
        if(number(thermo,key)<0)throw std::invalid_argument(std::string("Negative accounting diagnostic: ")+key);
    const double energy_uncertainty=std::abs(number(thermo,"energy_balance_residual_j"))+
        std::abs(number(thermo,"electrical_energy_j")-number(thermo,"trapezoid_electrical_energy_j"));
    const double entropy_uncertainty=std::abs(number(thermo,"entropy_balance_residual_j_per_k"))+
        std::abs(number(thermo,"generated_entropy_j_per_k")-number(thermo,"trapezoid_entropy_j_per_k"));
    if(!near(number(thermo,"energy_accounting_uncertainty_j"),energy_uncertainty,1e-8)||
       !near(number(thermo,"entropy_accounting_uncertainty_j_per_k"),entropy_uncertainty,1e-8)||
       !near(number(thermo,"committed_entropy_j_per_k"),number(thermo,"generated_entropy_j_per_k")+
          number(thermo,"terminal_commitment_j_per_k")-number(thermo,"initial_commitment_j_per_k"),1e-8))
        throw std::invalid_argument("Thermodynamic accounting summary mismatch");
    const bool thermodynamics=flag(thermo,"reported_integrals_verified")&&number(thermo,"max_instantaneous_energy_residual_w")<1e-6&&
        number(thermo,"normalized_energy_balance_residual")<=balance_relative&&
        number(thermo,"normalized_entropy_balance_residual")<=balance_relative&&
        number(thermo,"max_normalized_motor_energy_residual")<=balance_relative&&number(thermo,"minimum_entropy_generation_rate_w_per_k")>=-1e-10;

    auto& hold=row.at("initial_trim_observation");
    const unsigned held_steps=count(hold,"checked_steps",10),hold_samples=count(hold,"accepted_identification_samples",10);
    if(flag(hold,"required")!=scenario.autopilot||number(hold,"duration_s")!=.1||
       hold.at("accounting_scope")!="The hold is inside the 70-second mission clock and its energy/entropy integrals")
        throw std::invalid_argument("Initial observation-hold policy mismatch");
    const auto& initial_input=hold.at("initial_input_rad");
    if(!initial_input.is_array()||initial_input.size()!=4)throw std::invalid_argument("Invalid prepared-trim command array");
    for(const auto& value:initial_input)if(!value.is_number()||!std::isfinite(value.get<double>()))
        throw std::invalid_argument("Nonfinite prepared-trim command");
    const double hold_energy=number(hold,"electrical_energy_j"),hold_entropy=number(hold,"generated_entropy_j_per_k");
    const double input_error=number(hold,"max_command_deviation_rad");
    if(input_error<0||hold_energy<0||hold_entropy<0||hold_energy>number(thermo,"electrical_energy_j")+1e-8||
       hold_entropy>number(thermo,"generated_entropy_j_per_k")+1e-8)
        throw std::invalid_argument("Observation-hold accounting is outside the full mission integrals");
    if(!scenario.autopilot&&(hold_samples!=0||hold_energy!=0||hold_entropy!=0))
        throw std::invalid_argument("Disabled trial incorrectly received an initial observation hold");
    if(scenario.autopilot&&steps>0) {
        const auto& checkpoint=row.at("parameter_estimation").at("checkpoints").at(1);
        if(hold_samples!=checkpoint.at(3).get<unsigned>())throw std::invalid_argument("Initial hold identification count disagrees with the causal checkpoint");
    }
    const bool observation_hold_ok=near(number(hold,"start_time_s"),0,1e-9)&&input_error<=1e-12&&
        held_steps==(scenario.autopilot?std::min(steps,10U):0)&&near(number(hold,"end_time_s"),held_steps*dt,1e-8);
    hold["passed"]=observation_hold_ok;

    const auto& metrics=row.at("metrics");
    for(const char* key:{"rms_error","max_error","endpoint_error","final_speed","max_altitude"})
        if(number(metrics,key)<0)throw std::invalid_argument(std::string("Negative flight metric: ")+key);
    const bool completed=flag(row,"completed"),touchdown=flag(row,"touchdown"),crashed=flag(row,"crashed");
    if(number(thermo,"electrical_energy_j")<=0||number(thermo,"committed_entropy_j_per_k")<=0)
        throw std::invalid_argument("Positive mission energy and completed-process entropy are required for comparisons");
    const bool flight=scenario.autopilot?completed&&touchdown&&number(metrics,"rms_error")<.55&&number(metrics,"max_error")<1.4&&
        number(metrics,"endpoint_error")<.25&&number(metrics,"final_speed")<.25:
        !completed&&number(metrics,"rms_error")>2&&number(metrics,"max_altitude")<.30;
    const bool checks=flag(row,"finite")&&flag(row,"controls_bounded")&&flag(row,"telemetry_consistent")&&
        flag(row,"initial_dynamics_consistent")&&flag(row,"input_slew_consistent");
    row["flight_criteria_passed"]=flight;row["physical_constraints_passed"]=physical;thermo["passed"]=thermodynamics;
    row["passed"]=checks&&thermodynamics&&physical&&flight&&optimizer_ok&&parameters_ok&&observation_hold_ok&&steps==7000&&!crashed;
}

using MergeInput=std::pair<std::string,Json>;
Json merge_reports(const std::vector<MergeInput>& inputs,const Json& model) {
    if(inputs.empty())throw std::invalid_argument("--merge requires report files");
    const Json expected_header=report_header(model);
    const auto required=selected_trials("","");
    std::map<std::string,TrialSpec> specifications;
    for(const auto& spec:required)specifications.emplace(trial_key(*spec.first,spec.second),spec);
    std::map<std::string,Json> collected;
    Json input_provenance=Json::array();bool input_replays=true;
    for(const auto& [path,input]:inputs) {
        if(!input.is_object()||!finite_json(input))throw std::invalid_argument("Invalid/nonfinite merge report: "+path);
        if(input.at("schema_version")!=3)throw std::invalid_argument("Report schema mismatch: "+path);
        if(input.at("model")!=model)throw std::invalid_argument("Report model/source mismatch (stale or different build): "+path);
        for(const char* key:{"thresholds","simulator","controller","reset_replay_scope","savings_are_separate_from_test_pass","contact_boundary","timing_note"})
            if(input.at(key)!=expected_header.at(key))throw std::invalid_argument(std::string("Report policy mismatch: ")+key+" in "+path);
        const std::string filter=input.at("scenario_filter").get<std::string>(),mode_filter=input.at("mode_filter").get<std::string>();
        if(flag(input,"complete_evaluation")!=(filter.empty()&&mode_filter.empty()))throw std::invalid_argument("Report completeness/filter mismatch: "+path);
        const auto declared=selected_trials(filter,mode_filter);
        std::set<std::string> remaining;for(const auto& spec:declared)remaining.insert(trial_key(*spec.first,spec.second));
        const auto& rows=input.at("scenarios");if(!rows.is_array())throw std::invalid_argument("Report scenarios must be an array: "+path);
        if(!input.at("comparisons").is_array())throw std::invalid_argument("Report comparisons must be an array: "+path);
        Json input_keys=Json::array();
        for(const auto& row:rows) {
            const std::string key=row.at("scenario_id").get<std::string>()+"/"+row.at("mode").get<std::string>();
            const auto spec=specifications.find(key);
            if(spec==specifications.end())throw std::invalid_argument("Unknown trial in merge: "+key);
            validate_identity(row,*spec->second.first,spec->second.second);
            if(collected.contains(key))throw std::invalid_argument("Duplicate trial in merge: "+key);
            if(!remaining.erase(key))throw std::invalid_argument("Trial does not match shard filters: "+key);
            collected.emplace(key,row);input_keys.push_back(key);
        }
        if(!remaining.empty())throw std::invalid_argument("Missing declared trial in shard: "+*remaining.begin()+" in "+path);
        const bool replay=flag(input,"reset_replay_identical");input_replays=input_replays&&replay;
        input_provenance.push_back({{"path",path},{"trial_keys",input_keys},{"reported_passed",flag(input,"passed")},
            {"reset_replay_identical",replay},{"source_provenance",input.at("model").at("provenance")}});
    }
    for(const auto& spec:required) {
        const auto key=trial_key(*spec.first,spec.second);
        if(!collected.contains(key))throw std::invalid_argument("Missing required trial in merge: "+key);
    }
    if(collected.size()!=required.size())throw std::invalid_argument("Merge coverage count mismatch");
    Json report=report_header(model);bool passed=input_replays;
    for(const auto& spec:required) {
        auto row=collected.at(trial_key(*spec.first,spec.second));
        try {reaudit_trial(row,*spec.first,spec.second);}
        catch(const std::exception& error){throw std::invalid_argument(trial_key(*spec.first,spec.second)+": "+error.what());}
        passed=passed&&flag(row,"passed");report["scenarios"].push_back(std::move(row));
    }
    // Re-run the cheap deterministic replay in this process, never merely copy
    // a shard's reset result. Failed shard replays also remain failures.
    const bool current_replay=reset_replay();
    report["reset_replay_identical"]=input_replays&&current_replay;
    report["comparisons"]=comparisons(report.at("scenarios"));report["passed"]=passed&&current_replay;
    report["merge"]={{"method","Exact-coverage native report audit; no new flight integrations"},
        {"inputs",input_provenance},{"required_trial_count",required.size()},{"scenario_count",scenarios.size()},
        {"source_matches_current_binary",true},{"current_process_reset_replay_identical",current_replay},
        {"pass_policy","Recomputed trial, optimizer-counter, feasibility and objective-contribution gates; input top-level passed flags are provenance only"}};
    return report;
}

Json merge_self_test(const Json& model) {
    // These deliberately incomplete synthetic identities test rejection paths;
    // they cannot constitute a valid flight report or pass the deep row audit.
    auto partial=report_header(model,"nominal","pid");
    const auto& scenario=scenarios.front();
    partial["passed"]=false;partial["reset_replay_identical"]=true;partial["comparisons"]=Json::array();
    partial["scenarios"].push_back({{"scenario_id",scenario.id},{"scenario_name",scenario.name},{"name",scenario.name+" / pid"},
        {"mode","pid"},{"controller_mode","pid_baseline"},{"autopilot",true},{"seed",scenario.seed},
        {"mass_scale",scenario.mass_scale},{"extra_gusts_seconds",scenario.extra_gusts}});
    Json results=Json::array();
    const auto rejects=[&](const std::string& name,const std::vector<MergeInput>& inputs,const std::string& expected) {
        try {const auto unexpected=merge_reports(inputs,model);(void)unexpected;throw std::runtime_error("Invalid merge was accepted: "+name);}
        catch(const std::invalid_argument& error) {
            if(std::string(error.what()).find(expected)==std::string::npos)throw std::runtime_error("Wrong rejection for "+name+": "+error.what());
            results.push_back({{"name",name},{"passed",true},{"rejection",error.what()}});
        }
    };
    rejects("incomplete coverage",{{"synthetic-partial",partial}},"Missing required trial");
    rejects("duplicate trial",{{"synthetic-one",partial},{"synthetic-two",partial}},"Duplicate trial");
    auto stale=partial;stale["model"]["provenance"]["evaluator_source_sha256"]=std::string(64,'0');
    rejects("stale source",{{"synthetic-stale",stale}},"model/source mismatch");
    auto mismatch=partial;mismatch["scenarios"][0]["mass_scale"]=1.2;
    rejects("scenario specification mismatch",{{"synthetic-wrong-mass",mismatch}},"Scenario specification mismatch");
    const auto hybrid_fixture=[](unsigned accepted_count) {
        SolveAudit audit;
        for(unsigned decision=1;decision<=10;++decision) {
            const bool observation=decision==1;
            const bool attempted=!observation&&decision<=accepted_count+1;
            const unsigned solves=std::min(decision-1,accepted_count),landing=decision-1-solves;
            Json d{{"available",true},{"control_decisions",decision},{"solves",solves},{"accepted_solves",solves},{"fallback_solves",0},
                {"landing_support_decisions",landing},{"observation_hold_decisions",1},{"time",(decision-1)*helicopter::EntropyController::update_period},
                {"optimizer_attempted",attempted},{"accepted",attempted},{"converged",attempted},{"fallback",!attempted},
                {"applied_controller",attempted?"entropy_nmpc":observation?"trim_observation_hold":"pid_landing_support"},
                {"status",attempted?"Solve_Succeeded":observation?"observation_hold":"landing_support"},
                {"iterations",attempted?1:0},{"solve_ms",attempted?10.0:0.0},
                {"prediction_empty",!attempted},{"model_mass_scale",attempted?Json(1.0):Json()},
                {"observed_mass_scale",attempted?Json(1.0):Json()},
                {"action_difference_norm",nullptr},{"seed_constraint_violation",nullptr},
                {"seed_objective_j_per_k",nullptr},{"objective_j_per_k",nullptr},{"objective_reduction_j_per_k",nullptr},
                {"max_constraint_violation",nullptr},{"dynamics_residual",nullptr},{"objective_check_error",nullptr},
                {"intermediate_physical_violation",nullptr},{"intermediate_planning_violation",nullptr}};
            if(attempted)d.update({{"action_difference_norm",.01},{"seed_constraint_violation",0.0},
                {"seed_objective_j_per_k",.5},{"objective_j_per_k",.4},{"objective_reduction_j_per_k",.1},
                {"max_constraint_violation",0.0},{"dynamics_residual",0.0},{"objective_check_error",0.0},
                {"intermediate_physical_violation",0.0},{"intermediate_planning_violation",0.0}});
            if(!audit.observe(d))throw std::runtime_error("Synthetic control decision was not observed");
        }
        return audit;
    };
    const auto eight=hybrid_fixture(8);const auto eight_summary=eight.final_summary(true,100);
    const auto nine=hybrid_fixture(9);const auto nine_summary=nine.final_summary(true,100);
    if(!eight.counters_consistent||!nine.counters_consistent||flag(eight_summary,"participation_passed")||
       !flag(nine_summary,"participation_passed")||!near(number(eight_summary,"accepted_fraction"),.8)||
       !near(number(eight_summary,"attempt_acceptance_fraction"),1)||!near(number(eight_summary,"mean_solve_ms"),10)||
       count(eight_summary,"solves")!=8||count(eight_summary,"control_decisions")!=10||count(eight_summary,"landing_support_decisions")!=1||
       count(eight_summary,"observation_hold_decisions")!=1||count(eight_summary,"fallback_decisions")!=2||
       count(eight_summary,"pid_decisions")!=1||!flag(nine_summary,"observation_schedule_verified")||
       count(nine_summary,"expected_solves")!=9)
        throw std::runtime_error("Hybrid participation incorrectly excludes observation/landing support or includes them in solve timing");
    results.push_back({{"name","initial observation and planned landing remain in participation denominator"},{"passed",true},
        {"eight_of_ten_passes",false},{"nine_of_ten_passes",true},{"actual_attempt_mean_ms",10}});
    SolveAudit replay;
    for(auto record:eight.records) {
        record["available"]=true;
        if(record.at("decision")==10)record["landing_support_decisions"]=0;
        replay.observe(record);
    }
    if(replay.counters_consistent)throw std::runtime_error("Corrupt planned-landing counter was accepted");
    results.push_back({{"name","corrupt landing counter rejected"},{"passed",true}});
    SolveAudit repeated_hold;
    Json first_hold=eight.records.at(0);first_hold["available"]=true;
    repeated_hold.observe(first_hold);
    auto later_hold=first_hold;later_hold["decision"]=2;later_hold["control_decisions"]=2;
    later_hold["time"]=.1;later_hold["observation_hold_decisions"]=2;
    repeated_hold.observe(later_hold);
    if(repeated_hold.counters_consistent)throw std::runtime_error("Repeated or late observation hold was accepted");
    results.push_back({{"name","observation hold is permitted only once at the initial decision"},{"passed",true}});
    SolveAudit nonnull_hold;
    auto invented_objective=first_hold;invented_objective["objective_j_per_k"]=0.0;
    nonnull_hold.observe(invented_objective);
    if(nonnull_hold.counters_consistent)throw std::runtime_error("No-attempt observation hold carried an invented objective");
    results.push_back({{"name","observation hold has no optimizer candidate or objective"},{"passed",true}});
    SolveAudit wrong_parameter;
    for(auto record:eight.records) {
        record["available"]=true;
        if(record.at("decision")==2)record["model_mass_scale"]=.8;
        wrong_parameter.observe(record);
    }
    if(wrong_parameter.model_parameter_consistent)throw std::runtime_error("Wrong optimizer mass parameter was accepted");
    results.push_back({{"name","optimizer parameter must match prior observed estimate"},{"passed",true}});

    auto identification=ParameterAudit{}.summary(1.6);
    const double information=10*9.81*9.81,cross=1.6*information;
    identification.update({{"observation_steps",10},{"accepted_samples",10},{"sum_q_squared",information},{"sum_q_dot_force",cross},
        {"final_mass_kg",1.6},{"final_mass_scale",.8},{"final_absolute_mass_error_kg",0.0},{"final_relative_mass_error",0.0},
        {"checkpoints",Json::array({Json::array({0,0,0,0,0,2.0}),Json::array({.1,cross,information,10,0,1.6})})}});
    Json decisions=Json::array({{{"time",0.0},{"optimizer_attempted",true},{"observed_mass_scale",1.0}}});
    if(!reaudit_parameters(identification,10,.8,decisions))throw std::runtime_error("Valid synthetic identification checkpoints were rejected");
    const auto rejects_parameter=[&](const std::string& name,Json evidence,Json observations,const std::string& expected_error) {
        try {reaudit_parameters(evidence,10,.8,observations);throw std::runtime_error("Invalid parameter evidence was accepted: "+name);}
        catch(const std::invalid_argument& error) {
            if(std::string(error.what()).find(expected_error)==std::string::npos)throw;
            results.push_back({{"name",name},{"passed",true},{"rejection",error.what()}});
        }
    };
    auto future_parameter=decisions;future_parameter[0]["observed_mass_scale"]=.8;
    rejects_parameter("future mass estimate cannot inform an earlier decision",identification,future_parameter,"unavailable at its decision time");
    auto corrupt_checkpoint=identification;corrupt_checkpoint["checkpoints"][1][5]=2.4;
    rejects_parameter("reported mass must match cumulative least squares",corrupt_checkpoint,decisions,"maximum omits a checkpoint discrepancy");
    auto rejected_change=identification;rejected_change["checkpoints"][1][3]=0;rejected_change["checkpoints"][1][4]=10;
    rejects_parameter("rejected samples cannot change identification statistics",rejected_change,decisions,"Rejected samples changed");
    return {{"passed",true},{"test","Native report merger rejection paths"},{"synthetic_fixtures",true},
        {"validation_evidence",false},{"note","Structural tests only; these fixtures are not simulated-flight evidence"},{"checks",results}};
}
} // namespace

int main(int argc,char** argv) {
    try {
        std::string output,filter,mode_filter;
        std::vector<std::string> merge_paths;
        bool test_merge=false,scenario_option=false,mode_option=false;
        const std::string usage="Usage: helicopter-evaluate [--scenario NAME] [--mode pid|entropy] [--output report.json] | --merge report.json ... [--output merged.json] | --test-merge";
        for(int i=1;i<argc;++i) {
            const std::string option=argv[i];
            if((option=="--output"||option=="--scenario"||option=="--mode")&&i+1<argc) {
                const std::string value=argv[++i];
                if(value.starts_with("--"))throw std::invalid_argument(usage);
                if(option=="--output")output=value;
                else if(option=="--scenario"){filter=value;scenario_option=true;}
                else {mode_filter=value;mode_option=true;}
            } else if(option=="--merge") {
                const auto before=merge_paths.size();
                while(i+1<argc&&!std::string(argv[i+1]).starts_with("--"))merge_paths.emplace_back(argv[++i]);
                if(merge_paths.size()==before)throw std::invalid_argument("--merge requires report files");
            } else if(option=="--test-merge")test_merge=true;
            else throw std::invalid_argument(usage);
        }
        if(!merge_paths.empty()&&(scenario_option||mode_option))throw std::invalid_argument("--merge cannot be combined with --scenario or --mode");
        if(test_merge&&(!merge_paths.empty()||scenario_option||mode_option||!output.empty()))
            throw std::invalid_argument("--test-merge is standalone and does not write flight reports");
        const auto model=helicopter::Simulation::model();
        if(test_merge){std::cout<<merge_self_test(model).dump(2)<<'\n';return 0;}
        if(!finite_json(model)||model.at("provenance").at("model_source_sha256").get<std::string>().size()!=64||
           model.at("provenance").at("evaluator_source_sha256").get<std::string>().size()!=64)
            throw std::runtime_error("Invalid current model provenance");
        Json report;
        if(!merge_paths.empty()) {
            std::vector<MergeInput> inputs;
            const auto output_path=output.empty()?std::filesystem::path{}:std::filesystem::weakly_canonical(output);
            const auto temporary_path=output.empty()?std::filesystem::path{}:std::filesystem::weakly_canonical(output+".tmp");
            for(const auto& path:merge_paths) {
                const auto input_path=std::filesystem::weakly_canonical(path);
                if(!output.empty()&&(input_path==output_path||input_path==temporary_path))
                    throw std::invalid_argument("Merge output must not replace an input report: "+path);
                std::ifstream input(path);if(!input)throw std::runtime_error("Cannot open merge report: "+path);
                Json document=Json::parse(input);
                inputs.emplace_back(path,std::move(document));
            }
            report=merge_reports(inputs,model);
        } else {
            report=report_header(model,filter,mode_filter);bool passed=true;
            for(const auto& [scenario,mode]:selected_trials(filter,mode_filter)) {
                auto result=run(*scenario,mode);passed=passed&&flag(result,"passed");report["scenarios"].push_back(std::move(result));
            }
            // Only PID is deterministic by contract. Finite CPU-budget optimizer
            // iterates and wall-time diagnostics are not byte-compared.
            const bool deterministic=reset_replay();report["reset_replay_identical"]=deterministic;
            report["comparisons"]=comparisons(report.at("scenarios"));report["passed"]=passed&&deterministic;
        }
        const std::string text=report.dump(2)+"\n";
        if(!output.empty()) {
            const std::string temporary=output+".tmp";
            {std::ofstream file(temporary);if(!file)throw std::runtime_error("Cannot open output: "+temporary);file<<text;file.close();if(!file)throw std::runtime_error("Cannot write output: "+temporary);}
            std::filesystem::rename(temporary,output);
        }
        std::cout<<text;return report.at("passed").get<bool>()?0:1;
    }catch(const std::exception& error){std::cerr<<"Evaluation failed: "<<error.what()<<'\n';return 2;}
}
