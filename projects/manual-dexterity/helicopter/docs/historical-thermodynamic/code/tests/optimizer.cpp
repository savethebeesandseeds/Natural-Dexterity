#include "entropy_controller.hpp"
#include "flight_constraints.hpp"
#include "mass_scaled_dynamics.hpp"
#include <casadi/casadi.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using helicopter::EntropyController;
using helicopter::EntropyResult;
namespace rotor=helicopter::rotor;

void require(bool condition,const std::string& message) {
  if(!condition)throw std::runtime_error(message);
}
bool finite_json(const nlohmann::json& value) {
  if(value.is_number())return std::isfinite(value.get<double>());
  if(value.is_array()||value.is_object())
    for(const auto& child:value)if(!finite_json(child))return false;
  return true;
}
double number(const nlohmann::json& object,const std::string& key) {
  require(object.contains(key)&&object[key].is_number(),"Missing numeric optimizer diagnostic: "+key);
  const double result=object[key].get<double>();
  require(std::isfinite(result),"Nonfinite optimizer diagnostic: "+key);
  return result;
}

// Re-integrate the command that will actually drive the vehicle, independently
// of the optimizer's state decision variables and its reported feasibility.
double check_first_action(const EntropyResult& result,const rotor::State<double>& initial,
                         const rotor::Input<double>& previous,const rotor::Vec3<double>& wind,
                         const rotor::Params& p,double tolerance) {
  require(result.accepted,"Cannot audit an unaccepted action as an optimized action");
  for(int j=0;j<rotor::NU;++j) {
    require(std::isfinite(result.command[j]),"Accepted command is not finite");
    require(result.command[j]>=p.input_min[j]-1e-10&&result.command[j]<=p.input_max[j]+1e-10,
            "Accepted command is outside physical pitch bounds");
    require(std::abs(result.command[j]-previous[j])<=p.input_rate_max[j]*EntropyController::interval+1e-8,
            "Accepted command violates its first-interval pitch slew limit");
  }
  auto state=initial;
  double worst=0;
  for(int sample=0;sample<=5;++sample) {
    const auto violation=rotor::physical_violation(state,result.command,wind,p);
    require(violation.finite,"Accepted action produces nonfinite physical state");
    worst=std::max(worst,violation.maximum);
    require(violation.maximum<=tolerance,"Accepted action violates "+violation.worst);
    const auto rotation=rotor::detail::rotation(state);
    require(std::cos(EntropyController::planning_tilt_max_rad)-rotation[8]<=tolerance,
            "Accepted action violates the internal tilt margin between control knots");
    const double airspeed=std::hypot(state[3]-wind[0],state[4]-wind[1],state[5]-wind[2]);
    require(airspeed*airspeed/(EntropyController::planning_airspeed_max_m_s*
            EntropyController::planning_airspeed_max_m_s)-1<=tolerance,
            "Accepted action violates the internal airspeed margin between control knots");
    if(sample<5)state=rotor::integrate_rk4(state,result.command,wind,p,EntropyController::interval/5);
  }
  return worst;
}

void check_scaled_derivatives() {
  const rotor::Params nominal;
  const auto symbolic_state=casadi::SX::sym("state",rotor::NX);
  const auto symbolic_input=casadi::SX::sym("input",rotor::NU);
  const auto symbolic_wind=casadi::SX::sym("wind",3);
  const auto symbolic_ratio=casadi::SX::sym("ratio");
  rotor::State<casadi::SX> sx;rotor::Input<casadi::SX> su;rotor::Vec3<casadi::SX> sw;
  for(int j=0;j<rotor::NX;++j)sx[j]=symbolic_state(j);
  for(int j=0;j<rotor::NU;++j)su[j]=symbolic_input(j);
  for(int j=0;j<3;++j)sw[j]=symbolic_wind(j);
  const auto scaled=rotor::mass_scaled_derivative(sx,su,sw,nominal,symbolic_ratio);
  const casadi::Function symbolic("scaled_derivative_test",
    {symbolic_state,symbolic_input,symbolic_wind,symbolic_ratio},
    {casadi::SX::vertcat(std::vector<casadi::SX>(scaled.begin(),scaled.end()))});
  for(double ratio:{.5,.8,1.0,1.2,1.5}) {
    auto parameters=nominal;parameters.mass*=ratio;
    for(auto& inertia:parameters.inertia)inertia*=ratio;
    auto state=rotor::initial_state(parameters);state[2]=2.5;
    // Nonzero, unequal body rates exercise the gyroscopic correction: dividing
    // the complete nominal angular derivative by the ratio would fail here.
    state[10]=.31;state[11]=-.27;state[12]=.19;
    state[3]=.4;state[4]=-.2;state[5]=.1;
    state[7]=.03;state[8]=-.02;state[9]=.01;
    state[6]=std::sqrt(1-state[7]*state[7]-state[8]*state[8]-state[9]*state[9]);
    const rotor::Input<double> input{state[13]+.002,.003,-.004,state[16]-.001};
    const rotor::Vec3<double> wind{.3,-.15,.02};
    const auto expected=rotor::derivative(state,input,wind,parameters);
    const auto actual=rotor::mass_scaled_derivative(state,input,wind,nominal,ratio);
    const auto symbolic_actual=symbolic(std::vector<casadi::DM>{
      casadi::DM(std::vector<double>(state.begin(),state.end())),
      casadi::DM(std::vector<double>(input.begin(),input.end())),
      casadi::DM(std::vector<double>(wind.begin(),wind.end())),casadi::DM(ratio)})[0].nonzeros();
    for(int j=0;j<rotor::NX;++j) {
      const double tolerance=1e-10*std::max(1.0,std::abs(expected[j]));
      require(std::abs(actual[j]-expected[j])<=tolerance,"Mass-scaled numeric derivative disagrees with ordinary scaled parameters");
      require(std::abs(symbolic_actual[j]-expected[j])<=tolerance,"Mass-scaled symbolic derivative disagrees with ordinary scaled parameters");
    }
  }
}
}

int main() {
  try {
    check_scaled_derivatives();
    const rotor::Params p;
    auto initial=rotor::initial_state(p);
    // The same mechanical trim at mission hover altitude avoids ground contact.
    initial[2]=2.5;
    const rotor::Input<double> previous{initial[13],initial[14],initial[15],initial[16]};
    const rotor::Vec3<double> wind{};
    const rotor::Reference hold{{initial[0],initial[1],initial[2]},{},{}};
    const std::vector<rotor::Reference> reference(EntropyController::horizon_steps+1,hold);
    const double tolerance=EntropyController::description().at("feasibility_tolerance").get<double>();
    require(rotor::physical_violation(initial,previous,wind,p).maximum==0,"Nominal test fixture is not physically feasible");
    rotor::BaselineController pid;
    pid.reset(previous);
    const auto baseline=pid.command(initial,hold,wind,p,EntropyController::interval);
    EntropyController controller(p);
    const auto observation=controller.observation_hold(0.0);
    const auto& observation_diagnostics=observation.diagnostics;
    require(!observation.accepted&&observation_diagnostics.at("fallback").get<bool>()&&
            !observation_diagnostics.at("optimizer_attempted").get<bool>()&&
            !observation_diagnostics.at("converged").get<bool>(),"Observation hold fabricated optimized control");
    require(observation_diagnostics.at("status")=="observation_hold"&&
            observation_diagnostics.at("applied_controller")=="trim_observation_hold", "Observation hold lacks its explicit controller identity");
    require(observation_diagnostics.at("solves")==0&&observation_diagnostics.at("control_decisions")==1&&
            observation_diagnostics.at("observation_hold_decisions")==1&&observation_diagnostics.at("landing_support_decisions")==0,
            "Observation hold was omitted from participation or counted as an optimizer attempt");
    require(observation_diagnostics.at("prediction").empty()&&observation_diagnostics.at("objective_j_per_k").is_null()&&
            observation_diagnostics.at("intermediate_planning_violation").is_null()&&
            observation_diagnostics.at("solve_ms")==0&&observation_diagnostics.at("iterations")==0,
            "Observation hold fabricated solver diagnostics");
    controller.reset();
    nlohmann::json nominal_attempts=nlohmann::json::array();
    EntropyResult selected;
    bool meaningful_solution=false;
    double independently_checked_violation=0;
    unsigned accepted_count=0,fallback_count=0;
    // Finite-budget solves may stop differently across hosts. Three attempts is
    // a fixed allowance, not a relaxation of any physical or contribution test.
    for(unsigned attempt=1;attempt<=3&&!meaningful_solution;++attempt) {
      auto result=controller.solve(initial,previous,wind,reference,10.0);
      const auto& d=result.diagnostics;
      require(finite_json(d),"Nominal diagnostics contain nonfinite numeric data");
      require(d.at("accepted").get<bool>()==result.accepted,"Acceptance flag disagrees with the returned result");
      require(d.at("fallback").get<bool>()!=result.accepted,"Fallback flag disagrees with acceptance");
      result.accepted?++accepted_count:++fallback_count;
      require(d.at("solves").get<unsigned>()==attempt&&
              d.at("accepted_solves").get<unsigned>()==accepted_count&&
              d.at("fallback_solves").get<unsigned>()==fallback_count,"Nominal solve counters are inconsistent");
      nominal_attempts.push_back(d);
      if(!result.accepted) {
        require(d.at("prediction").empty(),"Rejected nominal result exposes an accepted prediction");
        continue;
      }
      require(number(d,"max_constraint_violation")<=tolerance,"Accepted solve exceeds its original constraint tolerance");
      require(number(d,"intermediate_physical_violation")<=tolerance,"Accepted solve violates intermediate physical bounds");
      require(number(d,"intermediate_planning_violation")<=tolerance,"Accepted solve violates intermediate planning margins");
      require(number(d,"dynamics_residual")<=tolerance,"Accepted solve exceeds its shooting-defect tolerance");
      require(number(d,"seed_constraint_violation")<=tolerance,"Nominal entropy comparison requires a feasible seed");
      const double seed=number(d,"seed_objective_j_per_k"),objective=number(d,"objective_j_per_k");
      const double objective_error=number(d,"objective_check_error");
      require(objective>=0&&seed>0,"Entropy objective has an invalid physical sign");
      require(objective_error<=1e-4,"Independent entropy quadrature disagrees with the NLP objective");
      require(std::abs(number(d,"objective_reduction_j_per_k")-(seed-objective))<1e-10,
              "Reported entropy reduction does not match the physical objectives");
      double difference=0;
      for(int j=0;j<rotor::NU;++j)difference+=std::pow(result.command[j]-baseline[j],2);
      difference=std::sqrt(difference);
      require(std::abs(number(d,"action_difference_norm")-difference)<1e-10,
              "Reported optimized action change disagrees with an independent PID calculation");
      independently_checked_violation=check_first_action(result,initial,previous,wind,p,tolerance);
      const auto& prediction=d.at("prediction");
      require(prediction.size()==EntropyController::horizon_steps+1,"Accepted prediction has the wrong horizon length");
      for(std::size_t k=0;k<prediction.size();++k) {
        require(std::abs(prediction[k].at("time").get<double>()-(10+k*EntropyController::interval))<1e-9,
                "Prediction timestamps do not match the solved horizon");
        for(int j=0;j<3;++j) {
          const double tube=k==EntropyController::horizon_steps?.10:j==2?.20:.4;
          require(std::abs(prediction[k]["position"][j].get<double>()-hold.position[j])<=tube*(1+tolerance),
                  "Accepted prediction violates the declared tracking/terminal position tolerance");
        }
        for(const auto& temperature:prediction[k].at("temperature_k"))
          require(temperature.get<double>()>=p.ambient_temperature-tolerance*(p.temperature_max-p.ambient_temperature)&&
                  temperature.get<double>()<=p.temperature_max+tolerance*(p.temperature_max-p.ambient_temperature),
                  "Accepted prediction violates motor temperature bounds");
      }
      meaningful_solution=difference>1e-6&&seed-objective>std::max(1e-5,10*objective_error);
      if(meaningful_solution)selected=std::move(result);
    }
    if(!meaningful_solution) std::cerr<<nominal_attempts.dump(2)<<'\n';
    require(meaningful_solution,"No independently feasible, entropy-reducing action distinct from PID was obtained in three solves");

    // The fixed initial temperature cannot fall from 500 K to the allowed range
    // in one integration interval. A good optimizer must reject this trajectory.
    auto hot=initial;hot[21]=500;hot[22]=500;
    require(rotor::physical_violation(hot,previous,wind,p).maximum>0,"Overtemperature fixture is unexpectedly feasible");
    const auto rejected=controller.solve(hot,previous,wind,reference,10.0);
    require(!rejected.accepted,"Impossible overtemperature initial condition was accepted");
    require(rejected.diagnostics.at("fallback").get<bool>(),"Rejected overtemperature solve did not select fallback");
    require(rejected.diagnostics.at("prediction").empty(),"Rejected overtemperature solve retained an accepted prediction");
    require(!rejected.diagnostics.at("fallback_reason").get<std::string>().empty(),"Fallback lacks a reason");
    require(finite_json(rejected.diagnostics),"Overtemperature rejection emitted nonfinite diagnostics");

    const auto support=controller.landing_support(10.1);
    require(!support.accepted&&support.diagnostics.at("fallback").get<bool>(),"Landing support was reported as optimized control");
    require(!support.diagnostics.at("optimizer_attempted").get<bool>()&&support.diagnostics.at("prediction").empty(),
            "Landing support fabricated an optimizer attempt or prediction");
    require(support.diagnostics.at("applied_controller")=="pid_landing_support", "Landing support lacks its explicit controller identity");
    require(support.diagnostics.at("solves")==rejected.diagnostics.at("solves"),"Landing support inflated actual solve count");
    require(support.diagnostics.at("control_decisions").get<unsigned>()==rejected.diagnostics.at("control_decisions").get<unsigned>()+1&&
            support.diagnostics.at("landing_support_decisions").get<unsigned>()==1,"Landing support disappeared from control participation");
    require(support.diagnostics.at("objective_j_per_k").is_null(),"Landing support fabricated an optimized objective");

    controller.reset();
    const auto after_reset=controller.solve(initial,previous,wind,reference,10.0);
    const auto& reset=after_reset.diagnostics;
    require(reset.at("solves").get<unsigned>()==1,"Reset did not clear solve count");
    require(reset.at("control_decisions").get<unsigned>()==1&&reset.at("landing_support_decisions").get<unsigned>()==0&&
            reset.at("observation_hold_decisions").get<unsigned>()==0,
            "Reset did not clear observation/landing-support/control-decision counts");
    require(reset.at("accepted_solves").get<unsigned>()==(after_reset.accepted?1u:0u)&&
            reset.at("fallback_solves").get<unsigned>()==(after_reset.accepted?0u:1u),
            "Reset did not clear acceptance/fallback counters");
    if(after_reset.accepted)check_first_action(after_reset,initial,previous,wind,p,tolerance);
    else require(reset.at("prediction").empty(),"Rejected post-reset solve retained a prediction");

    nlohmann::json scaled_attempts=nlohmann::json::array();
    for(double ratio:{.8,1.2}) {
      auto parameters=p;parameters.mass*=ratio;
      for(auto& inertia:parameters.inertia)inertia*=ratio;
      auto state=rotor::initial_state(parameters);state[2]=2.5;
      const rotor::Input<double> input{state[13],state[14],state[15],state[16]};
      const rotor::Reference target{{state[0],state[1],state[2]},{},{}};
      const std::vector<rotor::Reference> refs(EntropyController::horizon_steps+1,target);
      controller.reset();bool accepted=false;
      for(int attempt=0;attempt<3&&!accepted;++attempt) {
        const auto result=controller.solve(state,input,wind,refs,10.0,ratio);
        require(result.diagnostics.at("model_mass_scale").get<double>()==ratio,"Optimizer diagnostic lost the supplied mass ratio");
        scaled_attempts.push_back(result.diagnostics);
        if(result.accepted) {
          check_first_action(result,state,input,wind,parameters,tolerance);
          require(number(result.diagnostics,"max_constraint_violation")<=tolerance&&
                  number(result.diagnostics,"dynamics_residual")<=tolerance&&
                  number(result.diagnostics,"intermediate_planning_violation")<=tolerance&&
                  number(result.diagnostics,"objective_check_error")<=1e-4,
                  "Mass-scaled accepted solution failed its independent dynamics/objective audit");
          accepted=true;
        } else require(result.diagnostics.at("prediction").empty(),"Rejected mass-scaled solve retained a prediction");
      }
      require(accepted,"No independently feasible solution for an explicitly supplied nonnominal mass ratio");
    }

    const nlohmann::json report={
      {"passed",true},{"test","Native entropy optimizer contribution, feasibility, rejection and reset"},
      {"nominal_attempts",nominal_attempts},{"selected_nominal",selected.diagnostics},
      {"first_action_independent_physical_violation",independently_checked_violation},
      {"overtemperature_rejected",true},{"overtemperature_diagnostics",rejected.diagnostics},
      {"landing_support_counted_separately",true},{"landing_support_diagnostics",support.diagnostics},
      {"observation_hold_counted_separately",true},{"observation_hold_diagnostics",observation_diagnostics},
      {"reset_counters_correct",true},{"post_reset_diagnostics",reset},
      {"mass_scaled_derivatives_verified",true},{"mass_scaled_attempts",scaled_attempts},
      {"timing_note","Wall times are diagnostic only; this test imposes no real-time success condition"}};
    std::cout<<report.dump(2)<<'\n';
    return 0;
  } catch(const std::exception& error) {
    std::cout<<nlohmann::json{{"passed",false},{"test","Native entropy optimizer"},{"error",error.what()}}.dump(2)<<'\n';
    return 1;
  }
}
