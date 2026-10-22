#include "entropy_controller.hpp"
#include "flight_constraints.hpp"
#include "optimizer_stage.hpp"
#include "mass_scaled_dynamics.hpp"
#include <casadi/casadi.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace helicopter {
namespace {
using casadi::SX; using casadi::MX; using casadi::DM; using casadi::Function; using casadi::Slice;
using rotor::NX; using rotor::NU;
constexpr int N=EntropyController::horizon_steps;
constexpr double H=EntropyController::interval;
constexpr int NV=N*(NX+NU)+NX, NP=NX+NU+3+6*(N+1)+2;
constexpr int TIME=NP-2, MASS_RATIO=NP-1;
constexpr std::array<double,NX> scale{1,1,1,1,1,1,1,1,1,1,1,1,1,.1,.1,.1,.1,5,5,.1,.1,20,20};
int xi(int k) { return k*(NX+NU); }
int ui(int k) { return xi(k)+NX; }
std::vector<double> scaled_variables(std::vector<double> values,bool to_physical=false) {
  for(int k=0;k<=N;++k) {
    for(int j=0;j<NX;++j) values[xi(k)+j]*=to_physical?scale[j]:1/scale[j];
    if(k<N)for(int j=0;j<NU;++j) values[ui(k)+j]*=to_physical?.1:10;
  }
  return values;
}
template<class S,int M> std::array<S,M> slice_array(const S& value,int offset) {
  std::array<S,M> result; for(int i=0;i<M;++i) result[i]=value(offset+i); return result;
}
SX column(const rotor::State<SX>& x) { return SX::vertcat(std::vector<SX>(x.begin(),x.end())); }
struct Integrated { rotor::State<double> x; double entropy=0; double physical_violation=0,planning_violation=0; };
struct Rollout { std::vector<double> values; double objective=0; double physical_violation=0,planning_violation=0; };
// This numerical rollout is separate from IPOPT's state variables. Only controls
// are taken from its candidate; the original nonlinear ODE is integrated again.
Integrated integrate(rotor::State<double> x,const rotor::Input<double>& u,
                     const rotor::Vec3<double>& w,const rotor::Params& p) {
  double entropy=0,physical_violation=0,planning_violation=0; constexpr double h=H/5;
  for(int step=0;step<5;++step) {
    const auto a=rotor::derivative(x,u,w,p); auto bstate=x;
    for(int i=0;i<NX;++i) bstate[i]=x[i]+h*.5*a[i];
    const auto b=rotor::derivative(bstate,u,w,p); auto cstate=x;
    for(int i=0;i<NX;++i) cstate[i]=x[i]+h*.5*b[i];
    const auto c=rotor::derivative(cstate,u,w,p); auto dstate=x;
    for(int i=0;i<NX;++i) dstate[i]=x[i]+h*c[i];
    const auto d=rotor::derivative(dstate,u,w,p);
    entropy+=h/6*(rotor::evaluate(x,u,w,p).entropy_rate+2*rotor::evaluate(bstate,u,w,p).entropy_rate+
                  2*rotor::evaluate(cstate,u,w,p).entropy_rate+rotor::evaluate(dstate,u,w,p).entropy_rate);
    for(int i=0;i<NX;++i) x[i]+=h/6*(a[i]+2*b[i]+2*c[i]+d[i]);
    double q=0; for(int i=6;i<10;++i) q+=x[i]*x[i]; q=std::sqrt(q);
    for(int i=6;i<10;++i) x[i]/=q;
    physical_violation=std::max(physical_violation,rotor::physical_violation(x,u,w,p).maximum);
    const double tilt_margin=std::cos(EntropyController::planning_tilt_max_rad)-
      (1-2*(x[7]*x[7]+x[8]*x[8]));
    planning_violation=std::max(planning_violation,std::isfinite(tilt_margin)?tilt_margin:
      std::numeric_limits<double>::infinity());
    double airspeed_squared=0;
    for(int j=0;j<3;++j)airspeed_squared+=(x[3+j]-w[j])*(x[3+j]-w[j]);
    const double airspeed_margin=airspeed_squared/
      (EntropyController::planning_airspeed_max_m_s*EntropyController::planning_airspeed_max_m_s)-1;
    planning_violation=std::max(planning_violation,std::isfinite(airspeed_margin)?airspeed_margin:
      std::numeric_limits<double>::infinity());
  }
  return {x,entropy,physical_violation,planning_violation};
}
}

struct EntropyController::Impl {
  rotor::Params p;
  Function solver, constraints, objective;
  std::vector<double> lower,upper,gl,gu,previous_solution,dual_x,dual_g;
  double previous_solution_time=std::numeric_limits<double>::quiet_NaN();
  bool previous_solution_accepted=false;
  unsigned solves=0,accepted=0,fallbacks=0;
  unsigned control_decisions=0,landing_support_decisions=0,observation_hold_decisions=0;
  EntropyResult no_optimizer_decision(double time,bool observation);
  explicit Impl(const rotor::Params& parameters):p(parameters) {
    auto decision=MX::sym("trajectory",NV), par=MX::sym("conditions",NP);
    std::vector<MX> physical(NV);
    for(int k=0;k<=N;++k) {
      for(int j=0;j<NX;++j) physical[xi(k)+j]=scale[j]*decision(xi(k)+j);
      if(k<N)for(int j=0;j<NU;++j)physical[ui(k)+j]=.1*decision(ui(k)+j);
    }
    const MX v=MX::vertcat(physical);
    auto sx=SX::sym("state",NX), su=SX::sym("pitch",NU), sw=SX::sym("wind",3);
    const auto sr=SX::sym("mass_ratio");
    auto x=slice_array<SX,NX>(sx,0); const auto u=slice_array<SX,NU>(su,0);
    const auto wind=slice_array<SX,3>(sw,0); SX quadrature=0;
    constexpr double h=H/5;
    for(int substep=0;substep<1;++substep) {
      const auto a=rotor::mass_scaled_derivative(x,u,wind,p,sr); auto xb=x;
      for(int i=0;i<NX;++i) xb[i]=x[i]+h*.5*a[i];
      const auto b=rotor::mass_scaled_derivative(xb,u,wind,p,sr); auto xc=x;
      for(int i=0;i<NX;++i) xc[i]=x[i]+h*.5*b[i];
      const auto c=rotor::mass_scaled_derivative(xc,u,wind,p,sr); auto xd=x;
      for(int i=0;i<NX;++i) xd[i]=x[i]+h*c[i];
      const auto d=rotor::mass_scaled_derivative(xd,u,wind,p,sr);
      quadrature+=h/6*(rotor::evaluate(x,u,wind,p).entropy_rate+2*rotor::evaluate(xb,u,wind,p).entropy_rate+
                        2*rotor::evaluate(xc,u,wind,p).entropy_rate+rotor::evaluate(xd,u,wind,p).entropy_rate);
      for(int i=0;i<NX;++i) x[i]+=h/6*(a[i]+2*b[i]+2*c[i]+d[i]);
      SX q=0; for(int i=6;i<10;++i) q+=x[i]*x[i]; q=sqrt(q);
      for(int i=6;i<10;++i) x[i]/=q;
    }
    const casadi::Dict compact{{"max_num_dir",4}};
    Function kernel("rk4_kernel",{sx,su,sw,sr},{column(x),quadrature},compact);
    const auto terminal_derivative=rotor::mass_scaled_derivative(slice_array<SX,NX>(sx,0),u,wind,p,sr);
    std::vector<SX> terminal_rates;
    const std::vector<int> terminal_indices{3,4,5,10,11,12,13,14,15,16,17,18,19,20};
    const std::vector<double> terminal_limits{.6,.6,.6,.8,.8,.8,.03,.15,.15,.08,.4,.4,.15,.15};
    for(std::size_t j=0;j<terminal_indices.size();++j)
      terminal_rates.push_back(terminal_derivative[terminal_indices[j]]/terminal_limits[j]);
    Function settled("terminal_rates",{sx,su,sw,sr},{SX::vertcat(terminal_rates)},compact);
    auto mx=MX::sym("state",NX),mu=MX::sym("pitch",NU),mw=MX::sym("wind",3);
    const auto mr=MX::sym("mass_ratio");
    MX propagated=mx,total_entropy=0;std::vector<MX> substep_limits;
    for(int k=0;k<5;++k) {
      const auto result=kernel(std::vector<MX>{propagated,mu,mw,mr});
      propagated=result[0];total_entropy+=result[1];
      substep_limits.push_back(1-2*(propagated(7)*propagated(7)+propagated(8)*propagated(8))-
                            std::cos(EntropyController::planning_tilt_max_rad));
      MX airspeed_squared=0;
      for(int j=0;j<3;++j)airspeed_squared+=(propagated(3+j)-mw(j))*(propagated(3+j)-mw(j));
      substep_limits.push_back(airspeed_squared/
        (EntropyController::planning_airspeed_max_m_s*EntropyController::planning_airspeed_max_m_s));
    }
    Function step("flight_step",{mx,mu,mw,mr},{propagated,total_entropy,MX::vertcat(substep_limits)},compact);
    const auto stage=make_optimizer_stage(p);
    lower.assign(NV,-std::numeric_limits<double>::infinity());
    upper.assign(NV,std::numeric_limits<double>::infinity());
    std::vector<MX> g;
    auto bound=[&](const MX& value,double lo,double hi){g.push_back(value);gl.push_back(lo);gu.push_back(hi);};
    for(int j=0;j<NX;++j) bound((v(j)-par(j))/scale[j],0,0);
    const MX w=par(Slice(NX+NU,NX+NU+3)),mass_ratio=par(MASS_RATIO); MX cost=0;
    for(int k=0;k<=N;++k) {
      const auto state=slice_array<MX,NX>(v,xi(k));
      rotor::Input<MX> input;
      for(int j=0;j<NU;++j) input[j]=k<N ? v(ui(k)+j) : v(ui(N-1)+j);
      if(k>0) {
        const MX landing=par(TIME)+k*H>=68-1e-8;
        const MX approach=par(TIME)+k*H>=65-1e-8;
        const auto checked=stage.function(std::vector<MX>{v(Slice(xi(k),xi(k)+NX)),
          v(Slice(ui(std::min(k,N-1)),ui(std::min(k,N-1))+NU)),w,
          par(Slice(NX+NU+3+6*k,NX+NU+3+6*(k+1))),landing,MX(k==N),approach})[0];
        for(std::size_t j=0;j<stage.lower.size();++j) bound(checked(j),stage.lower[j],stage.upper[j]);
        if(k==N) {
          for(int j=0;j<3;++j) {
            bound((state[3+j]-par(NX+NU+3+6*k+3+j))/.25,-1,1);
            bound(state[10+j]/.5,-1,1);
          }
          bound(state[9]/.10,-1,1); // near-zero heading, qz half-angle.
          const auto rates=settled(std::vector<MX>{v(Slice(xi(k),xi(k)+NX)),v(Slice(ui(N-1),ui(N-1)+NU)),w,mass_ratio})[0];
          for(std::size_t j=0;j<terminal_indices.size();++j)bound(rates(j),-1,1);
        }
      }
      if(k<N) {
        for(int j=0;j<NU;++j) {
          lower[ui(k)+j]=p.input_min[j]/.1; upper[ui(k)+j]=p.input_max[j]/.1;
          const MX prior=k==0 ? par(NX+j) : v(ui(k-1)+j);
          bound((input[j]-prior)/(p.input_rate_max[j]*H),-1,1);
        }
        const auto next=step(std::vector<MX>{v(Slice(xi(k),xi(k)+NX)),v(Slice(ui(k),ui(k)+NU)),w,mass_ratio});
        for(int j=0;j<NX;++j) bound((v(xi(k+1)+j)-next[0](j))/scale[j],0,0);
        for(int substep=0;substep<5;++substep) {
          bound(next[2](2*substep),0,2);
          bound(next[2](2*substep+1),0,1);
        }
        cost+=next[1];
      }
    }
    cost+=rotor::terminal_entropy_commitment(slice_array<MX,NX>(v,xi(N)),p);
    constraints=Function("original_constraints",{decision,par},{MX::vertcat(g)});
    objective=Function("physical_entropy",{decision,par},{cost});
    casadi::Dict options;
    options["jit"]=true;
    options["jit_serialize"]="embed";
    options["compiler"]="shell";
    options["jit_options"]=casadi::Dict{{"flags","-Og -fno-inline"}};
    options["print_time"]=false; options["error_on_fail"]=false;
    options["ipopt.print_level"]=0; options["ipopt.sb"]="yes";
    options["ipopt.max_iter"]=150; options["ipopt.max_cpu_time"]=10.0;
    options["ipopt.nlp_scaling_method"]="none";
    options["ipopt.tol"]=1e-5; options["ipopt.constr_viol_tol"]=1e-8;
    options["ipopt.acceptable_tol"]=1e-3; options["ipopt.acceptable_constr_viol_tol"]=1e-8;
    options["ipopt.acceptable_iter"]=3;
    options["ipopt.bound_relax_factor"]=0.0;
    options["ipopt.hessian_approximation"]="exact";
    options["ipopt.mu_strategy"]="adaptive";
    options["ipopt.warm_start_init_point"]="yes";
    options["ipopt.warm_start_bound_push"]=1e-6;
    options["ipopt.warm_start_mult_bound_push"]=1e-6;
    options["ipopt.warm_start_slack_bound_push"]=1e-6;
    options["ipopt.linear_solver"]="mumps";
    // A source-bound native-function cache avoids compiling the same derivative
    // kernels on every reset/process launch. Only nominal published parameters
    // use this cache; other parameter sets are constructed independently.
    const bool cacheable=rotor::model_metadata(p)==rotor::model_metadata(rotor::Params{});
    const std::string cache="/workspace/build/entropy-solver-" HELICOPTER_MODEL_SOURCE_SHA256 ".casadi";
    if(cacheable&&std::filesystem::exists(cache)) {
      try { solver=Function::load(cache); } catch(const std::exception&) { /* Rebuild incomplete cache. */ }
    }
    if(solver.is_null()) {
      solver=casadi::nlpsol("entropy_nmpc","ipopt",casadi::MXDict{{"x",decision},{"p",par},{"f",cost},{"g",MX::vertcat(g)}},options);
      if(cacheable) solver.save(cache);
    }
  }
  double violation(const std::vector<double>& v,const std::vector<double>& par) const {
    const auto normalized=scaled_variables(v);
    const auto g=constraints(std::vector<DM>{DM(normalized),DM(par)})[0].nonzeros();
    double worst=0;
    for(std::size_t i=0;i<g.size();++i) {
      if(!std::isfinite(g[i])) return 1e30;
      worst=std::max({worst,gl[i]-g[i],g[i]-gu[i]});
    }
    for(int i=0;i<NV;++i) {
      if(!std::isfinite(v[i])) return 1e30;
      worst=std::max({worst,lower[i]-normalized[i],normalized[i]-upper[i]});
    }
    return worst;
  }
  int warm_start_shift(double time) const {
    if(previous_solution.empty()||!std::isfinite(time)||!std::isfinite(previous_solution_time))return -1;
    const double elapsed=time-previous_solution_time;
    // Inputs are held on the prediction grid. A repeated same-time solve uses
    // stage zero; fallbacks age the stored plan by every elapsed interval.
    constexpr double grid_tolerance=1e-8;
    if(elapsed < -H*grid_tolerance || elapsed/H >= N-grid_tolerance)return -1;
    return static_cast<int>(std::floor(std::max(0.0,elapsed/H)+grid_tolerance));
  }
  std::vector<double> shifted_guess(const rotor::State<double>& initial,
      const rotor::Input<double>& previous,const rotor::Vec3<double>& wind,
      const rotor::Params& prediction_params,int shift) const {
    std::vector<double> values(NV);
    auto prior=previous;
    for(int k=0;k<=N;++k) {
      if(k+shift<=N) {
        std::copy_n(previous_solution.begin()+xi(k+shift),NX,values.begin()+xi(k));
      } else {
        rotor::State<double> tail_state;
        rotor::Input<double> tail_input;
        std::copy_n(values.begin()+xi(k-1),NX,tail_state.begin());
        std::copy_n(values.begin()+ui(k-1),NU,tail_input.begin());
        const auto extended=integrate(tail_state,tail_input,wind,prediction_params);
        std::copy(extended.x.begin(),extended.x.end(),values.begin()+xi(k));
      }
      if(k<N) {
        for(int j=0;j<NU;++j) {
          const double command=previous_solution[ui(std::min(k+shift,N-1))+j];
          prior[j]=std::clamp(command,std::max(p.input_min[j],prior[j]-p.input_rate_max[j]*H),
                                     std::min(p.input_max[j],prior[j]+p.input_rate_max[j]*H));
          values[ui(k)+j]=prior[j];
        }
      }
    }
    // Multiple shooting can correct small defects between measured X0 and the
    // shifted state nodes. Repropagating the entire open-loop trajectory here
    // would amplify those errors before the solver has a chance to correct them.
    std::copy(initial.begin(),initial.end(),values.begin());
    return values;
  }
  Rollout rollout(const rotor::State<double>& initial,
      const rotor::Input<double>& previous,const rotor::Vec3<double>& w,
      const std::vector<rotor::Reference>& reference,const rotor::Params& prediction_params,
      int warm_shift=-1) const {
    std::vector<double> values(NV); auto state=initial; auto prior=previous;
    rotor::BaselineController pid; pid.reset(previous);
    double entropy=0,physical_violation=0,planning_violation=0;
    for(int k=0;k<N;++k) {
      std::copy(state.begin(),state.end(),values.begin()+xi(k));
      auto command=pid.command(state,reference[k],w,prediction_params,H);
      if(warm_shift>=0) {
        const int shifted=std::min(k+warm_shift,N-1);
        for(int j=0;j<NU;++j) command[j]=previous_solution[ui(shifted)+j];
      }
      for(int j=0;j<NU;++j) {
        command[j]=std::clamp(command[j],std::max(p.input_min[j],prior[j]-p.input_rate_max[j]*H),
                             std::min(p.input_max[j],prior[j]+p.input_rate_max[j]*H));
        values[ui(k)+j]=command[j];
      }
      auto integrated=integrate(state,command,w,prediction_params); state=integrated.x; entropy+=integrated.entropy; prior=command;
      physical_violation=std::max(physical_violation,integrated.physical_violation);
      planning_violation=std::max(planning_violation,integrated.planning_violation);
    }
    std::copy(state.begin(),state.end(),values.begin()+xi(N));
    return {values,entropy+rotor::terminal_entropy_commitment(state,prediction_params),physical_violation,planning_violation};
  }
};

EntropyController::EntropyController(const rotor::Params& p):impl_(std::make_unique<Impl>(p)) {}
EntropyController::~EntropyController()=default;
void EntropyController::reset() {
  impl_->previous_solution.clear();impl_->dual_x.clear();impl_->dual_g.clear();
  impl_->previous_solution_time=std::numeric_limits<double>::quiet_NaN();
  impl_->previous_solution_accepted=false;
  impl_->solves=impl_->accepted=impl_->fallbacks=0;
  impl_->control_decisions=impl_->landing_support_decisions=impl_->observation_hold_decisions=0;
}

EntropyResult EntropyController::Impl::no_optimizer_decision(double time,bool observation) {
  if(!std::isfinite(time))throw std::invalid_argument("No-optimizer decision time must be finite");
  auto& s=*this;
  if(observation&&(s.control_decisions!=0||std::abs(time)>1e-9))
    throw std::invalid_argument("Observation hold is only valid for the initial scheduled decision");
  ++s.control_decisions;
  if(observation)++s.observation_hold_decisions;else ++s.landing_support_decisions;
  EntropyResult result;
  result.diagnostics={{"available",true},{"time",time},{"status",observation?"observation_hold":"landing_support"},
    {"accepted",false},{"converged",false},{"fallback",true},{"optimizer_attempted",false},
    {"applied_controller",observation?"trim_observation_hold":"pid_landing_support"},
    {"fallback_reason",observation?"Prepared trim is held for one causal observation interval before optimization":
      "Planned PID landing support; skid contact lies outside the free-flight optimizer model"},
    {"iterations",0},{"solve_ms",0.0},{"control_interval_ms",update_period*1000},{"within_control_interval",true},
    {"seed_objective_j_per_k",nullptr},{"seed_constraint_violation",nullptr},{"objective_j_per_k",nullptr},
    {"objective_reduction_j_per_k",nullptr},{"objective_check_error",nullptr},{"max_constraint_violation",nullptr},
    {"dynamics_residual",nullptr},{"intermediate_physical_violation",nullptr},{"intermediate_planning_violation",nullptr},{"action_difference_norm",nullptr},
    {"warm_start",false},{"initial_guess_constraint_violation",nullptr},{"warm_start_source","none"},
    {"warm_start_shift_steps",nullptr},{"warm_start_plan_time",nullptr},{"rejected_candidate_retained_for_warm_start",false},
    {"prediction",nlohmann::json::array()},
    {"solves",s.solves},{"accepted_solves",s.accepted},{"fallback_solves",s.fallbacks},
    {"control_decisions",s.control_decisions},{"landing_support_decisions",s.landing_support_decisions},
    {"observation_hold_decisions",s.observation_hold_decisions}};
  return result;
}

EntropyResult EntropyController::landing_support(double time) {return impl_->no_optimizer_decision(time,false);}
EntropyResult EntropyController::observation_hold(double time) {return impl_->no_optimizer_decision(time,true);}

EntropyResult EntropyController::solve(const rotor::State<double>& initial,
    const rotor::Input<double>& previous,const rotor::Vec3<double>& wind,
    const std::vector<rotor::Reference>& reference,double time,double mass_ratio) {
  if(reference.size()!=N+1) throw std::invalid_argument("NMPC needs N+1 reference samples");
  if(!std::isfinite(mass_ratio)||mass_ratio<.5||mass_ratio>1.5)
    throw std::invalid_argument("NMPC mass ratio outside [0.5,1.5]");
  const auto started=std::chrono::steady_clock::now();
  auto& s=*impl_; ++s.solves;++s.control_decisions;
  auto prediction_params=s.p;
  prediction_params.mass*=mass_ratio;
  for(auto& inertia:prediction_params.inertia)inertia*=mass_ratio;
  std::vector<double> par(NP); std::copy(initial.begin(),initial.end(),par.begin());
  std::copy(previous.begin(),previous.end(),par.begin()+NX); std::copy(wind.begin(),wind.end(),par.begin()+NX+NU);
  par[TIME]=time;par[MASS_RATIO]=mass_ratio;
  for(int k=0;k<=N;++k) for(int j=0;j<3;++j) {
    par[NX+NU+3+6*k+j]=reference[k].position[j];
    par[NX+NU+3+6*k+3+j]=reference[k].velocity[j];
  }
  auto seed=s.rollout(initial,previous,wind,reference,prediction_params); const auto pid_seed=seed;
  double seed_violation=std::max({s.violation(seed.values,par),seed.physical_violation,seed.planning_violation});
  double comparison_cost=seed.objective,comparison_violation=seed_violation;
  bool warm_start=false;
  const int warm_shift=s.warm_start_shift(time);
  if(warm_shift<0&&!s.previous_solution.empty()) {
    s.previous_solution.clear();s.dual_x.clear();s.dual_g.clear();
    s.previous_solution_time=std::numeric_limits<double>::quiet_NaN();
    s.previous_solution_accepted=false;
  }
  double available_warm_violation=std::numeric_limits<double>::infinity();
  double available_warm_objective=std::numeric_limits<double>::infinity();
  if(warm_shift>=0) {
    // Independent propagation determines whether this plan is a valid objective
    // comparator. It does not replace the multiple-shooting state-node seed.
    auto warm=s.rollout(initial,previous,wind,reference,prediction_params,warm_shift);
    const double warm_violation=std::max({s.violation(warm.values,par),warm.physical_violation,warm.planning_violation});
    available_warm_violation=warm_violation;available_warm_objective=warm.objective;
    if(warm_violation<=1e-4&&(comparison_violation>1e-4||warm.objective<comparison_cost)) {
      comparison_cost=warm.objective;comparison_violation=warm_violation;
    }
    auto guess=s.shifted_guess(initial,previous,wind,prediction_params,warm_shift);
    const double guess_violation=s.violation(guess,par);
    if(guess_violation<.1||guess_violation<seed_violation) {
      seed.values=std::move(guess);seed_violation=guess_violation;
      warm_start=true;
    }
  }
  EntropyResult result;
  for(int j=0;j<NU;++j) result.command[j]=pid_seed.values[ui(0)+j];
  auto& d=result.diagnostics;
  d={{"available",true},{"time",time},{"status","not_solved"},{"accepted",false},{"converged",false},
     {"optimizer_attempted",true},{"applied_controller","pid_fallback"},{"model_mass_scale",mass_ratio},
     {"iterations",0},{"solve_ms",0.0},{"seed_objective_j_per_k",comparison_cost},{"seed_constraint_violation",comparison_violation},
     {"warm_start",warm_start},{"initial_guess_constraint_violation",seed_violation},
     {"warm_start_source",warm_start?(s.previous_solution_accepted?"accepted_solution":"rejected_candidate"):"none"},
     {"warm_start_shift_steps",warm_start?nlohmann::json(warm_shift):nlohmann::json(nullptr)},
     {"warm_start_plan_time",warm_start?nlohmann::json(s.previous_solution_time):nlohmann::json(nullptr)},
     {"rejected_candidate_retained_for_warm_start",false},
     {"objective_j_per_k",nullptr},{"max_constraint_violation",nullptr},{"intermediate_planning_violation",nullptr},
     {"fallback",true},{"fallback_reason","No accepted solution"},
     {"prediction",nlohmann::json::array()}};
  try {
    casadi::DMDict arguments{{"x0",DM(scaled_variables(seed.values))},{"p",DM(par)},
       {"lbx",DM(s.lower)},{"ubx",DM(s.upper)},{"lbg",DM(s.gl)},{"ubg",DM(s.gu)}};
    // Multipliers belong to their original stage rows. Reuse them only when the
    // accepted primal plan has not shifted; shifted starts use primal values.
    if(warm_start&&warm_shift==0&&s.previous_solution_accepted&&!s.dual_x.empty()) {
      arguments["lam_x0"]=DM(s.dual_x);arguments["lam_g0"]=DM(s.dual_g);
    }
    const auto answer=s.solver(arguments);
    const auto stats=s.solver.stats();
    const std::string status=stats.at("return_status").to_string();
    d["status"]=status; d["iterations"]=static_cast<int>(stats.at("iter_count").to_int());
    d["converged"]=status=="Solve_Succeeded"||status=="Solved_To_Acceptable_Level";
    auto candidate=scaled_variables(answer.at("x").nonzeros(),true); const double original_violation=s.violation(candidate,par);
    // Reintegrate controls in ordinary double precision. State decision values
    // from the optimizer cannot hide shooting defects or constraint violations.
    auto state=initial; double entropy=0,dynamic_error=0,intermediate_violation=0,intermediate_planning_violation=0;
    std::vector<double> verified=candidate;
    for(int k=0;k<=N;++k) {
      for(int j=0;j<NX;++j) {
        dynamic_error=std::max(dynamic_error,std::abs(candidate[xi(k)+j]-state[j])/scale[j]);
        verified[xi(k)+j]=state[j];
      }
      d["prediction"].push_back({{"time",time+k*H},{"position",{state[0],state[1],state[2]}},
                                {"temperature_k",{state[21],state[22]}}});
      if(k<N) {
        rotor::Input<double> input; for(int j=0;j<NU;++j) input[j]=candidate[ui(k)+j];
        const auto integrated=integrate(state,input,wind,prediction_params); state=integrated.x; entropy+=integrated.entropy;
        intermediate_violation=std::max(intermediate_violation,integrated.physical_violation);
        intermediate_planning_violation=std::max(intermediate_planning_violation,integrated.planning_violation);
      }
    }
    entropy+=rotor::terminal_entropy_commitment(state,prediction_params);
    const double checked_violation=std::max({s.violation(verified,par),intermediate_violation,intermediate_planning_violation});
    d["intermediate_physical_violation"]=intermediate_violation;
    d["intermediate_planning_violation"]=intermediate_planning_violation;
    d["max_constraint_violation"]=std::max(original_violation,checked_violation);
    d["dynamics_residual"]=dynamic_error; d["objective_j_per_k"]=entropy;
    d["objective_reduction_j_per_k"]=comparison_cost-entropy;
    d["objective_check_error"]=std::abs(entropy-static_cast<double>(answer.at("f")));
    const bool feasible=std::isfinite(entropy)&&std::max(original_violation,checked_violation)<=1e-4&&dynamic_error<=1e-4&&
      d["objective_check_error"].get<double>()<=1e-4;
    const bool nonworse=comparison_violation>1e-4||entropy<=comparison_cost+1e-7;
    result.accepted=feasible&&nonworse;
    d["accepted"]=result.accepted; d["fallback"]=!result.accepted;
    d["fallback_reason"]=result.accepted ? "" : !feasible ? "Independently checked constraints or dynamics failed" : "No improvement over feasible seed";
    if(result.accepted) {
      s.previous_solution=std::move(verified);
      s.previous_solution_time=time;s.previous_solution_accepted=true;
      s.dual_x=answer.at("lam_x").nonzeros();s.dual_g=answer.at("lam_g").nonzeros();
      for(int j=0;j<NU;++j) result.command[j]=candidate[ui(0)+j];
    } else {
      // A rejected solution can help the next nonlinear solve, but never supplies
      // an executable command or prediction. Retain only the independently
      // re-integrated finite trajectory and never replace a feasible warm plan
      // with an infeasible one merely because its objective looks attractive.
      const auto all_finite=[](const std::vector<double>& values) {
        return std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);});
      };
      const bool candidate_feasible=checked_violation<=1e-4;
      const bool warm_feasible=available_warm_violation<=1e-4;
      const bool better_warm_candidate=warm_feasible ?
        candidate_feasible&&entropy<available_warm_objective-1e-7 :
        candidate_feasible||checked_violation<available_warm_violation-1e-9||
        (std::abs(checked_violation-available_warm_violation)<=1e-9&&entropy<available_warm_objective-1e-7);
      const bool retain=std::isfinite(time)&&std::isfinite(entropy)&&std::isfinite(checked_violation)&&
        checked_violation<=.01&&all_finite(candidate)&&all_finite(verified)&&better_warm_candidate;
      if(retain) {
        s.previous_solution=std::move(verified);
        s.previous_solution_time=time;s.previous_solution_accepted=false;
        s.dual_x.clear();s.dual_g.clear();
        d["rejected_candidate_retained_for_warm_start"]=true;
      }
      d["prediction"]=nlohmann::json::array();
    }
  } catch(const std::exception& error) {
    result.accepted=false;d["accepted"]=false;d["fallback"]=true;d["converged"]=false;
    d["status"]="solver_exception";d["fallback_reason"]=error.what();
  }
  if(!result.accepted)d["prediction"]=nlohmann::json::array();
  d["solve_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  d["control_interval_ms"]=update_period*1000;
  d["within_control_interval"]=d["solve_ms"].get<double>()<=update_period*1000;
  double difference=0; for(int j=0;j<NU;++j) difference+=std::pow(result.command[j]-pid_seed.values[ui(0)+j],2);
  d["action_difference_norm"]=std::sqrt(difference);
  if(result.accepted) ++s.accepted; else ++s.fallbacks;
  d["applied_controller"]=result.accepted?"entropy_nmpc":"pid_fallback";
  d["solves"]=s.solves; d["accepted_solves"]=s.accepted; d["fallback_solves"]=s.fallbacks;
  d["control_decisions"]=s.control_decisions;d["landing_support_decisions"]=s.landing_support_decisions;
  d["observation_hold_decisions"]=s.observation_hold_decisions;
  return result;
}

nlohmann::json EntropyController::description() {
  return {{"name","Entropy NMPC with PID landing support"},{"implementation","Native C++ CasADi 3.7.2 / IPOPT, direct multiple shooting"},
    {"architecture","One counted prepared-trim observation interval, then airborne entropy optimization with explicitly counted PID fallback and landing support. The predictor does not model skid contact."},
    {"objective","Integral of modeled physical entropy generation plus terminal motor-cooldown entropy and stored-wake relaxation; no tracking-error cost"},
    {"horizon_steps",N},{"interval_s",H},{"horizon_s",N*H},{"update_period_s",update_period},
    {"integration","RK4, five substeps per prediction interval, quaternion normalization"},
    {"prediction_physical_check_step_s",H/5},{"execution","Synchronous simulation; measured wall time may exceed the simulated control interval"},
    {"planning_tilt_max_rad",planning_tilt_max_rad},{"physical_tilt_max_rad",rotor::Params{}.tilt_max_rad},
    {"planning_tilt_enforcement","Future prediction knots and every 0.02 s RK4 substep output, independently re-integrated; the physical tilt limit is unchanged."},
    {"planning_airspeed_max_m_s",planning_airspeed_max_m_s},{"physical_airspeed_max_m_s",rotor::Params{}.airspeed_max},
    {"planning_airspeed_enforcement","Future prediction knots and every 0.02 s RK4 substep output use current-wind persistence, independently re-integrated. The 4.7 m/s bound is a heuristic reserve below the unchanged 5.0 m/s physical bound, not a robust-control proof for changing wind."},
    {"tracking_tube_m",{.4,.4,.20}},{"mission_tracking_tube_m",{.6,.6,.35}},
    {"planning_margins","Internal position corridors are tighter than the unchanged independently audited mission corridors; approach and landing bounds further restrict them."},
    {"terminal_position_tolerance_m",.10},{"terminal_velocity_tolerance_m_s",.25},
    {"handover_settling",{{"from_predicted_time_s",65},{"position_xy_tolerance_m",.05},
      {"velocity_xy_tolerance_m_s",.05},{"tilt_max_deg",5},{"body_rate_max_rad_s",.10}}},
    {"landing_from_s",68},{"landing_position_tolerance_m",{.10,.10,.025}},{"landing_velocity_tolerance_m_s",.10},
    {"landing_tilt_max_deg",5},{"landing_body_rate_max_rad_s",.20},
    {"terminal_settling",{{"acceleration_max_m_s2",.6},{"angular_acceleration_max_rad_s2",.8},
      {"actuator_rate_max_rad_s",{.03,.15,.15,.08}},{"inflow_rate_max_m_s2",.4},{"flap_rate_max_rad_s",.15}}},
    {"solver_cpu_budget_s",10.0},{"solver_iteration_limit",150},
    {"solver_constraint_tolerance",1e-8},{"solver_acceptable_constraint_tolerance",1e-8},
    {"variable_scaling","SI plant variables are normalized in the NLP; objective remains J/K"},
    {"feasibility_tolerance",1e-4},{"constraint_residual_units","dimensionless; physical constraints scaled by their declared limits"},
    {"state_source","Exact simulated state; measured-current-wind persistence forecast; no hardware estimator claim"},
    {"mass_parameterization","Runtime causal mass-ratio estimate scales nominal mass and all principal inertias together; the declared uncertainty family preserves geometry. Entropy and physical limits are unchanged."},
    {"fallback","Explicit PID on rejected, failed, or infeasible solve; planned PID landing support has its own status and count."},
    {"landing_support",{{"controller","PID"},{"scope","Latched descent/contact handover selected by the simulation's geometry/reference guard; no entropy-optimal contact-control claim."},
      {"optimizer_attempted",false},{"counter","landing_support_decisions"}}},
    {"observation_hold",{{"controller","Prepared trim hold"},{"duration_s",update_period},{"optimizer_attempted",false},
      {"scope","Initial scheduled decision only; both compared modes hold the same prepared trim to collect causal mass observations."},
      {"counter","observation_hold_decisions"}}},
    {"control_accounting",{{"control_decisions","All scheduled NMPC-mode decisions, including initial observation hold and planned PID landing support"},
      {"solves","Actual optimizer attempts; accepted_solves plus fallback_solves"},
      {"identity","control_decisions = solves + landing_support_decisions + observation_hold_decisions"},
      {"participation","NMPC-applied decisions divided by all control_decisions; initial observation and planned support stay in the denominator"}}},
    {"optimality","Local finite-budget nonlinear solution; independently feasible is distinct from solver convergence; no global optimum claim"}};
}
}
