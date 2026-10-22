#include "mass_estimator.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using helicopter::MassEstimator;
using Vec3=std::array<double,3>;
using Observer=bool(MassEstimator::*)(const Vec3&,const Vec3&,const Vec3&,double,bool);
static_assert(std::is_same_v<decltype(&MassEstimator::observe),Observer>,
              "Estimator observations must not gain a plant-parameter input");

void require(bool condition,const std::string& message) {
  if(!condition)throw std::runtime_error(message);
}
void close(double actual,double expected,double tolerance,const std::string& message) {
  if(!std::isfinite(actual)||std::abs(actual-expected)>tolerance)
    throw std::runtime_error(message+": actual="+std::to_string(actual)+", expected="+std::to_string(expected));
}

// Synthetic plant uses Newton's force balance directly, without the helicopter
// derivative, model integration, estimator formula, or scenario metadata.
Vec3 advance_velocity(const Vec3& before,const Vec3& force,double mass,double dt) {
  Vec3 after{};
  for(int j=0;j<3;++j)after[j]=before[j]+dt*(force[j]/mass-(j==2?9.81:0));
  return after;
}

void learns_mass_from_motion() {
  for(double actual_mass:{1.6,2.0,2.4}) {
    MassEstimator estimate;
    close(estimate.mass_kg(),2.0,0,"Default prior must be nominal, independent of the fixture");
    Vec3 velocity{.3,-.2,.1};
    for(int i=0;i<200;++i) {
      const double dt=.006+.001*(i%9);
      const Vec3 force{2.0*std::sin(.07*i),-1.2*std::cos(.13*i),19.0+1.5*std::sin(.11*i)};
      const auto after=advance_velocity(velocity,force,actual_mass,dt);
      require(estimate.observe(force,velocity,after,dt,false),"Valid airborne observation was rejected");
      velocity=after;
    }
    close(estimate.mass_kg(),actual_mass,1e-10,"Mass learned from force and measured velocity increments");
    close(estimate.mass_scale(),actual_mass/2.0,1e-10,"Identified common scale");
    require(estimate.samples()==200,"Every valid observed step must be counted once");
    const auto metadata=estimate.metadata();
    require(metadata.at("accepted_samples")==200&&metadata.at("estimate_available")==true&&
            metadata.at("using_nominal_prior")==false,"Estimator provenance must reflect actual observations");
  }
}

void hover_is_informative_and_data_accumulate() {
  MassEstimator estimate;
  const Vec3 velocity{};
  require(estimate.observe({0,0,1.6*9.81},velocity,velocity,.01,false),"Hover is informative through gravity");
  close(estimate.mass_kg(),1.6,1e-12,"Hover force identifies mass even with zero acceleration");
  require(estimate.observe({0,0,2.4*9.81},velocity,velocity,.01,false),"Second informative observation");
  close(estimate.mass_kg(),2.0,1e-12,"Equal-information observations accumulate instead of replacing the fit");
  require(estimate.samples()==2,"Accumulated sample count");
}

void invalid_samples_leave_estimate_unchanged() {
  MassEstimator estimate;
  const Vec3 velocity{};
  const Vec3 hover_force{0,0,1.6*9.81};
  require(estimate.observe(hover_force,velocity,velocity,.01,false),"Prime estimator");
  const auto reject=[&](const Vec3& force,const Vec3& before,const Vec3& after,double dt,bool contact) {
    require(!estimate.observe(force,before,after,dt,contact),"Invalid observation must be rejected");
    close(estimate.mass_kg(),1.6,1e-12,"Rejected observation changed the learned mass");
    require(estimate.samples()==1,"Rejected observation changed the accepted count");
  };
  // Ground support is deliberately absent from the supplied rotor force.
  reject({0,0,2.4*9.81},velocity,velocity,.01,true);
  reject(hover_force,velocity,velocity,0,false);
  reject(hover_force,velocity,velocity,-.01,false);
  const double nan=std::numeric_limits<double>::quiet_NaN();
  const double inf=std::numeric_limits<double>::infinity();
  reject(hover_force,velocity,velocity,nan,false);
  reject({nan,0,19.62},velocity,velocity,.01,false);
  reject(hover_force,{0,inf,0},velocity,.01,false);
  reject(hover_force,velocity,{0,0,nan},.01,false);
  const auto freefall=advance_velocity(velocity,{0,0,0},1.6,.01);
  reject({0,0,0},velocity,freefall,.01,false);
  reject({0,0,.8*9.81},velocity,velocity,.01,false);
  reject({0,0,3.2*9.81},velocity,velocity,.01,false);
  require(estimate.metadata().at("rejected_samples")==10,"Rejected observation count");
}

void reset_and_instances_do_not_share_hidden_mass() {
  MassEstimator a,b;
  const Vec3 velocity{};
  require(a.observe({0,0,1.6*9.81},velocity,velocity,.01,false),"First independent fixture");
  require(b.observe({0,0,2.4*9.81},velocity,velocity,.01,false),"Second independent fixture");
  close(a.mass_kg(),1.6,1e-12,"Independent estimator A");
  close(b.mass_kg(),2.4,1e-12,"Independent estimator B");
  a.reset();
  close(a.mass_kg(),2.0,0,"Reset returns to nominal prior");
  close(a.mass_scale(),1.0,0,"Reset scale");
  require(a.samples()==0&&a.metadata().at("information")==0&&
          a.metadata().at("using_nominal_prior")==true,"Reset removes accumulated identification data");
  require(a.observe({0,0,2.4*9.81},velocity,velocity,.01,false),"Observation after reset");
  close(a.mass_kg(),2.4,1e-12,"Old measurements must not survive reset");
}

void common_inertia_scale_preserves_gyro_term() {
  // Scaling all principal inertias does not scale the Euler gyroscopic term.
  // This independent algebra fixture guards the proposed runtime parameterization.
  const Vec3 inertia{.060,.095,.090},omega{.4,-.3,.2},torque{.12,-.08,.03};
  for(double scale:{.8,1.0,1.2})for(int j=0;j<3;++j) {
    const int k=(j+1)%3,l=(j+2)%3;
    const double gyro=(inertia[l]-inertia[k])*omega[k]*omega[l];
    const double nominal_acceleration=(torque[j]-gyro)/inertia[j];
    const double parameterized=nominal_acceleration/scale+(1.0/scale-1.0)*gyro/inertia[j];
    const double directly_scaled=(torque[j]-scale*gyro)/(scale*inertia[j]);
    close(parameterized,directly_scaled,1e-12,"Common inertia scaling must preserve gyroscopic acceleration");
  }
}
} // namespace

int main() {
  const std::vector<std::pair<std::string,std::function<void()>>> checks{
    {"mass identified from causal force and velocity observations",learns_mass_from_motion},
    {"hover information and accumulated least squares",hover_is_informative_and_data_accumulate},
    {"contact, invalid, uninformative and out-of-range samples rejected",invalid_samples_leave_estimate_unchanged},
    {"reset and no shared scenario or plant-mass input",reset_and_instances_do_not_share_hidden_mass},
    {"common inertia scale preserves Euler gyroscopic term",common_inertia_scale_preserves_gyro_term}
  };
  int failures=0;
  for(const auto& [name,test]:checks) {
    try {test();std::cout<<"PASS "<<name<<'\n';}
    catch(const std::exception& error){++failures;std::cerr<<"FAIL "<<name<<": "<<error.what()<<'\n';}
  }
  std::cout<<checks.size()-failures<<'/'<<checks.size()<<" mass-estimator checks passed\n";
  return failures?1:0;
}
