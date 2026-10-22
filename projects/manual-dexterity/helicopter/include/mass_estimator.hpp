#pragma once

#include "rotor_model.hpp"

#include <cmath>
#include <cstdint>
#include <string>

namespace helicopter {

// Causal identification of one constant mass from free-flight observations.
// The inputs contain no plant parameters, scenario ID, or true mass. Applied
// force excludes gravity and contact and includes rotor force and airframe drag.
// Contact must describe the whole observed interval, not merely its endpoint.
class MassEstimator {
 public:
  static constexpr double minimum_mass_scale=.5;
  static constexpr double maximum_mass_scale=1.5;
  // Threshold on q dot q, in (m/s^2)^2, where q=dv/dt+g*e_z.
  static constexpr double minimum_information=1e-8;

  void reset() {
    information_=cross_information_=0;
    mass_=nominal_mass();
    accepted_samples_=rejected_samples_=0;
    last_rejection_.clear();
  }

  bool observe(const rotor::Vec3<double>& force_world,
               const rotor::Vec3<double>& velocity_before,
               const rotor::Vec3<double>& velocity_after,
               double dt,bool contact) {
    const auto reject=[&](const char* reason) {
      ++rejected_samples_;last_rejection_=reason;return false;
    };
    if(contact)return reject("contact_interval");
    if(!std::isfinite(dt)||dt<=0)return reject("invalid_timestep");

    double information=0,cross_information=0;
    const double gravity=rotor::Params{}.gravity;
    for(int j=0;j<3;++j) {
      if(!std::isfinite(force_world[j])||!std::isfinite(velocity_before[j])||
         !std::isfinite(velocity_after[j]))return reject("nonfinite_observation");
      const double q=(velocity_after[j]-velocity_before[j])/dt+(j==2?gravity:0.0);
      if(!std::isfinite(q))return reject("nonfinite_specific_acceleration");
      information+=q*q;
      cross_information+=q*force_world[j];
    }
    if(!std::isfinite(information)||!std::isfinite(cross_information))
      return reject("nonfinite_information");
    if(information<=minimum_information)return reject("uninformative_interval");

    // Reject inconsistent/out-of-domain observations; clipping them would
    // silently bias the fit and make an invalid sample look informative.
    const double observed_mass=cross_information/information;
    const double minimum_mass=minimum_mass_scale*nominal_mass();
    const double maximum_mass=maximum_mass_scale*nominal_mass();
    if(!std::isfinite(observed_mass)||observed_mass<minimum_mass||observed_mass>maximum_mass)
      return reject("mass_outside_admissible_range");

    const double next_information=information_+information;
    const double next_cross_information=cross_information_+cross_information;
    const double next_mass=next_cross_information/next_information;
    if(!std::isfinite(next_information)||!std::isfinite(next_cross_information)||
       !std::isfinite(next_mass)||next_mass<minimum_mass||next_mass>maximum_mass)
      return reject("invalid_accumulated_estimate");

    information_=next_information;cross_information_=next_cross_information;
    mass_=next_mass;++accepted_samples_;last_rejection_.clear();
    return true;
  }

  [[nodiscard]] double mass_scale() const {return mass_/nominal_mass();}
  [[nodiscard]] double mass_kg() const {return mass_;}
  [[nodiscard]] std::uint64_t samples() const {return accepted_samples_;}
  [[nodiscard]] nlohmann::json metadata() const {
    return {{"mass_kg",mass_kg()},{"mass_scale",mass_scale()},
      {"accepted_samples",accepted_samples_},{"rejected_samples",rejected_samples_},
      {"estimate_available",accepted_samples_>0},{"using_nominal_prior",accepted_samples_==0},
      {"nominal_prior_mass_kg",nominal_mass()},{"admissible_mass_scale",{minimum_mass_scale,maximum_mass_scale}},
      {"information",information_},{"information_units","(m/s^2)^2"},
      {"minimum_sample_information",minimum_information},{"last_rejection",last_rejection_},
      {"method","Accumulated least squares: F = m * (dv/dt + g e_z); m = sum(q dot F) / sum(q dot q)"},
      {"inertia_assumption","Inertia uses the identified mass scale times nominal inertia; common scaling is assumed, not independently identified"}};
  }

 private:
  static double nominal_mass() {return rotor::Params{}.mass;}
  double information_=0,cross_information_=0;
  double mass_=nominal_mass();
  std::uint64_t accepted_samples_=0,rejected_samples_=0;
  std::string last_rejection_;
};

} // namespace helicopter
