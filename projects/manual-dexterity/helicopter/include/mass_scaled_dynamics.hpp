#pragma once

#include "rotor_model.hpp"

namespace helicopter::rotor {

// The declared uncertainty family scales total mass and all three principal
// inertias together while preserving geometry. Aerodynamics and entropy terms
// depend on the measured state, so only rigid-body accelerations change.
// Keeping the ratio symbolic lets one compiled NLP serve every causal estimate.
template<class S> State<S> mass_scaled_derivative(const State<S>& x,
    const Input<S>& u,const Vec3<S>& wind,const Params& nominal,const S& ratio) {
  auto result=derivative(x,u,wind,nominal);
  const Vec3<S> omega{x[10],x[11],x[12]};
  const Vec3<S> angular_momentum{S(nominal.inertia[0])*omega[0],
    S(nominal.inertia[1])*omega[1],S(nominal.inertia[2])*omega[2]};
  const auto gyroscopic=detail::cross(omega,angular_momentum);
  for(int j=0;j<3;++j) {
    const S gravity=j==2?S(nominal.gravity):S(0);
    result[3+j]=(result[3+j]+gravity)/ratio-gravity;
    // omega x (ratio I omega)/(ratio I) is unchanged; scale only the
    // applied-torque contribution rather than dividing the entire derivative.
    const S gyro_acceleration=gyroscopic[j]/S(nominal.inertia[j]);
    result[10+j]=(result[10+j]+gyro_acceleration)/ratio-gyro_acceleration;
  }
  return result;
}

} // namespace helicopter::rotor
