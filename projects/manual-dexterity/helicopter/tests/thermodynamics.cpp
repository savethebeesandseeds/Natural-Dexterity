#include "rotor_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using helicopter::rotor::Params;
using State = helicopter::rotor::State<double>;
using Input = helicopter::rotor::Input<double>;
using Vec3 = std::array<double, 3>;
namespace model = helicopter::rotor;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double relative, double absolute,
           const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > absolute + relative * std::max(std::abs(actual), std::abs(expected))) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected));
    }
}

double dot(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Independent quaternion expansion: do not use the model's coordinate helpers.
Vec3 body_to_world(const State& x, const Vec3& b) {
    const double w=x[6], qx=x[7], qy=x[8], qz=x[9];
    return {
        (1-2*qy*qy-2*qz*qz)*b[0] + (2*qx*qy-2*w*qz)*b[1] + (2*qx*qz+2*w*qy)*b[2],
        (2*qx*qy+2*w*qz)*b[0] + (1-2*qx*qx-2*qz*qz)*b[1] + (2*qy*qz-2*w*qx)*b[2],
        (2*qx*qz-2*w*qy)*b[0] + (2*qy*qz+2*w*qx)*b[1] + (1-2*qx*qx-2*qy*qy)*b[2]
    };
}

State example(const Params& p) {
    State x{};
    x[2]=2.5;
    x[3]=.55; x[4]=-.25; x[5]=.12;
    const double roll=.12, pitch=-.08, yaw=.2;
    const double cr=std::cos(roll/2), sr=std::sin(roll/2);
    const double cp=std::cos(pitch/2), sp=std::sin(pitch/2);
    const double cy=std::cos(yaw/2), sy=std::sin(yaw/2);
    x[6]=cr*cp*cy+sr*sp*sy;
    x[7]=sr*cp*cy-cr*sp*sy;
    x[8]=cr*sp*cy+sr*cp*sy;
    x[9]=cr*cp*sy-sr*sp*cy;
    x[10]=.18; x[11]=-.11; x[12]=.09;
    x[13]=.10; x[14]=.04; x[15]=-.03; x[16]=.13;
    x[17]=3.0; x[18]=4.0; x[19]=.02; x[20]=-.04;
    x[21]=p.ambient_temperature+12;
    x[22]=p.ambient_temperature+9;
    return x;
}

Input actual_input(const State& x) {
    return {x[13],x[14],x[15],x[16]};
}

double mechanical_energy(const State& x, const Params& p) {
    double result=p.mass*p.gravity*x[2];
    for (int axis=0; axis<3; ++axis) {
        result+=.5*p.mass*x[3+axis]*x[3+axis];
        result+=.5*p.inertia[axis]*x[10+axis]*x[10+axis];
    }
    return result;
}

double inflow_energy(const State& x, const Params& p) {
    return .5*p.main_inflow_mass*x[17]*x[17] +
           .5*p.tail_inflow_mass*x[18]*x[18];
}

double total_stored_energy(const State& x, const Params& p) {
    return mechanical_energy(x,p)+inflow_energy(x,p)+
           p.main_thermal_capacity*(x[21]-p.ambient_temperature)+
           p.tail_thermal_capacity*(x[22]-p.ambient_temperature);
}

double mechanical_energy_rate(const State& x, const State& dx, const Params& p) {
    double result=p.mass*p.gravity*dx[2];
    for (int axis=0; axis<3; ++axis) {
        result+=p.mass*x[3+axis]*dx[3+axis];
        result+=p.inertia[axis]*x[10+axis]*dx[10+axis];
    }
    return result;
}

double wake_energy_rate(const State& x, const State& dx, const Params& p) {
    return p.main_inflow_mass*x[17]*dx[17] +
           p.tail_inflow_mass*x[18]*dx[18];
}

double thermal_energy_rate(const State& dx, const Params& p) {
    return p.main_thermal_capacity*dx[21]+p.tail_thermal_capacity*dx[22];
}

State displaced(const State& x, const State& direction, double h) {
    State result=x;
    for (int i=0; i<model::NX; ++i) result[i]+=h*direction[i];
    return result;
}

void ledger_and_entropy() {
    Params p;
    for (int scenario=0; scenario<12; ++scenario) {
        State x=example(p);
        x[3]+=.045*scenario;
        x[4]-=.017*scenario;
        x[10]+=.011*scenario;
        x[13]+=.002*scenario;
        x[16]+=.003*scenario;
        x[17]+=.045*scenario;
        x[18]+=.060*scenario;
        x[21]+=.8*scenario;
        x[22]+=.6*scenario;
        const Vec3 wind{.35-.05*scenario,-.12+.025*scenario,.02};
        const Input u=actual_input(x);
        const auto f=model::evaluate(x,u,wind,p);
        const auto dx=model::derivative(x,u,wind,p);
        const Vec3 rotor_world=body_to_world(x,f.force_body);
        const Vec3 velocity{x[3],x[4],x[5]};
        const Vec3 omega{x[10],x[11],x[12]};
        Vec3 relative{},total_force{};
        for (int axis=0; axis<3; ++axis) {
            relative[axis]=velocity[axis]-wind[axis];
            total_force[axis]=rotor_world[axis]+f.drag_world[axis];
            close(dx[axis],velocity[axis],1e-11,1e-11,"Kinematic velocity");
            close(p.mass*dx[3+axis],total_force[axis]-(axis==2?p.mass*p.gravity:0),
                  1e-10,1e-9,"World-frame force law");
        }
        const double wind_power=dot(total_force,wind);
        const double drag_loss=-dot(f.drag_world,relative);
        const double angular_loss=-dot(f.damping_body,omega);
        close(f.wind_power,wind_power,1e-10,1e-9,"External wind work");
        close(f.drag_dissipation,drag_loss,1e-10,1e-9,"Air-relative drag dissipation");
        close(f.angular_dissipation,angular_loss,1e-10,1e-9,"Angular dissipation");

        // The left side uses external ports; the right side differentiates
        // independently defined storage functions, not reported residual fields.
        const double storage_rate=mechanical_energy_rate(x,dx,p)+
                                  wake_energy_rate(x,dx,p)+thermal_energy_rate(dx,p);
        const double rejected=f.wake_dissipation+drag_loss+angular_loss+
                              f.main_heat_out+f.tail_heat_out;
        close(f.electrical_power+wind_power,storage_rate+rejected,
              1e-9,1e-7,"First law from state derivatives");
        close(f.stored_wake_energy,inflow_energy(x,p),1e-11,1e-10,"Inflow storage");

        // Entropy accumulation in the motor bodies plus entropy received by the
        // ambient reservoir independently reconstructs generation.
        const double entropy_storage_rate=p.main_thermal_capacity*dx[21]/x[21]+
                                          p.tail_thermal_capacity*dx[22]/x[22];
        const double reservoir_entropy_rate=rejected/p.ambient_temperature;
        close(f.entropy_rate,entropy_storage_rate+reservoir_entropy_rate,
              1e-9,1e-9,"Second law from entropy storage and ambient inflow");
        for (const double irreversible : {f.entropy_motor,f.entropy_heat_transfer,
             f.entropy_wake,f.entropy_drag,f.main_copper_loss,f.tail_copper_loss,
             f.main_braking_loss,f.tail_braking_loss}) {
            require(std::isfinite(irreversible)&&irreversible>=-1e-11,
                    "Negative or nonfinite irreversible term");
        }

        const double h=1e-5;
        const double commitment_rate=(model::terminal_entropy_commitment(displaced(x,dx,h),p)-
                                      model::terminal_entropy_commitment(displaced(x,dx,-h),p))/(2*h);
        close(f.entropy_rate+commitment_rate,
              (f.electrical_power+wind_power-mechanical_energy_rate(x,dx,p))/p.ambient_temperature,
              2e-7,2e-8,"Terminal completion closes the entropy ledger");
    }
}

void passive_cooldown() {
    Params p;
    State start=example(p);
    start[21]=p.ambient_temperature+42;
    start[22]=p.ambient_temperature+27;
    const double tau_main=p.main_thermal_capacity/p.main_heat_transfer;
    const double tau_tail=p.tail_thermal_capacity/p.tail_heat_transfer;
    const double duration=20*std::max(tau_main,tau_tail);
    const auto cooled=[&](double t) {
        State x=start;
        x[21]=p.ambient_temperature+42*std::exp(-t/tau_main);
        x[22]=p.ambient_temperature+27*std::exp(-t/tau_tail);
        return x;
    };
    const auto entropy_rate=[&](double t) {
        const State x=cooled(t);
        return p.main_heat_transfer*(x[21]-p.ambient_temperature)*
                   (1/p.ambient_temperature-1/x[21])+
               p.tail_heat_transfer*(x[22]-p.ambient_temperature)*
                   (1/p.ambient_temperature-1/x[22]);
    };
    // Simpson quadrature of reservoir/body entropy fluxes, independent of the
    // model's closed-form exergy function and controller dynamics.
    constexpr int intervals=4000;
    const double dt=duration/intervals;
    double integral=entropy_rate(0)+entropy_rate(duration);
    for (int i=1; i<intervals; ++i) integral+=(i%2?4:2)*entropy_rate(i*dt);
    integral*=dt/3;
    const State end=cooled(duration);
    close(integral,(model::thermal_exergy(start,p)-model::thermal_exergy(end,p))/
                    p.ambient_temperature,1e-8,1e-9,"Exact passive cooldown entropy");

    const double heat_to_ambient=p.main_thermal_capacity*(start[21]-end[21])+
                                 p.tail_thermal_capacity*(start[22]-end[22]);
    const double body_entropy_change=p.main_thermal_capacity*std::log(end[21]/start[21])+
                                    p.tail_thermal_capacity*std::log(end[22]/start[22]);
    close(integral,body_entropy_change+heat_to_ambient/p.ambient_temperature,
          1e-8,1e-9,"Cooling first/second law analytic solution");
    State ambient=start;
    ambient[21]=ambient[22]=p.ambient_temperature;
    close(model::thermal_exergy(ambient,p),0,0,1e-10,"Ambient thermal exergy");
    require(model::thermal_exergy(start,p)>0,"Hot storage has positive remaining exergy");
    close(model::terminal_entropy_commitment(start,p),
          (model::thermal_exergy(start,p)+inflow_energy(start,p))/p.ambient_temperature,
          1e-11,1e-10,"Wake and thermal completion counted exactly once");
}

void spinning_rotors_and_work() {
    Params p;
    State zero{};
    zero[2]=2.5; zero[6]=1;
    zero[21]=zero[22]=p.ambient_temperature;
    const Vec3 calm{};
    const auto idle=model::evaluate(zero,Input{},calm,p);
    const double pi=std::acos(-1.0);
    const double main_profile=p.air_density*pi*p.main_radius*p.main_radius*
        std::pow(p.main_omega*p.main_radius,3)*p.main_solidity*p.main_profile_cd/8;
    const double tail_profile=p.air_density*pi*p.tail_radius*p.tail_radius*
        std::pow(p.tail_omega*p.tail_radius,3)*p.tail_solidity*p.tail_profile_cd/8;
    close(idle.main_profile_power,main_profile,1e-10,1e-9,"Analytic main profile power");
    close(idle.tail_profile_power,tail_profile,1e-10,1e-9,"Analytic tail profile power");
    require(idle.main_profile_power>0&&idle.tail_profile_power>0,
            "Governed spinning rotors require profile power at zero collective");
    require(idle.main_thrust<1e-3&&idle.tail_thrust<1e-3,
            "Zero pitch/inflow must not invent material thrust");
    require(idle.electrical_power>main_profile+tail_profile,
            "Spinning rotors require positive electrical power including losses");

    State hover=zero;
    hover[13]=.10; hover[16]=.13;
    const Input command=actual_input(hover);
    // Find a steady inflow for this fixed-pitch, motionless fixture. Momentum
    // theory is checked separately below rather than assumed in the iteration.
    for (int i=0; i<1000; ++i) {
        const auto f=model::evaluate(hover,command,calm,p);
        hover[17]=.9*hover[17]+.1*f.main_inflow_equilibrium;
        hover[18]=.9*hover[18]+.1*f.tail_inflow_equilibrium;
    }
    const auto h=model::evaluate(hover,command,calm,p);
    require(h.main_thrust>1&&h.tail_thrust>0,"Loaded hover fixture has rotor thrust");
    close(h.main_inflow_dot,0,0,1e-7,"Steady main inflow");
    close(h.tail_inflow_dot,0,0,1e-7,"Steady tail inflow");
    const double ideal_main=std::pow(h.main_thrust,1.5)/
        std::sqrt(2*p.air_density*pi*p.main_radius*p.main_radius);
    const double ideal_tail=std::pow(h.tail_thrust,1.5)/
        std::sqrt(2*p.air_density*pi*p.tail_radius*p.tail_radius);
    close(h.main_induced_power,p.main_induced_factor*ideal_main,2e-6,1e-6,
          "Independent main hover momentum power");
    close(h.tail_induced_power,p.tail_induced_factor*ideal_tail,2e-6,1e-6,
          "Independent tail hover momentum power");
    require(h.electrical_power>h.shaft_power&&h.shaft_power>0,
            "Loaded hover cannot have free or negative electrical power");

    State rotating=hover;
    rotating[10]=.35; rotating[11]=-.22; rotating[12]=.15;
    rotating[19]=.08; rotating[20]=-.06;
    const auto r=model::evaluate(rotating,command,calm,p);
    const auto dx=model::derivative(rotating,command,calm,p);
    const Vec3 omega{rotating[10],rotating[11],rotating[12]};
    const double rotational_work=dot(r.torque_body,omega);
    require(std::abs(rotational_work)>1e-3,"Fixture exercises rotor moment work");
    close(r.shaft_power-wake_energy_rate(rotating,dx,p)-r.wake_dissipation,
          rotational_work,1e-9,1e-7,"Rotor moments have a shaft-power cost");
    close(r.main_shaft_power,p.main_omega*r.main_aero_torque,1e-10,1e-9,
          "Main shaft torque-speed relation");
    close(r.tail_shaft_power,p.tail_omega*r.tail_aero_torque,1e-10,1e-9,
          "Tail shaft torque-speed relation");
}

void finite_difference_storage() {
    Params p;
    const State x=example(p);
    const Input u=actual_input(x);
    const Vec3 wind{.3,-.2,.05};
    const auto f=model::evaluate(x,u,wind,p);
    const auto dx=model::derivative(x,u,wind,p);
    const double external_rate=f.electrical_power+f.wind_power-f.wake_dissipation-
        f.drag_dissipation-f.angular_dissipation-f.main_heat_out-f.tail_heat_out;
    const auto error=[&](double step) {
        return (total_stored_energy(displaced(x,dx,step),p)-total_stored_energy(x,p))/step-
               external_rate;
    };
    const double coarse=error(1e-3),fine=error(5e-4);
    require(std::abs(coarse)>1e-6,"Finite difference fixture has measurable curvature");
    close(fine,.5*coarse,2e-5,2e-6,"Storage derivative converges under timestep refinement");
    double curvature=0;
    for (int axis=0; axis<3; ++axis) {
        curvature+=p.mass*dx[3+axis]*dx[3+axis]+
                   p.inertia[axis]*dx[10+axis]*dx[10+axis];
    }
    curvature+=p.main_inflow_mass*dx[17]*dx[17]+p.tail_inflow_mass*dx[18]*dx[18];
    close(coarse,.5e-3*curvature,2e-5,2e-6,
          "Finite-step residual is the known energy curvature, not missing power");
}

void braking_and_thermal_feedback() {
    Params p;
    State x{};
    x[2]=2.5; x[6]=1;
    x[17]=6; x[18]=8;
    x[21]=x[22]=p.ambient_temperature;
    const Vec3 wind{};
    const Input command{};
    const auto braking=model::evaluate(x,command,wind,p);
    const auto dx=model::derivative(x,command,wind,p);
    require(braking.main_shaft_power<0,
            "High stored inflow and collective cut exercise negative shaft work");
    require(braking.main_braking_loss>0&&braking.electrical_power>=0,
            "Non-regenerative drive must dissipate returned work without charging the supply");
    require(wake_energy_rate(x,dx,p)<0,"Braking releases previously stored inflow energy");
    const double storage=mechanical_energy_rate(x,dx,p)+wake_energy_rate(x,dx,p)+
                         thermal_energy_rate(dx,p);
    close(braking.electrical_power,
          storage+braking.wake_dissipation+braking.drag_dissipation+
          braking.angular_dissipation+braking.main_heat_out+braking.tail_heat_out,
          1e-9,1e-7,"First law also holds while the shaft returns work");

    State cold=example(p),hot=cold;
    hot[21]+=10; hot[22]+=10;
    const Input u=actual_input(cold);
    const auto a=model::evaluate(cold,u,wind,p);
    const auto b=model::evaluate(hot,u,wind,p);
    close(a.shaft_power,b.shaft_power,1e-12,1e-10,"Temperature change preserves mechanical demand");
    require(b.main_copper_loss>a.main_copper_loss&&b.tail_copper_loss>a.tail_copper_loss&&
            b.electrical_power>a.electrical_power,
            "Hotter winding resistance must affect electrical demand, not just its display");
    const double predicted_increment=10*p.resistance_alpha*
        (a.main_current*a.main_current*p.main_resistance+
         a.tail_current*a.tail_current*p.tail_resistance);
    close(b.electrical_power-a.electrical_power,predicted_increment,1e-10,1e-9,
          "Independent Joule-loss temperature sensitivity");
}
} // namespace

int main() {
    const std::vector<std::pair<std::string,std::function<void()>>> checks{
        {"energy, entropy, wind, and terminal balances",ledger_and_entropy},
        {"analytic passive cooling and terminal storage",passive_cooldown},
        {"hover, spinning rotors, and cyclic mechanical work",spinning_rotors_and_work},
        {"independent finite-difference energy derivative",finite_difference_storage},
        {"non-regenerative braking and resistance feedback",braking_and_thermal_feedback}
    };
    int failures=0;
    for (const auto& [name,check] : checks) {
        try {
            check();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << (checks.size()-static_cast<std::size_t>(failures)) << '/'
              << checks.size() << " thermodynamic checks passed\n";
    return failures==0?0:1;
}
