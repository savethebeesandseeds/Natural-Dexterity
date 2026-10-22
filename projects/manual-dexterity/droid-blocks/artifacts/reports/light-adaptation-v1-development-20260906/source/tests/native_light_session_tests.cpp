#include "droid/light_session.hpp"
#include "droid/construction.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using Json = nlohmann::json;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejected(F&& call) { bool failed=false; try { call(); } catch(const std::exception&) { failed=true; } check(failed,"invalid command accepted"); }
int main() {
    try {
        droid::ConstructionWorld world;
        world.rebuild(droid::ConstructionWorld::light_assembly());
        const auto first_observation=world.observation();
        droid::LightControl baseline;
        baseline.reset("zero");
        const double first_effort=baseline.act(first_observation);
        baseline.observe(world.step(first_effort));
        const auto baseline_before=baseline.diagnostics();
        rejected([&] { (void)baseline.act(first_observation); });
        auto corrupted=world.observation();
        corrupted["reward_sensors"][1]["observations"]["illuminance_lux"]=1.0;
        rejected([&] { (void)baseline.act(corrupted); });
        check(baseline.diagnostics()==baseline_before,"invalid baseline observation changed control");
        rejected([&] { baseline.freeze_learning(); });
        check(baseline.diagnostics()==baseline_before,"nonlearning freeze changed baseline state");
        baseline.observe(world.step(baseline.act(world.observation())));
        droid::LightControl frozen_control, paired_control;
        frozen_control.reset("learner", 29); paired_control.reset("learner", 29);
        droid::ConstructionWorld frozen_world, paired_world;
        frozen_world.rebuild(droid::ConstructionWorld::light_assembly());
        paired_world.rebuild(droid::ConstructionWorld::light_assembly());
        const auto fresh_control = frozen_control.diagnostics();
        rejected([&] { frozen_control.freeze_learning(); });
        check(frozen_control.diagnostics()==fresh_control,"uninitialized control freeze changed state");
        for (std::size_t step=0; step<32; ++step) {
            const double command=frozen_control.act(frozen_world.observation());
            check(command==paired_control.act(paired_world.observation()),"paired native prefix command mismatch");
            if (step==5) {
                const auto pending=frozen_control.diagnostics();
                rejected([&] { frozen_control.freeze_learning(); });
                check(frozen_control.diagnostics()==pending,"pending control freeze changed state");
            }
            frozen_control.observe(frozen_world.step(command));
            paired_control.observe(paired_world.step(command));
            check(frozen_control.diagnostics()==paired_control.diagnostics(),"paired native controller prefix mismatch");
        }
        const auto prefix=frozen_control.diagnostics();
        const auto parameter_hash=prefix.at("parameter_fingerprint_sha256");
        frozen_control.freeze_learning();
        const auto frozen=frozen_control.diagnostics();
        frozen_control.freeze_learning();
        check(frozen_control.diagnostics()==frozen,"forwarded freeze is not idempotent");
        check(frozen.at("learning_frozen")==true && frozen.at("parameter_fingerprint_sha256")==parameter_hash,
              "forwarded freeze changed parameters or omitted its flag");
        frozen_world.set_sun(Json{{"position_m",{-0.3,0,0.2}},{"intensity_lux",1000}});
        for (std::size_t step=32; step<75; ++step) {
            frozen_control.observe(frozen_world.step(frozen_control.act(frozen_world.observation())));
            check(frozen_control.diagnostics().at("parameter_fingerprint_sha256")==parameter_hash,
                  "native frozen continuation changed parameters");
        }
        check(frozen_control.diagnostics().at("step")==75 &&
              frozen_control.diagnostics().at("unique_light_samples").get<std::size_t>() > prefix.at("unique_light_samples").get<std::size_t>(),
              "freeze stopped native sensing or control time");
        frozen_control.reset("learner",29);
        check(frozen_control.diagnostics()==fresh_control,"LightControl reset retained freeze state");
        droid::LightSession session;
        auto initial = session.state();
        check(initial.at("status") == "ready", "initial state");
        check(initial.at("assembly").at("schema") == "construction_kit_v2", "missing physical light");
        rejected([&] { (void)session.control(Json{{"action","freeze_learning"}}); });
        check(session.state()==initial,"development intervention leaked into browser API");
        rejected([&] { (void)session.control(Json{{"action","start"},{"mode","oracle"}}); });
        check(session.state() == initial, "invalid start mutated state");
        (void)session.control(Json{{"action","start"},{"mode","learner"}});
        session.advance_steps(75);
        auto running = session.state();
        check(running.at("status") == "running", "learner failed early");
        check(std::abs(running.at("elapsed_s").get<double>()-1.5)<1e-9, "continuous step count");
        rejected([&] { (void)session.control(Json{{"action","build"},{"assembly",initial.at("assembly")}}); });
        check(session.state() == running, "running build mutated state");
        auto moved = session.control(Json{{"action","sun"},{"sun",Json{{"position_m",{-0.3,0,0.2}},{"intensity_lux",1000}}}});
        check(moved.at("elapsed_s") == running.at("elapsed_s"), "sun reset time");
        auto mechanics_before=running.at("physics").at("diagnostics");
        auto mechanics_after=moved.at("physics").at("diagnostics");
        mechanics_before.erase("instantaneous_light_lux");
        mechanics_after.erase("instantaneous_light_lux");
        check(mechanics_before==mechanics_after && moved.at("physics").at("geometry")==running.at("physics").at("geometry"), "sun moved mechanics");
        check(moved.at("learner") == running.at("learner"), "sun reset learning");
        check(moved.at("observation") == running.at("observation"), "sun bypassed sensor delay");
        auto paused = session.control(Json{{"action","pause"}});
        session.advance_steps(100);
        check(session.state() == paused, "paused world advanced");
        rejected([&] { (void)session.control(Json{{"action","sun"},{"sun",Json{{"position_m",{3,0,0}},{"intensity_lux",1000}}}}); });
        check(session.state() == paused, "bad sun changed state");
        (void)session.control(Json{{"action","play"}});
        session.advance_steps(25);
        auto after = session.state();
        check(after.at("elapsed_s") == 2.0, "resume reset body");
        check(after.at("history").size() == 20, "history continuity");
        auto reset = session.control(Json{{"action","reset"}});
        check(reset.at("elapsed_s") == 0 && reset.at("history").empty(), "reset history");
        check(reset.at("physics").at("sun") == after.at("physics").at("sun"), "reset lost current sun");
        auto malformed = initial.at("assembly"); malformed["light"]["side"] = -1;
        rejected([&] { (void)session.control(Json{{"action","build"},{"assembly",malformed}}); });
        check(session.state() == reset, "invalid assembly destroyed ready body");
        auto body = initial.at("assembly"); body["blocks"].push_back(Json{{"id","weight"},{"segment",1},{"slot",0},{"side",1}});
        auto rebuilt = session.control(Json{{"action","build"},{"assembly",body}});
        check(rebuilt.at("assembly") == body, "assembly replacement");
        for (const std::string mode : {"zero","rhythm","random"}) {
            (void)session.control(Json{{"action","start"},{"mode",mode}});
            session.advance_steps(6000);
            const auto final = session.state();
            check(final.at("status") == "completed", "baseline failed full horizon");
            check(final.at("elapsed_s") == 120.0, "baseline unequal horizon");
            check(final.at("history").size() == 1200, "history bounds");
            session.advance_steps(10);
            check(session.state() == final, "completed run advanced");
        }
        std::cout << "light session contract passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
