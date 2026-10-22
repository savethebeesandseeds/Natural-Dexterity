#include "droid/construction.hpp"
#include "droid/light_session.hpp"
#include "droid/policy.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Json = nlohmann::json;
constexpr double kDt = .02;
constexpr std::size_t kSteps = 6000, kMove = 3000;
Json sun(bool left) { return Json{{"position_m",{left ? -.30 : .30,0,.20}},{"intensity_lux",1000}}; }
struct Metrics {
    double reward{}, light_reward{}, imu_reward{}, lux_integral{}, energy{}, max_current{}, max_temperature{}, max_speed{};
    std::size_t steps{}, vetoes{}, flagged{}, unique_light{};
    std::uint64_t last_sequence{};
    void add(const Json& transition, const Json& physical, double energy_delta) {
        ++steps;
        energy += energy_delta;
        reward += transition.at("reward").get<double>();
        for (const auto& component : transition.at("reward_components")) {
            if (component.at("family_id") == "ambient_light_v0") light_reward += component.at("transition_reward").get<double>();
            if (component.at("family_id") == "imu_6axis_v0") imu_reward += component.at("transition_reward").get<double>();
        }
        const auto& feedback = transition.at("observation").at("actuator_feedback").at(0).at("feedback");
        max_current = std::max(max_current,std::abs(feedback.at("current_a").get<double>()));
        max_temperature = std::max(max_temperature,feedback.at("temperature_c").get<double>());
        max_speed = std::max(max_speed,std::abs(feedback.at("velocity_rad_s").get<double>()));
        if (!feedback.at("fault_flags").empty()) ++flagged;
        if (transition.at("safety").at("veto") == true) ++vetoes;
        const auto& light = physical.at("light_sensor");
        if (light.at("valid") == true) {
            lux_integral += kDt * light.at("observations").at("illuminance_lux").get<double>();
            auto seq = light.at("sequence").get<std::uint64_t>();
            if (seq != last_sequence) { ++unique_light; last_sequence = seq; }
        }
    }
    Json json() const {
        return Json{{"steps",steps},{"duration_s",steps*kDt},{"integrated_sensor_reward",reward},
            {"integrated_light_reward",light_reward},{"integrated_imu_reward",imu_reward},
            {"mean_sensor_reward_rate",steps ? reward/(steps*kDt) : 0},
            {"mean_received_lux",steps ? lux_integral/(steps*kDt) : 0},
            {"unique_light_samples",unique_light},{"electrical_energy_j",energy},
            {"max_current_a",max_current},{"max_temperature_c",max_temperature},
            {"max_motor_speed_rad_s",max_speed},{"native_veto_steps",vetoes},{"feedback_flag_steps",flagged}};
    }
};
Json trial(const Json& assembly,const std::string& mode,std::uint64_t seed) {
    droid::ConstructionWorld world;
    world.rebuild(assembly);
    world.set_sun(sun(false));
    droid::LightControl controller;
    controller.reset(mode,seed);
    Metrics all, before, after, first_tail, last_tail;
    Json trace=Json::array(), efforts=Json::array(), curve=Json::array(), at_move;
    std::string error;
    std::size_t steps=0;
    double previous_energy=0;
    try {
        for (;steps<kSteps;) {
            if (steps==kMove) { at_move=controller.diagnostics(); world.set_sun(sun(true)); }
            const double effort=controller.act(world.observation());
            const Json transition=world.step(effort);
            ++steps;
            const Json physical=world.state();
            efforts.push_back(effort);
            trace.push_back(transition.at("observation"));
            const double total_energy=physical.at("diagnostics").at("electrical_energy_j");
            const double energy_delta=total_energy-previous_energy;
            previous_energy=total_energy;
            all.add(transition,physical,energy_delta);
            if (steps<=kMove) before.add(transition,physical,energy_delta); else after.add(transition,physical,energy_delta);
            if (steps>2000 && steps<=3000) first_tail.add(transition,physical,energy_delta);
            if (steps>5000) last_tail.add(transition,physical,energy_delta);
            if (steps%5==0) curve.push_back(Json{{"time_s",steps*kDt},
                {"illuminance_lux",physical.at("light_sensor").at("observations").at("illuminance_lux")},
                {"sensor_reward_rate",physical.at("reward").at("rate")},{"effort",effort}});
            controller.observe(transition);
        }
    } catch (const std::exception& exception) { error=exception.what(); }
    bool replay_matches=true;
    std::string replay_error;
    std::size_t replayed_steps=0;
    try {
        droid::ConstructionWorld replay;
        replay.rebuild(assembly);
        replay.set_sun(sun(false));
        droid::LightControl replay_controller;
        replay_controller.reset(mode,seed);
        for (std::size_t i=0;i<efforts.size();++i) {
            if (i==kMove) replay.set_sun(sun(true));
            const double effort=replay_controller.act(replay.observation());
            if (effort!=efforts.at(i).get<double>()) { replay_matches=false; break; }
            const auto result=replay.step(effort);
            if (result.at("observation")!=trace.at(i)) { replay_matches=false; break; }
            ++replayed_steps;
            replay_controller.observe(result);
        }
    } catch(const std::exception& exception) {
        replay_error=exception.what();
        replay_matches=!error.empty() && replay_error==error && replayed_steps==efforts.size();
    }
    return Json{{"mode",mode},{"seed",seed},{"completed",steps==kSteps&&error.empty()},
        {"completed_steps",steps},{"error",error},{"exact_controller_and_observation_replay",replay_matches},
        {"replay_error",replay_error},{"replay_scope","recorded command/observation prefix; completed determines full horizon"},{"observation_trace_sha256",droid::sha256_hex(trace.dump())},
        {"whole_run",all.json()},{"before_move",before.json()},{"after_move",after.json()},
        {"first_phase_final_20s",first_tail.json()},{"second_phase_final_20s",last_tail.json()},
        {"controller_before_move",at_move},{"controller_final",controller.diagnostics()},
        {"efforts",efforts},{"curve",curve}};
}
void save(const std::filesystem::path& output,const Json& report) {
    const auto next=output/"progress.next.json";
    std::ofstream stream(next,std::ios::binary|std::ios::trunc);
    stream << report.dump(2) << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("cannot save light experiment report");
    std::filesystem::rename(next,output/"report.json");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3||std::string(argv[1])!="--output")
            throw std::invalid_argument("Usage: droid-light-experiment --output NEW_DIRECTORY");
        const std::filesystem::path output(argv[2]);
        if(!std::filesystem::create_directory(output)) throw std::invalid_argument("output directory already exists; preserve previous evidence");
        Json report{{"schema","light_learning_development_v1"},{"complete",false},
            {"exposure","Known development bodies, seeds and light schedule; no frozen-v3 challenge exposure"},
            {"protocol",Json{{"steps",kSteps},{"control_dt_s",kDt},{"sun_move_step",kMove},
                {"initial_sun",sun(false)},{"moved_sun",sun(true)},{"reset_at_sun_move",false}}},
            {"physics_spec",droid::ConstructionWorld::light_specification()},
            {"control_spec",droid::LightControl::specification()},{"source_sha256",Json::object()},
            {"constructions",Json::array()}};
        for(const std::string file:{"src/construction.cpp","include/droid/construction.hpp",
            "src/light_learner.cpp","include/droid/light_learner.hpp","src/light_session.cpp",
            "include/droid/light_session.hpp","src/light_experiment_cli.cpp","docs/LIGHT_LEARNING_EXPERIMENT.md",
            "config/module_catalog.json","models/droid.xml","config/learning_experiment_v3.json",
            "artifacts/light-search-policy-v3-seed0.json"}) {
            report["source_sha256"][file]=droid::sha256_file(file);
            const auto target=output/"source"/file;
            std::filesystem::create_directories(target.parent_path());
            std::filesystem::copy_file(file,target);
        }
        Json original=droid::ConstructionWorld::light_assembly();
        Json weighted=original;
        weighted["name"]="Sun creature with a side weight";
        weighted["blocks"].push_back(Json{{"id","weight"},{"segment",1},{"slot",0},{"side",1}});
        save(output,report);
        std::size_t failures=0;
        for(const auto& assembly:{original,weighted}) {
            report["constructions"].push_back(Json{{"assembly",assembly},{"assembly_sha256",droid::sha256_hex(assembly.dump())},{"trials",Json::array()}});
            for(const std::string mode:{"learner","zero","rhythm","random"}) {
                for(std::uint64_t seed=1;seed<=(mode=="learner"||mode=="random"?3u:1u);++seed) {
                    Json result=trial(assembly,mode,seed);
                    if(result.at("completed")!=true||result.at("exact_controller_and_observation_replay")!=true) ++failures;
                    std::cout << assembly.at("name").get<std::string>() << " / " << mode << " / " << seed
                        << " reward=" << result.at("whole_run").at("integrated_sensor_reward")
                        << " final-light=" << result.at("second_phase_final_20s").at("mean_received_lux")
                        << " error=" << result.at("error") << std::endl;
                    report["constructions"].back()["trials"].push_back(std::move(result));
                    save(output,report);
                }
            }
        }
        report["failed_trials"]=failures;
        report["complete"]=true;
        save(output,report);
        return failures?1:0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
