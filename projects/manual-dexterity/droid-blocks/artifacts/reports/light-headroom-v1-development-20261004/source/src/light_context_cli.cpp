#include "droid/construction.hpp"
#include "droid/light_context.hpp"
#include "droid/light_session.hpp"
#include "droid/policy.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using Mode = droid::LightContextMode;
constexpr std::array<Mode,4> kModes{Mode::instantaneous,Mode::history,Mode::light_only,Mode::no_light_history};
constexpr double kDt=.02;
constexpr std::size_t kSteps=600,kFrames=120,kExamples=22;
constexpr std::string_view kReference="artifacts/reports/light-learning-v2-development-20260905/report.json";
constexpr std::string_view kReferenceSha="931a26b17b399456bb5461293e382ae9880486f1b38157007500eead65998230";
constexpr std::string_view kProtocol="docs/LIGHT_CONTEXT_EXPERIMENT.md";
constexpr std::string_view kProtocolSha="3e1218d1e0000cca5020962aeb73ba773f92bdcc1012a178ba4d238389d87dff";
void require(bool value,const std::string& message) { if(!value) throw std::runtime_error(message); }
Json read_json(const fs::path& path) {
    std::ifstream input(path,std::ios::binary);
    require(static_cast<bool>(input),"cannot read "+path.generic_string());
    return Json::parse(input);
}
std::string hash(const Json& value) { return droid::sha256_hex(value.dump()); }
bool same_effort(double left,double right) { return std::bit_cast<std::uint64_t>(left)==std::bit_cast<std::uint64_t>(right); }
Json sun(bool is_a) { return Json{{"position_m",{is_a?.30:-.30,0,.20}},{"intensity_lux",1000}}; }
Json non_light_observation(Json observation) {
    bool found=false;
    for(auto& sensor:observation.at("reward_sensors"))
        if(sensor.at("family_id")=="ambient_light_v0") { sensor.erase("observations");found=true; }
    require(found,"non-light projection missing light entry");
    return observation;
}
Json mechanical_state(Json state) {
    require(state.at("schema")=="construction_state_v2","mechanical projection requires construction v2");
    state.erase("sun");state.erase("reward");
    state.at("diagnostics").erase("instantaneous_light_lux");
    state.at("light_sensor").erase("observations");
    return state;
}
void write_new(const fs::path& path,const Json& value) {
    require(!fs::exists(path),"refusing to overwrite "+path.generic_string());
    fs::create_directories(path.parent_path());
    std::ofstream output(path,std::ios::binary);
    require(static_cast<bool>(output),"cannot create "+path.generic_string());
    output<<value.dump()<<'\n';output.close();
    require(static_cast<bool>(output),"cannot finish "+path.generic_string());
}
void save(const fs::path& output,const Json& report) {
    const fs::path next=output/"progress.next.json";
    std::ofstream stream(next,std::ios::binary|std::ios::trunc);
    stream<<report.dump(2)<<'\n';stream.close();
    require(static_cast<bool>(stream),"cannot save context report");
    fs::rename(next,output/"report.json");
}
void snapshot(const fs::path& output,Json& report,const fs::path& relative) {
    require(!relative.is_absolute(),"source snapshot path must be relative");
    for(const auto& part:relative) require(part!="..","source snapshot escaped project root");
    const std::string name=relative.generic_string();
    if(report.at("source_sha256").contains(name)) return;
    const std::string before=droid::sha256_file(name);
    const fs::path target=output/"source"/relative;
    fs::create_directories(target.parent_path());
    require(fs::copy_file(relative,target),"cannot snapshot "+name);
    require(droid::sha256_file(target.string())==before&&droid::sha256_file(name)==before,"source changed while snapshotted: "+name);
    report["source_sha256"][name]=before;
}
void snapshot_sources(const fs::path& output,Json& report,const Json& reference) {
    std::set<fs::path> files;
    for(const std::string directory:{"src","include/droid","tests"})
        for(const auto& entry:fs::recursive_directory_iterator(directory))
            if(entry.is_regular_file()&&(entry.path().extension()==".cpp"||entry.path().extension()==".hpp")) files.insert(entry.path());
    for(const std::string file:{"CMakeLists.txt","run.sh","Dockerfile","setup.sh","config/module_catalog.json","models/droid.xml",
        "tools/summarize-light-context.mjs","config/learning_experiment_v3.json","artifacts/light-search-policy-v3-seed0.json","docs/LIGHT_CONTEXT_EXPERIMENT.md"}) files.insert(file);
    if(fs::is_regular_file("build/light-context-qa/runtime.json")) files.insert("build/light-context-qa/runtime.json");
    files.insert(fs::path(kReference));
    for(const auto& [relative,expected]:reference.at("source_sha256").items()) {
        const fs::path archived=fs::path(kReference).parent_path()/"source"/relative;
        require(droid::sha256_file(archived.string())==expected.get<std::string>(),"archived v2 source differs: "+relative);
        files.insert(archived);
    }
    for(const auto& file:files) snapshot(output,report,file);
}
struct Metrics {
    double reward{},energy{},max_current{},max_temperature{},max_speed{};
    std::size_t steps{},vetoes{},flagged{};
    void add(const Json& transition,const Json& physical) {
        ++steps;reward+=transition.at("reward").get<double>();
        energy=physical.at("diagnostics").at("electrical_energy_j").get<double>();
        const auto& feedback=transition.at("observation").at("actuator_feedback").at(0).at("feedback");
        max_current=std::max(max_current,std::abs(feedback.at("current_a").get<double>()));
        max_temperature=std::max(max_temperature,feedback.at("temperature_c").get<double>());
        max_speed=std::max(max_speed,std::abs(feedback.at("velocity_rad_s").get<double>()));
        if(!feedback.at("fault_flags").empty()) ++flagged;
        if(transition.at("safety").at("veto")==true) ++vetoes;
    }
    Json json() const { return Json{{"steps",steps},{"duration_s",steps*kDt},{"integrated_sensor_reward",reward},
        {"electrical_energy_j",energy},{"max_current_a",max_current},{"max_temperature_c",max_temperature},
        {"max_motor_speed_rad_s",max_speed},{"native_veto_steps",vetoes},{"feedback_flag_steps",flagged}}; }
};
struct Run {
    Json result;
    Json requests=Json::array(),efforts=Json::array(),transitions=Json::array(),observations=Json::array(),non_light=Json::array(),mechanical_hashes=Json::array();
    std::vector<std::string> mechanical_serialized;
};
Run run_trial(const Json& assembly,std::size_t body,std::uint64_t seed,bool is_a,const fs::path& output) {
    const std::string context=is_a?"A":"B";
    const std::string id="body-"+std::to_string(body)+"-probe-"+std::to_string(seed)+"-"+context;
    Run run;
    Json frames=Json::array(),initial_observation,initial_physical,final_physical,final_diagnostics;
    std::unique_ptr<droid::ConstructionWorld> world;
    std::unique_ptr<droid::LightControl> controller;
    std::size_t steps=0,accepted_steps=0,attempted_step=0;
    double attempted_effort=0,requested=0,previous_effort=0;
    std::mt19937_64 request_rng(seed);
    bool attempted_accepted=false;
    std::string error;
    Metrics metrics;
    try {
        world=std::make_unique<droid::ConstructionWorld>();
        world->rebuild(assembly);world->set_sun(sun(is_a));world->reset();
        controller=std::make_unique<droid::LightControl>();controller->reset("random",seed);
        initial_observation=world->observation();initial_physical=world->state();
        while(steps<kSteps) {
            if(steps>=5&&(steps-5)%20==0) requested=(static_cast<int>(request_rng()%3)-1)*.15;
            const Json current_observation=world->observation();
            const double effort=controller->act(current_observation);
            require(same_effort(effort,droid::LightLearner::bounded_effort(requested,previous_effort,current_observation)),"reconstructed request/governor differs from random controller");
            attempted_step=steps+1;attempted_effort=effort;attempted_accepted=false;
            const Json transition=world->step(effort);++steps;
            // Preserve native output before any later diagnostic can fail.
            run.requests.push_back(requested);run.efforts.push_back(effort);run.transitions.push_back(transition);
            run.observations.push_back(transition.at("observation"));
            run.non_light.push_back(non_light_observation(transition.at("observation")));
            const Json physical=world->state();
            const std::string mechanical=mechanical_state(physical).dump();
            run.mechanical_serialized.push_back(mechanical);run.mechanical_hashes.push_back(droid::sha256_hex(mechanical));
            metrics.add(transition,physical);controller->observe(transition);
            if((steps-1)%5==0) {
                const auto frame=droid::project_light_context_frame(transition.at("observation"),effort);
                frames.push_back(Json{{"frame_index",frames.size()},{"step",steps},{"time_s",steps*kDt},
                    {"executed_effort",effort},{"observation",transition.at("observation")},{"features",frame}});
            }
            ++accepted_steps;attempted_accepted=true;previous_effort=effort;
        }
    } catch(const std::exception& exception) { error=exception.what(); }
    std::vector<std::string> snapshot_errors;
    try { if(controller) final_diagnostics=controller->diagnostics();else throw std::runtime_error("not initialized"); }
    catch(const std::exception& exception) { snapshot_errors.push_back("controller: "+std::string(exception.what())); }
    try { if(world) final_physical=world->state();else throw std::runtime_error("not initialized"); }
    catch(const std::exception& exception) { snapshot_errors.push_back("public physical state: "+std::string(exception.what())); }
    if(!snapshot_errors.empty()&&error.empty()) error="final state snapshot failed";
    std::string replay_error;
    std::size_t replayed_steps=0;
    bool replay_matches=false;
    Json replay_requests=Json::array(),replay_efforts=Json::array(),replay_observations=Json::array(),replay_transitions=Json::array();
    Json replay_final_physical,replay_final_controller;
    try {
        require(error.empty()&&accepted_steps==kSteps&&snapshot_errors.empty(),"incomplete original; full replay cannot be certified");
        droid::ConstructionWorld replay;
        replay.rebuild(assembly);replay.set_sun(sun(is_a));replay.reset();
        droid::LightControl replay_controller;replay_controller.reset("random",seed);
        std::mt19937_64 replay_request_rng(seed);
        double replay_request=0,replay_previous_effort=0;
        require(replay.observation().dump()==initial_observation.dump(),"replay initial observation differs");
        require(replay.state().dump()==initial_physical.dump(),"replay initial public physical state differs");
        for(std::size_t i=0;i<kSteps;++i) {
            if(i>=5&&(i-5)%20==0) replay_request=(static_cast<int>(replay_request_rng()%3)-1)*.15;
            require(same_effort(replay_request,run.requests.at(i).get<double>()),"replay requested command differs");
            const Json replay_observation=replay.observation();
            const double effort=replay_controller.act(replay_observation);
            require(same_effort(effort,droid::LightLearner::bounded_effort(replay_request,replay_previous_effort,replay_observation)),"replay reconstructed request/governor differs");
            require(same_effort(effort,run.efforts.at(i).get<double>()),"replay command bits differ at step "+std::to_string(i+1));
            const Json transition=replay.step(effort);
            require(transition.dump()==run.transitions.at(i).dump(),"replay full serialized transition differs at step "+std::to_string(i+1));
            require(mechanical_state(replay.state()).dump()==run.mechanical_serialized.at(i),"replay mechanics differ at step "+std::to_string(i+1));
            replay_controller.observe(transition);
            replay_requests.push_back(replay_request);replay_efforts.push_back(effort);replay_observations.push_back(transition.at("observation"));replay_transitions.push_back(transition);
            replay_previous_effort=effort;
            ++replayed_steps;
        }
        replay_final_controller=replay_controller.diagnostics();replay_final_physical=replay.state();
        require(replay_final_controller.dump()==final_diagnostics.dump(),"replay final controller differs");
        require(replay_final_physical.dump()==final_physical.dump(),"replay final public physical state differs");
        replay_matches=replayed_steps==kSteps;
    } catch(const std::exception& exception) { replay_error=exception.what(); }
    const std::string trace_path="traces/"+id+".json";
    Json trace{{"schema","light_context_trace_v1"},{"trial_id",id},{"initial_observation",initial_observation},
        {"initial_public_physical_snapshot",initial_physical},{"requested_efforts",run.requests},{"efforts",run.efforts},{"transitions",run.transitions},
        {"mechanical_state_sha256",run.mechanical_hashes},{"final_public_physical_snapshot",final_physical},{"controller_final",final_diagnostics}};
    write_new(output/trace_path,trace);
    const auto nullable_hash=[](const Json& value)->Json { return value.is_object()?Json(hash(value)):Json(nullptr); };
    run.result=Json{{"trial_id",id},{"body_index",body},{"probe_seed",seed},{"context",context},
        {"split",seed<=108?"train":"test"},{"mode","random"},
        {"completed",steps==kSteps&&accepted_steps==kSteps&&error.empty()&&frames.size()==kFrames},
        {"completed_steps",steps},{"accepted_steps",accepted_steps},{"error",error},
        {"recorded_transition_count",run.transitions.size()},{"recorded_effort_count",run.efforts.size()},{"recorded_requested_effort_count",run.requests.size()},
        {"last_attempted_action",Json{{"step",attempted_step},{"effort",attempted_effort},{"requested_effort",requested},{"fully_accepted",attempted_accepted}}},
        {"snapshot_errors",snapshot_errors},{"final_controller_snapshot_available",final_diagnostics.is_object()},
        {"final_public_physical_snapshot_available",final_physical.is_object()},
        {"exact_controller_and_observation_replay",replay_matches},{"exact_full_transition_replay",replay_matches},
        {"requested_effort_trace_sha256",hash(run.requests)},{"effort_trace_sha256",hash(run.efforts)},{"observation_trace_sha256",hash(run.observations)},
        {"transition_trace_sha256",hash(run.transitions)},{"non_light_observation_trace_sha256",hash(run.non_light)},
        {"mechanical_state_trace_sha256",hash(run.mechanical_hashes)},
        {"initial_public_mechanical_snapshot_sha256",initial_physical.is_object()?Json(hash(mechanical_state(initial_physical))):Json(nullptr)},
        {"final_public_physical_snapshot_sha256",nullable_hash(final_physical)},{"final_controller_snapshot_sha256",nullable_hash(final_diagnostics)},
        {"replay",Json{{"completed_steps",replayed_steps},{"error",replay_error},{"requested_effort_trace_sha256",hash(replay_requests)},{"effort_trace_sha256",hash(replay_efforts)},
            {"observation_trace_sha256",hash(replay_observations)},{"transition_trace_sha256",hash(replay_transitions)},
            {"final_public_physical_snapshot_sha256",nullable_hash(replay_final_physical)},
            {"final_controller_snapshot_sha256",nullable_hash(replay_final_controller)},
            {"scope","600 exact command bits, serialized full transitions, every public mechanical projection and final physical/controller states"}}},
        {"trace",Json{{"path",trace_path},{"sha256",droid::sha256_file((output/trace_path).string())}}},
        {"frames",std::move(frames)},{"metrics",metrics.json()},{"controller_final",final_diagnostics}};
    return run;
}
Json pair_result(const Run& a,const Run& b,std::size_t body,std::uint64_t seed) {
    bool commands_equal=a.efforts.size()==kSteps&&b.efforts.size()==kSteps;
    for(std::size_t i=0;commands_equal&&i<kSteps;++i) commands_equal=same_effort(a.efforts.at(i).get<double>(),b.efforts.at(i).get<double>());
    const bool mechanics_equal=a.mechanical_serialized.size()==kSteps&&b.mechanical_serialized.size()==kSteps&&
        a.mechanical_serialized==b.mechanical_serialized&&
        a.result.at("initial_public_mechanical_snapshot_sha256")==b.result.at("initial_public_mechanical_snapshot_sha256");
    bool requests_equal=a.requests.size()==kSteps&&b.requests.size()==kSteps;
    for(std::size_t i=0;requests_equal&&i<kSteps;++i) requests_equal=same_effort(a.requests.at(i).get<double>(),b.requests.at(i).get<double>());
    Json checks{{"identical_requests",requests_equal},{"identical_commands",commands_equal},
        {"identical_non_light_observations",a.non_light.size()==kSteps&&b.non_light.size()==kSteps&&a.non_light.dump()==b.non_light.dump()},
        {"identical_mechanical_states",mechanics_equal},
        {"both_full_replays",a.result.at("exact_full_transition_replay")==true&&b.result.at("exact_full_transition_replay")==true}};
    bool valid=a.result.at("completed")==true&&b.result.at("completed")==true;
    for(const auto& [name,passed]:checks.items()) { (void)name;valid=valid&&passed==true; }
    return Json{{"body_index",body},{"probe_seed",seed},{"trial_a",a.result.at("trial_id")},{"trial_b",b.result.at("trial_id")},
        {"valid",valid},{"checks",checks}};
}
std::vector<double> example_features(const Json& trial,std::size_t end_frame,Mode mode) {
    require(end_frame>=10&&end_frame<trial.at("frames").size(),"example endpoint outside causal history");
    std::array<droid::LightContextFrame,11> history{};
    for(std::size_t i=0;i<history.size();++i) history[i]=trial.at("frames").at(end_frame-10+i).at("features").get<droid::LightContextFrame>();
    return droid::light_context_features(mode,history);
}
Json example_metadata(const Json& trial,std::size_t end_frame) {
    return Json{{"trial_id",trial.at("trial_id")},{"probe_seed",trial.at("probe_seed")},{"context",trial.at("context")},
        {"frame_index",end_frame},{"step",trial.at("frames").at(end_frame).at("step")},
        {"time_s",trial.at("frames").at(end_frame).at("time_s")}};
}
struct Score {
    std::size_t true_a{},false_b{},false_a{},true_b{},ambiguous{};
    double brier{};
    void add(bool is_a,const droid::LightContextPrediction& prediction) {
        if(is_a) { if(prediction.predicts_a) ++true_a;else ++false_b; }
        else { if(prediction.predicts_a) ++false_a;else ++true_b; }
        const double error=prediction.probability_a-(is_a?1.0:0.0);brier+=error*error;
        if(prediction.probability_a>=.4&&prediction.probability_a<=.6) ++ambiguous;
    }
    Json json() const {
        const auto a=true_a+false_b,b=true_b+false_a,count=a+b;
        require(a>0&&b>0,"score requires both labels");
        return Json{{"example_count",count},{"a_count",a},{"b_count",b},{"true_a",true_a},{"false_b",false_b},
            {"false_a",false_a},{"true_b",true_b},
            {"balanced_accuracy",.5*(static_cast<double>(true_a)/static_cast<double>(a)+static_cast<double>(true_b)/static_cast<double>(b))},
            {"brier_score",brier/static_cast<double>(count)},{"ambiguous_count",ambiguous},
            {"ambiguous_fraction",static_cast<double>(ambiguous)/static_cast<double>(count)}};
    }
};
using Models=std::map<Mode,std::unique_ptr<droid::LightContextDecoder>>;
void evaluate(Json& report,const std::array<Models,2>& models) {
    report["evaluation"]=Json{{"bodies",Json::array()},{"completed",false},{"negative_control_passed",false},{"readiness_passed",true}};
    Json& evaluation=report.at("evaluation");
    std::map<Mode,Score> aggregate;
    for(std::size_t body=0;body<2;++body) {
        evaluation["bodies"].push_back(Json{{"body_index",body},{"decoders",Json::object()}});
        Json& body_result=evaluation["bodies"].back();
        for(Mode mode:kModes) {
            Score overall;
            std::map<std::uint64_t,Score> by_probe;
            body_result["decoders"][droid::light_context_mode_id(mode)]=Json{{"completed",false},{"predictions",Json::array()}};
            Json& predictions=body_result["decoders"][droid::light_context_mode_id(mode)]["predictions"];
            std::map<std::pair<std::uint64_t,std::size_t>,double> negative_a;
            for(const Json& trial:report.at("trials")) {
                if(trial.at("body_index")!=body||trial.at("split")!="test") continue;
                const auto seed=trial.at("probe_seed").get<std::uint64_t>();
                require(seed>=109&&seed<=112,"non-withheld sequence reached evaluation");
                const bool is_a=trial.at("context")=="A";
                for(std::size_t j=0;j<kExamples;++j) {
                    const std::size_t endpoint=10+5*j;
                    const auto prediction=models.at(body).at(mode)->predict(example_features(trial,endpoint,mode));
                    overall.add(is_a,prediction);by_probe[seed].add(is_a,prediction);aggregate[mode].add(is_a,prediction);
                    Json item=example_metadata(trial,endpoint);
                    item["probability_a"]=prediction.probability_a;item["predicts_a"]=prediction.predicts_a;
                    item["neighbor_count"]=prediction.neighbor_count;
                    item["squared_mean_distance_cutoff"]=prediction.squared_mean_distance_cutoff;
                    predictions.push_back(std::move(item));
                    if(mode==Mode::no_light_history) {
                        const auto key=std::make_pair(seed,endpoint);
                        if(is_a) negative_a[key]=prediction.probability_a;
                        else require(negative_a.contains(key)&&same_effort(negative_a.at(key),prediction.probability_a),
                            "light-removed paired probabilities differ");
                    }
                }
            }
            require(predictions.size()==176&&by_probe.size()==4,"withheld example membership differs");
            Json grouped=Json::array();
            double probe_min=1,probe_max=0,probe_mean=0;
            bool ready=overall.json().at("balanced_accuracy").get<double>()>=.75;
            for(const auto& [seed,score]:by_probe) {
                const Json metrics=score.json();
                const double accuracy=metrics.at("balanced_accuracy").get<double>();
                probe_min=std::min(probe_min,accuracy);probe_max=std::max(probe_max,accuracy);probe_mean+=accuracy/4.0;
                ready=ready&&accuracy>=.65;
                if(mode==Mode::no_light_history) require(accuracy==.5,"light-removed probe negative control differs from chance");
                grouped.push_back(Json{{"probe_seed",seed},{"metrics",metrics}});
            }
            if(mode==Mode::no_light_history) require(overall.json().at("balanced_accuracy")==.5,"light-removed negative control differs from chance");
            body_result["decoders"][droid::light_context_mode_id(mode)]=Json{
                {"completed",true},{"overall",overall.json()},{"by_probe",grouped},{"predictions",predictions},{"readiness_passed",ready},
                {"probe_balanced_accuracy_mean",probe_mean},{"probe_balanced_accuracy_min",probe_min},{"probe_balanced_accuracy_max",probe_max}};
        }
        const auto& history=body_result.at("decoders").at("history");
        const auto& instantaneous=body_result.at("decoders").at("instantaneous");
        body_result["readiness_passed"]=history.at("readiness_passed");
        body_result["history_minus_instantaneous_balanced_accuracy"]=history.at("overall").at("balanced_accuracy").get<double>()-
            instantaneous.at("overall").at("balanced_accuracy").get<double>();
        body_result["paired_probe_differences"]=Json::array();
        for(std::size_t i=0;i<4;++i) body_result["paired_probe_differences"].push_back(Json{
            {"probe_seed",history.at("by_probe").at(i).at("probe_seed")},
            {"history_minus_instantaneous_balanced_accuracy",history.at("by_probe").at(i).at("metrics").at("balanced_accuracy").get<double>()-
                instantaneous.at("by_probe").at(i).at("metrics").at("balanced_accuracy").get<double>()}});
        if(body_result.at("readiness_passed")!=true) evaluation["readiness_passed"]=false;
        // This body remains in report even if later scoring fails.
    }
    evaluation["overall"]=Json::object();
    for(Mode mode:kModes) evaluation["overall"][droid::light_context_mode_id(mode)]=aggregate.at(mode).json();
    evaluation["history_minus_instantaneous_balanced_accuracy"]=aggregate.at(Mode::history).json().at("balanced_accuracy").get<double>()-
        aggregate.at(Mode::instantaneous).json().at("balanced_accuracy").get<double>();
    evaluation["negative_control_passed"]=true;
    evaluation["completed"]=true;
}
void fit_and_evaluate(const fs::path& output,Json& report) {
    std::array<Models,2> models;
    Json artifact{{"schema","light_context_models_v1"},{"saved_before_evaluation",true},{"bodies",Json::array()}};
    for(std::size_t body=0;body<2;++body) {
        Json body_models{{"body_index",body},{"training_example_metadata",Json::array()},{"models",Json::object()}};
        for(const auto& trial:report.at("trials")) {
            if(trial.at("body_index")!=body||trial.at("split")!="train") continue;
            for(std::size_t j=0;j<kExamples;++j) body_models["training_example_metadata"].push_back(example_metadata(trial,10+5*j));
        }
        require(body_models.at("training_example_metadata").size()==352,"training metadata membership differs");
        for(Mode mode:kModes) {
            std::vector<droid::LightContextExample> training;
            for(const auto& trial:report.at("trials")) {
                if(trial.at("body_index")!=body||trial.at("split")!="train") continue;
                const auto seed=trial.at("probe_seed").get<std::uint64_t>();
                require(seed>=101&&seed<=108,"withheld sequence reached fitting");
                for(std::size_t j=0;j<kExamples;++j)
                    training.push_back(droid::LightContextExample{example_features(trial,10+5*j,mode),trial.at("context")=="A"});
            }
            require(training.size()==352,"training examples differ from protocol");
            models[body][mode]=std::make_unique<droid::LightContextDecoder>(mode,training);
            body_models["models"][droid::light_context_mode_id(mode)]=models[body][mode]->serialize();
        }
        artifact["bodies"].push_back(std::move(body_models));
    }
    const fs::path model_path=output/"training-models.json";
    write_new(model_path,artifact);
    const std::string sealed_hash=droid::sha256_file(model_path.string());
    report["training_models"]=Json{{"path","training-models.json"},{"sha256",sealed_hash},
        {"saved_before_evaluation",true},{"unchanged_after_evaluation",false},{"training_examples_per_body",352},{"heldout_examples_per_body",176}};
    report["stage"]="models_sealed";save(output,report);
    evaluate(report,models);
    require(droid::sha256_file(model_path.string())==sealed_hash,"training model artifact changed during evaluation");
    for(std::size_t body=0;body<2;++body)
        for(Mode mode:kModes) require(models[body].at(mode)->serialize().dump()==
            artifact.at("bodies").at(body).at("models").at(droid::light_context_mode_id(mode)).dump(),"prediction mutated a model");
    report["training_models"]["unchanged_after_evaluation"]=true;
    report["training_models"]["sha256_after_evaluation"]=sealed_hash;
    report["checks"]["models_immutable_during_evaluation"]=true;
}
} // namespace
int main(int argc,char** argv) {
    fs::path output;
    Json report;
    bool reserved=false;
    try {
        if(argc!=3||std::string(argv[1])!="--output")
            throw std::invalid_argument("Usage: droid-light-context --output NEW_DIRECTORY (from project root)");
        require(droid::sha256_file(std::string(kProtocol))==kProtocolSha,"predeclared context protocol hash differs");
        require(droid::sha256_file(std::string(kReference))==kReferenceSha,"fixed archived v2 report hash differs");
        const Json reference=read_json(fs::path(kReference));
        require(reference.at("complete")==true&&reference.at("failed_trials")==0&&reference.at("constructions").size()==2,"archived v2 reference is invalid");
        output=fs::path(argv[2]);
        require(fs::create_directory(output),"output directory already exists; preserve previous evidence");
        reserved=true;
        report=Json{{"schema","light_context_v1"},{"complete",false},{"finished",false},{"stage","collecting"},
            {"failed_trials",0},{"failed_pairs",0},{"source_sha256",Json::object()},
            {"bodies",Json::array()},{"trials",Json::array()},{"pairs",Json::array()},{"checks",Json::object()},
            {"protocol",Json{{"document",kProtocol},{"document_sha256",kProtocolSha},{"control_dt_s",kDt},
                {"steps",kSteps},{"duration_s",12},{"body_count",2},{"trial_count",48},{"replay_count",48},
                {"training_probe_seeds",{101,102,103,104,105,106,107,108}},{"heldout_probe_seeds",{109,110,111,112}},
                {"source_A",sun(true)},{"source_B",sun(false)},{"probe_mode","unchanged LightControl random"},
                {"initialization","rebuild assembly, set constant source, reset physics and acquisition queues"},
                {"frame_count",kFrames},{"frame_steps","1+5*k, k=0..119"},{"endpoint_frames","10+5*j, j=0..21"},
                {"history_frame_count",11},{"examples_per_trial",kExamples},{"training_examples_per_body",352},{"heldout_examples_per_body",176},
                {"primary_metric","held-out history balanced accuracy"},{"paired_comparison","history minus instantaneous balanced accuracy"},
                {"readiness_mean_threshold",.75},{"readiness_every_probe_threshold",.65},{"readiness_requires_both_bodies",true},
                {"native_state_caveat","paired canonical public mechanical projections are compared at all600steps; retained hashes are not complete private MuJoCo checkpoints"},
                {"interpretation_limit","supervised decodability for two known bodies and fixed sources under withheld entire random probe sequences; no autonomous retrieval or control claim"}}},
            {"reference_report",Json{{"path",kReference},{"sha256",kReferenceSha},{"snapshot","source/"+std::string(kReference)}}},
            {"compiler",__VERSION__},{"physics_spec",droid::ConstructionWorld::light_specification()},
            {"control_spec",droid::LightControl::specification()},{"decoder_spec",droid::light_context_specification()},
            {"module_catalog",read_json("config/module_catalog.json")}};
        snapshot_sources(output,report,reference);
        require(report.at("source_sha256").at(std::string(kProtocol))==kProtocolSha,"protocol changed before snapshot");
        for(std::size_t body=0;body<2;++body) {
            const auto& construction=reference.at("constructions").at(body);
            require(hash(construction.at("assembly"))==construction.at("assembly_sha256"),"reference assembly hash differs");
            report["bodies"].push_back(Json{{"body_index",body},{"assembly",construction.at("assembly")},{"assembly_sha256",construction.at("assembly_sha256")}});
        }
        save(output,report);
        std::size_t failed_trials=0,failed_pairs=0;
        for(std::size_t body=0;body<2;++body) {
            const Json assembly=report.at("bodies").at(body).at("assembly");
            for(std::uint64_t seed=101;seed<=112;++seed) {
                std::cout<<"body "<<body<<", probe "<<seed<<": A and independent replay"<<std::endl;
                Run a=run_trial(assembly,body,seed,true,output);
                std::cout<<"body "<<body<<", probe "<<seed<<": B and independent replay"<<std::endl;
                Run b=run_trial(assembly,body,seed,false,output);
                Json paired=pair_result(a,b,body,seed);
                for(const Run* trial:{&a,&b}) if(trial->result.at("completed")!=true||trial->result.at("exact_full_transition_replay")!=true) ++failed_trials;
                if(paired.at("valid")!=true) ++failed_pairs;
                report["trials"].push_back(std::move(a.result));report["trials"].push_back(std::move(b.result));
                report["pairs"].push_back(std::move(paired));report["failed_trials"]=failed_trials;report["failed_pairs"]=failed_pairs;
                save(output,report);
            }
        }
        report["stage"]="collected";
        report["checks"]["all_trials_complete"]=failed_trials==0&&report.at("trials").size()==48;
        report["checks"]["all_pairs_valid"]=failed_pairs==0&&report.at("pairs").size()==24;
        report["checks"]["all_full_replays"]=failed_trials==0;save(output,report);
        if(failed_trials==0&&failed_pairs==0) {
            fit_and_evaluate(output,report);
            report["checks"]["negative_control_passed"]=report.at("evaluation").at("negative_control_passed");
            report["complete"]=true;report["stage"]="evaluated";
        }
        report["finished"]=true;save(output,report);
        return report.at("complete")==true?0:1;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        if(reserved&&report.is_object()) {
            report["complete"]=false;report["fatal_error"]=error.what();
            try { save(output,report); } catch(...) { }
        }
        return 2;
    }
}



