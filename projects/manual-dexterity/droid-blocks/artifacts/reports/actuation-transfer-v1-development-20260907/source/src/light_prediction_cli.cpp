#include "droid/construction.hpp"
#include "droid/light_context.hpp"
#include "droid/light_learner.hpp"
#include "droid/light_prediction.hpp"
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
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
using Mode=droid::LightPredictionMode;
using Frame=droid::LightContextFrame;
constexpr std::array<Mode,3> kModes{Mode::current_input,Mode::history_input,Mode::history_no_input};
constexpr std::array<std::size_t,3> kHorizons{1,4,10};
constexpr std::array<std::size_t,7> kPrimary{1,2,3,5,7,9,11};
constexpr std::array<double,12> kScales{.15,1,1,10,.5,50,50,50,10,10,10,1000};
constexpr double kDt=.02;
constexpr std::size_t kSteps=1201,kIntervals=240,kFrames=241,kBodies=3;
constexpr std::string_view kReference="artifacts/reports/light-context-v1-development-20260906/report.json";
constexpr std::string_view kReferenceSha="f3f729120df6b13f79845ecbd4ee72409f111f76b1b525037d3c059e41f7c6ee";
constexpr std::string_view kProtocol="docs/LIGHT_PREDICTION_EXPERIMENT.md";
constexpr std::string_view kProtocolSha="7cc508c8cf1bc88e0b0f417b49df93022e895f8f90c9f1c91e528372696ef6be";
void require(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
Json read_json(const fs::path& path) {
    std::ifstream input(path,std::ios::binary);require(static_cast<bool>(input),"cannot read "+path.generic_string());
    return Json::parse(input);
}
std::string hash(const Json& value) { return droid::sha256_hex(value.dump()); }
Json object_hash(const Json& value) { return value.is_object()?Json(hash(value)):Json(nullptr); }
bool same_bits(double a,double b) { return std::bit_cast<std::uint64_t>(a)==std::bit_cast<std::uint64_t>(b); }
Json sun(bool a) { return Json{{"position_m",{a?.30:-.30,0,.20}},{"intensity_lux",1000}}; }
Json non_light_observation(Json observation) {
    bool found=false;
    for(auto& sensor:observation.at("reward_sensors")) if(sensor.at("family_id")=="ambient_light_v0") {
        sensor.erase("observations");found=true;
    }
    require(found,"non-light projection missing light entry");return observation;
}
Json mechanical_state(Json state) {
    require(state.at("schema")=="construction_state_v2","mechanical projection requires construction v2");
    state.erase("sun");state.erase("reward");state.at("diagnostics").erase("instantaneous_light_lux");
    state.at("light_sensor").erase("observations");return state;
}
void write_new(const fs::path& path,const Json& value) {
    require(!fs::exists(path),"refusing to overwrite "+path.generic_string());
    fs::create_directories(path.parent_path());std::ofstream stream(path,std::ios::binary);
    require(static_cast<bool>(stream),"cannot create "+path.generic_string());
    stream<<value.dump()<<'\n';stream.close();require(static_cast<bool>(stream),"cannot finish "+path.generic_string());
}
void save(const fs::path& output,const Json& report) {
    const fs::path next=output/"progress.next.json";
    std::ofstream stream(next,std::ios::binary|std::ios::trunc);stream<<report.dump(2)<<'\n';stream.close();
    require(static_cast<bool>(stream),"cannot save prediction report");fs::rename(next,output/"report.json");
}
void snapshot(const fs::path& output,Json& report,const fs::path& relative) {
    require(!relative.is_absolute(),"snapshot path must be relative");
    for(const auto& part:relative) require(part!="..","source snapshot escaped project root");
    const std::string name=relative.generic_string();if(report.at("source_sha256").contains(name)) return;
    const std::string before=droid::sha256_file(name);const fs::path target=output/"source"/relative;
    fs::create_directories(target.parent_path());require(fs::copy_file(relative,target),"cannot snapshot "+name);
    require(droid::sha256_file(target.string())==before&&droid::sha256_file(name)==before,"source changed during snapshot: "+name);
    report["source_sha256"][name]=before;
}
void snapshot_sources(const fs::path& output,Json& report) {
    std::set<fs::path> files;
    for(const std::string directory:{"src","include/droid","tests"})
        for(const auto& entry:fs::recursive_directory_iterator(directory))
            if(entry.is_regular_file()&&(entry.path().extension()==".cpp"||entry.path().extension()==".hpp")) files.insert(entry.path());
    for(const std::string file:{"CMakeLists.txt","run.sh","Dockerfile","setup.sh","config/module_catalog.json","models/droid.xml",
        "tools/summarize-light-prediction.mjs","config/learning_experiment_v3.json","artifacts/light-search-policy-v3-seed0.json",
        "docs/LIGHT_PREDICTION_EXPERIMENT.md"}) files.insert(file);
    if(fs::is_regular_file("build/light-prediction-qa/runtime.json")) files.insert("build/light-prediction-qa/runtime.json");
    files.insert(fs::path(kReference));for(const auto& file:files) snapshot(output,report,file);
}
struct SplitMix64 {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t value=(state+=0x9e3779b97f4a7c15ULL);
        value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
        value=(value^(value>>27))*0x94d049bb133111ebULL;
        return value^(value>>31);
    }
};
std::vector<double> request_tape(std::uint64_t seed) {
    SplitMix64 random{seed};constexpr std::array<std::size_t,4> lengths{1,2,4,8};
    std::vector<double> result;result.reserve(kIntervals);
    while(result.size()<kIntervals) {
        const std::size_t duration=lengths.at(static_cast<std::size_t>(random.next()%4));
        const double request=static_cast<double>(static_cast<int>(random.next()%3)-1);
        for(std::size_t i=0;i<duration&&result.size()<kIntervals;++i) result.push_back(request);
    }
    return result;
}
void validate_time(const Json& observation,std::size_t steps) {
    const double time=static_cast<double>(steps)*kDt;
    for(const auto& sensor:observation.at("reward_sensors")) {
        if(steps==0) { require(sensor.at("valid")==false&&sensor.at("sequence")==0,"initial sensor must be undelivered");continue; }
        require(sensor.at("valid")==true,"delivered sensor became invalid");
        const double period=sensor.at("sample_period_s").get<double>(),latency=sensor.at("latency_s").get<double>();
        const auto sequence=static_cast<std::uint64_t>(std::floor((time-latency+1e-8)/period))+1;
        const double sampled=static_cast<double>(sequence-1)*period;
        require(sensor.at("sequence").get<std::uint64_t>()==sequence&&
            std::abs(sensor.at("sample_time_s").get<double>()-sampled)<2e-6&&
            std::abs(sensor.at("delivered_time_s").get<double>()-sampled-latency)<2e-6&&
            std::abs(sensor.at("age_s").get<double>()-time+sampled)<2e-6,"sensor clock does not match native elapsed time");
    }
}
struct ProbeMetrics {
    double reward{},energy{},max_current{},max_temperature{},max_speed{};
    std::size_t steps{},vetoes{},flagged{};
    void add(const Json& transition,const Json& physical) {
        ++steps;reward+=transition.at("reward").get<double>();energy=physical.at("diagnostics").at("electrical_energy_j").get<double>();
        const auto& feedback=transition.at("observation").at("actuator_feedback").at(0).at("feedback");
        max_current=std::max(max_current,std::abs(feedback.at("current_a").get<double>()));
        max_temperature=std::max(max_temperature,feedback.at("temperature_c").get<double>());
        max_speed=std::max(max_speed,std::abs(feedback.at("velocity_rad_s").get<double>()));
        if(!feedback.at("fault_flags").empty()) ++flagged;
        if(transition.at("safety").at("veto")==true) ++vetoes;
    }
    Json json() const { return Json{{"steps",steps},{"duration_s",static_cast<double>(steps)*kDt},{"integrated_sensor_reward",reward},
        {"electrical_energy_j",energy},{"max_current_a",max_current},{"max_temperature_c",max_temperature},
        {"max_motor_speed_rad_s",max_speed},{"native_veto_steps",vetoes},{"feedback_flag_steps",flagged}}; }
};
struct Run {
    Json result,requests=Json::array(),efforts=Json::array(),transitions=Json::array(),observations=Json::array(),
        non_light=Json::array(),mechanical_hashes=Json::array();
    std::vector<std::string> mechanical_serialized;
};
Run run_trial(const Json& assembly,std::size_t body,std::uint64_t seed,bool is_a,const fs::path& output) {
    const std::string condition=is_a?"A":"B",id="body-"+std::to_string(body)+"-probe-"+std::to_string(seed)+"-"+condition;
    const std::vector<double> tape=request_tape(seed); // Generated before any world exists.
    Run run;Json frames=Json::array(),initial_observation,initial_physical,final_physical;
    std::unique_ptr<droid::ConstructionWorld> world;
    std::size_t steps=0,accepted_steps=0,attempted_step=0;
    double previous=0,requested=0,attempted_effort=0;bool attempted_accepted=false;
    std::string error;ProbeMetrics metrics;
    try {
        world=std::make_unique<droid::ConstructionWorld>();world->rebuild(assembly);world->set_sun(sun(is_a));world->reset();
        initial_observation=world->observation();initial_physical=world->state();
        while(steps<kSteps) {
            requested=steps==0?0.0:.15*tape.at((steps-1)/5);
            const Json observation=world->observation();validate_time(observation,steps);
            const double effort=droid::LightLearner::bounded_effort(requested,previous,observation);
            attempted_step=steps+1;attempted_effort=effort;attempted_accepted=false;
            const Json transition=world->step(effort);++steps;
            run.requests.push_back(requested);run.efforts.push_back(effort);run.transitions.push_back(transition);
            run.observations.push_back(transition.at("observation"));run.non_light.push_back(non_light_observation(transition.at("observation")));
            const Json physical=world->state();const std::string mechanical=mechanical_state(physical).dump();
            run.mechanical_serialized.push_back(mechanical);run.mechanical_hashes.push_back(droid::sha256_hex(mechanical));
            metrics.add(transition,physical);
            require(transition.at("info").at("elapsed_control_dt_s")==kDt&&transition.at("info").at("physics_steps_executed")==10,
                "native transition duration differs");
            require(transition.at("terminated")==false&&transition.at("truncated")==false&&transition.at("safety").at("veto")==false,
                "native safety stopped probe");
            validate_time(transition.at("observation"),steps);
            (void)droid::LightLearner::bounded_effort(0.0,effort,transition.at("observation"));
            if((steps-1)%5==0) {
                frames.push_back(Json{{"index",frames.size()},{"step",steps},{"time_s",static_cast<double>(steps)*kDt},
                    {"output",droid::project_light_context_frame(transition.at("observation"),effort)}});
            }
            ++accepted_steps;attempted_accepted=true;previous=effort;
        }
    } catch(const std::exception& exception) { error=exception.what(); }
    std::vector<std::string> snapshot_errors;
    try { if(world) final_physical=world->state();else throw std::runtime_error("world not initialized"); }
    catch(const std::exception& exception) { snapshot_errors.push_back(exception.what()); }
    if(!snapshot_errors.empty()&&error.empty()) error="final public physical snapshot failed";
    Json replay_requests=Json::array(),replay_efforts=Json::array(),replay_observations=Json::array(),replay_transitions=Json::array(),
        replay_mechanical_hashes=Json::array(),replay_final_physical;
    std::size_t replay_steps=0;std::string replay_error;bool replay_exact=false;
    try {
        require(error.empty()&&accepted_steps==kSteps&&snapshot_errors.empty(),"incomplete original; full replay cannot be certified");
        const auto replay_tape=request_tape(seed);require(Json(replay_tape).dump()==Json(tape).dump(),"regenerated normalized requests differ");
        droid::ConstructionWorld replay;replay.rebuild(assembly);replay.set_sun(sun(is_a));replay.reset();
        require(replay.observation().dump()==initial_observation.dump(),"replay initial observation differs");
        require(replay.state().dump()==initial_physical.dump(),"replay initial physical snapshot differs");
        double replay_previous=0;
        for(std::size_t i=0;i<kSteps;++i) {
            const double replay_request=i==0?0.0:.15*replay_tape.at((i-1)/5);
            require(same_bits(replay_request,run.requests.at(i).get<double>()),"replay requested command bits differ");
            const Json observation=replay.observation();validate_time(observation,i);
            const double effort=droid::LightLearner::bounded_effort(replay_request,replay_previous,observation);
            require(same_bits(effort,run.efforts.at(i).get<double>()),"replay effort bits differ at step "+std::to_string(i+1));
            const Json transition=replay.step(effort);
            require(transition.dump()==run.transitions.at(i).dump(),"replay full serialized transition differs at step "+std::to_string(i+1));
            const std::string mechanical=mechanical_state(replay.state()).dump();
            require(mechanical==run.mechanical_serialized.at(i),"replay mechanics differ at step "+std::to_string(i+1));
            replay_requests.push_back(replay_request);replay_efforts.push_back(effort);replay_observations.push_back(transition.at("observation"));
            replay_transitions.push_back(transition);replay_mechanical_hashes.push_back(droid::sha256_hex(mechanical));
            replay_previous=effort;++replay_steps;
        }
        replay_final_physical=replay.state();require(replay_final_physical.dump()==final_physical.dump(),"replay final physical snapshot differs");
        replay_exact=replay_steps==kSteps;
    } catch(const std::exception& exception) { replay_error=exception.what(); }
    const std::string trace_path="traces/"+id+".json";
    const Json trace{{"schema","light_prediction_trace_v1"},{"trial_id",id},{"initial_observation",initial_observation},
        {"initial_public_physical_snapshot",initial_physical},{"requested_efforts",run.requests},{"efforts",run.efforts},
        {"transitions",run.transitions},{"mechanical_state_sha256",run.mechanical_hashes},{"final_public_physical_snapshot",final_physical}};
    write_new(output/trace_path,trace);
    run.result=Json{{"trial_id",id},{"body_index",body},{"seed",seed},{"condition",condition},{"split",seed<=208?"train":"test"},
        {"completed",steps==kSteps&&accepted_steps==kSteps&&error.empty()&&frames.size()==kFrames},{"error",error},
        {"completed_steps",steps},{"accepted_steps",accepted_steps},{"snapshot_errors",snapshot_errors},
        {"final_public_physical_snapshot_available",final_physical.is_object()},
        {"recorded_transition_count",run.transitions.size()},{"recorded_effort_count",run.efforts.size()},
        {"recorded_requested_effort_count",run.requests.size()},
        {"last_attempted_action",Json{{"step",attempted_step},{"requested_effort",requested},{"effort",attempted_effort},{"fully_accepted",attempted_accepted}}},
        {"request_tape",tape},{"request_tape_sha256",hash(Json(tape))},{"requested_effort_trace_sha256",hash(run.requests)},
        {"effort_trace_sha256",hash(run.efforts)},{"observation_trace_sha256",hash(run.observations)},
        {"transition_trace_sha256",hash(run.transitions)},{"non_light_observation_trace_sha256",hash(run.non_light)},
        {"mechanical_state_trace_sha256",hash(run.mechanical_hashes)},
        {"initial_public_mechanical_snapshot_sha256",initial_physical.is_object()?Json(hash(mechanical_state(initial_physical))):Json(nullptr)},
        {"final_public_physical_snapshot_sha256",object_hash(final_physical)},
        {"exact_full_transition_replay",replay_exact},
        {"replay",Json{{"completed_steps",replay_steps},{"error",replay_error},{"exact",replay_exact},
            {"requested_effort_trace_sha256",hash(replay_requests)},{"effort_trace_sha256",hash(replay_efforts)},
            {"observation_trace_sha256",hash(replay_observations)},{"transition_trace_sha256",hash(replay_transitions)},
            {"mechanical_state_trace_sha256",hash(replay_mechanical_hashes)},{"final_public_physical_snapshot_sha256",object_hash(replay_final_physical)}}},
        {"trace",Json{{"path",trace_path},{"sha256",droid::sha256_file((output/trace_path).string())}}},
        {"frames",std::move(frames)},{"metrics",metrics.json()}};
    return run;
}
Json pair_result(const Run& a,const Run& b,std::size_t body,std::uint64_t seed) {
    bool efforts=a.efforts.size()==kSteps&&b.efforts.size()==kSteps,requests=a.requests.size()==kSteps&&b.requests.size()==kSteps;
    for(std::size_t i=0;efforts&&i<kSteps;++i) efforts=same_bits(a.efforts.at(i).get<double>(),b.efforts.at(i).get<double>());
    for(std::size_t i=0;requests&&i<kSteps;++i) requests=same_bits(a.requests.at(i).get<double>(),b.requests.at(i).get<double>());
    Json checks{{"requests",requests},{"efforts",efforts},
        {"non_light_observations",a.non_light.size()==kSteps&&b.non_light.size()==kSteps&&a.non_light.dump()==b.non_light.dump()},
        {"mechanical_states",a.mechanical_serialized.size()==kSteps&&b.mechanical_serialized.size()==kSteps&&
            a.mechanical_serialized==b.mechanical_serialized&&a.result.at("initial_public_mechanical_snapshot_sha256")==b.result.at("initial_public_mechanical_snapshot_sha256")},
        {"replays",a.result.at("exact_full_transition_replay")==true&&b.result.at("exact_full_transition_replay")==true}};
    bool valid=a.result.at("completed")==true&&b.result.at("completed")==true;
    for(const auto& [name,passed]:checks.items()) { (void)name;valid=valid&&passed==true; }
    return Json{{"body_index",body},{"seed",seed},{"trial_a",a.result.at("trial_id")},{"trial_b",b.result.at("trial_id")},{"valid",valid},{"checks",checks}};
}
std::vector<double> state_at(const Json& trial,std::size_t origin,Mode mode) {
    const std::size_t taps=droid::light_prediction_taps(mode);require(origin+1>=taps,"insufficient causal history");
    std::vector<Frame> outputs;std::vector<double> requests;
    for(std::size_t index=origin+1-taps;index<=origin;++index) outputs.push_back(trial.at("frames").at(index).at("output").get<Frame>());
    if(droid::light_prediction_uses_input(mode))
        for(std::size_t index=origin+1-taps;index<origin;++index) requests.push_back(trial.at("request_tape").at(index).get<double>());
    return droid::light_prediction_state(mode,outputs,requests);
}
using Models=std::map<Mode,std::unique_ptr<droid::LightPredictionModel>>;
struct Score {
    std::array<long double,12> squares{};
    std::size_t origins{},failed{};
    void add(const Frame& actual,const Json& prediction) {
        ++origins;
        if(!prediction.is_array()||prediction.size()!=12) { ++failed;return; }
        for(std::size_t i=0;i<12;++i) {
            const double predicted=prediction.at(i).get<double>();
            require(std::isfinite(predicted)&&std::isfinite(actual[i]),"score received nonfinite stored output");
            const long double error=static_cast<long double>(predicted)-actual[i];squares[i]+=error*error;
        }
    }
};
Json finite_number(long double value) {
    if(!std::isfinite(value)||std::abs(value)>std::numeric_limits<double>::max()) return nullptr;
    return static_cast<double>(value);
}
Json aggregate_scores(const std::vector<Score>& scores) {
    std::array<long double,12> means{};
    std::size_t origins=0,failed=0,eligible=0;
    for(const Score& score:scores) {
        origins+=score.origins;failed+=score.failed;
        if(score.origins==0) continue;
        ++eligible;
        for(std::size_t channel=0;channel<12;++channel) means[channel]+=score.squares[channel]/static_cast<long double>(score.origins);
    }
    Json result{{"origins",origins},{"failed_origins",failed},{"eligible_trials",eligible},{"complete",failed==0&&eligible>0},
        {"mse",nullptr},{"rmse",nullptr},{"channels_mse",nullptr},{"channels_rmse",nullptr},{"channels_physical_rmse",nullptr},{"metric_overflow",false}};
    if(failed>0||eligible==0) return result;
    long double mse=0;
    for(auto& mean:means) mean/=static_cast<long double>(eligible);
    for(std::size_t channel:kPrimary) mse+=means[channel]/static_cast<long double>(kPrimary.size());
    result["mse"]=finite_number(mse);result["rmse"]=finite_number(std::sqrt(mse));
    result["channels_mse"]=Json::array();result["channels_rmse"]=Json::array();result["channels_physical_rmse"]=Json::array();
    bool overflow=result.at("mse").is_null()||result.at("rmse").is_null();
    for(std::size_t channel=0;channel<12;++channel) {
        const Json squared=finite_number(means[channel]),root=finite_number(std::sqrt(means[channel])),
            physical=finite_number(std::sqrt(means[channel])*kScales[channel]);
        result["channels_mse"].push_back(squared);result["channels_rmse"].push_back(root);result["channels_physical_rmse"].push_back(physical);
        overflow=overflow||squared.is_null()||root.is_null()||physical.is_null();
    }
    result["metric_overflow"]=overflow;result["complete"]=!overflow;return result;
}
void attach_changed(Json& metrics,const Json& changed) {
    metrics["changed_request_rmse"]=changed.at("rmse");metrics["changed_request_mse"]=changed.at("mse");
    metrics["changed_request_origins"]=changed.at("origins");metrics["changed_request_eligible_trials"]=changed.at("eligible_trials");
    metrics["changed_request_failed_origins"]=changed.at("failed_origins");metrics["changed_request_complete"]=changed.at("complete");
    metrics["changed_request_channels_rmse"]=changed.at("channels_rmse");
    metrics["changed_request_channels_physical_rmse"]=changed.at("channels_physical_rmse");
}
Json relative_improvement(const Json& candidate,const Json& reference) {
    if(!candidate.is_number()||!reference.is_number()||reference.get<double>()==0) return nullptr;
    return 1.0-candidate.get<double>()/reference.get<double>();
}
bool at_most(const Json& candidate,const Json& reference,double fraction) {
    return candidate.is_number()&&reference.is_number()&&candidate.get<double>()<=fraction*reference.get<double>();
}
Json range_of(const Json& records,const std::string& field) {
    double low=std::numeric_limits<double>::infinity(),high=-low;
    for(const auto& record:records) {
        if(!record.at(field).is_number()) return nullptr;
        const double value=record.at(field).get<double>();low=std::min(low,value);high=std::max(high,value);
    }
    return records.empty()?Json(nullptr):Json{{"min",low},{"max",high}};
}
void evaluate(const fs::path& output,Json& report,const std::array<Models,kBodies>& models) {
    report["evaluation"]=Json{{"complete",false},{"readiness_passed",false},{"simpler_candidate_readiness_passed",false},
        {"failed_forecasts",0},{"bodies",Json::array()}};
    Json& evaluation=report.at("evaluation");
    std::size_t total_failures=0;
    for(std::size_t body=0;body<kBodies;++body) {
        evaluation["bodies"].push_back(Json{{"body_index",body},{"complete",false},{"forecasts",Json::array()},{"metrics",Json::array()}});
        Json& body_result=evaluation["bodies"].back();
        for(const Json& trial:report.at("trials")) {
            if(trial.at("body_index")!=body||trial.at("split")!="test") continue;
            const auto seed=trial.at("seed").get<std::uint64_t>();require(seed>=209&&seed<=212,"training seed reached withheld evaluation");
            for(std::size_t origin=10;origin<=230;origin+=10) {
                std::array<double,10> future{};Json actual=Json::array();bool changed=false;
                for(std::size_t step=0;step<10;++step) {
                    future[step]=trial.at("request_tape").at(origin+step).get<double>();
                    actual.push_back(trial.at("frames").at(origin+step+1).at("output"));
                    if(step>0&&future[step]!=future[0]) changed=true;
                }
                body_result["forecasts"].push_back(Json{{"trial_id",trial.at("trial_id")},{"seed",seed},{"condition",trial.at("condition")},
                    {"origin_index",origin},{"origin_time_s",trial.at("frames").at(origin).at("time_s")},
                    {"initial_output",trial.at("frames").at(origin).at("output")},{"actual",actual},{"requests",future},
                    {"changed_future_request",changed},{"predictions",Json::object()},{"failures",Json::object()}});
                Json& item=body_result["forecasts"].back();
                for(Mode mode:kModes) {
                    const std::string id(droid::light_prediction_mode_id(mode));item["predictions"][id]=Json::array();
                    std::size_t step=0;
                    try {
                        auto state=state_at(trial,origin,mode);
                        for(;step<10;++step) {
                            const Frame prediction=models.at(body).at(mode)->predict(state,future[step]);
                            for(double value:prediction) require(std::isfinite(value),"model returned nonfinite output");
                            item["predictions"][id].push_back(prediction);
                            state=models.at(body).at(mode)->advance(state,future[step]);
                        }
                    } catch(const std::exception& error) {
                        item["failures"][id]=Json{{"error",error.what()},{"attempted_step",step+1},
                            {"recorded_predictions",item.at("predictions").at(id).size()}};
                        ++total_failures;evaluation["failed_forecasts"]=total_failures;
                    }
                }
                item["predictions"]["persistence"]=Json::array();
                for(std::size_t step=0;step<10;++step) item["predictions"]["persistence"].push_back(item.at("initial_output"));
            }
            // Keep completed scoring prefixes reviewable even if a later trial fails.
            save(output,report);
        }
        require(body_result.at("forecasts").size()==184,"withheld forecast origin membership differs");
        for(const std::string id:{"current_input","history_input","history_no_input","persistence"}) {
            body_result["metrics"].push_back(Json{{"model_id",id},{"horizons",Json::array()},{"failed_forecasts",0}});
            Json& model_metrics=body_result["metrics"].back();std::size_t model_failures=0;
            for(const auto& forecast:body_result.at("forecasts")) if(forecast.at("failures").contains(id)) ++model_failures;
            model_metrics["failed_forecasts"]=model_failures;
            for(std::size_t horizon:kHorizons) {
                std::vector<Score> all_scores,changed_scores;std::map<std::uint64_t,std::vector<Score>> pair_scores,pair_changed;
                Json trial_metrics=Json::array();
                for(const Json& trial:report.at("trials")) {
                    if(trial.at("body_index")!=body||trial.at("split")!="test") continue;
                    Score score,changed_score;
                    for(const Json& forecast:body_result.at("forecasts")) {
                        if(forecast.at("trial_id")!=trial.at("trial_id")) continue;
                        const auto& predictions=forecast.at("predictions").at(id);
                        const Json predicted=predictions.size()>=horizon?predictions.at(horizon-1):Json(nullptr);
                        const Frame actual=forecast.at("actual").at(horizon-1).get<Frame>();
                        score.add(actual,predicted);if(forecast.at("changed_future_request")==true) changed_score.add(actual,predicted);
                    }
                    require(score.origins==23,"trial scoring origin count differs");
                    const auto seed=trial.at("seed").get<std::uint64_t>();
                    all_scores.push_back(score);changed_scores.push_back(changed_score);
                    pair_scores[seed].push_back(score);pair_changed[seed].push_back(changed_score);
                    Json metrics=aggregate_scores({score});metrics["trial_id"]=trial.at("trial_id");metrics["seed"]=seed;metrics["condition"]=trial.at("condition");
                    attach_changed(metrics,aggregate_scores({changed_score}));trial_metrics.push_back(std::move(metrics));
                }
                require(all_scores.size()==8&&pair_scores.size()==4,"heldout trial membership differs");
                Json metrics=aggregate_scores(all_scores);attach_changed(metrics,aggregate_scores(changed_scores));
                metrics["steps"]=horizon;metrics["seconds"]=static_cast<double>(horizon)*.1;metrics["trials"]=trial_metrics;
                metrics["seed_pairs"]=Json::array();
                for(const auto& [seed,scores]:pair_scores) {
                    require(scores.size()==2,"seed pair is incomplete");Json paired=aggregate_scores(scores);
                    paired["seed"]=seed;attach_changed(paired,aggregate_scores(pair_changed.at(seed)));metrics["seed_pairs"].push_back(std::move(paired));
                }
                metrics["trial_rmse_range"]=range_of(metrics.at("trials"),"rmse");
                metrics["seed_pair_rmse_range"]=range_of(metrics.at("seed_pairs"),"rmse");
                model_metrics["horizons"].push_back(std::move(metrics));
            }
        }
        auto& metrics=body_result.at("metrics");
        for(std::size_t model=0;model<metrics.size();++model) {
            for(std::size_t h=0;h<kHorizons.size();++h) {
                auto& value=metrics.at(model).at("horizons").at(h);
                const auto& persistence=metrics.at(3).at("horizons").at(h);
                const auto& no_input=metrics.at(2).at("horizons").at(h);
                value["relative_improvement_over_persistence"]=relative_improvement(value.at("rmse"),persistence.at("rmse"));
                value["relative_improvement_over_no_input"]=relative_improvement(value.at("rmse"),no_input.at("rmse"));
                value["changed_request_relative_improvement_over_no_input"]=relative_improvement(value.at("changed_request_rmse"),no_input.at("changed_request_rmse"));
                for(std::size_t i=0;i<value.at("trials").size();++i) {
                    value["trials"][i]["relative_improvement_over_persistence"]=relative_improvement(value.at("trials").at(i).at("rmse"),persistence.at("trials").at(i).at("rmse"));
                    value["trials"][i]["relative_improvement_over_no_input"]=relative_improvement(value.at("trials").at(i).at("rmse"),no_input.at("trials").at(i).at("rmse"));
                }
                for(std::size_t i=0;i<value.at("seed_pairs").size();++i)
                    value["seed_pairs"][i]["relative_improvement_over_persistence"]=relative_improvement(value.at("seed_pairs").at(i).at("rmse"),persistence.at("seed_pairs").at(i).at("rmse"));
            }
            if(model>=2) { metrics[model]["readiness"]=Json{{"candidate",false},{"passed",false}};continue; }
            const auto& medium=metrics.at(model).at("horizons").at(1);
            const auto& long_horizon=metrics.at(model).at("horizons").at(2);
            const auto& quiet=metrics.at(3).at("horizons").at(2);
            const auto& blind=metrics.at(2).at("horizons").at(2);
            bool every_trial=true;
            for(std::size_t i=0;i<8;++i) every_trial=every_trial&&at_most(long_horizon.at("trials").at(i).at("rmse"),quiet.at("trials").at(i).at("rmse"),1.0);
            Json checks{{"no_failed_forecasts",metrics.at(model).at("failed_forecasts")==0},
                {"rmse_04s_at_most_08_persistence",at_most(medium.at("rmse"),metrics.at(3).at("horizons").at(1).at("rmse"),.8)},
                {"rmse_1s_at_most_08_persistence",at_most(long_horizon.at("rmse"),quiet.at("rmse"),.8)},
                {"every_trial_1s_no_worse_than_persistence",every_trial},
                {"rmse_1s_at_most_09_no_input",at_most(long_horizon.at("rmse"),blind.at("rmse"),.9)},
                {"changed_request_subset_nonempty",long_horizon.at("changed_request_origins").get<std::size_t>()>0},
                {"changed_request_rmse_1s_at_most_09_no_input",at_most(long_horizon.at("changed_request_rmse"),blind.at("changed_request_rmse"),.9)}};
            bool ready=true;for(const auto& [name,passed]:checks.items()) { (void)name;ready=ready&&passed==true; }
            metrics[model]["readiness"]=Json{{"candidate",true},{"passed",ready},{"checks",checks}};
        }
        body_result["readiness_passed"]=metrics.at(1).at("readiness").at("passed");
        body_result["simpler_candidate_readiness_passed"]=metrics.at(0).at("readiness").at("passed");
        body_result["complete"]=true;save(output,report);
    }
    bool history_ready=true,current_ready=true;
    for(const auto& body:evaluation.at("bodies")) {
        history_ready=history_ready&&body.at("readiness_passed")==true;
        current_ready=current_ready&&body.at("simpler_candidate_readiness_passed")==true;
    }
    evaluation["readiness_passed"]=history_ready;evaluation["simpler_candidate_readiness_passed"]=current_ready;
    evaluation["complete"]=true;
}
void fit_and_evaluate(const fs::path& output,Json& report) {
    std::array<Models,kBodies> models;
    Json artifact{{"schema","light_prediction_models_v1"},{"saved_before_evaluation",true},{"bodies",Json::array()}};
    for(std::size_t body=0;body<kBodies;++body) {
        Json body_models{{"body_index",body},{"training_membership",Json::array()},{"models",Json::array()}};
        for(const auto& trial:report.at("trials")) {
            if(trial.at("body_index")!=body||trial.at("split")!="train") continue;
            for(std::size_t origin=10;origin<=239;++origin)
                body_models["training_membership"].push_back(Json{{"trial_id",trial.at("trial_id")},{"origin_index",origin}});
        }
        require(body_models.at("training_membership").size()==3680,"training membership differs");
        for(Mode mode:kModes) {
            std::vector<droid::LightPredictionExample> examples;examples.reserve(3680);
            for(const auto& trial:report.at("trials")) {
                if(trial.at("body_index")!=body||trial.at("split")!="train") continue;
                const auto seed=trial.at("seed").get<std::uint64_t>();require(seed>=201&&seed<=208,"withheld sequence reached fitting");
                for(std::size_t origin=10;origin<=239;++origin)
                    examples.push_back(droid::LightPredictionExample{state_at(trial,origin,mode),trial.at("request_tape").at(origin).get<double>(),
                        trial.at("frames").at(origin+1).at("output").get<Frame>()});
            }
            require(examples.size()==3680,"training row count differs");
            models[body][mode]=std::make_unique<droid::LightPredictionModel>(mode,examples);
            body_models["models"].push_back(models[body][mode]->serialize());
        }
        artifact["bodies"].push_back(std::move(body_models));
    }
    const fs::path model_path=output/"models.json";write_new(model_path,artifact);
    const std::string sealed_hash=droid::sha256_file(model_path.string());
    report["training_models"]=Json{{"path","models.json"},{"sha256",sealed_hash},{"saved_before_evaluation",true},
        {"unchanged_after_evaluation",false},{"training_rows_per_body",3680}};
    report["stage"]="models_sealed";save(output,report);
    evaluate(output,report,models);
    require(droid::sha256_file(model_path.string())==sealed_hash,"sealed model artifact changed during evaluation");
    for(std::size_t body=0;body<kBodies;++body)
        for(std::size_t i=0;i<kModes.size();++i) require(models.at(body).at(kModes[i])->serialize().dump()==
            artifact.at("bodies").at(body).at("models").at(i).dump(),"forecast mutated fitted model");
    report["training_models"]["unchanged_after_evaluation"]=true;
    report["training_models"]["sha256_after_evaluation"]=sealed_hash;
    report["checks"]["models_immutable_during_evaluation"]=true;
}
} // namespace
int main(int argc,char** argv) {
    fs::path output;Json report;bool reserved=false;
    try {
        if(argc!=3||std::string(argv[1])!="--output")
            throw std::invalid_argument("Usage: droid-light-prediction --output NEW_DIRECTORY (from project root)");
        require(droid::sha256_file(std::string(kProtocol))==kProtocolSha,"locked prediction protocol hash differs");
        require(droid::sha256_file(std::string(kReference))==kReferenceSha,"reference context report hash differs");
        const Json reference=read_json(fs::path(kReference));
        require(reference.at("complete")==true&&reference.at("failed_trials")==0&&reference.at("bodies").size()==2,"context reference is invalid");
        output=fs::path(argv[2]);require(fs::create_directory(output),"output directory already exists; preserve previous evidence");reserved=true;
        report=Json{{"schema","light_prediction_v1"},{"complete",false},{"finished",false},{"stage","collecting"},
            {"failed_trials",0},{"failed_pairs",0},{"source_sha256",Json::object()},{"bodies",Json::array()},
            {"trials",Json::array()},{"pairs",Json::array()},{"checks",Json::object()},
            {"protocol",Json{{"path",kProtocol},{"sha256",kProtocolSha},{"control_dt_s",kDt},{"sample_dt_s",.1},
                {"first_frame_time_s",.02},{"steps",kSteps},{"duration_s",24.02},{"frame_count",kFrames},
                {"requested_intervals",kIntervals},{"body_count",kBodies},{"trial_count",72},{"replay_count",72},
                {"training_seeds",{201,202,203,204,205,206,207,208}},{"heldout_seeds",{209,210,211,212}},
                {"source_A",sun(true)},{"source_B",sun(false)},{"training_origins","10..239 inclusive"},
                {"forecast_origins","10,20,...,230"},{"horizon_steps",kHorizons},{"primary_channels",kPrimary},
                {"physical_channel_scales",kScales},{"request_normalization",.15},
                {"channel_units",{"effort","dimensionless","dimensionless","rad/s","A","m/s^2","m/s^2","m/s^2","rad/s","rad/s","rad/s","lux"}},
                {"request_generator","SplitMix64(seed); duration lengths[next()%4], then amplitude int(next()%3)-1; truncate to 240 intervals"},
                {"forecast_input","known normalized requested tape only; future emitted effort is predicted output"},
                {"plant_scope","unchanged governor + motor + mechanics + local sensor sampling"},
                {"public_state_limit","every public mechanical projection is compared; these are not complete private MuJoCo checkpoints"},
                {"interpretation_limit","per-body affine delay-state identification under held-out stationary-source probes; not physical-state observation, source-switch prediction or closed-loop control"}}},
            {"reference_report",Json{{"path",kReference},{"sha256",kReferenceSha},{"snapshot","source/"+std::string(kReference)}}},
            {"compiler",__VERSION__},{"physics_spec",droid::ConstructionWorld::light_specification()},
            {"governor_spec",droid::LightLearner::specification()},{"predictor_spec",droid::light_prediction_specification()},
            {"module_catalog",read_json("config/module_catalog.json")}};
        snapshot_sources(output,report);
        require(report.at("source_sha256").at(std::string(kProtocol))==kProtocolSha,"protocol changed before snapshot");
        for(std::size_t body=0;body<2;++body) {
            const auto& item=reference.at("bodies").at(body);require(hash(item.at("assembly"))==item.at("assembly_sha256"),"reference assembly hash differs");
            report["bodies"].push_back(Json{{"body_index",body},{"assembly",item.at("assembly")},{"assembly_sha256",item.at("assembly_sha256")}});
        }
        Json longer=reference.at("bodies").at(0).at("assembly");longer["segments"]={4,3};longer["name"]="Sun creature with a longer first beam";
        report["bodies"].push_back(Json{{"body_index",2},{"assembly",longer},{"assembly_sha256",hash(longer)}});save(output,report);
        std::size_t failed_trials=0,failed_pairs=0;
        for(std::size_t body=0;body<kBodies;++body) {
            const Json assembly=report.at("bodies").at(body).at("assembly");
            for(std::uint64_t seed=201;seed<=212;++seed) {
                std::cout<<"body "<<body<<", probe "<<seed<<": A and independent replay"<<std::endl;
                Run a=run_trial(assembly,body,seed,true,output);
                if(a.result.at("completed")!=true||a.result.at("exact_full_transition_replay")!=true) ++failed_trials;
                report["trials"].push_back(a.result);report["failed_trials"]=failed_trials;save(output,report);
                std::cout<<"body "<<body<<", probe "<<seed<<": B and independent replay"<<std::endl;
                Run b=run_trial(assembly,body,seed,false,output);
                if(b.result.at("completed")!=true||b.result.at("exact_full_transition_replay")!=true) ++failed_trials;
                report["trials"].push_back(b.result);Json pair=pair_result(a,b,body,seed);
                if(pair.at("valid")!=true) ++failed_pairs;
                report["pairs"].push_back(std::move(pair));report["failed_trials"]=failed_trials;report["failed_pairs"]=failed_pairs;save(output,report);
            }
        }
        report["stage"]="collected";report["checks"]["all_trials_complete"]=failed_trials==0&&report.at("trials").size()==72;
        report["checks"]["all_pairs_valid"]=failed_pairs==0&&report.at("pairs").size()==36;
        report["checks"]["all_full_replays"]=failed_trials==0;save(output,report);
        if(failed_trials==0&&failed_pairs==0) {
            fit_and_evaluate(output,report);
            for(const auto& [relative,expected]:report.at("source_sha256").items()) {
                require(droid::sha256_file(relative)==expected.get<std::string>(),"source changed during experiment: "+relative);
                require(droid::sha256_file((output/"source"/relative).string())==expected.get<std::string>(),"archived source changed: "+relative);
            }
            report["checks"]["source_snapshots_unchanged"]=true;report["complete"]=true;report["stage"]="evaluated";
        }
        report["finished"]=true;save(output,report);return report.at("complete")==true?0:1;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        if(reserved&&report.is_object()) { report["complete"]=false;report["fatal_error"]=error.what();try { save(output,report); } catch(...) {} }
        return 2;
    }
}

