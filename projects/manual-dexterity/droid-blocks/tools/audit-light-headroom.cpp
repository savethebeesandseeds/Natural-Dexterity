#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

// A separate implementation: inspect recorded transitions without linking the
// runner, learner, simulator or its Metrics/decisions implementation. Exact
// physics/controller replay and SHA-256 archive verification are separate work.
namespace {
using J = nlohmann::json;
namespace fs = std::filesystem;
constexpr std::size_t kSteps = 3200, kLateStart = 2400;
constexpr double kDt = .02, kMargin = 3.2;
constexpr std::array<std::size_t, 4> kWindows{320, 640, 1600, 3200};
constexpr std::array<std::uint64_t, 3> kSeeds{101, 202, 303};
constexpr std::array<const char*, 6> kManual{
    "quiet", "constant_negative", "constant_positive", "square_2s", "square_4s", "sensor_search_hold"};

struct Audit {
    J issues = J::array();
    std::size_t failed_checks{};
    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++failed_checks;
            if (issues.size() < 1000) issues.push_back(message);
        }
    }
};
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
double num(const J& value) {
    require(value.is_number(), "expected numeric field");
    const double result = value.get<double>();
    require(std::isfinite(result), "nonfinite numeric field");
    return result;
}
bool near(double a, double b, double tolerance = 2e-9) {
    return std::isfinite(a) && std::isfinite(b) &&
        std::abs(a-b) <= tolerance * std::max({1., std::abs(a), std::abs(b)});
}
void compare(Audit& audit, const J& actual, const J& expected, const std::string& where) {
    if (actual.is_number() && expected.is_number()) {
        audit.check(near(num(actual), num(expected)), where + ": number differs");
    } else if (expected.is_object()) {
        audit.check(actual.is_object(), where + ": expected object");
        if (!actual.is_object()) return;
        audit.check(actual.size() == expected.size(), where + ": field count differs");
        for (const auto& [key, value] : expected.items()) {
            audit.check(actual.contains(key), where + ": missing " + key);
            if (actual.contains(key)) compare(audit, actual.at(key), value, where + "/" + key);
        }
    } else if (expected.is_array()) {
        audit.check(actual.is_array() && actual.size() == expected.size(), where + ": array size differs");
        if (actual.is_array()) for (std::size_t i=0; i<std::min(actual.size(),expected.size()); ++i)
            compare(audit, actual.at(i), expected.at(i), where + "/" + std::to_string(i));
    } else audit.check(actual == expected, where + ": value differs");
}
fs::path evidence_path(const fs::path& root, const J& relative) {
    require(relative.is_string(), "evidence path is not text");
    const std::string name = relative.get<std::string>();
    require(!name.empty() && name.find('\\') == std::string::npos &&
        name.find(':') == std::string::npos && name.find('\0') == std::string::npos,
        "invalid portable evidence path");
    const fs::path path(name);
    require(!path.is_absolute() && !path.has_root_name() && !path.has_root_directory(), "absolute evidence path");
    fs::path cursor = root;
    for (const auto& part : path) {
        require(part != ".." && part != "." && !part.empty(), "evidence path traversal");
        cursor /= part;
        require(!fs::is_symlink(cursor), "evidence symlink is unsupported");
    }
    require(fs::is_regular_file(cursor), "missing evidence file: " + name);
    return cursor;
}
J file_json(const fs::path& path, std::uintmax_t bound) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= bound, "JSON file missing or too large: " + path.string());
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot read JSON file");
    return J::parse(input);
}
// getline normally allocates an unbounded string; cap each row before parsing.
bool line_bounded(std::istream& input, std::string& line) {
    line.clear();
    char ch;
    while (input.get(ch)) {
        if (ch == '\n') return true;
        require(line.size() < 2*1024*1024, "JSONL record exceeds 2 MiB");
        line.push_back(ch);
    }
    require(!input.bad(), "tape read failed");
    return !line.empty();
}
const J& sensor(const J& observation, const char* family) {
    const J* found = nullptr;
    require(observation.at("reward_sensors").is_array() && observation.at("reward_sensors").size()==2,
        "expected exactly two sensors");
    for (const auto& item : observation.at("reward_sensors")) if (item.at("family_id") == family) {
        require(found == nullptr, "duplicate sensor family"); found = &item;
    }
    require(found != nullptr, std::string("missing sensor ") + family);
    return *found;
}
const J& feedback(const J& observation) {
    const auto& motors = observation.at("actuator_feedback");
    require(motors.is_array() && motors.size()==1, "expected exactly one actuator");
    require(motors.at(0).at("module_id") == "motor-0001", "unexpected actuator identity");
    return motors.at(0).at("feedback");
}
struct Totals {
    std::size_t steps{}, vetoes{}, flags{}, invalid_sensor{}, invalid_motor{}, lux_endpoints{};
    double reward{}, light{}, imu{}, lux_sum{}, energy{}, current{}, temperature{}, speed{}, absolute_effort{}, max_effort{};
    void add(const J& transition, double effort, Audit& audit, const std::string& where) {
        ++steps;
        audit.check(transition.at("info").at("physics_steps_executed") == 10, where + ": physics step count");
        audit.check(near(num(transition.at("info").at("elapsed_control_dt_s")), kDt,1e-12), where + ": duration");
        audit.check(transition.at("truncated") == false, where + ": unexpected truncation");
        double parts=0.; std::set<std::string> families;
        require(transition.at("reward_components").is_array() && transition.at("reward_components").size()==2,
            "expected exactly two reward components");
        for (const auto& component : transition.at("reward_components")) {
            const auto family=component.at("family_id").get<std::string>();
            require(families.insert(family).second, "duplicate reward component");
            const double part=num(component.at("transition_reward"));
            const double rate=num(component.at("reward_rate"));
            audit.check(std::abs(part)<=kDt+1e-9 && std::abs(rate)<=1.+1e-9,where + ": bounded sensor vote");
            const auto& sample=sensor(transition.at("observation"),family.c_str());
            audit.check(component.at("module_id")==sample.at("module_id") && component.at("valid")==sample.at("valid"),
                where + ": component/sample identity");
            parts+=part;
            if (family=="ambient_light_v0") light+=part;
            else if (family=="imu_6axis_v0") imu+=part;
            else throw std::runtime_error("unknown reward family");
        }
        audit.check(near(parts,num(transition.at("reward")),1e-12),where + ": component sum");
        // Independent accumulation is from components, not the runner's total.
        reward+=parts;
        const auto& observation=transition.at("observation");
        for (const auto& item:observation.at("reward_sensors")) if (item.at("valid")!=true) ++invalid_sensor;
        const auto& lamp=sensor(observation,"ambient_light_v0");
        if (lamp.at("valid")==true) { lux_sum+=num(lamp.at("observations").at("illuminance_lux")); ++lux_endpoints; }
        if (observation.at("actuator_feedback").at(0).at("valid")!=true) ++invalid_motor;
        const auto& fb=feedback(observation);
        current=std::max(current,std::abs(num(fb.at("current_a"))));
        temperature=std::max(temperature,num(fb.at("temperature_c")));
        speed=std::max(speed,std::abs(num(fb.at("velocity_rad_s"))));
        if (transition.at("safety").at("veto")==true || transition.at("terminated")==true) ++vetoes;
        if (!transition.at("safety").at("flags").empty() || !fb.at("fault_flags").empty()) ++flags;
        absolute_effort+=std::abs(effort); max_effort=std::max(max_effort,std::abs(effort));
    }
    bool safe() const { return !vetoes && !flags && !invalid_sensor && !invalid_motor; }
    J json() const {
        return J{{"steps",steps},{"duration_s",steps*kDt},{"integrated_sensor_reward",reward},
            {"integrated_light_reward",light},{"integrated_imu_reward",imu},
            {"mean_sensor_reward_rate",steps?reward/(steps*kDt):0.},
            {"sampled_mean_received_lux",steps?lux_sum/static_cast<double>(steps):0.},
            {"valid_light_control_endpoints",lux_endpoints},{"electrical_energy_j",energy},
            {"sampled_max_current_a",current},{"sampled_max_temperature_c",temperature},
            {"sampled_max_motor_speed_rad_s",speed},
            {"mean_absolute_effort",steps?absolute_effort/static_cast<double>(steps):0.},
            {"max_absolute_effort",max_effort},{"native_veto_steps",vetoes},{"flagged_steps",flags},
            {"invalid_reward_sensor_samples",invalid_sensor},{"invalid_feedback_samples",invalid_motor}};
    }
};
J expected_cases() {
    J result=J::array();
    constexpr std::array<std::array<int,2>,3> lengths{{{4,3},{3,4},{4,4}}};
    constexpr std::array<const char*,3> names{"long-upper","long-lower","weighted"};
    for (std::size_t body=0;body<3;++body) for (int hinge:{0,1}) for (bool lower:{false,true}) for (int side:{-1,1}) {
        const std::string id=std::string(names[body])+"-hinge"+std::to_string(hinge)+
            (lower?"-lower-":"-high-")+(side<0?"left":"right");
        J assembly{{"schema","construction_kit_v2"},{"name","The same parts, rebuilt"},
            {"segments",lengths[body]},{"powered_hinge",hinge},{"blocks",J::array()},
            {"sensor",J{{"segment",1},{"slot",2},{"side",-1}}},
            {"light",J{{"segment",1},{"slot",2},{"side",1},{"face",1}}}};
        if (body==2) assembly["blocks"].push_back(J{{"id","block-1"},{"segment",1},{"slot",0},{"side",-1}});
        result.push_back(J{{"id",id},{"primary",lower},{"assembly",assembly},
            {"sun",J{{"position_m",{side*.3,0.,lower?-.3:.2}},{"intensity_lux",1000.}}}});
    }
    return result;
}
J expected_controller(std::size_t index) {
    if (index<kManual.size()) return J{{"mode",kManual[index]},{"seed",nullptr}};
    return J{{"mode","learner"},{"seed",kSeeds.at(index-kManual.size())}};
}
void sample_continuity(const J& observation,const J& previous,std::size_t step,Audit& audit,const std::string& where) {
    audit.check(observation.at("schema_version")=="construction_observation_v2",where+": observation schema");
    for (const char* family:{"imu_6axis_v0","ambient_light_v0"}) {
        const auto& current=sensor(observation,family);
        const auto& old=sensor(previous,family);
        audit.check(current.at("module_id")==old.at("module_id"),where+": sensor identity changed");
        const bool lamp=std::string(family)=="ambient_light_v0";
        audit.check(near(num(current.at("sample_period_s")),lamp?.1:.01,1e-12) &&
            near(num(current.at("latency_s")),lamp?.02:.006,1e-12),where+": acquisition contract");
        if (current.at("valid")!=true) continue;
        const double acquired=num(current.at("sample_time_s"));
        const double delivered=num(current.at("delivered_time_s"));
        audit.check(near(acquired+num(current.at("age_s")),step*kDt,1e-8),where+": sensor age/time");
        audit.check(acquired<=delivered && delivered<=step*kDt+1e-8,where+": future sensor sample");
        audit.check(near(delivered-acquired,lamp?.02:.006,1e-8),where+": delivery latency");
        if (old.at("valid")==true) {
            audit.check(current.at("sequence").get<std::uint64_t>()>=old.at("sequence").get<std::uint64_t>(),where+": sequence reversal");
            if (current.at("sequence")==old.at("sequence")) for (const char* key:{"sample_time_s","delivered_time_s","observations"})
                audit.check(current.at(key)==old.at(key),where+": mutated held sample");
            else audit.check(acquired>num(old.at("sample_time_s")),where+": acquisition time did not advance");
        }
    }
}
double governor(double request,double previous,const J& observation) {
    if (sensor(observation,"imu_6axis_v0").at("valid")!=true || sensor(observation,"ambient_light_v0").at("valid")!=true) return 0.;
    double target=std::clamp(request,-.35,.35);
    if (target!=0.) {
        const double sign=target>0.?1.:-1.;
        target=sign*std::min(std::abs(target),.05*std::max(0.,3.-sign*num(feedback(observation).at("velocity_rad_s"))));
    }
    return std::clamp(target,previous-.1,previous+.1);
}
double fixed_request(const std::string& mode,std::size_t zero_based) {
    if (zero_based<5 || mode=="quiet") return 0.;
    if (mode=="constant_negative") return -.15;
    if (mode=="constant_positive") return .15;
    const std::size_t offset=zero_based-5;
    if (mode=="square_2s") return (offset/50)%2? .15:-.15;
    if (mode=="square_4s") return (offset/100)%2? .15:-.15;
    throw std::runtime_error("not a fixed controller");
}
void physical_endpoint(Audit& audit,const J& physics,const J& transition,std::size_t step,double emitted,double reward,const J& scene,const std::string& where) {
    audit.check(near(num(physics.at("elapsed_s")),step*kDt,1e-8) && physics.at("tick")==step*10,where+": physical clock");
    audit.check(physics.at("assembly")==scene.at("assembly") && physics.at("sun")==scene.at("sun"),where+": physical scene changed");
    audit.check(near(num(physics.at("command")),emitted,1e-12),where+": physical command");
    audit.check(near(num(physics.at("reward").at("cumulative")),reward,2e-9),where+": physical cumulative reward");
    audit.check(physics.at("actuator")==feedback(transition.at("observation")),where+": physical actuator feedback");
    audit.check(physics.at("safety")==transition.at("safety"),where+": physical safety");
    for (const auto& [physical_key,family]:std::array<std::pair<const char*,const char*>,2>{{{"sensor","imu_6axis_v0"},{"light_sensor","ambient_light_v0"}}}) {
        const auto& recorded=physics.at(physical_key); const auto& observed=sensor(transition.at("observation"),family);
        for (const auto& [key,value]:observed.items()) audit.check(recorded.contains(key) && recorded.at(key)==value,where+": physical sensor differs: "+key);
    }
}

J trial_audit(const fs::path& root,const J& entry,const J& scene,Audit& audit) {
    const std::string id=entry.at("id").get<std::string>();
    audit.check(entry.at("assembly")==scene.at("assembly") && entry.at("sun")==scene.at("sun"),id+": retained trial case");
    audit.check(entry.at("trace").at("path")=="traces/"+id+".jsonl" &&
        entry.at("snapshots").at("path")=="snapshots/"+id+".json",id+": expected sidecar paths");
    const auto tape_path=evidence_path(root,entry.at("trace").at("path"));
    require(fs::file_size(tape_path)<=128*1024*1024,"transition tape exceeds 128 MiB");
    const auto snapshots_path=evidence_path(root,entry.at("snapshots").at("path"));
    const J snapshots=file_json(snapshots_path,32*1024*1024);
    require(snapshots.is_array() && !snapshots.empty() && snapshots.size()<=322,"invalid snapshot count");
    std::ifstream input(tape_path,std::ios::binary); require(static_cast<bool>(input),"cannot open tape");
    std::string line; require(line_bounded(input,line),"empty tape");
    const J initial=J::parse(line);
    audit.check(initial.at("type")=="initial" && initial.at("schema")=="light_headroom_tape_v1",id+": initial schema");
    audit.check(initial.at("case_id")==scene.at("id") && initial.at("controller")==entry.at("controller"),id+": initial identity");
    audit.check(initial.at("assembly")==scene.at("assembly") && initial.at("sun")==scene.at("sun"),id+": initial case");
    audit.check(initial.at("planned_steps")==kSteps && entry.at("planned_steps")==kSteps,id+": planned budget");
    audit.check(initial.at("physics").at("elapsed_s")==0. && initial.at("physics").at("tick")==0 &&
        initial.at("physics").at("reward").at("cumulative")==0.,id+": reset clock/reward");
    audit.check(initial.at("physics").at("assembly")==scene.at("assembly") && initial.at("physics").at("sun")==scene.at("sun"),id+": reset physical case");
    audit.check(snapshots.at(0)==J{{"step",0},{"physics",initial.at("physics")}},id+": reset snapshot");
    for (const char* family:{"imu_6axis_v0","ambient_light_v0"}) audit.check(sensor(initial.at("observation"),family).at("valid")==false,id+": reset must have fresh queues");
    const auto& initial_fb=feedback(initial.at("observation"));
    audit.check(num(initial_fb.at("position_rad"))==0. && num(initial_fb.at("velocity_rad_s"))==0.,id+": reset shaft state");
    J previous=initial.at("observation"),windows=J::object(),final,last_transition;
    Totals totals,late;
    std::size_t records=1,snapshot_index=1;
    double previous_effort=0.,late_energy_origin=0.;
    std::array<double,4> search_scores{};
    std::size_t selected=0;
    const std::string mode=entry.at("controller").at("mode").get<std::string>();
    bool stopped=false;
    while (line_bounded(input,line)) {
        ++records; require(records<=kSteps+2,"tape exceeds declared budget");
        const J row=J::parse(line);
        if (row.at("type")=="final") { final=row; break; }
        require(row.at("type")=="step","unknown tape row");
        audit.check(!stopped,id+": interaction continued after safety stop");
        const std::size_t step=totals.steps+1;
        audit.check(row.at("step")==step,id+": nonconsecutive step");
        const double effort=num(row.at("effort"));
        audit.check(std::abs(effort)<=.15+1e-12 && std::abs(effort-previous_effort)<=.1+1e-12,id+": emitted effort envelope");
        double request=0.; bool independent_request=false;
        if (mode!="learner" && mode!="sensor_search_hold") { request=fixed_request(mode,step-1); independent_request=true; }
        else if (mode=="sensor_search_hold") {
            independent_request=true;
            if (step-1>=5) {
                const std::size_t offset=step-1-5;
                std::size_t target=offset<800?offset/200:selected;
                if (offset==800) { selected=static_cast<std::size_t>(std::max_element(search_scores.begin(),search_scores.end())-search_scores.begin()); target=selected; }
                const double pi=std::acos(-1.);
                const std::array<double,4> angles{0.,pi/2,-pi/2,pi};
                const auto& fb=feedback(previous);
                const double delta=angles.at(target)-num(fb.at("position_rad"));
                request=std::clamp(.30*std::atan2(std::sin(delta),std::cos(delta))-.05*num(fb.at("velocity_rad_s")),-.15,.15);
            }
        }
        if (independent_request) audit.check(near(governor(request,previous_effort,previous),effort,1e-12),id+": independently governed command at "+std::to_string(step));
        if (row.contains("requested_effort")) {
            const double retained_request=num(row.at("requested_effort"));
            if (independent_request) audit.check(near(retained_request,request,1e-12),id+": requested effort differs");
            else {
                audit.check(retained_request==-.15 || retained_request==0. || retained_request==.15,id+": learner request vocabulary");
                audit.check(near(governor(retained_request,previous_effort,previous),effort,1e-12),id+": learner requested/emitted governor");
            }
        } else audit.check(false,id+": missing requested_effort");
        const auto& transition=row.at("transition");
        sample_continuity(transition.at("observation"),previous,step,audit,id);
        totals.add(transition,effort,audit,id);
        if (step>kLateStart) late.add(transition,effort,audit,id+" late");
        if (mode=="sensor_search_hold" && step-1>=5) {
            const std::size_t offset=step-1-5;
            if (offset<800 && offset%200>=150) search_scores.at(offset/200)+=num(transition.at("reward"));
            if (offset==799) selected=static_cast<std::size_t>(std::max_element(search_scores.begin(),search_scores.end())-search_scores.begin());
        }
        const bool native_stopped=transition.at("terminated")==true || transition.at("safety").at("veto")==true ||
            !transition.at("safety").at("flags").empty() || !feedback(transition.at("observation")).at("fault_flags").empty();
        stopped=stopped||native_stopped;
        if (snapshot_index<snapshots.size() && snapshots.at(snapshot_index).at("step")==step) {
            physical_endpoint(audit,snapshots.at(snapshot_index).at("physics"),transition,step,effort,totals.reward,scene,id+" snapshot");
            totals.energy=num(snapshots.at(snapshot_index).at("physics").at("diagnostics").at("electrical_energy_j"));
            if (step==kLateStart) late_energy_origin=totals.energy;
            ++snapshot_index;
        }
        if (std::find(kWindows.begin(),kWindows.end(),step)!=kWindows.end()) windows[std::to_string(step)]=totals.json();
        previous=transition.at("observation"); previous_effort=effort; last_transition=transition;
    }
    require(final.is_object(),"missing final tape record; retained prefix is not complete");
    require(!line_bounded(input,line),"extra record after final");
    totals.energy=num(final.at("physics").at("diagnostics").at("electrical_energy_j"));
    late.energy=late.steps?totals.energy-late_energy_origin:0.;
    audit.check(snapshot_index==snapshots.size(),id+": unmatched physical snapshots");
    audit.check(entry.at("snapshots").at("count")==snapshots.size() && entry.at("trace").at("records")==records,id+": sidecar counts");
    audit.check(snapshots.size()==1+totals.steps/10,id+": mandatory snapshot schedule");
    for (std::size_t n=0;n<snapshots.size();++n) audit.check(snapshots.at(n).at("step")==n*10,id+": snapshot order");
    audit.check(entry.at("recorded_steps")==totals.steps,id+": recorded prefix count");
    audit.check(entry.at("completed_steps")==totals.steps && totals.steps<=kSteps,id+": step count");
    const bool complete=totals.steps==kSteps && final.at("error")=="";
    const bool safe=complete && totals.safe();
    audit.check(entry.at("completed")==complete && final.at("completed")==complete && entry.at("error")==final.at("error"),id+": completion metadata");
    audit.check(entry.at("safe")==safe,id+": safety metadata");
    compare(audit,entry.at("metrics"),totals.json(),id+" metrics");
    compare(audit,final.at("metrics"),totals.json(),id+" final metrics");
    compare(audit,entry.at("windows"),windows,id+" windows");
    if (totals.steps) {
        // A final state must agree with the final observation even if no periodic
        // snapshot was due at this failed prefix's last control.
        physical_endpoint(audit,final.at("physics"),last_transition,totals.steps,previous_effort,totals.reward,scene,id+" final");
    }
    compare(audit,entry.at("late_window"),J{{"start_step",2400},{"end_step",3200},{"metrics",late.json()}},id+" late window");
    if (mode=="sensor_search_hold") {
        compare(audit,final.at("controller_final").at("scores"),J(search_scores),id+" search scores");
        audit.check(final.at("controller_final").at("best_target_index")==selected,id+": selected search target");
    }
    audit.check(near(totals.reward,totals.light+totals.imu),id+": total integrated components");
    return J{{"id",id},{"controller",entry.at("controller")},{"completed",complete},{"safe_from_tape",safe},
        {"recomputed_metrics",totals.json()},{"recomputed_late_window",late.json()},
        {"search_target_scores",mode=="sensor_search_hold"?J(search_scores):J(nullptr)},
        {"search_selected_target",mode=="sensor_search_hold"?J(selected):J(nullptr)}};
}

J audit_directory(const fs::path& root,Audit& audit) {
    const J report=file_json(root/"report.json",128*1024*1024);
    audit.check(report.at("schema")=="light_headroom_development_v1","report schema");
    const J expected=expected_cases();
    audit.check(report.at("cases").is_array() && report.at("cases").size()==24,"expected exactly 24 cases");
    audit.check(report.at("manifest").at("expected_runs")==216,"manifest expected216 runs");
    audit.check(report.at("manifest").at("steps")==kSteps && near(num(report.at("manifest").at("native_dt_s")),kDt,1e-12),"manifest budget");
    audit.check(report.at("manifest").at("cases")==expected,"manifest exact case matrix");
    compare(audit,report.at("manifest").at("learner_seeds"),J(kSeeds),"manifest learner seeds");
    compare(audit,report.at("manifest").at("windows_steps"),J(kWindows),"manifest windows");
    audit.check(report.at("source_sha256").contains("tools/audit-light-headroom.cpp") &&
        report.at("source_sha256").contains("docs/LIGHT_HEADROOM_EXPERIMENT.md"),"protocol/auditor missing from source archive");
    audit.check(report.at("source_sha256")==report.at("source_sha256_after_collection"),"source inputs changed during collection");
    audit.check(report.at("source_sha256").at("docs/LIGHT_HEADROOM_EXPERIMENT.md")==report.at("protocol").at("sha256"),"recorded protocol archive identity");
    audit.check(report.at("manifest").at("manual_quiet_steps")==5 &&
        report.at("manifest").at("snapshot_interval_steps")==10,"manifest schedule");
    J outputs=J::array(),contrasts=J::array();
    std::size_t runs=0,failed=0,primary_count=0,passing_headroom=0,passing_fixed=0,passing_search=0;
    std::array<std::size_t,3> learner_passes{};
    std::set<std::string> seen_cases,seen_trials;
    for (std::size_t i=0;i<report.at("cases").size();++i) {
        require(i<expected.size(),"too many cases");
        const auto& scene=report.at("cases").at(i);
        const std::string case_id=scene.at("id");
        audit.check(seen_cases.insert(case_id).second,"duplicate case "+case_id);
        for (const char* key:{"id","primary","assembly","sun"}) audit.check(scene.at(key)==expected.at(i).at(key),case_id+"/"+key+": exact case differs");
        audit.check(scene.at("trials").is_array() && scene.at("trials").size()==9,case_id+": expected nine trials");
        J case_out{{"case_id",case_id},{"trials",J::array()}};
        std::vector<J> checked;
        for (std::size_t n=0;n<scene.at("trials").size();++n) {
            ++runs;
            const auto& entry=scene.at("trials").at(n);
            const std::string trial_id=entry.at("id");
            audit.check(n<9,case_id+": extra trial");
            if (n<9) compare(audit,entry.at("controller"),expected_controller(n),trial_id+" controller");
            if (n<9) {
                const J control=expected_controller(n);
                const std::string label=control.at("mode").get<std::string>()+
                    (control.at("seed").is_null()?"":"-"+std::to_string(control.at("seed").get<std::uint64_t>()));
                audit.check(trial_id==case_id+"--"+label,trial_id+": expected ordered trial ID");
            }
            audit.check(entry.at("case_id")==case_id && seen_trials.insert(trial_id).second,trial_id+": trial identity");
            const std::size_t before=audit.failed_checks;
            J result;
            try { result=trial_audit(root,entry,scene,audit); }
            catch (const std::exception& error) { audit.check(false,trial_id+": "+error.what()); result=J{{"id",trial_id},{"safe_from_tape",false},{"completed",false},{"controller",entry.at("controller")}}; }
            result["record_consistent"]=audit.failed_checks==before;
            result["valid_witness"]=result.at("safe_from_tape")==true && result.at("record_consistent")==true;
            if (result.at("valid_witness")!=true) ++failed;
            checked.push_back(result); case_out["trials"].push_back(result);
        }
        const bool primary=expected.at(i).at("primary");
        if (primary) ++primary_count;
        J deltas=J::array(); bool headroom=false,fixed=false,search=false;
        J best_id=nullptr,best_delta=nullptr,feedback_id=nullptr,feedback_delta=nullptr;
        std::array<bool,3> learned{};
        if (!checked.empty() && checked.at(0).contains("recomputed_metrics")) {
            const double quiet=num(checked.at(0).at("recomputed_metrics").at("integrated_sensor_reward"));
            for (std::size_t n=0;n<checked.size();++n) {
                const auto& item=checked[n];
                if (!item.contains("recomputed_metrics")) continue;
                const double gain=num(item.at("recomputed_metrics").at("integrated_sensor_reward"))-quiet;
                const bool pair=checked.at(0).at("valid_witness")==true && item.at("valid_witness")==true;
                const bool passes=pair && gain>kMargin;
                deltas.push_back(J{{"trial_id",item.at("id")},{"safe_pair",pair},
                    {"integrated_reward_minus_quiet",gain},{"reward_rate_minus_quiet",gain/64.},{"clears_threshold",passes}});
                if (n>=1 && n<=4) fixed=fixed||passes;
                if (n==5) search=passes;
                if (n>=6 && n<=8) learned[n-6]=passes;
                if (n>=1 && n<=4 && pair && (best_delta.is_null() || gain>num(best_delta))) { best_id=item.at("id"); best_delta=gain; }
                if (n==5 && pair) { feedback_id=item.at("id"); feedback_delta=gain; }
            }
        }
        headroom=fixed||search;
        if (primary) { if (headroom) ++passing_headroom; if (fixed) ++passing_fixed; if (search) ++passing_search;
            for (std::size_t n=0;n<3;++n) if (learned[n]) ++learner_passes[n]; }
        contrasts.push_back(J{{"case_id",case_id},{"primary",primary},{"headroom",headroom},
            {"fixed_headroom",fixed},{"feedback_headroom",search},
            {"best_safe_fixed_trial",best_id},{"best_safe_fixed_delta",best_delta},
            {"safe_feedback_trial",feedback_id},{"safe_feedback_delta",feedback_delta},{"contrasts",deltas}});
        outputs.push_back(case_out);
    }
    audit.check(runs==216 && report.at("completed_runs")==runs,"report run count");
    audit.check(report.at("failed_runs")==failed,"report failed run count");
    audit.check(report.at("complete")==true,"report is an incomplete preserved batch");
    J seed_gates=J::array(); bool competent=primary_count==12;
    for (std::size_t n=0;n<3;++n) { const bool pass=primary_count==12 && learner_passes[n]==12; competent=competent&&pass;
        seed_gates.push_back(J{{"seed",kSeeds[n]},{"passing_primary_cases",learner_passes[n]},{"passed",pass}}); }
    const bool headroom=primary_count==12 && passing_headroom==12;
    if (report.contains("comparison")) {
        const auto& comparison=report.at("comparison");
        audit.check(near(num(comparison.at("integrated_reward_threshold")),kMargin,1e-12),"comparison margin");
        audit.check(comparison.at("primary_case_count")==primary_count,"comparison primary count");
        audit.check(comparison.at("headroom_gate").at("passed")==headroom &&
            comparison.at("headroom_gate").at("passing_primary_cases")==passing_headroom,"comparison headroom gate");
        audit.check(comparison.at("learner_competence_gate").at("passed")==competent,"comparison learner gate");
        compare(audit,comparison.at("learner_competence_gate").at("seeds"),seed_gates,"comparison learner seeds");
        compare(audit,comparison.at("cases"),contrasts,"comparison case results");
    } else audit.check(false,"missing retained comparison");
    return J{{"schema","light_headroom_independent_audit_v1"},{"input",root.generic_string()},
        {"retained_trials",runs},{"inconsistent_or_unsafe_trials",failed},
        {"primary_case_count",primary_count},{"integrated_reward_threshold",kMargin},
        {"headroom_gate",J{{"passed",headroom},{"passing_primary_cases",passing_headroom},
            {"passing_fixed_schedule_cases",passing_fixed},{"passing_search_hold_cases",passing_search}}},
        {"learner_competence_gate",J{{"passed",competent},{"seeds",seed_gates}}},
        {"comparisons",contrasts},{"cases",outputs},
        {"scope","Independent bounded JSONL accounting, permitted manual requests/governor, endpoint consistency and gate recomputation. No physical or learner replay and no SHA-256 verification; those require the separate native verifier/archive check."}};
}
void write_new(const fs::path& path,const J& value) {
    require(fs::is_directory(path.parent_path()),"audit output parent does not exist");
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    require(fd>=0,"refusing existing/unwritable audit output");
    const std::string bytes=value.dump(2)+"\n";
    std::size_t offset=0;
    while(offset<bytes.size()) { const ssize_t written=::write(fd,bytes.data()+offset,bytes.size()-offset);
        if(written<=0) { ::close(fd); throw std::runtime_error("failed writing audit output; preserve partial file"); }
        offset+=static_cast<std::size_t>(written); }
    require(::close(fd)==0,"failed closing audit output");
}
}
int main(int argc,char** argv) {
    fs::path input,output;
    try {
        require(argc==5,"Usage: audit-light-headroom --input BATCH_DIR --output NEW_FILE");
        for(int i=1;i<argc;i+=2) { const std::string argument=argv[i];
            if(argument=="--input" && input.empty()) input=argv[i+1];
            else if(argument=="--output" && output.empty()) output=argv[i+1];
            else throw std::runtime_error("invalid or duplicate audit argument"); }
        require(!input.empty() && !output.empty(),"missing audit argument");
        require(!fs::exists(output),"refusing existing audit output");
        input=fs::canonical(input); output=fs::absolute(output);
        Audit audit; J result;
        try { result=audit_directory(input,audit); }
        catch(const std::exception& error) { audit.check(false,error.what()); result=J{{"schema","light_headroom_independent_audit_v1"},{"input",input.generic_string()}}; }
        result["issues"]=audit.issues; result["failed_checks"]=audit.failed_checks;
        result["integrity_passed"]=audit.failed_checks==0;
        write_new(output,result);
        std::cout<<J{{"integrity_passed",audit.failed_checks==0},{"issues",audit.issues.size()},
            {"failed_checks",audit.failed_checks},{"output",output.generic_string()}}.dump()<<'\n';
        return audit.failed_checks==0?0:1;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
