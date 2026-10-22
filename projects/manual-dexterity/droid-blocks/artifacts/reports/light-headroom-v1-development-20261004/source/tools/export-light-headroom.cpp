#include "droid/policy.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>

// Export retained, independently accounted evidence. This executable does not
// construct a simulator or controller, and never synthesizes physical frames.
namespace {
using J = nlohmann::json;
namespace fs = std::filesystem;
constexpr double kDt = .02, kDuration = 64., kGain = 3.2;
constexpr std::array<const char*, 6> kModes{
    "quiet", "constant_negative", "constant_positive", "square_2s", "square_4s", "sensor_search_hold"};
constexpr std::array<const char*, 6> kTitles{
    "Quiet motor", "Constant negative effort", "Constant positive effort", "Square wave · 2 s", "Square wave · 4 s", "Sensor search and hold"};
constexpr std::array<std::uint64_t, 3> kSeeds{101, 202, 303};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
double number(const J& value) {
    require(value.is_number(), "expected numeric evidence field");
    const double result = value.get<double>();
    require(std::isfinite(result), "nonfinite evidence field");
    return result;
}
bool near(double a, double b, double tolerance = 2e-9) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tolerance*std::max({1., std::abs(a), std::abs(b)});
}
void compare(const J& actual, const J& expected, const std::string& where) {
    if (actual.is_number() && expected.is_number()) require(near(number(actual), number(expected)), where+": numeric evidence differs");
    else if (expected.is_object()) {
        require(actual.is_object() && actual.size()==expected.size(), where+": object fields differ");
        for (const auto& [key,value] : expected.items()) compare(actual.at(key),value,where+"/"+key);
    } else if (expected.is_array()) {
        require(actual.is_array() && actual.size()==expected.size(), where+": array size differs");
        for (std::size_t i=0;i<expected.size();++i) compare(actual.at(i),expected.at(i),where+"/"+std::to_string(i));
    } else require(actual==expected,where+": evidence differs");
}
bool present(const fs::path& path) { return fs::symlink_status(path).type()!=fs::file_type::not_found; }
fs::path unlinked_absolute(const fs::path& path) {
    const fs::path absolute=fs::absolute(path).lexically_normal();
    fs::path cursor;
    for (const auto& component:absolute) {
        cursor/=component;
        require(!fs::is_symlink(fs::symlink_status(cursor)),"symlink path is unsupported: "+cursor.generic_string());
    }
    return absolute;
}
fs::path evidence_path(const fs::path& root,const J& relative) {
    require(relative.is_string(),"evidence path is not text");
    const std::string name=relative.get<std::string>();
    require(!name.empty() && name.find('\\')==std::string::npos && name.find(':')==std::string::npos &&
        name.find('\0')==std::string::npos,"invalid portable evidence path");
    const fs::path path(name);
    require(!path.is_absolute() && !path.has_root_name() && !path.has_root_directory(),"absolute evidence path");
    fs::path cursor=root;
    for (const auto& component:path) {
        require(component!=".." && component!="." && !component.empty(),"evidence path traversal");
        cursor/=component;
        require(!fs::is_symlink(fs::symlink_status(cursor)),"evidence symlink is unsupported");
    }
    require(fs::is_regular_file(cursor),"missing evidence file: "+name);
    return cursor;
}
J file_json(const fs::path& path,std::uintmax_t bound) {
    require(fs::is_regular_file(path) && fs::file_size(path)<=bound,"JSON missing or too large: "+path.generic_string());
    std::ifstream input(path,std::ios::binary);
    require(static_cast<bool>(input),"cannot read JSON: "+path.generic_string());
    return J::parse(input);
}
bool digest(const J& value) {
    if (!value.is_string()) return false;
    const std::string text=value.get<std::string>();
    return text.size()==64 && std::all_of(text.begin(),text.end(),[](char ch) { return (ch>='0' && ch<='9') || (ch>='a' && ch<='f'); });
}
void hash_matches(const fs::path& path,const J& wanted) {
    require(digest(wanted) && droid::sha256_file(path)==wanted.get<std::string>(),"evidence hash differs: "+path.generic_string());
}
void write_new(const fs::path& path,const J& value) {
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    require(fd>=0,"refusing existing/unwritable export: "+path.generic_string());
    const std::string bytes=value.dump()+"\n";
    std::size_t offset=0;
    while (offset<bytes.size()) {
        const ssize_t written=::write(fd,bytes.data()+offset,bytes.size()-offset);
        if (written<0 && errno==EINTR) continue;
        if (written<=0) { ::close(fd); throw std::runtime_error("export write failed; preserve partial file: "+path.generic_string()); }
        offset+=static_cast<std::size_t>(written);
    }
    if (::fsync(fd)!=0) { ::close(fd); throw std::runtime_error("cannot flush export; preserve partial file"); }
    require(::close(fd)==0,"cannot close export; preserve partial file");
}
J controller_spec(std::size_t index) {
    return index<kModes.size() ? J{{"mode",kModes[index]},{"seed",nullptr}} : J{{"mode","learner"},{"seed",kSeeds.at(index-kModes.size())}};
}
std::string controller_id(std::size_t index) {
    return index<kModes.size() ? kModes[index] : "learner-"+std::to_string(kSeeds.at(index-kModes.size()));
}
J controllers() {
    J result=J::array();
    for (std::size_t i=0;i<9;++i) {
        J item{{"id",controller_id(i)},{"title",i<kModes.size()?std::string(kTitles[i]):"Native learner · seed "+std::to_string(kSeeds.at(i-kModes.size()))},
            {"kind",i==0?"quiet":i<kModes.size()?"handwritten":"learner"},{"mode",controller_spec(i).at("mode")}};
        if (i>=kModes.size()) item["seed"]=kSeeds.at(i-kModes.size());
        result.push_back(std::move(item));
    }
    return result;
}
std::string scene_title(const J& scene) {
    const auto& assembly=scene.at("assembly");
    const auto& lengths=assembly.at("segments");
    const auto& position=scene.at("sun").at("position_m");
    return std::to_string(lengths.at(0).get<int>())+" + "+std::to_string(lengths.at(1).get<int>())+" studs"+
        (assembly.at("blocks").empty()?"":" · weighted")+" · motor "+(assembly.at("powered_hinge")==0?"at anchor":"at elbow")+
        " · lamp "+(number(position.at(2))<0?"below":"above")+" "+(number(position.at(0))<0?"left":"right");
}
void validate_binding(const fs::path& input,const J& report,const J& audit) {
    require(report.at("schema")=="light_headroom_development_v1" && report.at("complete")==true,"input is not a completed retained headroom batch");
    require(report.at("cases").is_array() && report.at("cases").size()==24 && report.at("completed_runs")==216,"expected 24 cases and 216 retained trials");
    require(audit.at("schema")=="light_headroom_independent_audit_v1" && audit.at("integrity_passed")==true &&
        audit.at("issues").is_array() && audit.at("issues").empty(),"independent accounting did not pass");
    require(fs::path(audit.at("input").get<std::string>()).lexically_normal()==input,"independent audit belongs to a different batch path");
    require(audit.at("retained_trials")==216 && audit.at("primary_case_count")==12 && audit.at("cases").size()==24,"independent audit has a different scope");
    const auto& comparison=report.at("comparison");
    require(near(number(comparison.at("integrated_reward_threshold")),kGain,1e-12) && comparison.at("primary_case_count")==12,"declared headroom criterion differs");
    require(comparison.at("headroom_gate").at("passed")==audit.at("headroom_gate").at("passed") &&
        comparison.at("headroom_gate").at("passing_primary_cases")==audit.at("headroom_gate").at("passing_primary_cases"),"headroom gate differs from audit");
    compare(comparison.at("learner_competence_gate"),audit.at("learner_competence_gate"),"learner gate");
    compare(comparison.at("cases"),audit.at("comparisons"),"complete case contrasts");
    std::set<std::string> case_ids,trial_ids;
    std::size_t primary_count=0,failed=0;
    for (std::size_t c=0;c<24;++c) {
        const auto& scene=report.at("cases").at(c); const auto& checked=audit.at("cases").at(c);
        const std::string case_id=scene.at("id");
        require(case_ids.insert(case_id).second && checked.at("case_id")==case_id,"ambiguous audit case identity");
        require(scene.at("trials").is_array() && scene.at("trials").size()==9 && checked.at("trials").size()==9,"each case needs nine declared trials");
        require(scene.at("primary").is_boolean() && scene.at("primary").get<bool>()==(number(scene.at("sun").at("position_m").at(2))<0),"primary case does not match lower lamp");
        if (scene.at("primary")==true) ++primary_count;
        for (std::size_t n=0;n<9;++n) {
            const auto& trial=scene.at("trials").at(n); const auto& audited=checked.at("trials").at(n);
            const std::string id=trial.at("id");
            require(!id.empty() && std::all_of(id.begin(),id.end(),[](char ch) { return (ch>='a'&&ch<='z') || (ch>='A'&&ch<='Z') || (ch>='0'&&ch<='9') || ch=='-' || ch=='_'; }),"unsafe trial filename");
            require(id==case_id+"--"+controller_id(n) && trial_ids.insert(id).second && audited.at("id")==id && trial.at("case_id")==case_id,"trial identity/order differs");
            require(trial.at("controller")==controller_spec(n) && audited.at("controller")==trial.at("controller"),"trial controller differs");
            require(trial.at("assembly")==scene.at("assembly") && trial.at("sun")==scene.at("sun"),"trial scene differs");
            require(trial.at("completed").is_boolean() && trial.at("safe").is_boolean() &&
                audited.at("record_consistent")==true && audited.at("completed")==trial.at("completed") && audited.at("safe_from_tape")==trial.at("safe"),"trial completion/safety differs from audit");
            compare(trial.at("metrics"),audited.at("recomputed_metrics"),id+" metrics");
            require(trial.at("late_window").at("start_step")==2400 && trial.at("late_window").at("end_step")==3200,"late window scope differs");
            compare(trial.at("late_window").at("metrics"),audited.at("recomputed_late_window"),id+" late window");
            require(trial.at("planned_steps")==3200 && trial.at("metrics").at("steps")==trial.at("completed_steps") &&
                near(number(trial.at("metrics").at("duration_s")),number(trial.at("completed_steps"))*kDt),"trial duration/horizon differs");
            require(trial.at("replay").at("exact_full_transition_tape")==true,"full native tape replay failed");
            if (trial.at("completed")==true && trial.at("safe")==true)
                require(trial.at("replay").at("exact_request_controller_replay")==true,"completed safe trial lacks exact controller replay");
            if (trial.at("safe")!=true || trial.at("replay").at("exact_request_controller_replay")!=true) ++failed;
            require(trial.at("snapshots").at("path")=="snapshots/"+id+".json" && trial.at("trace").at("path")=="traces/"+id+".jsonl","sidecar path differs from trial identity");
            (void)evidence_path(input,trial.at("trace").at("path"));
        }
    }
    require(primary_count==12 && trial_ids.size()==216 && report.at("failed_runs")==failed,"retained batch counts differ");
    require(report.at("source_sha256").is_object() && !report.at("source_sha256").empty() &&
        report.at("source_sha256")==report.at("source_sha256_after_collection"),"source archive identity changed during collection");
    require(droid::sha256_hex(report.at("manifest").dump())==report.at("manifest_sha256"),"native manifest checksum differs");
    for (const auto& [path,hash]:report.at("source_sha256").items()) hash_matches(evidence_path(input/"source",path),hash);
    require(report.at("source_sha256").at("docs/LIGHT_HEADROOM_EXPERIMENT.md")==report.at("protocol").at("sha256"),"protocol source identity differs");
}
J playback(const fs::path& input,const J& trial) {
    const fs::path source=evidence_path(input,trial.at("snapshots").at("path"));
    hash_matches(source,trial.at("snapshots").at("sha256"));
    const J snapshots=file_json(source,32*1024*1024);
    const auto steps=trial.at("metrics").at("steps").get<std::size_t>();
    require(snapshots.is_array() && !snapshots.empty() && snapshots.size()==1+steps/10 &&
        snapshots.size()==trial.at("snapshots").at("count"),"physical snapshot schedule/count differs");
    J frames=J::array();
    for (std::size_t n=0;n<snapshots.size();++n) {
        const auto& snapshot=snapshots.at(n); const auto& physics=snapshot.at("physics");
        require(snapshot.at("step")==n*10 && physics.is_object(),"physical snapshot order differs");
        const double time=n*10*kDt; const double reward=number(physics.at("reward").at("cumulative"));
        require(near(number(physics.at("elapsed_s")),time,1e-8) && physics.at("assembly")==trial.at("assembly") &&
            physics.at("sun")==trial.at("sun"),"physical snapshot clock/scene differs");
        frames.push_back(J{{"time_s",time},{"cumulative_reward",reward},{"physics",physics}});
    }
    if (trial.at("completed")==true) require(near(number(frames.back().at("time_s")),kDuration) &&
        near(number(frames.back().at("cumulative_reward")),number(trial.at("metrics").at("integrated_sensor_reward"))),"complete trial snapshot endpoint differs");
    // Re-read after parsing so an input changed during export cannot be published.
    hash_matches(source,trial.at("snapshots").at("sha256"));
    return J{{"schema","light_headroom_playback_v1"},{"id",trial.at("id")},{"source_snapshot_sha256",trial.at("snapshots").at("sha256")},
        {"native_dt_s",kDt},{"snapshot_interval_s",.2},{"frames",std::move(frames)}};
}
int export_directory(fs::path input,fs::path audit_path,fs::path web_root) {
    input=unlinked_absolute(input); audit_path=unlinked_absolute(audit_path); web_root=unlinked_absolute(web_root);
    require(fs::is_directory(input) && fs::is_directory(web_root),"batch and web root directories must exist");
    input=fs::canonical(input);
    const fs::path traces=web_root/"headroom-traces",index=web_root/"headroom-data.json",next=web_root/"headroom-data.next.json";
    require(!present(traces) && !present(index) && !present(next),"refusing existing headroom export destination; preserve it");
    const fs::path report_path=evidence_path(input,"report.json");
    const std::string report_hash=droid::sha256_file(report_path),audit_hash=droid::sha256_file(audit_path);
    const J report=file_json(report_path,16*1024*1024),audit=file_json(audit_path,32*1024*1024);
    validate_binding(input,report,audit);
    // Validate all snapshot assets before creating any destination; load only one
    // trial at a time, including when writing, to bound peak memory use.
    for (const auto& scene:report.at("cases")) for (const auto& trial:scene.at("trials")) (void)playback(input,trial);
    require(droid::sha256_file(report_path)==report_hash && droid::sha256_file(audit_path)==audit_hash,"report/audit changed during validation");
    require(!present(index) && !present(next) && fs::create_directory(traces),"refusing changed/existing destination");
    J view{{"schema","light_headroom_view_v1"},{"generated_by","tools/export-light-headroom.cpp"},
        {"cases",J::array()},{"controllers",controllers()},{"trials",J::array()},
        {"protocol",J{{"duration_s",kDuration},{"gain_threshold",kGain},{"native_dt_s",kDt},{"snapshot_interval_s",.2},
            {"learner_seeds",kSeeds},{"scope","Anchored two-link mechanisms; lower lamps are primary and upper lamps diagnostic. This experiment establishes no transfer or hardware result."},
            {"native_manifest",report.at("manifest")},{"physics_spec",report.at("physics_spec")},{"control_spec",report.at("control_spec")}}},
        {"summary",J{{"integrity_passed",true},{"headroom_gate_passed",audit.at("headroom_gate").at("passed")},
            {"learner_gate_passed",audit.at("learner_competence_gate").at("passed")},{"headroom_gate",audit.at("headroom_gate")},
            {"learner_competence_gate",audit.at("learner_competence_gate")},{"native_comparison",report.at("comparison")}}},
        {"provenance",J{{"report_path",report_path.generic_string()},{"report_sha256",report_hash},{"audit_path",audit_path.generic_string()},
            {"audit_sha256",audit_hash},{"independent_audit_integrity_passed",true},{"audit_scope",audit.at("scope")},
            {"protocol",report.at("protocol")},{"source_archive_files",report.at("source_sha256").size()},
            {"source_archive_manifest_sha256",droid::sha256_hex(report.at("source_sha256").dump())},
            {"snapshot_inputs",J::array()},{"exported_playbacks",J::array()}}}};
    for (const auto& scene:report.at("cases")) {
        view["cases"].push_back(J{{"id",scene.at("id")},{"title",scene_title(scene)},{"primary",scene.at("primary")},
            {"case",J{{"assembly",scene.at("assembly")},{"sun",scene.at("sun")}}}});
        for (std::size_t n=0;n<9;++n) {
            const auto& trial=scene.at("trials").at(n); const auto& metrics=trial.at("metrics");
            const std::string filename=trial.at("id").get<std::string>()+".json";
            const fs::path destination=traces/filename; const std::string url="/headroom-traces/"+filename;
            write_new(destination,playback(input,trial));
            view["trials"].push_back(J{{"id",trial.at("id")},{"case_id",scene.at("id")},{"controller_id",controller_id(n)},
                {"error",trial.at("error")},{"metrics",J{{"duration_s",metrics.at("duration_s")},{"native_reward",metrics.at("integrated_sensor_reward")},
                    {"energy_j",metrics.at("electrical_energy_j")},{"complete",trial.at("completed")},{"safe",trial.at("safe")},
                    {"safety_stops",metrics.at("native_veto_steps").get<std::uint64_t>()+metrics.at("flagged_steps").get<std::uint64_t>()},
                    {"native_veto_steps",metrics.at("native_veto_steps")},{"flagged_steps",metrics.at("flagged_steps")},
                    {"invalid_reward_sensor_samples",metrics.at("invalid_reward_sensor_samples")},{"invalid_feedback_samples",metrics.at("invalid_feedback_samples")}}},
                {"native_metrics",metrics},{"windows",trial.at("windows")},{"late_window",trial.at("late_window")},
                {"trace_url",url},{"replay",J{{"exact",trial.at("replay").at("exact_full_transition_tape")},
                    {"controller_exact",trial.at("replay").at("exact_request_controller_replay")},{"native",trial.at("replay")}}},
                {"source_trace",trial.at("trace")},{"source_snapshots",trial.at("snapshots")}});
            view["provenance"]["snapshot_inputs"].push_back(J{{"trial_id",trial.at("id")},{"path",trial.at("snapshots").at("path")},{"sha256",trial.at("snapshots").at("sha256")}});
            view["provenance"]["exported_playbacks"].push_back(J{{"trial_id",trial.at("id")},{"url",url},{"sha256",droid::sha256_file(destination)}});
        }
    }
    require(view.at("trials").size()==216 && droid::sha256_file(report_path)==report_hash &&
        droid::sha256_file(audit_path)==audit_hash,"source evidence changed while exporting; preserve unpublished partial export");
    write_new(next,view);
    // Atomic, no-overwrite publication. A competing index or unsupported rename
    // leaves the complete temporary index and all traces available for diagnosis.
    require(::syscall(SYS_renameat2,AT_FDCWD,next.c_str(),AT_FDCWD,index.c_str(),RENAME_NOREPLACE)==0,
        "cannot atomically publish new index; preserve headroom-data.next.json and traces");
    std::cout<<J{{"exported",true},{"cases",24},{"trials",216},{"index",index.generic_string()},
        {"index_sha256",droid::sha256_file(index)},{"headroom_gate_passed",view.at("summary").at("headroom_gate_passed")},
        {"learner_gate_passed",view.at("summary").at("learner_gate_passed")}}.dump()<<'\n';
    return 0;
}
}
int main(int argc,char** argv) {
    try {
        require(argc==7,"Usage: export-light-headroom --input BATCH_DIR --audit AUDIT_JSON --web-root EXISTING_WEB_DIRECTORY");
        fs::path input,audit,web;
        for (int i=1;i<argc;i+=2) {
            const std::string key=argv[i];
            if (key=="--input" && input.empty()) input=argv[i+1];
            else if (key=="--audit" && audit.empty()) audit=argv[i+1];
            else if (key=="--web-root" && web.empty()) web=argv[i+1];
            else throw std::runtime_error("invalid or duplicate exporter argument");
        }
        require(!input.empty() && !audit.empty() && !web.empty(),"missing exporter argument");
        return export_directory(input,audit,web);
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
