#include "humanoid/simulation.hpp"
#include <ATen/Parallel.h>
#include <torch/cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
void write_json(const fs::path& path, const Json& value) {
  std::ofstream out(path); out.exceptions(std::ios::failbit | std::ios::badbit);
  out << value.dump(2) << '\n';
}
void preview(const fs::path& path, const Json& frames) {
  Json player = Json::array();
  for (const auto& frame : frames) {
    const auto& state = frame.at("state");
    player.push_back({{"image", frame.at("image")}, {"time", frame.at("nominal_seconds")},
      {"distance", state.at("distance")}, {"speed", state.at("speed")},
      {"left", state.at("left_steps")}, {"right", state.at("right_steps")},
      {"alternations", state.at("alternating_count")}, {"fall", state.at("fall")}});
  }
  std::ofstream out(path); out.exceptions(std::ios::failbit | std::ios::badbit);
  out << R"HTML(<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Offline humanoid replay</title>
<style>body{margin:24px auto;padding:0 16px;max-width:850px;background:#101723;color:#e5edf8;font:16px system-ui}h1{font-size:24px}img{display:block;width:100%;max-width:640px;background:#000}button,input{margin:14px 8px 14px 0}input{width:70%}a{color:#a7ceff}#stats{line-height:1.6;font-variant-numeric:tabular-nums}</style>
<h1>Offline checkpoint replay</h1><p>Actual MuJoCo frames · 0.5 m/s command · CUDA inference · 10 frames/s</p>
<img id="frame" alt="Recorded MuJoCo humanoid simulation" width="640" height="426">
<button id="play" type="button">Play</button><input id="seek" type="range" min="0" value="0" aria-label="Recorded frame">
<div id="stats"></div><p><a href="report.json">Final measurements and provenance</a> · <a href="frames.json">Per-frame states</a></p>
<p>This page plays saved images. It does not run or control a live simulation.</p>
<script>
const frames=)HTML" << player.dump() << R"HTML(;
let index=0,playing=false,timer;
const picture=document.getElementById('frame'),seek=document.getElementById('seek'),button=document.getElementById('play'),stats=document.getElementById('stats');
seek.max=frames.length-1;
function show(){const f=frames[index];picture.src=f.image;seek.value=index;stats.textContent=`${f.time.toFixed(2)} s · Forward distance ${f.distance.toFixed(3)} m · Speed ${f.speed.toFixed(3)} m/s · Touchdowns L/R ${f.left}/${f.right} · Alternations ${f.alternations} · ${f.fall?'Fallen':'Upright'} · Frame ${index+1}/${frames.length}`;}
function pause(){playing=false;clearTimeout(timer);button.textContent='Play';}
function next(){if(!playing)return;if(index+1>=frames.length){pause();return;}timer=setTimeout(()=>{index++;show();next();},1000*(frames[index+1].time-frames[index].time));}
button.onclick=()=>{if(playing){pause();return;}if(index+1>=frames.length){index=0;show();}playing=true;button.textContent='Pause';next();};
seek.oninput=()=>{pause();index=Number(seek.value);show();};show();
</script></html>
)HTML";
}
}  // namespace

int main(int argc, char** argv) {
  fs::path output, policy;
  uint64_t seed = 8000000000ULL;
  bool owns_output = false;
  Json report = Json::object();
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "humanoid-render-proof --policy FILE --output NEW_DIR [--seed 8000000000]\n"
                     "Fixed20s at0.5m/s, CUDA inference,640x426 JPEGs at10Hz; stops on fall without reset.\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("Missing value for " + key);
      const std::string value = argv[i];
      if (key == "--policy") policy = value;
      else if (key == "--output") output = value;
      else if (key == "--seed") {
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }))
          throw std::invalid_argument("Seed must be an unsigned decimal integer");
        seed = std::stoull(value);
      } else throw std::invalid_argument("Unknown argument " + key);
    }
    if (policy.empty() || output.empty() || !fs::is_regular_file(policy))
      throw std::invalid_argument("An existing --policy FILE and --output NEW_DIR are required");
    policy = fs::canonical(policy); output = fs::absolute(output);
    if (fs::exists(output)) throw std::invalid_argument("Output directory must be new");
    at::set_num_threads(1); at::set_num_interop_threads(1);
    if (!torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable; no fallback");
    fs::create_directories(output.parent_path());
    if (!fs::create_directory(output)) throw std::runtime_error("Cannot create new output directory");
    owns_output = true; fs::create_directory(output / "frames");
    report = {{"schema", 1}, {"status", "rendering"}, {"mode", "offline_saved_checkpoint_replay"},
      {"policy_path", policy.string()}, {"seed", seed}, {"command_x", .5}, {"requested_seconds", 20},
      {"inference_device", "cuda"}, {"physics_device", "cpu"}, {"render_backend", "OSMesa CPU"},
      {"width", 640}, {"height", 426}, {"saved_frame_hz", 10}, {"automatic_resets", false},
      {"training_performed", false}, {"live_viewer_modified", false}};
    write_json(output / "report.json", report);
    const auto started = std::chrono::steady_clock::now();
    humanoid::WalkingSimulation sim("", "cuda", 640, 426, false);
    if (std::abs(sim.control_dt() - .02) > 1e-12) throw std::runtime_error("Expected50Hz controller");
    sim.load_policy(policy.string()); sim.reset(seed, .5, "trained");
    report["provenance"] = sim.provenance();
    report["policy_sha256"] = report["provenance"].at("active_policy_sha256");
    Json frames = Json::array();
    auto save_frame = [&](int interval) {
      std::ostringstream name; name << "frames/frame-" << std::setfill('0') << std::setw(4) << interval << ".jpg";
      const auto bytes = sim.render_jpeg(88);
      std::ofstream image(output / name.str(), std::ios::binary); image.exceptions(std::ios::failbit | std::ios::badbit);
      image.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      frames.push_back({{"image", name.str()}, {"control_interval", interval},
        {"nominal_seconds", interval * sim.control_dt()}, {"state", sim.state()}});
    };
    save_frame(0);
    int intervals = 0;
    while (intervals < 1000 && !sim.fallen()) {
      sim.step(); ++intervals;
      if (intervals % 5 == 0 || sim.fallen()) save_frame(intervals);
    }
    report["final_state"] = sim.state();
    report["completed_control_intervals"] = intervals;
    report["nominal_seconds"] = intervals * sim.control_dt();
    report["completed_twenty_seconds_without_fall"] = intervals == 1000 && !sim.fallen();
    report["frame_count"] = frames.size();
    report["frame_sampling"] = "Initial state and every fifth20ms control endpoint; an off-grid terminal fall is additionally saved. Actual simulation advances every interval, with no resets.";
    report["evidence_scope"] = "One continuous deterministic checkpoint replay; saved images and physical measurements permit visual assessment. Surviving20s alone does not establish walking.";
    if (sim.provenance().at("active_policy_sha256") != report["policy_sha256"])
      throw std::runtime_error("Policy file changed during replay");
    write_json(output / "frames.json", frames);
    preview(output / "preview.html", frames);
    report["wall_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    report["status"] = "complete";
    write_json(output / "report.json", report);
    std::cout << Json({{"output", output.string()}, {"frames", frames.size()}, {"final_state", report["final_state"]}}).dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Offline replay failed: " << error.what() << '\n';
    if (owns_output) try { report["status"] = "failed"; report["error"] = error.what(); write_json(output / "report.json", report); } catch (...) {}
    return 1;
  }
}
