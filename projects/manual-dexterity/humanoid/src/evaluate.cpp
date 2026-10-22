#include "humanoid/simulation.hpp"

#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
nlohmann::json benchmark_render(humanoid::WalkingSimulation& sim, int count,
                                const std::string& frame_output) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  nlohmann::json frames = nlohmann::json::array();
  std::vector<unsigned char> jpeg;
  double cold_ms = 0, warm_sum_ms = 0;
  for (int frame = 0; frame < count; ++frame) {
    sim.step();
    const auto before_render = Clock::now();
    jpeg = sim.render_jpeg(82);
    const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - before_render).count();
    if (frame == 0) cold_ms = wall_ms;
    else warm_sum_ms += wall_ms;
    const auto state = sim.state();
    const nlohmann::json measured = {
      {"frame", frame}, {"time", state["time"]}, {"render_frame_ms", wall_ms},
      {"jpeg_bytes", jpeg.size()}, {"timing_ms", state["timing_ms"]}
    };
    frames.push_back(measured);
    std::cout << measured.dump() << std::endl;
  }
  if (!frame_output.empty()) {
    const std::filesystem::path path = frame_output;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream image(path, std::ios::binary);
    image.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(jpeg.size()));
    if (!image) throw std::runtime_error("Cannot write benchmark JPEG");
  }
  return {
    {"schema_version", 1}, {"mode", "render_benchmark"}, {"provenance", sim.provenance()},
    {"summary", {{"frames", count}, {"cold_frame_ms", cold_ms},
      {"warm_mean_frame_ms", count > 1 ? nlohmann::json(warm_sum_ms / (count - 1)) : nlohmann::json(nullptr)},
      {"width", 640}, {"height", 426}}},
    {"frames", frames}, {"frame_output", frame_output},
    {"wall_seconds", std::chrono::duration<double>(Clock::now() - started).count()},
    {"timing_scope", "Native CPU OSMesa drawing/readback and JPEG encoding; concurrent system workloads can affect timing"}
  };
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string root, device = "cuda", output = "runs/evaluation.json", policy_path, frame_output;
    std::size_t trials = 5;
    int render_frames = 0;
    bool output_given = false;
    double seconds = 20, command = .5;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--help") {
        std::cout << "humanoid-evaluate [--asset-root PATH] [--device cuda|cpu] [--output FILE] "
                     "[--trials N>=5] [--seconds S>=20] [--command-x SPEED] [--policy-path FILE]\n"
                     "Render benchmark instead: --render-frames N (1..120) [--frame-output FILE.jpg]\n";
        return 0;
      }
      if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
      const std::string value = argv[++i];
      if (flag == "--asset-root") root = value;
      else if (flag == "--device") device = value;
      else if (flag == "--output") { output = value; output_given = true; }
      else if (flag == "--policy-path") policy_path = value;
      else if (flag == "--frame-output") frame_output = value;
      else if (flag == "--render-frames") {
        render_frames = std::stoi(value);
        if (render_frames < 1 || render_frames > 120)
          throw std::invalid_argument("Render benchmark requires 1..120 frames");
      }
      else if (flag == "--trials") trials = std::stoull(value);
      else if (flag == "--seconds") seconds = std::stod(value);
      else if (flag == "--command-x") command = std::stod(value);
      else throw std::invalid_argument("Unknown argument: " + flag);
    }
    if (!frame_output.empty() && render_frames == 0)
      throw std::invalid_argument("--frame-output requires --render-frames");
    if (render_frames && !output_given) output = "runs/render-benchmark.json";
    humanoid::WalkingSimulation sim(root, device, render_frames ? 640 : 960,
                                    render_frames ? 426 : 640, policy_path.empty());
    nlohmann::json report;
    if (render_frames) {
      if (!policy_path.empty()) sim.load_policy(policy_path);
      sim.reset(0, command, policy_path.empty() ? "pretrained" : "trained");
      report = benchmark_render(sim, render_frames, frame_output);
    } else {
      report = humanoid::evaluate(sim, trials, seconds, command, policy_path);
    }
    const std::filesystem::path path = output, temporary = output + ".tmp";
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    {
      std::ofstream stream(temporary);
      stream << report.dump(2) << '\n';
      if (!stream) throw std::runtime_error("Cannot write evaluation report");
    }
    std::filesystem::rename(temporary, path);
    std::cout << nlohmann::json({{"output", output}, {"summary", report["summary"]}}).dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Evaluation failed: " << error.what() << '\n';
    return 1;
  }
}
