#include "humanoid/simulation.hpp"
#include "humanoid/goal_criteria.hpp"
#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
std::string sha256(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot hash " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx || !EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr)) throw std::runtime_error("SHA256 init failed");
  std::array<char, 65536> buffer{};
  while (input.read(buffer.data(), buffer.size()) || input.gcount())
    if (!EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(input.gcount())))
      throw std::runtime_error("SHA256 update failed");
  if (!input.eof()) throw std::runtime_error("Checkpoint read failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (!EVP_DigestFinal_ex(ctx.get(), digest.data(), &length)) throw std::runtime_error("SHA256 final failed");
  std::ostringstream output;
  for (unsigned int i = 0; i < length; ++i)
    output << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return output.str();
}

void save(const fs::path& output, const Json& report) {
  const fs::path temporary = output.string() + ".tmp";
  std::ofstream stream(temporary);
  stream << report.dump(2) << '\n';
  if (!stream) throw std::runtime_error("Cannot save evaluation");
  stream.close();
  fs::rename(temporary, output);
}
}

int main(int argc, char** argv) {
  fs::path owned_output;
  try {
    std::vector<fs::path> runs, policies;
    std::string device = "cuda", assets;
    fs::path output;
    bool screen = false;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout << "humanoid-goal-evaluate --run DIR [--run DIR --run DIR] --output NEW.json\n"
          "  Optional --policy FILE for each --run; otherwise DIR/policy.pt\n"
          "  --device cuda|cpu --assets DIR\n"
          "  --screen: one run, two 20s trials at 0.5m/s, development seeds only; cannot pass goal\n"
          "Full evaluation: three distinct training seeds, five held-out resets each,\n"
          "0.25/0.5/0.75m/s, 60s, >=90% overall and >=4/5 every group.\n";
        return 0;
      }
      if (key == "--screen") { screen = true; continue; }
      if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + key);
      const std::string value = argv[++i];
      if (key == "--run") runs.emplace_back(value);
      else if (key == "--policy") policies.emplace_back(value);
      else if (key == "--output") output = value;
      else if (key == "--device") device = value;
      else if (key == "--assets") assets = value;
      else throw std::invalid_argument("Unknown option " + key);
    }
    if (runs.size() != (screen ? 1u : 3u) || output.empty() || fs::exists(output))
      throw std::invalid_argument("Supply one screen run or three full runs and a new output path");
    if (!policies.empty() && policies.size() != runs.size())
      throw std::invalid_argument("Supply exactly one explicit policy per training run");
    if (device != "cuda" && device != "cpu") throw std::invalid_argument("Invalid device");
    if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
    const auto started = std::chrono::steady_clock::now();
    const double seconds = screen ? 20 : 60;
    const int trials = screen ? 2 : 5;
    const std::uint64_t reset_base = screen ? 8000000000ULL : 9000000000ULL;
    const std::vector<double> commands = screen ? std::vector<double>{.5} : std::vector<double>{.25, .5, .75};
    Json report = {{"schema_version", 1}, {"mode", screen ? "development_screen" : "goal_evaluation"},
      {"status", "running"}, {"goal_achieved", false}, {"quantitative_gate_pass", false},
      {"evaluator_source_on_disk_sha256", sha256(__FILE__)}, {"evaluator_executable_sha256", sha256("/proc/self/exe")},
      {"visual_gait_review_required", true}, {"automatic_resets", false},
      {"protocol", {{"seconds", seconds}, {"trials_per_training_seed_and_command", trials},
        {"commands_m_s", commands}, {"reset_seed_base", reset_base},
        {"reset_distribution", "fixed canonical pose plus uniform +/-0.005 rad leg-joint noise"},
        {"distance_fraction_min", .7}, {"touchdowns_per_foot_min", 5}, {"alternations_min", 8},
        {"airborne_fraction_max", .1}, {"overall_success_min", .9}, {"group_success_min", .8},
        {"command_tracking_review", "Separate from the approved numeric walking gate: mean speed 70-130% of command and RMSE <=max(0.15m/s, 30% of command)"},
        {"fall", "pelvis height <0.45m OR pelvis up-axis cosine <0.5"},
        {"selection", "Development screens use separate seeds; final seeds reserved before training"}}},
      {"candidates", Json::array()}, {"groups", Json::array()}, {"results", Json::array()}};
    save(output, report);
    owned_output = output;
    std::set<std::uint64_t> training_seeds;
    std::set<std::string> checkpoint_hashes;
    std::vector<int> group_counts;
    int total_passed = 0;
    for (std::size_t candidate = 0; candidate < runs.size(); ++candidate) {
      const fs::path run = fs::canonical(runs[candidate]);
      const fs::path manifest = run / "run.json";
      std::ifstream input(manifest);
      if (!input) throw std::runtime_error("Missing training manifest " + manifest.string());
      const Json training = Json::parse(input);
      const auto seed = training.at("seed").get<std::uint64_t>();
      if (!training_seeds.insert(seed).second) throw std::invalid_argument("Training seeds must be distinct");
      if (training.value("expert_assistance", true) || training.value("teacher_parameters_copied", true)
          || training.value("initialization", "") != "random; no pretrained weights")
        throw std::invalid_argument("Manifest must establish unassisted random initialization");
      if (training.value("completed_steps", std::int64_t(0)) <= 0)
        throw std::invalid_argument("Manifest records no training transitions");
      const fs::path checkpoint = fs::canonical(policies.empty() ? run / "policy.pt" : policies[candidate]);
      const auto hash = sha256(checkpoint), manifest_hash = sha256(manifest);
      Json binding;
      if (checkpoint == run / "policy.pt") {
        if (training.value("policy_sha256", "") != hash)
          throw std::invalid_argument("Policy hash does not match the training manifest");
        binding = {{"kind", "training manifest final export"}, {"sha256", hash}};
      } else {
        const auto relative = checkpoint.lexically_relative(run);
        if (relative.empty() || *relative.begin() != "checkpoints" || checkpoint.filename() != "policy.pt")
          throw std::invalid_argument("Explicit policy must be a checkpoint inside its training run");
        const fs::path metadata_path = checkpoint.parent_path() / "metadata.json";
        std::ifstream checkpoint_input(metadata_path);
        if (!checkpoint_input) throw std::invalid_argument("Missing checkpoint metadata");
        binding = Json::parse(checkpoint_input);
        if (binding.value("policy_sha256", "") != hash || binding.at("seed").get<std::uint64_t>() != seed
            || binding.value("completed_steps", std::int64_t(0)) <= 0
            || binding.at("completed_steps").get<std::int64_t>() > training.at("completed_steps").get<std::int64_t>())
          throw std::invalid_argument("Checkpoint metadata is not bound to this training run");
        binding["metadata_path"] = metadata_path.string();
        binding["metadata_sha256"] = sha256(metadata_path);
      }
      if (!checkpoint_hashes.insert(hash).second) throw std::invalid_argument("Cannot count one checkpoint as multiple trained policies");
      humanoid::WalkingSimulation sim(assets, device, 640, 426, false);
      sim.load_policy(checkpoint.string());
      report["candidates"].push_back({{"training_seed", seed}, {"training_run", run.string()},
        {"training_manifest_sha256", manifest_hash}, {"training_manifest", training},
        {"policy_path", checkpoint.string()}, {"policy_sha256", hash}, {"checkpoint_binding", binding},
        {"provenance", sim.provenance()}});
      for (const double command : commands) {
        int passed = 0;
        for (int reset = 0; reset < trials; ++reset) {
          if (sha256(checkpoint) != hash) throw std::runtime_error("Checkpoint changed during evaluation");
          auto state = sim.reset(reset_base + reset, command, "trained");
          while (state.at("time").get<double>() + 1e-8 < seconds && !sim.fallen()) state = sim.step();
          if (sha256(checkpoint) != hash) throw std::runtime_error("Checkpoint changed during trial");
          if (!state.contains("foot_slip") || !state.contains("speed_tracking_rmse"))
            throw std::runtime_error("Evaluation requires foot-slip and speed-error telemetry");
          const double speed_error = state.at("speed_tracking_rmse").get<double>();
          const auto& slip = state.at("foot_slip").at("combined");
          const double support_seconds = slip.at("support_seconds").get<double>();
          if (!std::isfinite(speed_error) || speed_error < 0 || !std::isfinite(support_seconds)
              || (support_seconds > 0 && (!std::isfinite(slip.at("mean_speed").get<double>())
              || !std::isfinite(slip.at("rms_speed").get<double>()))))
            throw std::runtime_error("Invalid tracking/slip measurements");
          const auto reasons = humanoid::goal::failures(state, seconds, command);
          const bool success = reasons.empty();
          passed += success;
          state["training_seed"] = seed;
          state["candidate_index"] = candidate;
          state["policy_sha256"] = hash;
          state["walking_criteria_pass"] = success;
          state["mean_speed_to_command_ratio"] = state.at("mean_speed").get<double>() / command;
          state["command_tracking_review_pass"] = success
            && state["mean_speed_to_command_ratio"].get<double>() <= 1.3
            && speed_error <= std::max(.15, .3 * command);
          state["failure_reasons"] = reasons;
          report["results"].push_back(state);
          std::cout << Json({{"training_seed", seed}, {"reset_seed", reset_base + reset},
            {"command_x", command}, {"time", state["time"]}, {"distance", state["distance"]},
            {"alternations", state["alternating_count"]}, {"fall", state["fall"]},
            {"pass", success}, {"failure_reasons", reasons}}).dump() << std::endl;
          save(output, report);
        }
        group_counts.push_back(passed);
        total_passed += passed;
        report["groups"].push_back({{"training_seed", seed}, {"command_x", command},
          {"passed", passed}, {"trials", trials}});
      }
      if (binding.contains("metadata_path")
          && sha256(binding.at("metadata_path").get<std::string>()) != binding.at("metadata_sha256").get<std::string>())
        throw std::runtime_error("Checkpoint metadata changed during evaluation");
      const auto end_manifest_hash = sha256(manifest);
      report["candidates"][candidate]["training_manifest_end_sha256"] = end_manifest_hash;
      report["candidates"][candidate]["live_training_manifest_advanced"] = end_manifest_hash != manifest_hash;
      if (!screen && end_manifest_hash != manifest_hash)
        throw std::runtime_error("Training manifest changed during final evaluation; use a completed or frozen run");
    }
    report["status"] = "completed";
    report["summary"] = {{"passed", total_passed}, {"trials", report["results"].size()},
      {"training_seeds", training_seeds}, {"quantitative_gate_pass", !screen && humanoid::goal::aggregate_pass(group_counts, trials)}};
    report["quantitative_gate_pass"] = report["summary"]["quantitative_gate_pass"];
    report["interpretation"] = "Numeric acceptance is separate from required visual gait/slip review and verified training provenance; this report alone never marks the goal achieved.";
    report["wall_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    save(output, report);
    std::cout << report["summary"].dump() << std::endl;
    return 0;
  } catch (const std::exception& error) {
    if (!owned_output.empty()) {
      try {
        std::ifstream input(owned_output);
        auto failed = Json::parse(input);
        failed["status"] = "failed";
        failed["error"] = error.what();
        failed["goal_achieved"] = false;
        failed["quantitative_gate_pass"] = false;
        save(owned_output, failed);
      } catch (...) { /* Preserve the original failure and any partial report. */ }
    }
    std::cerr << "Goal evaluation failed: " << error.what() << '\n';
    return 1;
  }
}
