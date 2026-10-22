#include "walk_policy.hpp"
#include <torch/version.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using humanoid::walk::ActorCritic;
namespace {
struct Options { fs::path left, right, output; };
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help") {
      std::cout << "CPU numerical checkpoint comparison; no optimization or simulation.\n"
        "--left CHECKPOINT_DIRECTORY --right CHECKPOINT_DIRECTORY --output NEW_REPORT.json\n"
        "Exit 0: exact match; 2: numerical/mapping mismatch; 1: load or reporting error.\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("Missing value for " + flag);
    if (flag == "--left") o.left = argv[i];
    else if (flag == "--right") o.right = argv[i];
    else if (flag == "--output") o.output = argv[i];
    else throw std::invalid_argument("Unknown option " + flag);
  }
  if (o.left.empty() || o.right.empty() || o.output.empty()) throw std::invalid_argument("--left, --right and --output are required");
  o.left = fs::absolute(o.left); o.right = fs::absolute(o.right); o.output = fs::absolute(o.output);
  if (!fs::is_directory(o.left) || !fs::is_directory(o.right)) throw std::invalid_argument("Both checkpoints must be existing directories");
  if (fs::exists(o.output)) throw std::invalid_argument("Output report must be new");
  return o;
}
void write_json(const fs::path& path, const Json& value) {
  if (fs::exists(path)) throw std::runtime_error("Refusing to overwrite output report");
  fs::create_directories(path.parent_path());
  std::ofstream output(path); output.exceptions(std::ios::badbit | std::ios::failbit);
  output << value.dump(2) << '\n';
}
std::string sha256(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot hash " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("SHA256 initialization failed");
  std::array<char,65536> bytes{};
  while (input.read(bytes.data(), bytes.size()) || input.gcount())
    if (EVP_DigestUpdate(context.get(), bytes.data(), static_cast<size_t>(input.gcount())) != 1) throw std::runtime_error("SHA256 update failed");
  if (!input.eof()) throw std::runtime_error("SHA256 read failed");
  std::array<unsigned char,EVP_MAX_MD_SIZE> digest{}; unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) throw std::runtime_error("SHA256 final failed");
  std::ostringstream result;
  for (unsigned i = 0; i < length; ++i) result << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return result.str();
}
Json describe(const torch::Tensor& tensor) {
  Json value = {{"defined", tensor.defined()}};
  if (tensor.defined()) {
    value["shape"] = tensor.sizes().vec(); value["dtype"] = c10::toString(tensor.scalar_type());
    value["device"] = tensor.device().str(); value["layout"] = static_cast<int>(tensor.layout());
    value["elements"] = tensor.numel();
  }
  return value;
}
Json compare_tensor(const torch::Tensor& a, const torch::Tensor& b) {
  Json result = {{"left", describe(a)}, {"right", describe(b)}};
  bool equal = a.defined() == b.defined();
  if (a.defined() && b.defined()) {
    const bool compatible = a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type() && a.layout() == b.layout();
    equal = compatible && torch::equal(a, b);
    if (compatible) {
      result["mismatched_elements"] = torch::ne(a, b).sum().item<int64_t>();
      if (a.numel() && a.is_floating_point()) {
        const auto difference = (a.to(torch::kFloat64) - b.to(torch::kFloat64)).abs();
        result["max_abs_difference"] = torch::isfinite(difference).all().item<bool>() ? Json(difference.max().item<double>()) : Json(nullptr);
        result["left_all_finite"] = torch::isfinite(a).all().item<bool>();
        result["right_all_finite"] = torch::isfinite(b).all().item<bool>();
      }
    }
  }
  result["exact_match"] = equal;
  return result;
}
struct Snapshot {
  ActorCritic model;
  std::unique_ptr<torch::optim::Adam> optimizer;
  std::map<std::string,torch::Tensor> parameters;
  std::unordered_map<void*,std::string> names_by_impl;
  Json group_mappings = Json::array();
  bool mapping_valid = true;
  size_t unmapped_state_keys = 0;
  explicit Snapshot(const fs::path& directory) {
    model->to(torch::kCPU);
    // Same constructor and normal recursive parameter order as humanoid-walk-train.
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(.0003).eps(1e-5));
    torch::load(model, (directory / "model.pt").string(), torch::Device(torch::kCPU));
    torch::serialize::InputArchive archive;
    archive.load_from((directory / "optimizer.pt").string(), torch::Device(torch::kCPU));
    optimizer->load(archive);
    for (const auto& parameter : model->named_parameters()) {
      parameters.emplace(parameter.key(), parameter.value());
      if (!names_by_impl.emplace(parameter.value().unsafeGetTensorImpl(), parameter.key()).second) mapping_valid = false;
    }
    std::set<std::string> seen;
    for (const auto& group : optimizer->param_groups()) {
      Json mapping = Json::array();
      for (const auto& parameter : group.params()) {
        const auto found = names_by_impl.find(parameter.unsafeGetTensorImpl());
        if (found == names_by_impl.end()) { mapping.push_back(nullptr); mapping_valid = false; }
        else { mapping.push_back(found->second); if (!seen.insert(found->second).second) mapping_valid = false; }
      }
      group_mappings.push_back(mapping);
    }
    if (seen.size() != parameters.size()) mapping_valid = false;
    for (const auto& entry : optimizer->state()) if (!names_by_impl.count(entry.first)) ++unmapped_state_keys;
  }
};
Json adam_options(const torch::optim::OptimizerParamGroup& group) {
  const auto* options = dynamic_cast<const torch::optim::AdamOptions*>(&group.options());
  if (!options) return {{"type", "unexpected optimizer options"}};
  const auto [beta1,beta2] = options->betas();
  return {{"type", "AdamOptions"}, {"lr", options->lr()}, {"beta1", beta1}, {"beta2", beta2},
    {"eps", options->eps()}, {"weight_decay", options->weight_decay()}, {"amsgrad", options->amsgrad()},
    {"all_finite", std::isfinite(options->lr()) && std::isfinite(beta1) && std::isfinite(beta2) &&
                   std::isfinite(options->eps()) && std::isfinite(options->weight_decay())}};
}
Json compare_state(Snapshot& left, Snapshot& right, const std::string& name) {
  const auto lp = left.parameters.find(name), rp = right.parameters.find(name);
  if (lp == left.parameters.end() || rp == right.parameters.end()) return {{"exact_match",false},{"error","Missing model parameter"}};
  const auto ls = left.optimizer->state().find(lp->second.unsafeGetTensorImpl());
  const auto rs = right.optimizer->state().find(rp->second.unsafeGetTensorImpl());
  const bool has_left = ls != left.optimizer->state().end(), has_right = rs != right.optimizer->state().end();
  Json result = {{"present_left",has_left},{"present_right",has_right}};
  if (!has_left || !has_right) { result["exact_match"] = has_left == has_right; return result; }
  const auto* a = dynamic_cast<const torch::optim::AdamParamState*>(ls->second.get());
  const auto* b = dynamic_cast<const torch::optim::AdamParamState*>(rs->second.get());
  if (!a || !b) return {{"exact_match",false},{"error","Unexpected non-Adam parameter state"}};
  result["step"] = {{"left",a->step()},{"right",b->step()},{"exact_match",a->step()==b->step()}};
  result["exp_avg"] = compare_tensor(a->exp_avg(),b->exp_avg());
  result["exp_avg_sq"] = compare_tensor(a->exp_avg_sq(),b->exp_avg_sq());
  result["max_exp_avg_sq"] = compare_tensor(a->max_exp_avg_sq(),b->max_exp_avg_sq());
  result["exact_match"] = a->step()==b->step() && result["exp_avg"]["exact_match"].get<bool>() &&
    result["exp_avg_sq"]["exact_match"].get<bool>() && result["max_exp_avg_sq"]["exact_match"].get<bool>();
  return result;
}
Json provenance(const fs::path& directory) {
  Json result = {{"path",directory.string()},{"model_file_sha256",sha256(directory/"model.pt")},
    {"optimizer_file_sha256",sha256(directory/"optimizer.pt")}};
  if (fs::exists(directory/"metadata.json")) {
    std::ifstream input(directory/"metadata.json"); Json meta; input >> meta;
    result["metadata_sha256"] = sha256(directory/"metadata.json");
    for (const auto* key : {"seed","completed_steps","optimizer_updates","reward_version","learning_rate"})
      if (meta.contains(key)) result["metadata"][key] = meta[key];
  }
  return result;
}
int compare(const Options& o) {
  torch::set_num_threads(1); torch::manual_seed(0); torch::NoGradGuard guard;
  Snapshot left(o.left), right(o.right);
  Json report = {{"schema",1},{"libtorch",TORCH_VERSION},{"device","cpu"},{"training_performed",false},
    {"comparison","Exact tensor element equality with matching shape/dtype/layout; no tolerance; signed zero compares equal; NaNs do not match"},
    {"left",provenance(o.left)},{"right",provenance(o.right)}, {"parameters",Json::array()}, {"parameter_groups",Json::array()}};
  std::set<std::string> names;
  for (const auto& entry : left.parameters) names.insert(entry.first);
  for (const auto& entry : right.parameters) names.insert(entry.first);
  bool models_match = true, states_match = true, groups_match = true;
  size_t model_matches = 0, state_matches = 0;
  for (const auto& name : names) {
    const auto a = left.parameters.find(name), b = right.parameters.find(name);
    const auto model = compare_tensor(a == left.parameters.end() ? torch::Tensor{} : a->second,
                                      b == right.parameters.end() ? torch::Tensor{} : b->second);
    const auto state = compare_state(left,right,name);
    const bool model_match = model["exact_match"].get<bool>(), state_match = state["exact_match"].get<bool>();
    models_match &= model_match; states_match &= state_match; model_matches += model_match; state_matches += state_match;
    report["parameters"].push_back({{"name",name},{"model",model},{"adam_state",state},{"exact_match",model_match&&state_match}});
  }
  const auto left_groups = left.optimizer->param_groups().size(), right_groups = right.optimizer->param_groups().size();
  for (size_t i = 0; i < std::max(left_groups,right_groups); ++i) {
    const auto a = i < left_groups ? adam_options(left.optimizer->param_groups()[i]) : Json(nullptr);
    const auto b = i < right_groups ? adam_options(right.optimizer->param_groups()[i]) : Json(nullptr);
    const auto am = i < left_groups ? left.group_mappings[i] : Json(nullptr);
    const auto bm = i < right_groups ? right.group_mappings[i] : Json(nullptr);
    const bool options_match = !a.is_null() && !b.is_null() && a == b && a.value("all_finite",false);
    const bool mapping_match = !am.is_null() && !bm.is_null() && am == bm;
    groups_match &= options_match && mapping_match;
    report["parameter_groups"].push_back({{"index",i},{"left_options",a},{"right_options",b},{"options_exact_match",options_match},
      {"left_parameter_names",am},{"right_parameter_names",bm},{"ordered_mapping_exact_match",mapping_match}});
  }
  const bool mapping_valid = left.mapping_valid && right.mapping_valid && !left.unmapped_state_keys && !right.unmapped_state_keys;
  const bool optimizer_match = states_match && groups_match && mapping_valid;
  const bool all_match = models_match && optimizer_match;
  report["summary"] = {{"exact_match",all_match},{"model_parameters_match",models_match},{"optimizer_match",optimizer_match},
    {"adam_parameter_states_match",states_match},{"parameter_groups_match",groups_match},{"parameter_mapping_valid",mapping_valid},
    {"model_parameter_count_left",left.parameters.size()},{"model_parameter_count_right",right.parameters.size()},
    {"model_parameters_matching",model_matches},{"adam_states_matching",state_matches},
    {"optimizer_state_count_left",left.optimizer->state().size()},{"optimizer_state_count_right",right.optimizer->state().size()},
    {"unmapped_optimizer_state_keys_left",left.unmapped_state_keys},{"unmapped_optimizer_state_keys_right",right.unmapped_state_keys}};
  report["status"] = all_match ? "match" : "mismatch";
  report["interpretation"] = "Raw serialization hashes are recorded but do not determine numerical equality; Adam state is mapped to current named model parameters after normal-order restoration.";
  report["comparator_source_sha256"] = sha256(__FILE__);
  report["actor_header_sha256"] = sha256(fs::path(__FILE__).parent_path()/"walk_policy.hpp");
  if (fs::exists("/proc/self/exe")) report["comparator_executable_sha256"] = sha256("/proc/self/exe");
  write_json(o.output,report); std::cout << report["summary"].dump(2) << '\n';
  return all_match ? 0 : 2;
}
}  // namespace
int main(int argc, char** argv) {
  fs::path output;
  try { const auto options=parse(argc,argv); output=options.output; return compare(options); }
  catch (const std::exception& error) {
    std::cerr << "Checkpoint comparison failed: " << error.what() << '\n';
    if (!output.empty() && !fs::exists(output)) {
      try { write_json(output,{{"status","error"},{"error",error.what()},{"exact_match",false}}); } catch (...) {}
    }
    return 1;
  }
}
