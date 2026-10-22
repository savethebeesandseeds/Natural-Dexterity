#include "droid/pretrained_policy.hpp"
#include "droid/policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace droid {
namespace {
using Json = nlohmann::json;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
constexpr std::string_view kActorTensorSha{
    "f163453766fe2ca7f7bab75a0093238bbcfce682aaa8bf11529ddc1673d0454c"};
constexpr std::string_view kActorBytesSha{
    "73db0a5e562fe052c81894844fac08bdc35676da356b8acefd7e3602f4bb5e2b"};
constexpr std::string_view kProtocolSha{
    "4cf6fe175dfad1150c9264e2ab074f8df7fe575a978724bc7d3ffddb16915727"};
constexpr std::string_view kFullPolicySha{
    "30cb3f0efad3b0a6755cecafac73366aef1b925e2ee30f62b1bab989a04023dc"};
constexpr std::string_view kExporterSha{
    "2d2cd9e73cf29ec5f14f51608703347258664db39449ca806d761aac8345fa4b"};

void require(bool condition, std::string_view message) {
    if (!condition) throw std::invalid_argument(std::string(message));
}

void exact_fields(const Json& value, const std::set<std::string>& expected,
                  std::string_view context) {
    require(value.is_object(), std::string(context) + " must be an object");
    std::set<std::string> actual;
    for (const auto& [key, unused] : value.items()) {
        (void)unused;
        actual.insert(key);
    }
    require(actual == expected, std::string(context) + " contains unexpected or missing fields");
}

double finite_number(const Json& value, std::string_view context) {
    require(value.is_number(), std::string(context) + " must be a number");
    const double result = value.get<double>();
    require(std::isfinite(result), std::string(context) + " must be finite");
    return result;
}

void append_float_bytes(std::string& output, float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<char>((bits >> shift) & 0xffU));
}

std::string float_bytes(const std::vector<float>& values) {
    std::string result;
    result.reserve(values.size() * sizeof(float));
    for (float value : values) append_float_bytes(result, value);
    return result;
}

std::string tuple_shape(const std::vector<std::size_t>& shape) {
    std::string value{"("};
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index) value += ", ";
        value += std::to_string(shape[index]);
    }
    if (shape.size() == 1) value += ',';
    value += ')';
    return value;
}

void check_reference(const Json& provenance, std::string_view key,
                     std::string_view path, std::string_view hash) {
    const Json& ref = provenance.at(std::string(key));
    exact_fields(ref, {"path", "sha256"}, "pretrained provenance reference");
    require(ref.at("path") == path && ref.at("sha256") == hash,
            std::string("Pretrained trusted ") + std::string(key) + " provenance differs");
}

void validate_provenance(const Json& value) {
    exact_fields(value, {"selection_rule", "source_arm", "source_seed", "source_decisions",
                 "protocol", "source_seal", "source_result", "checkpoint", "native_audit",
                 "tensor_audit", "full_policy_parameter_sha256", "exporter_source_sha256"},
                 "pretrained provenance");
    require(value.at("selection_rule") == "first declared source seed, pooled arm; no target-score selection"
            && value.at("source_arm") == "pooled" && value.at("source_seed") == 101
            && value.at("source_decisions") == 98304, "Pretrained source selection differs");
    require(value.at("full_policy_parameter_sha256") == kFullPolicySha
            && value.at("exporter_source_sha256") == kExporterSha, "Pretrained tensor/exporter provenance differs");
    check_reference(value, "protocol", "artifacts/reports/generalization-study-v1-20261006/protocol.json", kProtocolSha);
    check_reference(value, "source_seal", "artifacts/reports/generalization-study-v1-20261006/sources/sealed-checkpoints.json",
                    "6404315c901c19972c7b190f797026431fc88ecd9006c6d0ebaa740feed18508");
    check_reference(value, "source_result", "artifacts/reports/generalization-study-v1-20261006/sources/source-pooled-101/result.json",
                    "b7239ade4c0853cfa4ddb4c798671ce33803669616e242b2d879d2827268880b");
    check_reference(value, "checkpoint", "artifacts/reports/generalization-study-v1-20261006/sources/source-pooled-101/policy.zip",
                    "4c7ce76f644dd07d41340ef311d5f797578fa58f0470e2b89aa49c0d3b93082c");
    check_reference(value, "native_audit", "artifacts/reports/generalization-study-v1-20261006/verification/native-replay-v1.json",
                    "92839768519ec6c3df1a7dc56f9ca25adcf563375886f26e3057fe679a19bf20");
    check_reference(value, "tensor_audit", "artifacts/reports/generalization-study-v1-20261006/verification/tensors-v1.json",
                    "35cbaf63da3a84ade402efc486573fb2b0dde520865b04407ad02faab14e5544");
}

struct Tensor {
    std::vector<std::size_t> shape;
    std::vector<float> data;
};
using TensorMap = std::map<std::string, Tensor, std::less<>>;

TensorMap read_tensors(const Json& document) {
    const std::map<std::string, std::vector<std::size_t>> shapes{
        {"action_net.bias", {3}}, {"action_net.weight", {3, 64}},
        {"mlp_extractor.policy_net.0.bias", {64}},
        {"mlp_extractor.policy_net.0.weight", {64, 204}},
        {"mlp_extractor.policy_net.2.bias", {64}},
        {"mlp_extractor.policy_net.2.weight", {64, 64}}};
    const Json& tensors = document.at("tensors");
    require(tensors.is_array() && tensors.size() == shapes.size(), "Expected exactly six pretrained actor tensors");
    TensorMap result;
    std::string raw, commitment, previous_name;
    std::size_t parameter_count = 0;
    for (const auto& item : tensors) {
        exact_fields(item, {"name", "shape", "data", "sha256"}, "pretrained tensor");
        require(item.at("name").is_string(), "Pretrained tensor name must be a string");
        const auto name = item.at("name").get<std::string>();
        require(shapes.contains(name) && (previous_name.empty() || previous_name < name),
                "Pretrained tensors must have unique sorted original names");
        previous_name = name;
        const auto& expected_shape = shapes.at(name);
        const Json& shape = item.at("shape");
        require(shape.is_array() && shape.size() == expected_shape.size(), "Pretrained tensor dimensions differ");
        std::size_t size = 1;
        for (std::size_t index = 0; index < shape.size(); ++index) {
            require(shape[index].is_number_integer() && shape[index].get<std::int64_t>() ==
                    static_cast<std::int64_t>(expected_shape[index]), "Pretrained tensor shape differs");
            size *= expected_shape[index];
        }
        const Json& data = item.at("data");
        require(data.is_array() && data.size() == size, "Pretrained tensor data size differs");
        std::vector<float> values;
        values.reserve(size);
        for (const auto& entry : data) {
            const double number = finite_number(entry, "pretrained weight");
            const float value = static_cast<float>(number);
            require(std::isfinite(value) && static_cast<double>(value) == number,
                    "Pretrained weight is not a losslessly represented float32 value");
            values.push_back(value);
        }
        const std::string bytes = float_bytes(values);
        require(item.at("sha256") == sha256_hex(bytes), "Pretrained individual tensor byte hash differs");
        raw += bytes;
        commitment += name;
        commitment.push_back('\0');
        commitment += tuple_shape(expected_shape);
        commitment.push_back('\0');
        commitment += "float32";
        commitment.push_back('\0');
        commitment += bytes;
        parameter_count += size;
        result.emplace(name, Tensor{expected_shape, std::move(values)});
    }
    require(parameter_count == 17475, "Pretrained actor parameter count differs");
    require(sha256_hex(raw) == kActorBytesSha && document.at("actor_bytes_sha256") == kActorBytesSha,
            "Pretrained actor bytes differ from the trusted v1 checkpoint");
    require(sha256_hex(commitment) == kActorTensorSha && document.at("actor_tensor_sha256") == kActorTensorSha,
            "Pretrained actor tensor digest differs from the trusted v1 checkpoint");
    return result;
}

template <std::size_t Inputs, std::size_t Outputs>
std::array<float, Outputs> dense(const std::array<float, Inputs>& input,
                                 const Tensor& weights, const Tensor& bias,
                                 bool activation) {
    std::array<float, Outputs> result{};
    for (std::size_t output = 0; output < Outputs; ++output) {
        float value = bias.data[output];
        for (std::size_t feature = 0; feature < Inputs; ++feature)
            value += weights.data[output * Inputs + feature] * input[feature];
        result[output] = activation ? std::tanh(value) : value;
        require(std::isfinite(result[output]), "Pretrained inference produced a nonfinite output");
    }
    return result;
}

const Json& sensor(const Json& observation, std::string_view family) {
    const Json& values = observation.at("reward_sensors");
    require(values.is_array(), "Local reward sensors must be an array");
    const Json* found = nullptr;
    for (const auto& value : values) {
        if (value.at("family_id") == family) {
            require(found == nullptr, "Local reward sensor family is duplicated");
            found = &value;
        }
    }
    require(found != nullptr, "Required local reward sensor family is missing");
    require(found->at("valid").is_boolean(), "Local sensor validity must be Boolean");
    return *found;
}

std::array<double, 3> local_vector(const Json& sensor_value, std::string_view name) {
    const Json& value = sensor_value.at("observations").at(std::string(name));
    require(value.is_array() && value.size() == 3, "Local IMU vector requires exactly three components");
    std::array<double, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = finite_number(value[index], "local IMU component");
    return result;
}
} // namespace

struct PretrainedPolicy::Impl {
    explicit Impl(const std::filesystem::path& path) {
        require(std::filesystem::is_regular_file(path), "Pretrained actor artifact does not exist");
        require(std::filesystem::file_size(path) <= 2 * 1024 * 1024, "Pretrained actor artifact is too large");
        std::ifstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("Cannot read pretrained actor artifact");
        const std::string bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        Json document = Json::parse(bytes);
        exact_fields(document, {"schema", "version", "policy_id", "dtype", "byte_order", "dimensions",
                     "activation", "action_requests", "decision_dt_s", "native_dt_s", "frame_count",
                     "frame_size", "input_clip", "actor_tensor_sha256", "actor_bytes_sha256",
                     "tensors", "provenance", "stochastic_contract"}, "pretrained actor artifact");
        require(document.at("schema") == "pretrained_actor_v1" && document.at("version") == 1
                && document.at("policy_id") == "pooled-seed101-v1", "Pretrained actor version differs");
        require(document.at("dtype") == "float32" && document.at("byte_order") == "little"
                && document.at("dimensions") == Json::array({204, 64, 64, 3})
                && document.at("activation") == "tanh", "Pretrained actor architecture differs");
        require(document.at("action_requests") == Json::array({-.15, 0., .15})
                && document.at("decision_dt_s") == .1 && document.at("native_dt_s") == .02
                && document.at("frame_count") == 12 && document.at("frame_size") == 17
                && document.at("input_clip") == Json::array({-5., 5.}), "Pretrained actor observation/control contract differs");
        const Json& stochastic = document.at("stochastic_contract");
        exact_fields(stochastic, {"law", "native_rng", "torch_seeded_trajectory_equivalence"}, "pretrained stochastic contract");
        require(stochastic.at("law") == "categorical_softmax"
                && stochastic.at("native_rng") == "versioned portable sampler; independent of Torch RNG"
                && stochastic.at("torch_seeded_trajectory_equivalence") == false, "Pretrained sampling contract differs");
        validate_provenance(document.at("provenance"));
        tensors = read_tensors(document);
        info = std::move(document);
        info.erase("tensors");
        info["artifact_file_sha256"] = sha256_hex(bytes);
        info["parameters"] = 17475;
        info["learning"] = false;
        info["weight_updates"] = 0;
    }
    TensorMap tensors;
    Json info;
};

PretrainedPolicy::PretrainedPolicy(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}
PretrainedPolicy::~PretrainedPolicy() = default;
Json PretrainedPolicy::metadata() const { return impl_->info; }
PretrainedOutput PretrainedPolicy::logits(const PretrainedInput& input) const {
    for (float value : input)
        require(std::isfinite(value) && value >= -5.f && value <= 5.f, "Pretrained input must be finite and clipped to [-5,5]");
    const auto& tensors = impl_->tensors;
    const auto first = dense<204, 64>(input, tensors.at("mlp_extractor.policy_net.0.weight"),
                                     tensors.at("mlp_extractor.policy_net.0.bias"), true);
    const auto second = dense<64, 64>(first, tensors.at("mlp_extractor.policy_net.2.weight"),
                                     tensors.at("mlp_extractor.policy_net.2.bias"), true);
    return dense<64, 3>(second, tensors.at("action_net.weight"), tensors.at("action_net.bias"), false);
}
PretrainedOutput PretrainedPolicy::probabilities(const PretrainedInput& input) const {
    PretrainedOutput result = logits(input);
    const float largest = *std::max_element(result.begin(), result.end());
    float total = 0.f;
    for (float& value : result) {
        value = std::exp(value - largest);
        total += value;
    }
    require(std::isfinite(total) && total > 0.f, "Pretrained softmax denominator is invalid");
    for (float& value : result) value /= total;
    return result;
}
int PretrainedPolicy::sample(const PretrainedInput& input, double uniform) const {
    return categorical(probabilities(input), uniform);
}
int PretrainedPolicy::categorical(const PretrainedOutput& probabilities, double uniform) {
    require(std::isfinite(uniform) && uniform >= 0. && uniform < 1., "Categorical uniform must lie in [0,1)");
    double total = 0.;
    int last_positive = -1;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        const float value = probabilities[index];
        require(std::isfinite(value) && value >= 0.f && value <= 1.f, "Categorical probabilities must be finite within [0,1]");
        total += static_cast<double>(value);
        if (value > 0.f) last_positive = static_cast<int>(index);
    }
    require(total > 0. && std::isfinite(total), "Categorical probability mass must be positive");
    double cumulative = 0.;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        cumulative += static_cast<double>(probabilities[index]) / total;
        if (uniform < cumulative) return static_cast<int>(index);
    }
    return last_positive; // Covers only final double-precision rounding residue.
}

PretrainedHistory::PretrainedHistory(const Json& initial_observation) { reset(initial_observation); }
void PretrainedHistory::reset(const Json& initial_observation) {
    const auto initial = frame(initial_observation, 0., 0., 0.);
    input_.fill(0.f);
    std::copy(initial.begin(), initial.end(), input_.end() - kPretrainedFrameSize);
    count_ = 1;
    completed_decisions_ = 0;
}
void PretrainedHistory::append(const Json& observation, double executed_effort,
                               double previous_request, double reward) {
    if (count_ == 0) throw std::logic_error("Reset pretrained history before appending a decision");
    const auto next = frame(observation, executed_effort, previous_request, reward);
    std::move(input_.begin() + kPretrainedFrameSize, input_.end(), input_.begin());
    std::copy(next.begin(), next.end(), input_.end() - kPretrainedFrameSize);
    count_ = std::min(count_ + 1, kPretrainedFrameCount);
    ++completed_decisions_;
}
const PretrainedInput& PretrainedHistory::values() const {
    if (count_ == 0) throw std::logic_error("Reset pretrained history before reading policy input");
    return input_;
}
std::size_t PretrainedHistory::count() const noexcept { return count_; }
Json PretrainedHistory::diagnostics() const {
    std::string bytes;
    bytes.reserve(input_.size() * sizeof(float));
    for (float value : input_) append_float_bytes(bytes, value);
    return Json{{"schema", "pretrained_history_v1"}, {"frames_present", count_},
                {"frame_count", kPretrainedFrameCount}, {"frame_size", kPretrainedFrameSize},
                {"completed_decisions", completed_decisions_}, {"order", "oldest_to_newest"},
                {"input_float32_le_sha256", sha256_hex(bytes)}};
}
std::array<float, kPretrainedFrameSize> PretrainedHistory::frame(
    const Json& observation, double executed_effort, double previous_request, double reward) {
    require(observation.is_object() && observation.at("schema_version") == "construction_observation_v2",
            "Expected the native construction_observation_v2 local boundary");
    const Json& imu = sensor(observation, "imu_6axis_v0");
    const Json& light = sensor(observation, "ambient_light_v0");
    const Json& feedback = observation.at("actuator_feedback");
    require(feedback.is_array() && feedback.size() == 1, "Pretrained actor requires exactly one local motor feedback sample");
    const Json& motor = feedback.at(0).at("feedback");
    const bool imu_valid = imu.at("valid").get<bool>();
    const bool light_valid = light.at("valid").get<bool>();
    const std::array<double, 3> force = imu_valid ? local_vector(imu, "specific_force_m_s2") : std::array<double, 3>{};
    const std::array<double, 3> gyro = imu_valid ? local_vector(imu, "angular_velocity_rad_s") : std::array<double, 3>{};
    const double lux = light_valid ? finite_number(light.at("observations").at("illuminance_lux"), "local illuminance") : 0.;
    const double position = finite_number(motor.at("position_rad"), "local encoder position");
    const std::array<double, kPretrainedFrameSize> values{
        executed_effort / .15, std::sin(position), std::cos(position),
        finite_number(motor.at("velocity_rad_s"), "local encoder velocity") / 10.,
        finite_number(motor.at("current_a"), "local motor current") / .5,
        force[0] / 50., force[1] / 50., force[2] / 50.,
        gyro[0] / 10., gyro[1] / 10., gyro[2] / 10., lux / 1000.,
        previous_request / .15, reward / (2. * .1),
        imu_valid ? 1. : 0., light_valid ? 1. : 0., 1.};
    std::array<float, kPretrainedFrameSize> result{};
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float value = static_cast<float>(values[index]);
        require(std::isfinite(values[index]) && std::isfinite(value), "Pretrained local frame contains a nonfinite value");
        result[index] = std::clamp(value, -5.f, 5.f);
    }
    return result;
}
} // namespace droid
