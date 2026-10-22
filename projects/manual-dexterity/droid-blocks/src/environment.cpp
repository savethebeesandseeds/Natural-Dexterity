#include "droid/environment.hpp"

#include "droid/simulation.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace droid {
namespace {

using Json = nlohmann::json;

struct JsonDifference {
    std::string path;
    std::string reason;
    bool expected_present{true};
    bool actual_present{true};
    Json expected;
    Json actual;
};

[[nodiscard]] std::string pointer_token(std::string_view token) {
    std::string escaped;
    escaped.reserve(token.size());
    for (const char character : token) {
        if (character == '~') {
            escaped += "~0";
        } else if (character == '/') {
            escaped += "~1";
        } else {
            escaped += character;
        }
    }
    return escaped;
}

[[nodiscard]] std::optional<JsonDifference> first_difference(
    const Json& expected,
    const Json& actual,
    const std::string& path) {
    // dump() distinguishes JSON integer, unsigned, and floating encodings. That
    // is stricter than nlohmann::json::operator== for numeric values and is the
    // canonical byte comparison used by the trace contract.
    if (expected.dump() == actual.dump()) {
        return std::nullopt;
    }

    if (expected.type() != actual.type()) {
        return JsonDifference{
            path,
            "JSON types differ (expected " +
                std::string(expected.type_name()) + ", actual " +
                std::string(actual.type_name()) + ")",
            true,
            true,
            expected,
            actual,
        };
    }

    if (expected.is_object()) {
        for (const auto& [key, expected_value] : expected.items()) {
            const std::string child_path =
                path + "/" + pointer_token(key);
            const auto actual_iterator = actual.find(key);
            if (actual_iterator == actual.end()) {
                return JsonDifference{
                    child_path,
                    "key is missing from replay output",
                    true,
                    false,
                    expected_value,
                    nullptr,
                };
            }
            if (auto difference = first_difference(
                    expected_value, *actual_iterator, child_path)) {
                return difference;
            }
        }
        for (const auto& [key, actual_value] : actual.items()) {
            if (!expected.contains(key)) {
                return JsonDifference{
                    path + "/" + pointer_token(key),
                    "replay output has an unexpected key",
                    false,
                    true,
                    nullptr,
                    actual_value,
                };
            }
        }
    } else if (expected.is_array()) {
        const std::size_t shared_size =
            std::min(expected.size(), actual.size());
        for (std::size_t index = 0; index < shared_size; ++index) {
            if (auto difference = first_difference(
                    expected.at(index),
                    actual.at(index),
                    path + "/" + std::to_string(index))) {
                return difference;
            }
        }
        if (expected.size() != actual.size()) {
            return JsonDifference{
                path,
                "array lengths differ",
                true,
                true,
                expected.size(),
                actual.size(),
            };
        }
    }

    return JsonDifference{
        path,
        "JSON values differ",
        true,
        true,
        expected,
        actual,
    };
}

[[nodiscard]] Json mismatch_report(
    std::string_view schema_version,
    std::string_view phase,
    std::optional<std::size_t> transition_index,
    std::size_t verified_steps,
    const JsonDifference& difference) {
    Json mismatch{
        {"phase", phase},
        {"path", difference.path.empty() ? "/" : difference.path},
        {"reason", difference.reason},
        {"expected_present", difference.expected_present},
        {"actual_present", difference.actual_present},
    };
    if (transition_index.has_value()) {
        mismatch["transition_index"] = *transition_index;
    }
    if (difference.expected_present) {
        mismatch["expected"] = difference.expected;
    }
    if (difference.actual_present) {
        mismatch["actual"] = difference.actual;
    }
    return Json{
        {"ok", false},
        {"schema_version", schema_version},
        {"verified_steps", verified_steps},
        {"mismatch", std::move(mismatch)},
    };
}

void require_object_member(
    const Json& value,
    std::string_view member,
    Json::value_t expected_type,
    std::string_view context) {
    const std::string key(member);
    if (!value.contains(key)) {
        throw std::invalid_argument(
            std::string(context) + " is missing '" + key + "'");
    }
    if (value.at(key).type() != expected_type) {
        throw std::invalid_argument(
            std::string(context) + "." + key + " has the wrong JSON type");
    }
}

void validate_reset_result(const Json& result) {
    if (!result.is_object()) {
        throw std::runtime_error("simulation reset result must be an object");
    }
    require_object_member(
        result, "observation", Json::value_t::object, "reset result");
    require_object_member(result, "info", Json::value_t::object, "reset result");
}

void validate_transition_result(const Json& result) {
    if (!result.is_object()) {
        throw std::runtime_error("simulation step result must be an object");
    }
    require_object_member(
        result, "observation", Json::value_t::object, "step result");
    require_object_member(
        result, "reward_components", Json::value_t::array, "step result");
    require_object_member(
        result, "safety", Json::value_t::object, "step result");
    require_object_member(
        result, "terminated", Json::value_t::boolean, "step result");
    require_object_member(
        result, "truncated", Json::value_t::boolean, "step result");
    require_object_member(result, "info", Json::value_t::object, "step result");
    if (!result.contains("reward") || !result.at("reward").is_number()) {
        throw std::runtime_error("step result.reward must be numeric");
    }
    if (!std::isfinite(result.at("reward").get<double>())) {
        throw std::runtime_error("step result.reward must be finite");
    }
}

void validate_trace_document(const Json& document) {
    if (!document.is_object()) {
        throw std::invalid_argument("environment trace must be a JSON object");
    }
    require_object_member(
        document,
        "schema_version",
        Json::value_t::string,
        "environment trace");
    const std::string schema_version =
        document.at("schema_version").get<std::string>();
    const bool legacy_v1 =
        schema_version == kLegacyEnvironmentTraceSchemaVersion;
    if (schema_version != kEnvironmentTraceSchemaVersion && !legacy_v1) {
        throw std::invalid_argument(
            "unsupported environment trace schema_version '" +
            schema_version + "'");
    }
    require_object_member(
        document, "assembly_id", Json::value_t::string, "environment trace");
    if (document.at("assembly_id").get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(
            "environment trace.assembly_id must not be empty");
    }
    if (!document.contains("seed") ||
        !(document.at("seed").is_number_unsigned() ||
          document.at("seed").is_number_integer())) {
        throw std::invalid_argument(
            "environment trace.seed must be a non-negative integer");
    }
    if (document.at("seed").is_number_integer() &&
        !document.at("seed").is_number_unsigned() &&
        document.at("seed").get<std::int64_t>() < 0) {
        throw std::invalid_argument(
            "environment trace.seed must be a non-negative integer");
    }
    if (!legacy_v1) {
        require_object_member(
            document,
            "episode_profile",
            Json::value_t::string,
            "environment trace");
        if (document.at("episode_profile").get_ref<const std::string&>().empty()) {
            throw std::invalid_argument(
                "environment trace.episode_profile must not be empty");
        }
    }
    if (!document.contains("reset")) {
        throw std::invalid_argument("environment trace is missing 'reset'");
    }
    validate_reset_result(document.at("reset"));
    if (!legacy_v1) {
        const Json& reset_info = document.at("reset").at("info");
        require_object_member(
            reset_info,
            "episode_profile",
            Json::value_t::string,
            "environment trace.reset.info");
        if (reset_info.at("episode_profile") != document.at("episode_profile")) {
            throw std::invalid_argument(
                "environment trace reset episode_profile does not match its "
                "top-level value");
        }
    }
    require_object_member(
        document, "steps", Json::value_t::array, "environment trace");

    const Json& steps = document.at("steps");
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const Json& entry = steps.at(index);
        if (!entry.is_object()) {
            throw std::invalid_argument(
                "environment trace.steps[" + std::to_string(index) +
                "] must be an object");
        }
        if (!entry.contains("index") ||
            !entry.at("index").is_number_unsigned() ||
            entry.at("index").get<std::size_t>() != index) {
            throw std::invalid_argument(
                "environment trace step indices must be contiguous from zero");
        }
        require_object_member(
            entry,
            "actions",
            Json::value_t::object,
            "environment trace step");
        for (const auto& [module_id, effort] : entry.at("actions").items()) {
            if (module_id.empty() || !effort.is_number() ||
                !std::isfinite(effort.get<double>())) {
                throw std::invalid_argument(
                    "environment trace step actions must map non-empty module "
                    "IDs to finite numbers");
            }
        }
        if (!entry.contains("control_dt_s") ||
            !entry.at("control_dt_s").is_number() ||
            !std::isfinite(entry.at("control_dt_s").get<double>()) ||
            entry.at("control_dt_s").get<double>() <= 0.0) {
            throw std::invalid_argument(
                "environment trace step control_dt_s must be finite and "
                "positive");
        }
        if (!entry.contains("transition")) {
            throw std::invalid_argument(
                "environment trace step is missing 'transition'");
        }
        validate_transition_result(entry.at("transition"));
        if (!legacy_v1) {
            const Json& transition_info =
                entry.at("transition").at("info");
            require_object_member(
                transition_info,
                "episode_profile",
                Json::value_t::string,
                "environment trace step transition.info");
            if (transition_info.at("episode_profile") !=
                document.at("episode_profile")) {
                throw std::invalid_argument(
                    "environment trace step episode_profile does not match "
                    "its top-level value");
            }
        }
    }
}

[[nodiscard]] Json legacy_comparable_result(
    const Json& expected,
    Json actual,
    bool legacy_v1) {
    if (legacy_v1 && expected.contains("info") &&
        expected.at("info").is_object() &&
        !expected.at("info").contains("episode_profile") &&
        actual.contains("info") && actual.at("info").is_object()) {
        actual.at("info").erase("episode_profile");
    }
    return actual;
}

[[nodiscard]] std::filesystem::path unique_sibling(
    const std::filesystem::path& destination,
    std::string_view suffix) {
    static std::atomic<std::uint64_t> sequence{0};
    const std::uint64_t timestamp = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        std::filesystem::path candidate = destination;
        candidate += std::string(suffix) + "." + std::to_string(timestamp) +
            "." + std::to_string(sequence.fetch_add(1)) + "." +
            std::to_string(attempt);
        std::error_code error;
        if (!std::filesystem::exists(candidate, error) && !error) {
            return candidate;
        }
    }
    throw std::runtime_error(
        "unable to reserve a temporary trace path beside " +
        destination.string());
}

void write_trace_atomically(
    const std::filesystem::path& destination,
    const Json& document) {
    if (destination.empty() || destination.filename().empty()) {
        throw std::invalid_argument("trace destination must name a file");
    }
    const std::filesystem::path parent = destination.has_parent_path()
        ? destination.parent_path()
        : std::filesystem::current_path();
    if (!std::filesystem::is_directory(parent)) {
        throw std::runtime_error(
            "trace destination directory does not exist: " +
            parent.string());
    }
    if (std::filesystem::is_directory(destination)) {
        throw std::runtime_error(
            "trace destination is a directory: " + destination.string());
    }
    if (std::filesystem::exists(destination)) {
        throw std::runtime_error(
            "refusing to overwrite existing trace: " +
            destination.string());
    }

    const std::filesystem::path temporary =
        unique_sibling(destination, ".tmp");
    try {
        {
            std::ofstream output(
                temporary, std::ios::binary | std::ios::out | std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "unable to open temporary trace file: " +
                    temporary.string());
            }
            output << document.dump(2) << '\n';
            output.flush();
            if (!output) {
                throw std::runtime_error(
                    "unable to write temporary trace file: " +
                    temporary.string());
            }
        }

        // A hard-link installation is atomic and has no replace-existing
        // operation: even a racing writer cannot make this call clobber an
        // earlier trace. The temporary name is on the same filesystem.
        std::error_code error;
        std::filesystem::create_hard_link(temporary, destination, error);
        if (error) {
            if (std::filesystem::exists(destination)) {
                throw std::runtime_error(
                    "refusing to overwrite existing trace: " +
                    destination.string());
            }
            throw std::runtime_error(
                "unable to install trace file " + destination.string() +
                ": " + error.message());
        }
        std::filesystem::remove(temporary, error);
        if (error) {
            throw std::runtime_error(
                "trace was saved, but its temporary link could not be "
                "removed: " + temporary.string() + ": " + error.message());
        }
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove(temporary, cleanup_error);
        throw;
    }
}

}  // namespace

struct DroidEnvironment::Impl {
    explicit Impl(const std::filesystem::path& path, bool realtime)
        : model_path(std::filesystem::absolute(path)),
          simulation(model_path, realtime) {}

    std::filesystem::path model_path;
    DroidSimulation simulation;
    mutable std::mutex mutex;
    bool episode_active{false};
    bool episode_done{false};
    bool recording_enabled{false};
    Json trace_document;
};

DroidEnvironment::DroidEnvironment(
    const std::filesystem::path& model_path,
    bool realtime)
    : impl_(std::make_unique<Impl>(model_path, realtime)) {}

DroidEnvironment::~DroidEnvironment() = default;

Json DroidEnvironment::spec() const {
    return impl_->simulation.agent_spec();
}

Json DroidEnvironment::body_experiment_spec() const {
    return impl_->simulation.body_experiment_spec();
}

Json DroidEnvironment::reset(
    std::string_view assembly_id,
    std::uint64_t seed,
    bool recording,
    std::string_view episode_profile) {
    if (assembly_id.empty()) {
        throw std::invalid_argument("assembly_id must not be empty");
    }

    std::scoped_lock lock(impl_->mutex);
    Json result =
        impl_->simulation.reset_agent(assembly_id, seed, episode_profile);
    validate_reset_result(result);

    impl_->episode_active = true;
    impl_->episode_done = false;
    impl_->recording_enabled = recording;
    if (recording) {
        impl_->trace_document = Json{
            {"schema_version", kEnvironmentTraceSchemaVersion},
            {"assembly_id", assembly_id},
            {"seed", seed},
            {"episode_profile", episode_profile},
            {"reset", result},
            {"steps", Json::array()},
        };
    } else {
        impl_->trace_document = nullptr;
    }
    return result;
}

Json DroidEnvironment::step(
    const std::map<std::string, double>& actions,
    double control_dt_s) {
    if (!std::isfinite(control_dt_s) || control_dt_s <= 0.0) {
        throw std::invalid_argument(
            "control_dt_s must be finite and positive");
    }
    for (const auto& [module_id, effort] : actions) {
        if (module_id.empty()) {
            throw std::invalid_argument("action module IDs must not be empty");
        }
        if (!std::isfinite(effort)) {
            throw std::invalid_argument(
                "action for '" + module_id + "' must be finite");
        }
    }

    std::scoped_lock lock(impl_->mutex);
    if (!impl_->episode_active) {
        throw std::logic_error("step requires a preceding agent reset");
    }
    if (impl_->episode_done) {
        throw std::logic_error(
            "episode has terminated or truncated; reset before stepping again");
    }

    Json transition = impl_->simulation.step_agent(actions, control_dt_s);
    validate_transition_result(transition);
    if (impl_->recording_enabled) {
        Json& steps = impl_->trace_document.at("steps");
        steps.push_back(Json{
            {"index", steps.size()},
            {"actions", actions},
            {"control_dt_s", control_dt_s},
            {"transition", transition},
        });
    }
    impl_->episode_done = transition.at("terminated").get<bool>() ||
        transition.at("truncated").get<bool>();
    return transition;
}

bool DroidEnvironment::recording() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->recording_enabled;
}

Json DroidEnvironment::trace() const {
    std::scoped_lock lock(impl_->mutex);
    if (!impl_->recording_enabled || impl_->trace_document.is_null()) {
        throw std::logic_error(
            "trace recording is disabled for the current episode");
    }
    return impl_->trace_document;
}

void DroidEnvironment::save_trace(const std::filesystem::path& path) const {
    const Json snapshot = trace();
    validate_trace_document(snapshot);
    write_trace_atomically(path, snapshot);
}

Json DroidEnvironment::load_trace(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("trace file not found: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open trace file: " + path.string());
    }
    Json document;
    try {
        input >> document;
    } catch (const Json::exception& error) {
        throw std::invalid_argument(
            "invalid JSON trace " + path.string() + ": " + error.what());
    }
    validate_trace_document(document);
    return document;
}

Json DroidEnvironment::replay_and_verify(const Json& recorded_trace) const {
    validate_trace_document(recorded_trace);

    const std::string trace_schema_version =
        recorded_trace.at("schema_version").get<std::string>();
    const bool legacy_v1 =
        trace_schema_version == kLegacyEnvironmentTraceSchemaVersion;
    DroidEnvironment replay(impl_->model_path, false);
    const std::string assembly_id =
        recorded_trace.at("assembly_id").get<std::string>();
    const std::uint64_t seed =
        recorded_trace.at("seed").get<std::uint64_t>();
    const std::string episode_profile = legacy_v1
        ? std::string(kFixedDemoEpisodeProfile)
        : recorded_trace.at("episode_profile").get<std::string>();
    Json actual_reset;
    try {
        actual_reset = replay.reset(
            assembly_id, seed, false, episode_profile);
    } catch (const std::exception& error) {
        return mismatch_report(
            trace_schema_version,
            "reset",
            std::nullopt,
            0,
            JsonDifference{
                "/reset",
                "replay reset raised: " + std::string(error.what()),
                true,
                true,
                recorded_trace.at("reset"),
                Json{{"replay_error", error.what()}},
            });
    }
    actual_reset = legacy_comparable_result(
        recorded_trace.at("reset"), std::move(actual_reset), legacy_v1);
    if (auto difference = first_difference(
            recorded_trace.at("reset"), actual_reset, "/reset")) {
        return mismatch_report(
            trace_schema_version, "reset", std::nullopt, 0, *difference);
    }

    const Json& steps = recorded_trace.at("steps");
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const Json& entry = steps.at(index);
        std::map<std::string, double> actions;
        for (const auto& [module_id, effort] : entry.at("actions").items()) {
            actions.emplace(module_id, effort.get<double>());
        }

        Json actual_transition;
        try {
            actual_transition = replay.step(
                actions, entry.at("control_dt_s").get<double>());
        } catch (const std::exception& error) {
            return mismatch_report(
                trace_schema_version,
                "step",
                index,
                index,
                JsonDifference{
                    "/steps/" + std::to_string(index) + "/transition",
                    "replay step raised: " + std::string(error.what()),
                    true,
                    true,
                    entry.at("transition"),
                    Json{{"replay_error", error.what()}},
                });
        }
        actual_transition = legacy_comparable_result(
            entry.at("transition"),
            std::move(actual_transition),
            legacy_v1);
        if (auto difference = first_difference(
                entry.at("transition"),
                actual_transition,
                "/steps/" + std::to_string(index) + "/transition")) {
            return mismatch_report(
                trace_schema_version, "step", index, index, *difference);
        }
    }

    return Json{
        {"ok", true},
        {"schema_version", trace_schema_version},
        {"verified_steps", steps.size()},
    };
}

Json DroidEnvironment::replay_and_verify_file(
    const std::filesystem::path& path) const {
    return replay_and_verify(load_trace(path));
}

Json DroidEnvironment::visualization_state() const {
    return impl_->simulation.state();
}

Json DroidEnvironment::catalog() const {
    return impl_->simulation.catalog();
}

Json DroidEnvironment::health() const {
    return impl_->simulation.health();
}

void DroidEnvironment::set_demo_running(bool running) {
    std::scoped_lock lock(impl_->mutex);
    impl_->simulation.set_running(running);
    if (running) {
        impl_->episode_active = false;
        impl_->episode_done = false;
    }
}

void DroidEnvironment::reset_visualization() {
    std::scoped_lock lock(impl_->mutex);
    impl_->simulation.reset();
    impl_->episode_active = false;
    impl_->episode_done = false;
}

}  // namespace droid
