#include "helicopter.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr std::size_t max_request_bytes = 1024;
constexpr std::size_t max_trail_points = 600;
constexpr std::size_t max_telemetry_samples = 800;
constexpr std::size_t max_evaluation_bytes = 8 * 1024 * 1024;
constexpr std::string_view evaluation_path = "/workspace/artifacts/evaluation.json";
std::atomic_bool shutdown_requested{false};
static_assert(std::atomic_bool::is_always_lock_free,
              "The signal flag must use lock-free atomic operations");

extern "C" void request_shutdown(int) {
    shutdown_requested.store(true, std::memory_order_relaxed);
}

void send_json(httplib::Response& response, int status, const Json& value) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_content(value.dump(), "application/json; charset=utf-8");
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool allowed_authority(std::string_view value) {
    return value == "localhost:43118" || value == "127.0.0.1:43118" ||
           value == "localhost:8080" || value == "127.0.0.1:8080";
}

bool allowed_control_request(const httplib::Request& request) {
    if (request.get_header_value_count("Host") != 1 ||
        !allowed_authority(lowercase(request.get_header_value("Host")))) {
        return false;
    }
    if (request.get_header_value_count("Origin") > 1 ||
        request.get_header_value_count("Sec-Fetch-Site") > 1) {
        return false;
    }
    if (request.has_header("Origin")) {
        const std::string origin = lowercase(request.get_header_value("Origin"));
        constexpr std::string_view scheme = "http://";
        if (!origin.starts_with(scheme) ||
            !allowed_authority(std::string_view(origin).substr(scheme.size()))) {
            return false;
        }
    }
    return lowercase(request.get_header_value("Sec-Fetch-Site")) != "cross-site";
}

std::string parse_action(const httplib::Request& request) {
    if (request.get_header_value_count("Content-Type") != 1) {
        throw std::invalid_argument("Content-Type must be application/json");
    }
    std::string content_type = lowercase(request.get_header_value("Content-Type"));
    content_type = content_type.substr(0, content_type.find(';'));
    while (!content_type.empty() && content_type.back() == ' ') {
        content_type.pop_back();
    }
    if (content_type != "application/json") {
        throw std::invalid_argument("Content-Type must be application/json");
    }
    if (request.body.empty() || request.body.size() > max_request_bytes) {
        throw std::invalid_argument("Expected a nonempty JSON control object");
    }

    unsigned action_keys = 0;
    Json body;
    try {
        body = Json::parse(request.body, [&action_keys](int, Json::parse_event_t event,
                                                      Json& parsed) {
            if (event == Json::parse_event_t::key && parsed == "action") {
                ++action_keys;
            }
            return true;
        });
    } catch (const Json::exception&) {
        throw std::invalid_argument("Malformed JSON control object");
    }
    if (!body.is_object() || body.size() != 1 || action_keys != 1 ||
        !body.contains("action") || !body.at("action").is_string()) {
        throw std::invalid_argument("Expected exactly one string field: action");
    }
    const auto action = body.at("action").get<std::string>();
    if (action != "play" && action != "pause" && action != "reset" &&
        action != "gust" && action != "hinf" && action != "pid") {
        throw std::invalid_argument("Action must be play, pause, reset, gust, hinf, or pid");
    }
    return action;
}

Json inspect_gpu() {
    Json result{{"available", false}, {"count", 0}, {"name", "Unavailable"},
                {"devices", Json::array()}, {"detail", "CUDA driver unavailable"}};
    const auto close_library = [](void* handle) noexcept { (void)dlclose(handle); };
    std::unique_ptr<void, decltype(close_library)> library(
        dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL), close_library);
    if (!library) {
        return result;
    }
    using Init = int (*)(unsigned);
    using DeviceCount = int (*)(int*);
    using DeviceGet = int (*)(int*, int);
    using DeviceName = int (*)(char*, int, int);
    const auto init = reinterpret_cast<Init>(dlsym(library.get(), "cuInit"));
    const auto count_devices = reinterpret_cast<DeviceCount>(
        dlsym(library.get(), "cuDeviceGetCount"));
    const auto get_device = reinterpret_cast<DeviceGet>(
        dlsym(library.get(), "cuDeviceGet"));
    const auto name_device = reinterpret_cast<DeviceName>(
        dlsym(library.get(), "cuDeviceGetName"));
    if (!init || !count_devices || !get_device || !name_device) {
        result["detail"] = "CUDA driver API symbols unavailable";
        return result;
    }
    const int init_status = init(0);
    if (init_status != 0) {
        result["detail"] = "cuInit failed with code " + std::to_string(init_status);
        return result;
    }
    int count = 0;
    const int count_status = count_devices(&count);
    if (count_status != 0 || count <= 0) {
        result["detail"] = "No accessible CUDA devices; query code " +
                           std::to_string(count_status);
        return result;
    }
    result["count"] = count;
    for (int ordinal = 0; ordinal < std::min(count, 32); ++ordinal) {
        int device = 0;
        if (get_device(&device, ordinal) != 0) {
            continue;
        }
        std::array<char, 256> name{};
        const bool named = name_device(name.data(), static_cast<int>(name.size()),
                                       device) == 0;
        name.back() = '\0';
        result["devices"].push_back(named ? std::string(name.data())
                                          : "CUDA device " + std::to_string(ordinal));
    }
    result["available"] = !result["devices"].empty();
    if (result["available"].get<bool>()) {
        result["name"] = result["devices"][0];
        result["detail"] = "CUDA driver access verified; flight physics and controller use CPU";
    } else {
        result["detail"] = "CUDA devices reported but device access failed";
    }
    return result;
}

// Six decimal places retain useful measurement precision while keeping mission
// history compact. Live state and the simulator retain their original doubles.
void round_telemetry_numbers(Json& value) {
    if (value.is_number_float()) {
        const double number = value.get<double>();
        if (!std::isfinite(number)) {
            throw std::runtime_error("Nonfinite telemetry value");
        }
        value = std::round(number * 1e6) / 1e6;
    } else if (value.is_structured()) {
        for (auto& child : value) {
            round_telemetry_numbers(child);
        }
    }
}

void sanitize_controller_numbers(Json& value) {
    if (value.is_number_float() && !std::isfinite(value.get<double>())) {
        // A failed controller update can have no finite diagnostic. Keep
        // that diagnostic explicitly missing rather than breaking safe fallback.
        value = nullptr;
    } else if (value.is_structured()) {
        for (auto& child : value) {
            sanitize_controller_numbers(child);
        }
    }
}

Json telemetry_sample(const Json& state) {
    const auto position = state.at("position").get<std::array<double, 3>>();
    const auto target = state.at("target").get<std::array<double, 3>>();
    std::array<double, 3> error{};
    for (std::size_t axis = 0; axis < error.size(); ++axis) {
        error[axis] = target[axis] - position[axis];
    }
    Json sample{{"time", state.at("time")}, {"position", position},
                {"target", target}, {"velocity", state.at("velocity")},
                {"reference", {{"velocity", state.at("reference").at("velocity")}}},
                {"attitude_deg", state.at("attitude_deg")},
                {"body_rates", state.at("body_rates")},
                {"controls", state.at("controls")}, {"actuators", state.at("actuators")},
                {"wind", state.at("wind")}, {"error", error},
                {"metrics", Json::object()}, {"solution", Json::object()}};
    for (const char* key : {"endpoint_error", "rms_error", "max_error",
                           "saturation_fraction", "max_tilt_deg"}) {
        sample["metrics"][key] = state.at("metrics").at(key);
    }
    for (const char* key : {"available", "time", "actual_thrust_n", "commanded_thrust_n"}) {
        sample["solution"][key] = state.at("solution").at(key);
    }
    if (state.contains("controller_mode")) {
        sample["controller_mode"] = state.at("controller_mode");
    }
    for (const char* key : {"input_rad", "actuator_pitch_rad", "inflow_m_s", "flap_rad", "measured_state", "sensor_noise_scale", "measurement_truth", "measurement_time"}) {
        if (state.contains(key)) {
            sample[key] = state.at(key);
        }
    }
    sample["mass_kg"] = state.at("mass_kg");
    sample["parameter_estimation"] = Json::object();
    for (const char* key : {"mass_kg", "mass_scale", "accepted_samples"}) {
        sample["parameter_estimation"][key] = state.at("parameter_estimation").at(key);
    }
    if (state.contains("thermodynamics") && state.at("thermodynamics").is_object()) {
        // Retain native component accounting and units without reconstructing a
        // thermodynamic model in the browser.
        sample["thermodynamics"] = state.at("thermodynamics");
    }
    if (state.contains("hinf") && state.at("hinf").is_object()) {
        sample["hinf"] = state.at("hinf");
        sanitize_controller_numbers(sample["hinf"]);
    }
    round_telemetry_numbers(sample);
    return sample;
}

void send_saved_evaluation(httplib::Response& response, const Json& model) {
    const auto fail = [&response](int status, std::string_view error) {
        send_json(response, status,
                  {{"source", "saved evaluation file"}, {"validation_saved", false},
                   {"source_verified", false}, {"live", false}, {"error", error}});
    };
    std::error_code file_error;
    const auto size = std::filesystem::file_size(evaluation_path, file_error);
    if (file_error) {
        const bool missing = file_error == std::errc::no_such_file_or_directory;
        fail(missing ? 404 : 500, missing ? "Saved evaluation file is missing"
                                         : "Unable to inspect saved evaluation file");
        return;
    }
    if (size == 0 || size > max_evaluation_bytes) {
        fail(500, "Saved evaluation file has an invalid size");
        return;
    }
    std::ifstream input(std::string(evaluation_path), std::ios::binary);
    if (!input) {
        fail(500, "Unable to read saved evaluation file");
        return;
    }
    // Read a bounded snapshot even if another process is replacing the report.
    std::string contents(max_evaluation_bytes + 1, '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    const auto read_size = static_cast<std::size_t>(input.gcount());
    if (input.bad() || read_size == 0 || read_size > max_evaluation_bytes) {
        fail(500, "Unable to read a complete bounded evaluation file");
        return;
    }
    contents.resize(read_size);
    Json report;
    try {
        report = Json::parse(contents);
    } catch (const Json::exception&) {
        fail(500, "Saved evaluation file contains invalid JSON");
        return;
    }
    bool valid = report.is_object() && report.value("schema_version", 0) == 4 && report.value("method", "") == "minimum_entropy_hinf" && report.contains("passed") &&
                 report.at("passed").is_boolean() && report.contains("scenarios") &&
                 report.at("scenarios").is_array() && !report.at("scenarios").empty();
    if (valid) {
        for (const auto& scenario : report.at("scenarios")) {
            valid = scenario.is_object() && scenario.contains("name") &&
                    scenario.at("name").is_string() && scenario.contains("passed") &&
                    scenario.at("passed").is_boolean() && scenario.contains("metrics") &&
                    scenario.at("metrics").is_object();
            if (!valid) {
                break;
            }
        }
    }
    if (!valid || (report.contains("model") && !report.at("model").is_object())) {
        fail(500, "Saved evaluation file has an invalid report schema");
        return;
    }
    const bool has_model = report.contains("model");
    const bool verified = has_model && report.at("model") == model;
    const std::string reason = !has_model ? "Saved report has no model provenance"
                              : verified ? "Saved model provenance matches this executable"
                                         : "Saved model provenance differs from this executable";
    send_json(response, 200,
              {{"source", "saved evaluation file"},
               {"artifact", "artifacts/evaluation.json"}, {"validation_saved", true},
               {"source_verified", verified}, {"verification_reason", reason},
               {"live", false}, {"report", std::move(report)}});
}

struct Session {
    mutable std::shared_mutex mutex;
    helicopter::Simulation simulation;
    bool running = true;
    unsigned ticks = 0;
    std::uint64_t run_id = 1;
    std::deque<std::array<double, 3>> trail;
    std::deque<Json> telemetry;
    bool history_complete = true;
    std::string error;
    Clock::time_point last_tick = Clock::now();
    const Json gpu = inspect_gpu();

    Session() { sample_history(); }

    // Callers hold the session mutex for all simulation and trail access.
    void sample_history() {
        const Json state = simulation.state();
        Json sample = telemetry_sample(state);
        trail.push_back(state.at("position").get<std::array<double, 3>>());
        if (trail.size() > max_trail_points) {
            trail.pop_front();
        }
        telemetry.push_back(std::move(sample));
        if (telemetry.size() > max_telemetry_samples) {
            telemetry.pop_front();
            history_complete = false;
        }
    }

    Json history() const {
        Json value{{"schema_version", 2}, {"run_id", run_id}, {"running", running},
                   {"finished", simulation.finished()}, {"step_seconds", 0.01},
                   {"sample_period_seconds", 0.1}, {"max_samples", max_telemetry_samples},
                   {"sample_count", telemetry.size()}, {"history_complete", history_complete},
                   {"samples", telemetry}};
        if (!error.empty()) {
            value["error"] = error;
        }
        return value;
    }

    Json snapshot() const {
        Json value = simulation.state();
        value["solution"]["position_integral_role"] =
            "PID diagnostic only; H-infinity feedback has its own disclosed dynamic state";
        value["run_id"] = run_id;
        value["running"] = running;
        value["trail"] = trail;
        value["gpu"] = gpu;
        value["execution"] = {{"physics", "CPU"}, {"controller", "CPU"},
                              {"renderer", "Browser WebGL"}};
        if (!error.empty()) {
            value["error"] = error;
        }
        return value;
    }
};

void run_simulation(Session& session, std::stop_token stop) {
    auto next_tick = Clock::now();
    while (!stop.stop_requested()) {
        next_tick += 10ms;
        {
            std::unique_lock lock(session.mutex);
            try {
                if (session.running && session.error.empty()) {
                    session.simulation.step();
                    if (++session.ticks % 10 == 0 || session.simulation.finished()) {
                        session.sample_history();
                    }
                    if (session.simulation.finished()) {
                        session.running = false;
                    }
                }
            } catch (const std::exception& error) {
                session.error = error.what();
                session.running = false;
                std::cerr << "Simulation paused after failure: " << error.what() << '\n';
            } catch (...) {
                session.error = "Unknown simulation failure";
                session.running = false;
                std::cerr << "Simulation paused after unknown failure\n";
            }
            // Health requests share this lock and can wait for a synchronous
            // solve. Timestamp its completion so a completed long solve is not
            // mistaken for an unresponsive worker when the request proceeds.
            session.last_tick = Clock::now();
        }
        // A slow host slows simulation time rather than running unbounded catch-up.
        const auto now = Clock::now();
        if (next_tick < now) {
            next_tick = now + 10ms;
        }
        std::this_thread::sleep_until(next_tick);
    }
}

int serve() {
    Session session;
    const Json model = helicopter::Simulation::model();
    httplib::Server server;
    server.set_payload_max_length(max_request_bytes);
    server.set_read_timeout(5);
    server.set_write_timeout(5);
    server.set_keep_alive_max_count(30);
    server.set_default_headers({
        {"Content-Security-Policy",
         "default-src 'self'; script-src 'self'; style-src 'self'; "
         "style-src-elem 'self' 'unsafe-inline'; style-src-attr 'none'; "
         "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
         "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"},
        {"X-Content-Type-Options", "nosniff"},
        {"X-Frame-Options", "DENY"},
        {"X-Robots-Tag", "noindex, nofollow"},
        {"Referrer-Policy", "no-referrer"},
        {"Cache-Control", "no-cache"}
    });
    // The style element exception serves the local Codex review overlay only.
    // This service is a local development viewer, not a production export.
    server.set_pre_routing_handler([](const httplib::Request& request,
                                      httplib::Response& response) {
        if (request.method == "POST" && !allowed_control_request(request)) {
            send_json(response, 403, {{"error", "Control requests require an allowed local host and origin"}});
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_exception_handler([](const httplib::Request&, httplib::Response& response,
                                   std::exception_ptr error) {
        try {
            if (error) {
                std::rethrow_exception(error);
            }
        } catch (const std::exception& failure) {
            std::cerr << "HTTP handler failure: " << failure.what() << '\n';
        } catch (...) {
            std::cerr << "Unknown HTTP handler failure\n";
        }
        send_json(response, 500, {{"error", "Internal service error"}});
    });
    server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (response.body.empty()) {
            const std::string error = response.status == 413 ? "Request body too large"
                                                             : "Request failed";
            send_json(response, response.status, {{"error", error}});
        }
    });
    server.Get("/healthz", [&session](const httplib::Request&, httplib::Response& response) {
        std::shared_lock lock(session.mutex);
        const bool responsive = Clock::now() - session.last_tick < 2s;
        const bool healthy = session.error.empty() && responsive;
        Json health{{"status", healthy ? "ok" : "error"},
                    {"running", session.running}, {"gpu", session.gpu},
                    {"physics", "CPU"}, {"controller", "CPU"},
                    {"renderer", "Browser WebGL"}};
        if (!session.error.empty()) {
            health["error"] = session.error;
        } else if (!responsive) {
            health["error"] = "Simulation worker is unresponsive";
        }
        send_json(response, healthy ? 200 : 503, health);
    });
    server.Get("/api/state", [&session](const httplib::Request&, httplib::Response& response) {
        std::shared_lock lock(session.mutex);
        const int status = session.error.empty() ? 200 : 503;
        const Json snapshot = session.snapshot();
        lock.unlock();
        send_json(response, status, snapshot);
    });
    server.Get("/api/model", [&model](const httplib::Request&, httplib::Response& response) {
        send_json(response, 200, model);
    });
    server.Get("/api/telemetry", [&session](const httplib::Request&, httplib::Response& response) {
        std::shared_lock lock(session.mutex);
        const int status = session.error.empty() ? 200 : 503;
        const Json history = session.history();
        lock.unlock();
        // JSON serialization must not hold the simulation's shared mutex.
        send_json(response, status, history);
    });
    server.Get("/api/evaluation", [&model](const httplib::Request&, httplib::Response& response) {
        send_saved_evaluation(response, model);
    });
    server.Post("/api/control", [&session](const httplib::Request& request,
                                          httplib::Response& response) {
        std::string action;
        try {
            action = parse_action(request);
        } catch (const std::invalid_argument& error) {
            send_json(response, 400, {{"error", error.what()}});
            return;
        }
        std::unique_lock lock(session.mutex);
        const bool new_run = action == "reset" || action == "hinf" || action == "pid";
        if (!session.error.empty() && !new_run && action != "pause") {
            send_json(response, 503, {{"error", "Simulation failed; reset is required"}});
            return;
        }
        if (new_run) {
            const auto previous_mode = session.simulation.state().value(
                "controller_mode", std::string("minimum_entropy_hinf"));
            const bool pid = action == "pid" ||
                             (action == "reset" && previous_mode == "pid_baseline");
            session.simulation.reset(0, 1.0, true,
                pid ? helicopter::ControllerMode::Pid : helicopter::ControllerMode::Hinf);
            session.trail.clear();
            session.telemetry.clear();
            session.history_complete = true;
            ++session.run_id;
            session.sample_history();
            session.ticks = 0;
            session.error.clear();
            session.running = true;
        } else if (action == "play") {
            if (session.simulation.finished()) {
                send_json(response, 409, {{"error", "Mission finished; reset to fly again"}});
                return;
            }
            session.running = true;
        } else if (action == "pause") {
            session.running = false;
        } else {
            if (session.simulation.finished()) {
                send_json(response, 409, {{"error", "Mission finished; reset before adding a gust"}});
                return;
            }
            session.simulation.gust();
        }
        const Json snapshot = session.snapshot();
        lock.unlock();
        send_json(response, 200, snapshot);
    });
    if (!server.set_mount_point("/vendor", "/opt/helicopter/web/vendor") ||
        !server.set_mount_point("/", "/workspace/web")) {
        throw std::runtime_error("Required web asset directories are missing");
    }
    if (!server.bind_to_port("0.0.0.0", 8080)) {
        throw std::runtime_error("Unable to bind HTTP server on port 8080");
    }
    std::jthread simulation_thread([&session](std::stop_token stop) {
        run_simulation(session, stop);
    });
    std::jthread signal_monitor([&server](std::stop_token stop) {
        while (!stop.stop_requested()) {
            if (shutdown_requested.load(std::memory_order_relaxed)) {
                server.stop();
                return;
            }
            std::this_thread::sleep_for(20ms);
        }
    });
    std::cout << "Helicopter viewer listening on 0.0.0.0:8080; "
              << "physics/controller=CPU, rendering=browser WebGL; GPU="
              << session.gpu.at("name").get<std::string>() << std::endl;
    const bool served = server.listen_after_bind();
    simulation_thread.request_stop();
    signal_monitor.request_stop();
    return served || shutdown_requested.load(std::memory_order_relaxed) ? 0 : 1;
}
} // namespace

int main() {
    std::signal(SIGTERM, request_shutdown);
    std::signal(SIGINT, request_shutdown);
    try {
        return serve();
    } catch (const std::exception& error) {
        std::cerr << "Helicopter service failed: " << error.what() << '\n';
        return 1;
    }
}
