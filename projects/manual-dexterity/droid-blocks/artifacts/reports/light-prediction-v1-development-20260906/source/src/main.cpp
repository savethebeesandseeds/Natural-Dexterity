#include "droid/environment.hpp"
#include "droid/construction_session.hpp"
#include "droid/light_session.hpp"
#include "droid/playground.hpp"
#include "droid/simulation.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaxRequestBytes = 4096;
constexpr std::string_view kDefaultHost{"0.0.0.0"};
constexpr int kDefaultPort = 8080;

volatile std::sig_atomic_t g_shutdown_requested = 0;

void request_shutdown(int) {
    g_shutdown_requested = 1;
}

struct Options {
    std::string host{kDefaultHost};
    int port{kDefaultPort};
    std::filesystem::path model{"models/droid.xml"};
    std::filesystem::path web_root{"web"};
    std::optional<std::size_t> probe_steps;
    bool help{false};
};

[[nodiscard]] std::string require_value(
    int& index,
    int argc,
    char** argv,
    std::string_view option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string(option) + " requires a value");
    }
    return argv[++index];
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(
    std::string_view text,
    std::string_view option) {
    Integer value{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument(
            std::string(option) + " must be an integer");
    }
    return value;
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "-h" || argument == "--help") {
            options.help = true;
        } else if (argument == "--host") {
            options.host = require_value(index, argc, argv, argument);
            if (options.host.empty()) {
                throw std::invalid_argument("--host cannot be empty");
            }
        } else if (argument == "--port") {
            options.port = parse_integer<int>(
                require_value(index, argc, argv, argument), argument);
            if (options.port < 0 || options.port > 65535) {
                throw std::invalid_argument(
                    "--port must be between 0 and 65535");
            }
        } else if (argument == "--model") {
            options.model = require_value(index, argc, argv, argument);
        } else if (argument == "--web-root") {
            options.web_root = require_value(index, argc, argv, argument);
        } else if (argument == "--probe-steps") {
            options.probe_steps = parse_integer<std::size_t>(
                require_value(index, argc, argv, argument), argument);
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument));
        }
    }
    return options;
}

void print_usage(std::ostream& output, std::string_view executable) {
    output
        << "Usage: " << executable << " [options]\n"
        << "\n"
        << "Run the native headless Droid Blocks simulator.\n"
        << "\n"
        << "Options:\n"
        << "  --host ADDRESS       HTTP bind address (default: 0.0.0.0)\n"
        << "  --port PORT          HTTP port, including 0 for any free port "
           "(default: 8080)\n"
        << "  --model PATH         MuJoCo XML model (default: models/droid.xml)\n"
        << "  --web-root PATH      Static web directory (default: web)\n"
        << "  --probe-steps N      Step exactly N times, print state JSON, exit\n"
        << "  -h, --help           Show this help\n";
}

void set_api_headers(httplib::Response& response) {
    response.set_header("Cache-Control", "no-store");
    response.set_header("Access-Control-Allow-Origin", "*");
    response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    response.set_header("Access-Control-Allow-Headers", "Content-Type");
}

void send_json(httplib::Response& response, int status, const Json& payload) {
    response.status = status;
    set_api_headers(response);
    response.set_content(payload.dump(), "application/json; charset=utf-8");
}

[[nodiscard]] int hexadecimal_value(char character) {
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
        return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
        return character - 'A' + 10;
    }
    return -1;
}

[[nodiscard]] std::optional<std::string> percent_decode(
    std::string_view encoded) {
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        if (encoded[index] != '%') {
            decoded.push_back(encoded[index]);
            continue;
        }
        if (index + 2 >= encoded.size()) {
            return std::nullopt;
        }
        const int high = hexadecimal_value(encoded[index + 1]);
        const int low = hexadecimal_value(encoded[index + 2]);
        if (high < 0 || low < 0) {
            return std::nullopt;
        }
        const char value = static_cast<char>((high << 4) | low);
        if (value == '\0') {
            return std::nullopt;
        }
        decoded.push_back(value);
        index += 2;
    }
    return decoded;
}

[[nodiscard]] bool path_is_within(
    const std::filesystem::path& root,
    const std::filesystem::path& candidate) {
    auto root_part = root.begin();
    auto candidate_part = candidate.begin();
    while (root_part != root.end() && candidate_part != candidate.end()) {
        if (*root_part != *candidate_part) {
            return false;
        }
        ++root_part;
        ++candidate_part;
    }
    return root_part == root.end();
}

[[nodiscard]] std::string content_type_for(
    const std::filesystem::path& path) {
    const std::string extension = path.extension().string();
    if (extension == ".html") {
        return "text/html; charset=utf-8";
    }
    if (extension == ".css") {
        return "text/css; charset=utf-8";
    }
    if (extension == ".js" || extension == ".mjs") {
        return "text/javascript; charset=utf-8";
    }
    if (extension == ".json") {
        return "application/json; charset=utf-8";
    }
    if (extension == ".svg") {
        return "image/svg+xml";
    }
    if (extension == ".png") {
        return "image/png";
    }
    if (extension == ".webp") {
        return "image/webp";
    }
    if (extension == ".ico") {
        return "image/x-icon";
    }
    if (extension == ".woff2") {
        return "font/woff2";
    }
    return "application/octet-stream";
}

void send_static(
    const httplib::Request& request,
    httplib::Response& response,
    const std::filesystem::path& canonical_web_root) {
    const std::optional<std::string> decoded = percent_decode(request.path);
    if (!decoded || decoded->find('\\') != std::string::npos) {
        send_json(response, 404, Json{{"error", "not found"}});
        return;
    }

    std::string relative_text = *decoded;
    while (!relative_text.empty() && relative_text.front() == '/') {
        relative_text.erase(relative_text.begin());
    }
    if (relative_text.empty()) {
        relative_text = "index.html";
    }

    std::filesystem::path relative;
    std::size_t offset = 0;
    while (offset <= relative_text.size()) {
        const std::size_t separator = relative_text.find('/', offset);
        const std::string_view component(
            relative_text.data() + offset,
            (separator == std::string::npos ? relative_text.size() : separator) -
                offset);
        if (component == ".." ||
            (!component.empty() && component.front() == '.')) {
            send_json(response, 404, Json{{"error", "not found"}});
            return;
        }
        if (!component.empty()) {
            relative /= std::string(component);
        }
        if (separator == std::string::npos) {
            break;
        }
        offset = separator + 1;
    }

    std::error_code error;
    const std::filesystem::path candidate = std::filesystem::weakly_canonical(
        canonical_web_root / relative, error);
    if (error || !path_is_within(canonical_web_root, candidate) ||
        !std::filesystem::is_regular_file(candidate, error) || error) {
        send_json(response, 404, Json{{"error", "not found"}});
        return;
    }

    std::ifstream input(candidate, std::ios::binary);
    if (!input) {
        send_json(
            response,
            500,
            Json{{"error", "unable to read static asset"}});
        return;
    }
    std::string content{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (!input.good() && !input.eof()) {
        send_json(
            response,
            500,
            Json{{"error", "unable to read static asset"}});
        return;
    }

    response.status = 200;
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Content-Type-Options", "nosniff");
    response.set_content(content, content_type_for(candidate));
}

template <typename Function>
void guarded_json(
    httplib::Response& response,
    Function&& function) {
    try {
        send_json(response, 200, std::forward<Function>(function)());
    } catch (const std::exception& error) {
        send_json(
            response,
            500,
            Json{{"error", "internal simulation error"},
                 {"detail", error.what()}});
    }
}

[[nodiscard]] bool parse_json_object(
    const httplib::Request& request,
    httplib::Response& response,
    Json& payload) {
    if (request.body.empty() || request.body.size() > kMaxRequestBytes) {
        send_json(response, 400, Json{{"error", "invalid request size"}});
        return false;
    }
    try {
        payload = Json::parse(request.body);
    } catch (const Json::exception&) {
        send_json(response, 400, Json{{"error", "invalid JSON"}});
        return false;
    }
    if (!payload.is_object()) {
        send_json(response, 400, Json{{"error", "JSON body must be an object"}});
        return false;
    }
    return true;
}

[[nodiscard]] bool has_only_fields(
    const Json& object,
    std::initializer_list<std::string_view> allowed,
    httplib::Response& response) {
    for (const auto& [key, unused] : object.items()) {
        (void)unused;
        bool found = false;
        for (const std::string_view candidate : allowed) {
            if (key == candidate) {
                found = true;
                break;
            }
        }
        if (!found) {
            send_json(
                response,
                400,
                Json{{"error", "unknown field"}, {"field", key}});
            return false;
        }
    }
    return true;
}

template <typename Function>
void guarded_agent_json(
    httplib::Response& response,
    Function&& function) {
    try {
        send_json(response, 200, std::forward<Function>(function)());
    } catch (const std::invalid_argument& error) {
        send_json(response, 400, Json{{"error", error.what()}});
    } catch (const std::logic_error& error) {
        send_json(response, 409, Json{{"error", error.what()}});
    } catch (const std::exception& error) {
        send_json(
            response,
            500,
            Json{{"error", "internal simulation error"},
                 {"detail", error.what()}});
    }
}

[[nodiscard]] int run_server(
    const Options& options,
    droid::PlaygroundSession& environment,
    droid::ConstructionSession& construction, droid::LightSession& light) {
    std::error_code error;
    const std::filesystem::path canonical_web_root =
        std::filesystem::canonical(options.web_root, error);
    if (error ||
        !std::filesystem::is_directory(canonical_web_root, error) || error) {
        throw std::runtime_error(
            "web root not found: " + options.web_root.string());
    }

    httplib::Server server;
    server.set_payload_max_length(kMaxRequestBytes);
    server.set_exception_handler(
        [](const httplib::Request&,
           httplib::Response& response,
           std::exception_ptr error) {
            try {
                if (error) {
                    std::rethrow_exception(error);
                }
            } catch (const Json::exception&) {
                send_json(response, 400, Json{{"error", "invalid JSON"}});
                return;
            } catch (const std::exception&) {
                send_json(
                    response,
                    500,
                    Json{{"error", "internal simulation error"}});
                return;
            } catch (...) {
                send_json(
                    response,
                    500,
                    Json{{"error", "internal simulation error"}});
                return;
            }
            send_json(
                response,
                500,
                Json{{"error", "internal simulation error"}});
        });

    server.Options(R"(/.*)", [](const httplib::Request&, auto& response) {
        response.status = 204;
        set_api_headers(response);
    });
    server.Get("/healthz", [&environment](const auto&, auto& response) {
        const Json health = environment.health();
        send_json(response, health.value("status", "error") == "ok" ? 200 : 503, health);
    });
    server.Get("/api/state", [&environment](const auto&, auto& response) {
        guarded_json(response, [&environment] {
            return environment.state();
        });
    });
    server.Get("/api/catalog", [&environment](const auto&, auto& response) {
        guarded_json(response, [&environment] { return environment.catalog(); });
    });
    server.Get("/api/agent/spec", [&environment](const auto&, auto& response) {
        guarded_json(response, [&environment] { return environment.spec(); });
    });
    server.Get("/api/agent/trace", [&environment](const auto&, auto& response) {
        guarded_agent_json(response, [&environment] {
            return environment.trace();
        });
    });
    server.Get("/api/construction/state", [&construction](const auto&, auto& response) {
        guarded_json(response, [&construction] { return construction.state(); });
    });
    server.Get("/api/construction/spec", [](const auto&, auto& response) {
        guarded_json(response, [] { return droid::ConstructionSession::specification(); });
    });
    server.Get("/api/light/state", [&light](const auto&, auto& response) {
        guarded_json(response, [&light] { return light.state(); });
    });
    server.Get("/api/light/spec", [](const auto&, auto& response) {
        guarded_json(response, [] { return droid::LightSession::specification(); });
    });
    server.Get(R"(/api/.*)", [](const auto&, auto& response) {
        send_json(response, 404, Json{{"error", "not found"}});
    });
    server.Post(
        "/api/control",
        [&environment](const httplib::Request& request, auto& response) {
            Json payload;
            if (!parse_json_object(request, response, payload)) {
                return;
            }
            if (!has_only_fields(payload, {"action"}, response)) {
                return;
            }

            const std::string action =
                payload.contains("action") &&
                    payload["action"].is_string()
                ? payload["action"].get<std::string>()
                : std::string{};
            guarded_agent_json(response, [&environment, &action] {
                return environment.legacy_control(action);
            });
        });
    server.Post(
        "/api/playground/control",
        [&environment](const httplib::Request& request, auto& response) {
            Json payload;
            if (!parse_json_object(request, response, payload) ||
                !has_only_fields(payload, {"action", "side", "variant", "strategy"}, response)) {
                return;
            }
            if (!payload.contains("action") || !payload.at("action").is_string()) {
                send_json(response, 400, Json{{"error", "action must be a string"}});
                return;
            }
            const std::string action = payload.at("action").get<std::string>();
            std::string side;
            if (payload.contains("side")) {
                if (!payload.at("side").is_string() ||
                    (payload.at("side") != "left" && payload.at("side") != "right")) {
                    send_json(response, 400, Json{{"error", "side must be left or right"}});
                    return;
                }
                side = payload.at("side").get<std::string>();
            }
            std::string variant;
            std::string strategy;
            for (const std::string field : {"variant", "strategy"}) {
                if (payload.contains(field) && (action != "body" || !payload.at(field).is_string() || payload.at(field).get<std::string>().empty())) {
                    send_json(response, 400, Json{{"error", field + " must be a nonempty string and is accepted only with body"}});
                    return;
                }
            }
            if (payload.contains("variant")) variant = payload.at("variant").get<std::string>();
            if (payload.contains("strategy")) strategy = payload.at("strategy").get<std::string>();
            guarded_agent_json(response, [&environment, &action, &side, &variant, &strategy] {
                return environment.control(action, side, variant, strategy);
            });
        });
    server.Post(
        "/api/agent/reset",
        [&environment](const httplib::Request& request, auto& response) {
            Json payload;
            if (!parse_json_object(request, response, payload) ||
                !has_only_fields(
                    payload,
                    {"assembly_id", "seed", "record", "episode_profile"},
                    response)) {
                return;
            }

            std::string assembly_id{"demo_rover_v0"};
            std::uint64_t seed = 0;
            bool record = false;
            std::string episode_profile(droid::kDefaultEpisodeProfile);
            try {
                if (payload.contains("assembly_id")) {
                    if (!payload.at("assembly_id").is_string()) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "assembly_id must be a string"}});
                        return;
                    }
                    assembly_id = payload.at("assembly_id").get<std::string>();
                }
                if (payload.contains("seed")) {
                    if (!payload.at("seed").is_number_unsigned()) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "seed must be an unsigned integer"}});
                        return;
                    }
                    seed = payload.at("seed").get<std::uint64_t>();
                }
                if (payload.contains("record")) {
                    if (!payload.at("record").is_boolean()) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "record must be a boolean"}});
                        return;
                    }
                    record = payload.at("record").get<bool>();
                }
                if (payload.contains("episode_profile")) {
                    if (!payload.at("episode_profile").is_string()) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "episode_profile must be a string"}});
                        return;
                    }
                    episode_profile =
                        payload.at("episode_profile").get<std::string>();
                }
            } catch (const Json::exception&) {
                send_json(
                    response,
                    400,
                    Json{{"error", "invalid reset field value"}});
                return;
            }

            guarded_agent_json(
                response,
                [&environment,
                 &assembly_id,
                 seed,
                 record,
                 &episode_profile] {
                    return environment.agent_reset(
                        assembly_id,
                        seed,
                        record,
                        episode_profile);
                });
        });
    server.Post(
        "/api/agent/step",
        [&environment](const httplib::Request& request, auto& response) {
            Json payload;
            if (!parse_json_object(request, response, payload) ||
                !has_only_fields(
                    payload, {"actions", "control_dt_s"}, response)) {
                return;
            }
            if (!payload.contains("actions") ||
                !payload.at("actions").is_object()) {
                send_json(
                    response,
                    400,
                    Json{{"error", "actions must be an object"}});
                return;
            }

            std::map<std::string, double> actions;
            try {
                for (const auto& [motor_id, raw_effort] :
                     payload.at("actions").items()) {
                    if (!raw_effort.is_number()) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "every action must be a number"},
                                 {"motor_id", motor_id}});
                        return;
                    }
                    const double effort = raw_effort.template get<double>();
                    if (!std::isfinite(effort)) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "every action must be finite"},
                                 {"motor_id", motor_id}});
                        return;
                    }
                    actions.emplace(motor_id, effort);
                }
            } catch (const Json::exception&) {
                send_json(
                    response,
                    400,
                    Json{{"error", "invalid action value"}});
                return;
            }

            double control_dt_s = 0.02;
            try {
                if (payload.contains("control_dt_s")) {
                    if (!payload.at("control_dt_s").is_number()) {
                        send_json(
                            response,
                            400,
                            Json{{"error", "control_dt_s must be a number"}});
                        return;
                    }
                    control_dt_s = payload.at("control_dt_s").get<double>();
                    if (!std::isfinite(control_dt_s) || control_dt_s <= 0.0) {
                        send_json(
                            response,
                            400,
                            Json{{"error",
                                  "control_dt_s must be finite and greater than zero"}});
                        return;
                    }
                }
            } catch (const Json::exception&) {
                send_json(
                    response,
                    400,
                    Json{{"error", "invalid control_dt_s value"}});
                return;
            }

            guarded_agent_json(response, [&environment, &actions, control_dt_s] {
                return environment.agent_step(actions, control_dt_s);
            });
        });
    server.Post("/api/construction/control", [&construction](const httplib::Request& request, auto& response) {
        Json payload;
        if (!parse_json_object(request, response, payload)) return;
        guarded_agent_json(response, [&construction, &payload] { return construction.control(payload); });
    });
    server.Post("/api/light/control", [&light](const httplib::Request& request, auto& response) {
        Json payload;
        if (!parse_json_object(request, response, payload)) return;
        guarded_agent_json(response, [&light, &payload] { return light.control(payload); });
    });
    server.Post(R"(/.*)", [](const auto&, auto& response) {
        send_json(response, 404, Json{{"error", "not found"}});
    });
    server.Get(
        R"(/.*)",
        [&canonical_web_root](const httplib::Request& request, auto& response) {
            send_static(request, response, canonical_web_root);
        });

    server.set_logger([](const auto& request, const auto& response) {
        std::clog << request.remote_addr << " - \"" << request.method << ' '
                  << request.path << "\" " << response.status << '\n';
    });

    int bound_port = options.port;
    const bool bound = options.port == 0
        ? (bound_port = server.bind_to_any_port(options.host)) > 0
        : server.bind_to_port(options.host, options.port);
    if (!bound) {
        throw std::runtime_error(
            "unable to bind HTTP server on " + options.host + ":" +
            std::to_string(options.port));
    }

    g_shutdown_requested = 0;
    std::signal(SIGINT, request_shutdown);
    std::signal(SIGTERM, request_shutdown);
    std::cout << "Droid Blocks running on http://" << options.host << ':'
              << bound_port << std::endl;

    std::atomic_bool listener_done{false};
    std::atomic_bool listener_ok{false};
    std::thread listener([&] {
        listener_ok.store(server.listen_after_bind(), std::memory_order_release);
        listener_done.store(true, std::memory_order_release);
    });

    while (!listener_done.load(std::memory_order_acquire) &&
           g_shutdown_requested == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (g_shutdown_requested != 0) {
        server.stop();
    }
    listener.join();
    return listener_ok.load(std::memory_order_acquire) ||
            g_shutdown_requested != 0
        ? 0
        : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.help) {
            print_usage(std::cout, argc > 0 ? argv[0] : "droid-blocks");
            return 0;
        }

        if (options.probe_steps) {
            droid::DroidSimulation simulation(options.model, false);
            simulation.reset();
            simulation.advance_steps(*options.probe_steps);
            std::cout << simulation.state().dump() << '\n';
            return 0;
        }

        droid::DroidEnvironment environment(options.model, true);
        droid::PlaygroundSession playground(
            environment, options.model,
            "artifacts/light-search-policy-v3-seed0.json", true);
        droid::ConstructionSession construction(true);
        droid::LightSession light(true);
        return run_server(options, playground, construction, light);
    } catch (const std::invalid_argument& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << "Try --help for usage.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
