// Local native MuJoCo viewer. Physics, inference, and rendering share one thread.
#include "humanoid/simulation.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr double kHorizonSeconds = 30.0;
constexpr std::size_t kHistoryLength = 160;
constexpr std::size_t kMaxCommands = 32;
constexpr std::size_t kMaxRequestBytes = 2048;
constexpr int kRenderWidth = 640;
constexpr int kRenderHeight = 426;
constexpr int kMaxCatchupSteps = 12;
std::atomic<bool> signal_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "Signal notification must use a lock-free atomic");

extern "C" void request_stop(int) { signal_stop.store(true, std::memory_order_relaxed); }

struct Options {
  std::string host = "0.0.0.0";
  int port = 8080;
  std::string device = "cuda";
  std::filesystem::path root = std::filesystem::current_path();
  std::string asset_root;
};

Options parse_options(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      std::cout << "humanoid-server [--host 0.0.0.0] [--port 8080] "
                   "[--device cpu|cuda] [--root PROJECT] [--asset-root ASSETS]\n";
      std::exit(0);
    }
    if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + option);
    const std::string value = argv[++i];
    if (option == "--host") result.host = value;
    else if (option == "--device") result.device = value;
    else if (option == "--root") result.root = value;
    else if (option == "--asset-root") result.asset_root = value;
    else if (option == "--port") {
      std::size_t consumed = 0;
      result.port = std::stoi(value, &consumed);
      if (consumed != value.size() || result.port < 1 || result.port > 65535)
        throw std::invalid_argument("Port must be an integer from 1 to 65535");
    } else throw std::invalid_argument("Unknown option: " + option);
  }
  if (result.device != "cpu" && result.device != "cuda")
    throw std::invalid_argument("Device must be cpu or cuda");
  if (result.host.empty()) throw std::invalid_argument("Host cannot be empty");
  result.root = std::filesystem::absolute(result.root);
  if (!std::filesystem::is_regular_file(result.root / "web" / "index.html"))
    throw std::invalid_argument("Project root has no web/index.html: " + result.root.string());
  return result;
}

struct Command {
  std::string action;
  std::optional<double> command_x;
  std::optional<std::string> policy;
  std::optional<std::uint64_t> seed;
};

std::filesystem::path trained_policy_path(const Options& options, const std::string& kind = "trained") {
  return options.root / "runs" / (kind == "distilled" ? "distilled-policy.pt" : "trained-policy.pt");
}

bool trained_policy_available(const Options& options, const std::string& kind = "trained") {
  std::error_code error;
  const auto path = trained_policy_path(options, kind);
  return std::filesystem::is_regular_file(path, error) && !error &&
         std::filesystem::file_size(path, error) > 0 && !error;
}

Command parse_command(const std::string& body) {
  if (body.empty() || body.size() > kMaxRequestBytes)
    throw std::invalid_argument("Expected a small JSON request");
  const auto message = Json::parse(body);
  if (!message.is_object()) throw std::invalid_argument("Expected a JSON object");
  for (const auto& item : message.items()) {
    if (item.key() != "action" && item.key() != "command_x" &&
        item.key() != "policy" && item.key() != "seed")
      throw std::invalid_argument("Unknown control field: " + item.key());
  }
  if (!message.contains("action") || !message["action"].is_string())
    throw std::invalid_argument("Missing action");
  Command result;
  result.action = message["action"].get<std::string>();
  if (result.action != "play" && result.action != "pause" && result.action != "reset")
    throw std::invalid_argument("Unknown action");
  if (result.action != "reset" && message.size() != 1)
    throw std::invalid_argument("Only reset accepts speed, policy, or seed");
  if (message.contains("command_x")) {
    if (!message["command_x"].is_number())
      throw std::invalid_argument("Speed must be a number from 0 to 0.8 m/s");
    const double speed = message["command_x"].get<double>();
    if (!std::isfinite(speed) || speed < 0.0 || speed > 0.8)
      throw std::invalid_argument("Speed must be a number from 0 to 0.8 m/s");
    result.command_x = speed;
  }
  if (message.contains("policy")) {
    if (!message["policy"].is_string()) throw std::invalid_argument("Unknown policy");
    const auto policy = message["policy"].get<std::string>();
    if (policy != "pretrained" && policy != "zero" && policy != "trained" && policy != "distilled")
      throw std::invalid_argument("Unknown policy");
    result.policy = policy;
  }
  if (message.contains("seed")) {
    if (!message["seed"].is_number_integer() || message["seed"].get<double>() < 0 ||
        message["seed"].get<double>() > 100000)
      throw std::invalid_argument("Seed must be an integer from 0 to 100000");
    result.seed = message["seed"].get<std::uint64_t>();
  }
  return result;
}

class Session {
 public:
  enum class EnqueueResult { accepted, full, unavailable };

  explicit Session(const Options& options)
      : worker_([this, options] { loop(options); }) {}
  ~Session() { stop(); }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  Json snapshot(bool history = true) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = snapshot_;
    if (!history) result.erase("history");
    return result;
  }

  Json health() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Json result = {{"ready", snapshot_.value("ready", false)},
                   {"playing", snapshot_.value("playing", false)}};
    for (const auto* field : {"time", "policy", "physics_device", "inference_device"}) {
      if (snapshot_.contains(field)) result[field] = snapshot_[field];
    }
    if (snapshot_.contains("error")) {
      const auto error = snapshot_["error"].get<std::string>();
      result["error"] = error.size() > 1024 ? error.substr(0, 1024) + "..." : error;
    }
    return result;
  }

  std::string frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jpeg_;
  }

  EnqueueResult enqueue(Command command) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_ || stopping_) return EnqueueResult::unavailable;
    if (commands_.size() >= kMaxCommands) return EnqueueResult::full;
    commands_.push_back(std::move(command));
    wake_.notify_one();
    return EnqueueResult::accepted;
  }

 private:
  void loop(const Options& options) noexcept {
    try {
      // Construction and destruction stay here so OpenGL context ownership never
      // crosses into the HTTP request threads, including when startup fails.
      auto sim = std::make_unique<humanoid::WalkingSimulation>(
          options.asset_root, options.device, kRenderWidth, kRenderHeight);
      sim->reset(0, 0.5, "pretrained");
      bool playing = true;
      double command_x = 0.5;
      std::string policy = "pretrained";
      std::uint64_t seed = 0;
      auto provenance = sim->provenance();
      std::string control_error;
      std::deque<Json> history;
      auto next_render = Clock::time_point::min();
      const auto tick = std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(sim->control_dt()));
      auto next_step = Clock::now();
      auto performance_started = next_step;
      double performance_sim_start = 0, realtime_factor = 0, render_fps = 0;
      double render_frame_ms = 0, last_rendered_time = -1;
      std::size_t performance_frames = 0;
      bool frame_dirty = true;
      double last_history_time = -1.0;
      while (true) {
        std::deque<Command> pending;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stopping_) break;
          pending.swap(commands_);
        }
        auto state = sim->state();
        for (const auto& command : pending) {
          if (command.action == "pause") {
            playing = false;
            frame_dirty = state.value("time", 0.0) != last_rendered_time;
            realtime_factor = 0;
          }
          else if (command.action == "play") {
            // A completed/fallen trial needs an explicit reset. Play never
            // advances beyond the horizon or silently starts another trial.
            playing = !state.value("fall", false) && state.value("time", 0.0) + 1e-9 < kHorizonSeconds;
            next_step = Clock::now();
            performance_started = next_step;
            performance_sim_start = state.value("time", 0.0);
            performance_frames = 0;
          } else if (command.action == "reset") {
            const double requested_speed = command.command_x.value_or(command_x);
            const auto requested_policy = command.policy.value_or(policy);
            const auto requested_seed = command.seed.value_or(seed);
            if (requested_policy == "trained" || requested_policy == "distilled") {
              try {
                if (!trained_policy_available(options, requested_policy))
                  throw std::runtime_error("The selected local checkpoint is not available yet");
                // The only accepted checkpoint is the project's fixed local
                // artifact. Load in isolation so a bad export cannot destroy
                // the current simulation. All GL/model lifetimes stay here.
                auto replacement = std::make_unique<humanoid::WalkingSimulation>(
                    options.asset_root, options.device, kRenderWidth, kRenderHeight, false);
                replacement->load_policy(trained_policy_path(options, requested_policy).string());
                auto replacement_state = replacement->reset(requested_seed, requested_speed, "trained");
                auto replacement_provenance = replacement->provenance();
                sim.swap(replacement);
                state = std::move(replacement_state);
                provenance = std::move(replacement_provenance);
              } catch (const std::exception& error) {
                playing = false;
                control_error = std::string("Could not load our trained policy: ") + error.what();
                std::cerr << control_error << std::endl;
                continue;
              }
            } else {
              state = sim->reset(requested_seed, requested_speed, requested_policy);
              provenance = sim->provenance();
            }
            command_x = requested_speed;
            policy = requested_policy;
            seed = requested_seed;
            history.clear();
            last_history_time = -1.0;
            playing = true;
            next_render = Clock::time_point::min();
            next_step = Clock::now();
            performance_started = next_step;
            performance_sim_start = state.value("time", 0.0);
            performance_frames = 0;
            realtime_factor = render_fps = 0;
            frame_dirty = true;
          }
          control_error.clear();
        }
        int catchup_steps = 0;
        // Keep the 20ms physics clock independent of frame cost. After a frame,
        // catch up with actual fixed control steps instead of slowing physics
        // to one step per rendered image. Bound work so commands stay responsive.
        while (playing && Clock::now() >= next_step && catchup_steps < kMaxCatchupSteps) {
          state = sim->step();
          next_step += tick;
          ++catchup_steps;
          if (state.value("fall", false) || state.value("time", 0.0) + 1e-9 >= kHorizonSeconds) {
            playing = false;
            frame_dirty = true;
          }
        }
        const double simulated_time = state.value("time", 0.0);
        if (state.value("fall", false) || simulated_time + 1e-9 >= kHorizonSeconds)
          playing = false;
        const double lag_ms = playing ? std::max(0.0, std::chrono::duration<double, std::milli>(
            Clock::now() - next_step).count()) : 0.0;
        // Catch-up takes priority over another frame when the renderer was slow.
        // A paused unchanged state serves the cached JPEG without any GL work.
        if (frame_dirty || (playing && Clock::now() >= next_render && lag_ms < 2 * sim->control_dt() * 1000)) {
          const auto render_started = Clock::now();
          const auto bytes = sim->render_jpeg(82);
          if (bytes.empty()) throw std::runtime_error("Renderer returned an empty JPEG");
          const auto render_finished = Clock::now();
          render_frame_ms = std::chrono::duration<double, std::milli>(render_finished - render_started).count();
          ++performance_frames;
          state = sim->state();  // include the newly measured rendering timings
          state["policy"] = policy;  // retain the public PPO/imitation distinction in history
          // Store telemetry before adding the response envelope. Pause does not
          // fabricate additional contact history while simulated time is frozen.
          if (simulated_time != last_history_time) {
            history.push_back(state);
            if (history.size() > kHistoryLength) history.pop_front();
            last_history_time = simulated_time;
          }
          {
            std::lock_guard<std::mutex> lock(mutex_);
            jpeg_.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
          }
          next_render = render_finished + std::chrono::milliseconds(67);
          last_rendered_time = simulated_time;
          frame_dirty = false;
        }
        const double performance_seconds = std::chrono::duration<double>(Clock::now() - performance_started).count();
        if (performance_seconds >= 1.0) {
          realtime_factor = playing ? (simulated_time - performance_sim_start) / performance_seconds : 0.0;
          render_fps = performance_frames / performance_seconds;
          performance_started = Clock::now();
          performance_sim_start = simulated_time;
          performance_frames = 0;
        }
        state["ready"] = true;
        state["playing"] = playing;
        state["command_x"] = command_x;
        state["policy"] = policy;
        state["seed"] = seed;
        state["provenance"] = provenance;
        state["horizon_s"] = kHorizonSeconds;
        state["history"] = history;
        state["trained_policy_available"] = trained_policy_available(options);
        state["distilled_policy_available"] = trained_policy_available(options, "distilled");
        state["control_error"] = control_error;
        state["viewer_performance"] = {
            {"realtime_factor", realtime_factor}, {"render_fps", render_fps},
            {"render_frame_ms", render_frame_ms}, {"catchup_steps", catchup_steps},
            {"lag_ms", lag_ms}, {"render_width", kRenderWidth}, {"render_height", kRenderHeight}};
        {
          std::unique_lock<std::mutex> lock(mutex_);
          snapshot_ = std::move(state);
          // Keep controller pacing independent of HTTP request frequency.
          const auto wake_at = playing ? next_step : Clock::now() + std::chrono::milliseconds(250);
          wake_.wait_until(lock, wake_at, [this] { return stopping_ || !commands_.empty(); });
        }
      }
      sim->close();
    } catch (const std::exception& error) {
      fail(error.what());
    } catch (...) {
      fail("Unknown native simulation failure");
    }
  }

  void fail(const std::string& error) {
    std::cerr << "Humanoid simulation failed: " << error << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
    snapshot_ = {{"ready", false}, {"playing", false}, {"error", error}};
    jpeg_.clear();
  }

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Command> commands_;
  Json snapshot_ = {{"ready", false}, {"playing", false}};
  std::string jpeg_;
  bool stopping_ = false;
  bool failed_ = false;
  // Must be last: the worker accesses all preceding members immediately.
  std::thread worker_;
};

void send_json(httplib::Response& response, int code, const Json& value) {
  response.status = code;
  response.set_content(value.dump(), "application/json; charset=utf-8");
}

std::string read_file(const std::filesystem::path& path, std::uintmax_t maximum) {
  if (std::filesystem::file_size(path) > maximum) throw std::runtime_error("File exceeds response limit");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open " + path.filename().string());
  std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  if (input.bad()) throw std::runtime_error("Cannot read " + path.filename().string());
  return data;
}

bool json_content_type(std::string type) {
  type = type.substr(0, type.find(';'));
  const auto start = type.find_first_not_of(" \t");
  if (start == std::string::npos) return false;
  const auto end = type.find_last_not_of(" \t");
  type = type.substr(start, end - start + 1);
  std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return std::tolower(c); });
  return type == "application/json";
}

int serve(const Options& options) {
  httplib::Server server;
  server.set_payload_max_length(kMaxRequestBytes);
  server.set_read_timeout(5);
  server.set_write_timeout(5);
  server.set_keep_alive_timeout(2);
  server.set_default_headers({
      {"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"},
      {"X-Robots-Tag", "noindex, nofollow"},
      {"Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
       "style-src-elem 'self' 'unsafe-inline'; style-src-attr 'none'; img-src 'self' blob:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"}});
  server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
    if (response.body.empty()) send_json(response, response.status, {{"error", "Request failed"}});
  });
  server.set_exception_handler([](const httplib::Request&, httplib::Response& response,
                                  std::exception_ptr exception) {
    try { if (exception) std::rethrow_exception(exception); }
    catch (const std::exception& error) { std::cerr << "HTTP request failed: " << error.what() << '\n'; }
    catch (...) { std::cerr << "HTTP request failed with unknown exception\n"; }
    send_json(response, 500, {{"error", "Viewer request failed; inspect container logs"}});
  });
  if (!server.bind_to_port(options.host, options.port))
    throw std::runtime_error("Cannot bind " + options.host + ":" + std::to_string(options.port));

  Session session(options);
  server.Get("/healthz", [&](const httplib::Request&, httplib::Response& response) {
    const auto state = session.health();
    send_json(response, state.value("ready", false) ? 200 : 503, state);
  });
  server.Get("/api/state", [&](const httplib::Request&, httplib::Response& response) {
    send_json(response, 200, session.snapshot());
  });
  server.Get("/frame\\.jpg", [&](const httplib::Request&, httplib::Response& response) {
    auto bytes = session.frame();
    response.status = bytes.empty() ? 503 : 200;
    response.set_content(std::move(bytes), "image/jpeg");
  });
  const auto evaluation_route = [&](const std::string& route, const std::string& filename) {
    server.Get(route, [&, filename](const httplib::Request&, httplib::Response& response) {
      const auto report = options.root / "runs" / filename;
      send_json(response, 200, std::filesystem::exists(report)
          ? Json::parse(read_file(report, 32 * 1024 * 1024))
          : Json{{"status", "No completed evaluation yet"}});
    });
  };
  evaluation_route("/api/evaluation", "evaluation.json");
  evaluation_route("/api/evaluation-trained", "evaluation-trained.json");
  evaluation_route("/api/evaluation-distilled", "evaluation-distilled.json");
  const auto static_file = [&](const std::string& route, const std::string& name, const std::string& mime) {
    server.Get(route, [&, name, mime](const httplib::Request&, httplib::Response& response) {
      response.set_content(read_file(options.root / "web" / name, 1024 * 1024), mime);
    });
  };
  static_file("/", "index.html", "text/html; charset=utf-8");
  static_file("/app\\.js", "app.js", "text/javascript; charset=utf-8");
  static_file("/style\\.css", "style.css", "text/css; charset=utf-8");
  server.Post("/api/control", [&](const httplib::Request& request, httplib::Response& response) {
    if (request.has_header("Origin") &&
        request.get_header_value("Origin") != "http://" + request.get_header_value("Host")) {
      send_json(response, 403, {{"error", "Origin mismatch"}});
      return;
    }
    if (!json_content_type(request.get_header_value("Content-Type"))) {
      send_json(response, 415, {{"error", "Expected application/json"}});
      return;
    }
    try {
      auto command = parse_command(request.body);
      if (command.policy && (*command.policy == "trained" || *command.policy == "distilled")
          && !trained_policy_available(options, *command.policy)) {
        send_json(response, 409, {{"error", "The selected local checkpoint is not available yet"}});
        return;
      }
      const auto result = session.enqueue(std::move(command));
      if (result == Session::EnqueueResult::full)
        send_json(response, 429, {{"error", "Control queue is full; retry shortly"}});
      else if (result == Session::EnqueueResult::unavailable)
        send_json(response, 503, {{"error", "Simulation is unavailable; inspect its reported error"}});
      else send_json(response, 202, {{"accepted", true}});
    } catch (const Json::exception&) {
      send_json(response, 400, {{"error", "Invalid control JSON"}});
    } catch (const std::invalid_argument& error) {
      send_json(response, 400, {{"error", error.what()}});
    }
  });

  // The signal handler only flips a lock-free flag. stop() and thread joins
  // happen in ordinary threads, avoiding signal-unsafe locks or allocation.
  std::atomic<bool> finished{false};
  std::thread shutdown([&] {
    while (!finished.load(std::memory_order_relaxed)) {
      if (signal_stop.load(std::memory_order_relaxed) && server.is_running()) {
        server.stop();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  std::cout << "Native humanoid viewer on " << options.host << ':' << options.port
            << "; policy device=" << options.device << std::endl;
  bool listened = false;
  try { listened = server.listen_after_bind(); }
  catch (...) {
    finished.store(true, std::memory_order_relaxed);
    shutdown.join();
    throw;
  }
  finished.store(true, std::memory_order_relaxed);
  shutdown.join();
  session.stop();
  return listened ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif
    return serve(options);
  } catch (const std::exception& error) {
    std::cerr << "humanoid-server: " << error.what() << std::endl;
    return 1;
  }
}
