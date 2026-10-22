#include "atari/atomic_io.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace std::chrono_literals;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
std::string read(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read test evidence " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
void write(const fs::path& path, const char* text) {
    std::ofstream output(path); output << text; output.close();
    require(bool(output), "Cannot write test evidence");
}
struct FakeClock {
    std::chrono::steady_clock::time_point time{};
    std::size_t sleeps = 0;
    atari::atomic_io_detail::Runtime runtime() {
        atari::atomic_io_detail::Runtime value;
        value.now = [this] { return time; };
        value.sleep = [this](auto duration) { require(duration.count() > 0, "Retry used a busy loop"); ++sleeps; time += duration; };
        return value;
    }
};
void cases(const fs::path& directory) {
    require(!fs::exists(directory) && fs::create_directories(directory), "Preserve existing atomic-I/O test evidence");
    const auto destination = directory / "progress.json", temporary = directory / "progress.json.tmp";
    atari::AtomicReplacePolicy policy{100ms, 2ms, 10ms};

    // Exercise an actual final rename after several different OS fault codes.
    // The old published value must remain readable through every failed try.
    write(destination, "old complete value"); write(temporary, "new complete value");
    FakeClock clock; auto runtime = clock.runtime(); std::size_t calls = 0;
    const std::vector<std::errc> transient{std::errc::permission_denied, std::errc::operation_not_permitted,
        std::errc::device_or_resource_busy, std::errc::text_file_busy,
        std::errc::resource_unavailable_try_again, std::errc::interrupted};
    runtime.rename = [&](const auto& from, const auto& to) {
        require(from == temporary && to == destination, "Retry changed rename paths");
        require(read(destination) == "old complete value" && read(temporary) == "new complete value", "Retry damaged an unpublished or existing file");
        if (calls < transient.size()) return std::make_error_code(transient[calls++]);
        ++calls; std::error_code error; fs::rename(from, to, error); return error;
    };
    const auto recovered = atari::atomic_io_detail::replace(temporary, destination, policy, runtime);
    require(recovered.attempts == transient.size() + 1 && clock.sleeps == transient.size(), "Transient error did not recover");
    require(read(destination) == "new complete value" && !fs::exists(temporary), "Successful retry did not publish exactly the prepared file");

    // A genuine nontransient OS error must not spend the retry budget, and its
    // original error code/paths must remain available to the native error report.
    for (auto error : {std::errc::read_only_file_system, std::errc::cross_device_link,
                       std::errc::no_space_on_device, std::errc::no_such_file_or_directory}) {
        write(destination, "preserved destination"); write(temporary, "preserved candidate");
        FakeClock permanent_clock; auto permanent_runtime = permanent_clock.runtime(); std::size_t attempts = 0;
        permanent_runtime.rename = [&](const auto&, const auto&) { ++attempts; return std::make_error_code(error); };
        bool failed = false;
        try { atari::atomic_io_detail::replace(temporary, destination, policy, permanent_runtime); }
        catch (const fs::filesystem_error& failure) {
            failed = true; require(failure.code() == error && failure.path1() == temporary && failure.path2() == destination,
                "Permanent failure lost OS error or file identities");
        }
        require(failed && attempts == 1 && permanent_clock.sleeps == 0, "Permanent error was retried or hidden");
        require(read(destination) == "preserved destination" && read(temporary) == "preserved candidate", "Permanent failure damaged evidence");
    }

    // An unreleased reader must lead to a bounded, visible failure. No rename
    // can occur at/after the deadline, and both old/candidate files survive.
    FakeClock timeout_clock; auto timeout_runtime = timeout_clock.runtime(); std::size_t timeout_attempts = 0;
    atari::AtomicReplacePolicy deadline_policy{25ms, 7ms, 20ms};
    timeout_runtime.rename = [&](const auto&, const auto&) {
        require(timeout_clock.time.time_since_epoch() < deadline_policy.timeout, "Rename occurred after retry deadline");
        ++timeout_attempts; return std::make_error_code(std::errc::permission_denied);
    };
    bool timed_out = false;
    try { atari::atomic_io_detail::replace(temporary, destination, deadline_policy, timeout_runtime); }
    catch (const fs::filesystem_error& failure) {
        timed_out = true; require(failure.code() == std::errc::permission_denied, "Timeout discarded the actual rename error");
        require(std::string(failure.what()).find("deadline exhausted") != std::string::npos, "Timeout is not visible in the diagnostic");
    }
    require(timed_out && timeout_attempts > 1 && timeout_clock.time.time_since_epoch() == deadline_policy.timeout,
        "Retry deadline was unbounded or no retries were attempted");
    require(read(destination) == "preserved destination" && read(temporary) == "preserved candidate", "Timeout removed or changed preserved files");

    // The deadline includes time spent in failed OS calls as well as backoff.
    FakeClock slow_clock; auto slow_runtime = slow_clock.runtime(); std::size_t slow_attempts = 0;
    slow_runtime.rename = [&](const auto&, const auto&) { ++slow_attempts; slow_clock.time += 30ms; return std::make_error_code(std::errc::device_or_resource_busy); };
    bool slow_failed = false;
    try { atari::atomic_io_detail::replace(temporary, destination, deadline_policy, slow_runtime); }
    catch (const fs::filesystem_error&) { slow_failed = true; }
    require(slow_failed && slow_attempts == 1 && slow_clock.sleeps == 0, "OS-call time was omitted from retry deadline");

    // Real I/O integration for first publication and replacement, including a
    // checkpoint-sized binary string with embedded NUL bytes and a JSON payload.
    const auto binary = directory / "checkpoint.pt";
    std::string bytes(64 * 1024, '\0'); bytes[0] = 'P'; bytes.back() = 'T';
    atari::write_text_atomically(binary, bytes);
    require(read(binary) == bytes, "Atomic write truncated binary data");
    atari::write_text_atomically(binary, "{\"status\":\"complete\"}\n");
    require(read(binary) == "{\"status\":\"complete\"}\n", "Atomic text replacement published wrong data");
    std::cout << "Atomic rename retry, permanent errors, bounded deadline, old-target preservation and real publication passed\n"
              << "Evidence directory: " << directory << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        // Real Windows bind-mount integration: root may hold DESTINATION through
        // FileShare.Read (without Delete), launch this command, then release it.
        // --replace never prepares or modifies either file before atomic rename.
        if (argc >= 2 && std::string(argv[1]) == "--replace") {
            if (argc != 4 && argc != 5) throw std::invalid_argument("Usage: atari-atomic-io-test --replace TEMPORARY DESTINATION [TIMEOUT_MS]");
            atari::AtomicReplacePolicy policy;
            if (argc == 5) {
                std::size_t consumed = 0; const std::string argument(argv[4]);
                policy.timeout = std::chrono::milliseconds(std::stoll(argument, &consumed));
                if (consumed != argument.size()) throw std::invalid_argument("Timeout must be integer milliseconds");
            }
            const auto result = atari::replace_file_atomically(argv[2], argv[3], policy);
            std::cout << "{\"status\":\"replaced\",\"attempts\":" << result.attempts
                      << ",\"elapsed_ms\":" << result.elapsed.count() << "}\n";
            return 0;
        }
        if (argc > 2) throw std::invalid_argument("Usage: atari-atomic-io-test [NEW_EVIDENCE_DIRECTORY]");
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        cases(argc == 2 ? fs::path(argv[1]) : fs::temp_directory_path() / ("atari-atomic-io-test-" + std::to_string(stamp)));
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
