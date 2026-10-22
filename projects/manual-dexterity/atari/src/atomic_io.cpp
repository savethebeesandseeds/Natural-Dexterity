#include "atari/atomic_io.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace atari {
namespace {
bool retryable(const std::error_code& error) {
    // Docker Desktop's Windows bind mount exposes sharing violations as EACCES.
    // EPERM can have the same transient origin; a real permissions problem will
    // remain bounded by the deadline. Missing files, EXDEV, ENOSPC and EROFS are
    // not transient sharing errors and therefore fail immediately.
    return error == std::errc::permission_denied || error == std::errc::operation_not_permitted ||
        error == std::errc::device_or_resource_busy || error == std::errc::text_file_busy ||
        error == std::errc::resource_unavailable_try_again || error == std::errc::interrupted;
}
void validate(const AtomicReplacePolicy& policy) {
    using namespace std::chrono;
    if (policy.timeout <= milliseconds::zero() || policy.timeout > minutes(1) ||
        policy.initial_delay <= milliseconds::zero() || policy.maximum_delay < policy.initial_delay ||
        policy.maximum_delay > minutes(1))
        throw std::invalid_argument("Atomic replacement requires a 1..60000 ms deadline and positive bounded retry delays");
}
} // namespace

AtomicReplaceResult atomic_io_detail::replace(const std::filesystem::path& temporary,
                                              const std::filesystem::path& destination,
                                              const AtomicReplacePolicy& policy, const Runtime& runtime) {
    validate(policy);
    if (temporary.empty() || destination.empty() || temporary.lexically_normal() == destination.lexically_normal())
        throw std::invalid_argument("Atomic replacement requires distinct nonempty temporary and destination paths");
    if (!runtime.rename || !runtime.now || !runtime.sleep) throw std::invalid_argument("Atomic replacement runtime is incomplete");
    const auto start = runtime.now(), deadline = start + policy.timeout;
    auto delay = policy.initial_delay;
    AtomicReplaceResult result;
    auto fail = [&](const std::error_code& error, bool timed_out) -> void {
        result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(runtime.now() - start);
        std::ostringstream message;
        message << "Atomic file replacement " << (timed_out ? "retry deadline exhausted" : "failed without retry")
                << " after " << result.attempts << " attempts in " << result.elapsed.count()
                << " ms (deadline " << policy.timeout.count() << " ms); destination and temporary preserved";
        throw std::filesystem::filesystem_error(message.str(), temporary, destination, error);
    };
    for (;;) {
        const auto error = runtime.rename(temporary, destination);
        ++result.attempts;
        if (!error) {
            result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(runtime.now() - start);
            return result;
        }
        if (!retryable(error)) fail(error, false);
        const auto now = runtime.now();
        if (now >= deadline) fail(error, true);
        // Ceil the remaining duration to milliseconds so a sub-ms remainder
        // cannot produce a zero-delay busy loop. Never issue a rename after the
        // retry deadline; the original destination survives unsuccessful tries.
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - now);
        runtime.sleep(std::min(delay, remaining));
        if (runtime.now() >= deadline) fail(error, true);
        delay = std::min(delay * 2, policy.maximum_delay);
    }
}

AtomicReplaceResult replace_file_atomically(const std::filesystem::path& temporary,
                                            const std::filesystem::path& destination,
                                            const AtomicReplacePolicy& policy) {
    atomic_io_detail::Runtime runtime;
    runtime.rename = [](const auto& from, const auto& to) {
        std::error_code error; std::filesystem::rename(from, to, error); return error;
    };
    runtime.now = [] { return std::chrono::steady_clock::now(); };
    runtime.sleep = [](auto duration) { std::this_thread::sleep_for(duration); };
    return atomic_io_detail::replace(temporary, destination, policy, runtime);
}

void write_text_atomically(const std::filesystem::path& destination, std::string_view contents,
                           const AtomicReplacePolicy& policy) {
    validate(policy);
    if (destination.empty()) throw std::invalid_argument("Atomic text destination is empty");
    if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
    auto temporary = destination; temporary += ".tmp";
    if (contents.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::length_error("Atomic text content exceeds stream size");
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        output.close();
        if (!output) throw std::filesystem::filesystem_error("Cannot completely write and close atomic temporary file",
            temporary, std::make_error_code(std::errc::io_error));
    }
    replace_file_atomically(temporary, destination, policy);
}
} // namespace atari
