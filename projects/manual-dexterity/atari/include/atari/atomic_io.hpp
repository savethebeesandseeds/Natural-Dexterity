#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <string_view>
#include <system_error>

namespace atari {

struct AtomicReplacePolicy {
    std::chrono::milliseconds timeout{10'000};
    std::chrono::milliseconds initial_delay{10};
    std::chrono::milliseconds maximum_delay{250};
};
struct AtomicReplaceResult {
    std::size_t attempts = 0;
    std::chrono::milliseconds elapsed{0};
};

// Publish a fully written, closed temporary file using rename only. A transient
// sharing/busy error retries until the bounded deadline; permanent errors fail
// immediately. Never unlink the destination or fall back to copying over it.
// On failure, both the previous destination and prepared temporary remain.
// The retry deadline does not interrupt an OS rename call already in progress.
AtomicReplaceResult replace_file_atomically(const std::filesystem::path& temporary,
                                            const std::filesystem::path& destination,
                                            const AtomicReplacePolicy& policy = {});

// Used for native JSON files. Checks write/close completion before publishing.
void write_text_atomically(const std::filesystem::path& destination, std::string_view contents,
                           const AtomicReplacePolicy& policy = {});

namespace atomic_io_detail {
// Injectable OS operation and clock keep fault tests deterministic and fast.
// Production always uses filesystem::rename and a steady monotonic clock.
struct Runtime {
    std::function<std::error_code(const std::filesystem::path&, const std::filesystem::path&)> rename;
    std::function<std::chrono::steady_clock::time_point()> now;
    std::function<void(std::chrono::milliseconds)> sleep;
};
AtomicReplaceResult replace(const std::filesystem::path& temporary,
                            const std::filesystem::path& destination,
                            const AtomicReplacePolicy& policy, const Runtime& runtime);
} // namespace atomic_io_detail
} // namespace atari
