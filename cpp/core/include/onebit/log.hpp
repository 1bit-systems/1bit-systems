// onebit/log.hpp — tiny C++23 logging facade.
//
// Header-only wrapper around std::println / std::print / std::format from
// <print> + <format>. Replaces ad-hoc fprintf/printf calls in the C++23
// tower with a single typesafe surface that:
//
//   - logs to stderr by default (service banners, errors, diagnostics),
//   - logs user-facing CLI text to stdout when explicitly requested,
//   - matches std::println's automatic '\n' / std::print's no-newline
//     contract so callers control line termination explicitly.
//
// Usage:
//
//     using onebit::log::eprintln;
//     using onebit::log::println;
//     using onebit::log::eprint;
//     using onebit::log::print;
//
//     eprintln("1bit-landing listening on {}:{} (lemond={})", bind, port, url);
//     println("PUT {} ({} bytes)", key, n);     // stdout, with '\n'
//     eprint("retrying...");                    // stderr, no '\n'
//
// Format-spec mapping vs printf:
//
//     %s         -> {}            %d / %u -> {}
//     %.3f       -> {:.3f}        %zu     -> {}
//     %llu       -> {}            %x      -> {:x}
//     %.2e       -> {:.2e}        %g      -> {}
//
// Note (GCC 15 / libstdc++ 15): std::print / std::println take FILE*
// only — there is no std::ostream overload yet (P3142 lands later).
// We accept that and use stderr / stdout directly.
//
// This header is intentionally header-only so it can be #included from
// any C++23 translation unit in the tower without dragging a new linkage
// dependency. Do NOT include it from a C++20 TU (e.g. anything in core/
// proper); it requires <print>.

#ifndef ONEBIT_LOG_HPP
#define ONEBIT_LOG_HPP

#include <cstdio>
#include <format>
#include <print>
#include <utility>

namespace onebit::log {

// stderr, '\n' appended automatically.
template <class... Args>
void eprintln(std::format_string<Args...> fmt, Args&&... args)
{
    std::println(stderr, fmt, std::forward<Args>(args)...);
}

// stderr, no '\n' (caller must include it in fmt if needed).
template <class... Args>
void eprint(std::format_string<Args...> fmt, Args&&... args)
{
    std::print(stderr, fmt, std::forward<Args>(args)...);
}

// stdout, '\n' appended automatically.
template <class... Args>
void println(std::format_string<Args...> fmt, Args&&... args)
{
    std::println(stdout, fmt, std::forward<Args>(args)...);
}

// stdout, no '\n'.
template <class... Args>
void print(std::format_string<Args...> fmt, Args&&... args)
{
    std::print(stdout, fmt, std::forward<Args>(args)...);
}

// Bare newline emitters — match std::println()/std::print() with no args.
inline void eprintln() { std::println(stderr); }
inline void println()  { std::println(stdout); }

} // namespace onebit::log

#endif // ONEBIT_LOG_HPP
