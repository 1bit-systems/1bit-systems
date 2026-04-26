#pragma once

// Lightweight error-code + value pair used across the onebit::power
// surface. We're on C++20 (no std::expected yet), so we roll a tiny
// std::variant-shaped helper instead of pulling tl::expected. Errors are
// propagated explicitly along the dispatch path; we do NOT throw out of
// the backend layer — exceptions are reserved for genuinely
// programmer-bug-class situations (e.g. doctest assertions).

#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace onebit::power {

enum class Error {
    Ok = 0,
    InvalidArgument,
    UnknownProfile,
    UnknownKnob,
    NotAvailable,        // EC sysfs absent, libryzenadj missing, etc.
    PermissionDenied,    // write to sysfs without root
    IoError,             // generic file read/write fail
    ParseError,          // bad TOML, bad number
    BackendError,        // upstream returned non-zero
    SymbolMissing,       // dlsym() returned null
};

[[nodiscard]] constexpr std::string_view error_name(Error e) noexcept
{
    switch (e) {
        case Error::Ok:                return "ok";
        case Error::InvalidArgument:   return "invalid argument";
        case Error::UnknownProfile:    return "unknown profile";
        case Error::UnknownKnob:       return "unknown knob";
        case Error::NotAvailable:      return "not available";
        case Error::PermissionDenied:  return "permission denied";
        case Error::IoError:           return "io error";
        case Error::ParseError:        return "parse error";
        case Error::BackendError:      return "backend error";
        case Error::SymbolMissing:     return "symbol missing";
    }
    return "unknown";
}

// Status: just an Error + diagnostic message. Used for void-returning
// operations.
struct Status {
    Error       code{Error::Ok};
    std::string message;

    [[nodiscard]] bool ok() const noexcept { return code == Error::Ok; }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

    static Status success() { return {Error::Ok, {}}; }
    static Status fail(Error c, std::string msg) { return {c, std::move(msg)}; }
};

// Result<T>: a value or a Status. We keep this minimal — no implicit
// conversions, no exception-on-bad-access. Callers check ok() first.
template <class T>
class Result {
public:
    Result(T v) : v_(std::move(v)) {}                      // NOLINT(*-explicit-*)
    Result(Status s) : v_(std::move(s)) {}                 // NOLINT(*-explicit-*)

    [[nodiscard]] bool   ok()      const noexcept { return v_.index() == 0; }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] const T& value()  const& { return std::get<0>(v_); }
    [[nodiscard]] T&       value()       & { return std::get<0>(v_); }
    [[nodiscard]] T        value()       && { return std::get<0>(std::move(v_)); }

    [[nodiscard]] const Status& status() const& { return std::get<1>(v_); }

    [[nodiscard]] Error error_code() const noexcept
    {
        return ok() ? Error::Ok : std::get<1>(v_).code;
    }

private:
    std::variant<T, Status> v_;
};

} // namespace onebit::power
