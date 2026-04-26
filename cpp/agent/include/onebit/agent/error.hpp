#pragma once

// 1bit-agent — error variant for the autonomous-agent daemon.
//
// Mirrors the variant style in cpp/core/include/onebit/core/error.hpp:
// std::expected<T, AgentError> on every fallible path; no exceptions on
// the dispatch loop. Sqlite, brain HTTP, adapter I/O, config parse, and
// tool dispatch each get a distinct alternative so the loop can branch
// on cause without string-matching.

#include <string>
#include <variant>

namespace onebit::agent {

struct ErrorConfig {
    std::string msg;       // human-readable, owned (TOML errors come pre-formatted)
};
struct ErrorSqlite {
    std::string msg;
    int         rc = 0;    // raw sqlite3 result code (SQLITE_*)
};
struct ErrorBrain {
    std::string msg;
    int         http_status = 0;  // 0 if not an HTTP-level failure
};
struct ErrorAdapter {
    std::string msg;
};
struct ErrorAdapterTimeout {};   // recv() returned empty within deadline; not fatal
struct ErrorAdapterClosed {};    // recv() observed graceful shutdown
struct ErrorTool {
    std::string name;
    std::string msg;
};
struct ErrorLoop {
    std::string msg;
};

class AgentError {
public:
    using Variant = std::variant<
        ErrorConfig,
        ErrorSqlite,
        ErrorBrain,
        ErrorAdapter,
        ErrorAdapterTimeout,
        ErrorAdapterClosed,
        ErrorTool,
        ErrorLoop>;

    explicit AgentError(Variant v) : v_(std::move(v)) {}

    [[nodiscard]] const Variant& variant() const noexcept { return v_; }
    [[nodiscard]] std::string    what() const;

    [[nodiscard]] bool is_timeout() const noexcept
    {
        return std::holds_alternative<ErrorAdapterTimeout>(v_);
    }
    [[nodiscard]] bool is_closed() const noexcept
    {
        return std::holds_alternative<ErrorAdapterClosed>(v_);
    }

    static AgentError config(std::string msg)        { return AgentError{ErrorConfig{std::move(msg)}}; }
    static AgentError sqlite(std::string msg, int rc){ return AgentError{ErrorSqlite{std::move(msg), rc}}; }
    static AgentError brain(std::string msg, int s = 0)
    {
        return AgentError{ErrorBrain{std::move(msg), s}};
    }
    static AgentError adapter(std::string msg)       { return AgentError{ErrorAdapter{std::move(msg)}}; }
    static AgentError adapter_timeout() noexcept     { return AgentError{ErrorAdapterTimeout{}}; }
    static AgentError adapter_closed() noexcept      { return AgentError{ErrorAdapterClosed{}}; }
    static AgentError tool(std::string name, std::string msg)
    {
        return AgentError{ErrorTool{std::move(name), std::move(msg)}};
    }
    static AgentError loop(std::string msg)          { return AgentError{ErrorLoop{std::move(msg)}}; }

private:
    Variant v_;
};

} // namespace onebit::agent
