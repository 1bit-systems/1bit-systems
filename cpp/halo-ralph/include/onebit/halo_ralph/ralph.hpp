#pragma once

// 1bit-halo-ralph — minimal ralph-loop agent. Points at any OpenAI-compatible
// chat-completions endpoint (lemond at :8180/v1 by default), runs a task
// prompt in a loop, optionally re-runs a test command each iteration and
// feeds its output back as context.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::halo_ralph {

inline constexpr std::string_view DEFAULT_BASE_URL = "http://localhost:8180/v1";
inline constexpr std::string_view DEFAULT_MODEL    = "1bit-halo-v2";
inline constexpr std::string_view DEFAULT_SYSTEM   =
    "You are a precise, terse coding assistant. "
    "Every turn, emit exactly one concrete action: a diff, a shell command, "
    "or a specific file change. No preamble, no apology, no commentary. "
    "If the previous turn included a failing test output, address it directly.";

inline constexpr std::uint32_t DEFAULT_MAX_ITER    = 5;
inline constexpr float         DEFAULT_TEMPERATURE = 0.3f;

struct Message {
    std::string role;
    std::string content;
};

struct Args {
    std::string                 task;
    std::string                 base_url    = std::string(DEFAULT_BASE_URL);
    std::string                 model       = std::string(DEFAULT_MODEL);
    std::optional<std::string>  api_key;
    std::uint32_t               max_iter    = DEFAULT_MAX_ITER;
    std::optional<std::string>  test_cmd;
    std::optional<std::string>  system;
    float                       temperature = DEFAULT_TEMPERATURE;
};

// Pure-data helpers (testable without a network).
[[nodiscard]] std::string serialize_chat_request(std::string_view             model,
                                                 const std::vector<Message>&  messages,
                                                 float                        temperature,
                                                 bool                         stream);

[[nodiscard]] std::optional<std::string>
parse_first_choice_content(std::string_view body);

// Split a base_url like "http://host:port/v1" into (scheme, host, port,
// base_path). Returns nullopt on malformed input.
struct ParsedUrl {
    bool        is_https = false;
    std::string host;
    int         port = 80;
    std::string base_path; // e.g. "/v1", may be empty; leading slash kept
};
[[nodiscard]] std::optional<ParsedUrl>
parse_base_url(std::string_view base_url);

struct TestCmdResult {
    int         exit_code = 0;
    std::string stdout_text;
    std::string stderr_text;
};

// Fork+exec `sh -c <cmd>`, capture stdout + stderr separately. Returns
// nullopt on spawn failure.
[[nodiscard]] std::optional<TestCmdResult>
run_test_cmd(std::string_view cmd);

// Format the user feedback message that gets pushed when a test command
// fails (Rust crate's literal string).
[[nodiscard]] std::string
format_test_failure_feedback(std::string_view cmd, const TestCmdResult& r);

// Result of run_loop().
enum class RunStatus {
    TestsPassed,   // exit 0
    GaveUp,        // exit 2 (loop exhausted without passing tests)
    NoTestCmd,     // exit 0 (single iteration completed, no test_cmd)
    HttpError,     // exit 1
};

// Drive the full ralph loop. Prints model output to stdout; test command
// output is appended to history as the next user turn. The CLI front-end
// translates RunStatus -> exit code.
[[nodiscard]] RunStatus run_loop(const Args& args);

} // namespace onebit::halo_ralph
