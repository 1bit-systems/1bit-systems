#pragma once

// 1bit-agent — TOML config schema.
//
// Schema:
//   [agent]
//   name             = "halo-helpdesk"
//   brain_url        = "http://127.0.0.1:8180"
//   system_prompt    = "You are halo-helpdesk..."
//   max_history      = 32              # turns kept in working window
//   max_tool_iters   = 5               # cap on brain<->tool round-trips
//   request_timeout_ms = 60000
//   stream           = true            # SSE on /v1/chat/completions
//   model            = "halo-1.58b"
//   temperature      = 0.2
//
//   [adapter]
//   kind             = "discord" | "telegram" | "web" | "stdin"
//   token            = "${ENV:DISCORD_TOKEN}" # ${ENV:NAME} is expanded
//   bind_host        = "127.0.0.1"           # adapter-specific; ignored by Discord
//   bind_port        = 8086
//
//   [memory]
//   sqlite_path      = "/var/lib/1bit-agent/halo-helpdesk.db"
//   keep_messages    = 10000           # trim threshold; 0 = no trim
//
//   [tools]
//   enabled          = ["repo_search", "url_fetch"]

#include "onebit/agent/error.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace onebit::agent {

struct AgentSection {
    std::string  name             = "1bit-agent";
    std::string  brain_url;            // required; canonical = http://127.0.0.1:8180
    std::string  system_prompt;
    std::string  model;                 // empty = let lemond pick the default recipe
    std::int32_t max_history      = 32;
    std::int32_t max_tool_iters   = 5;
    std::int32_t request_timeout_ms = 60000;
    bool         stream           = true;
    double       temperature      = 0.2;
};

struct AdapterSection {
    std::string   kind = "stdin";
    std::string   token;
    std::string   bind_host = "127.0.0.1";
    std::uint16_t bind_port = 0;
};

struct MemorySection {
    std::filesystem::path sqlite_path;
    std::int64_t          keep_messages = 0;
};

struct ToolsSection {
    std::vector<std::string> enabled;

    // Optional per-tool config sub-tables. Parsed from
    //   [tools.agent_consult]
    //   peer_name      = "halo-coder"
    //   peer_brain_url = "http://127.0.0.1:8180/v1"
    //   peer_model     = "Qwen3-8B-GGUF"
    // Empty strings = disabled / use ToolDef defaults.
    struct AgentConsultCfg {
        std::string peer_name;
        std::string peer_brain_url;
        std::string peer_model;
    } agent_consult;

    struct SpeakToEchoCfg {
        std::string echo_url    = "http://127.0.0.1:8181/v1/tts";
        bool        auto_speak  = false;
    } speak_to_echo;
};

struct Config {
    AgentSection   agent;
    AdapterSection adapter;
    MemorySection  memory;
    ToolsSection   tools;
};

// Loads a TOML file off disk. Returns AgentError::config(...) on parse
// or schema failure, with the source line where available. Performs
// ${ENV:VAR} substitution on string fields after parse.
[[nodiscard]] std::expected<Config, AgentError>
load_config(const std::filesystem::path& path);

// Same parse path against an in-memory buffer. Hot-path use is config
// reload; tests use this with synthetic strings.
[[nodiscard]] std::expected<Config, AgentError>
parse_config(std::string_view toml_text);

} // namespace onebit::agent
