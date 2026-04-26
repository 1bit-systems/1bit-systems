#pragma once

// 1bit-agent — sqlite-backed conversation log + key/value facts.
//
// Schema (created on open if missing):
//
//   CREATE TABLE IF NOT EXISTS messages (
//     id              INTEGER PRIMARY KEY AUTOINCREMENT,
//     channel         TEXT    NOT NULL,
//     user_id         TEXT    NOT NULL,
//     role            TEXT    NOT NULL,   -- system|user|assistant|tool
//     content         TEXT    NOT NULL,
//     tool_calls_json TEXT    NOT NULL DEFAULT '',
//     created_at      INTEGER NOT NULL    -- unix epoch seconds
//   );
//   CREATE INDEX IF NOT EXISTS idx_messages_channel_id
//       ON messages(channel, id);
//
//   CREATE TABLE IF NOT EXISTS facts (
//     key        TEXT PRIMARY KEY,
//     value      TEXT NOT NULL,
//     updated_at INTEGER NOT NULL
//   );
//
// pImpl per Core Guidelines I.27. sqlite3* never escapes the
// translation unit; callers see std::expected<T, AgentError> only.
//
// Thread-safety: Memory holds one connection. Callers must externally
// serialize calls (the agent loop is single-threaded by design).

#include "onebit/agent/error.hpp"
#include "onebit/agent/event.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::agent {

struct StoredMessage {
    std::int64_t  id = 0;
    std::string   channel;
    std::string   user_id;
    std::string   role;
    std::string   content;
    std::string   tool_calls_json;   // serialized; "" if none
    std::int64_t  created_at = 0;
};

class Memory {
public:
    // Opens (and creates if missing) a sqlite database at `path`. Pass
    // ":memory:" for an in-memory store (used by tests).
    [[nodiscard]] static std::expected<Memory, AgentError>
    open(const std::filesystem::path& path);

    ~Memory();
    Memory(const Memory&)            = delete;
    Memory& operator=(const Memory&) = delete;
    Memory(Memory&&) noexcept;
    Memory& operator=(Memory&&) noexcept;

    // ---- messages -----------------------------------------------------

    [[nodiscard]] std::expected<std::int64_t, AgentError>
    append_message(std::string_view channel,
                   std::string_view user_id,
                   std::string_view role,
                   std::string_view content,
                   std::string_view tool_calls_json,
                   std::int64_t     created_at);

    // Most recent N messages on `channel`, returned oldest-first
    // (chronological). N <= 0 returns empty.
    [[nodiscard]] std::expected<std::vector<StoredMessage>, AgentError>
    recent_messages(std::string_view channel, int n) const;

    // Drops messages where id < (max(id) - keep + 1). Returns rows
    // deleted. keep <= 0 is a no-op.
    [[nodiscard]] std::expected<std::int64_t, AgentError>
    trim_messages(std::int64_t keep);

    // ---- facts --------------------------------------------------------

    [[nodiscard]] std::expected<void, AgentError>
    upsert_fact(std::string_view key,
                std::string_view value,
                std::int64_t     updated_at);

    [[nodiscard]] std::expected<std::optional<std::string>, AgentError>
    get_fact(std::string_view key) const;

    [[nodiscard]] std::expected<std::int64_t, AgentError>
    fact_count() const;

    // ---- diagnostics --------------------------------------------------

    // Returns the sqlite library version the binary was linked against
    // (e.g. "3.45.1"). Useful for systemd journal banners + tests.
    [[nodiscard]] static std::string sqlite_version() noexcept;

private:
    Memory();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::agent
