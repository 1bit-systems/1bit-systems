#include "onebit/agent/memory.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace onebit::agent {

namespace {

constexpr const char* kSchemaSql = R"SQL(
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT    NOT NULL,
    user_id         TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    tool_calls_json TEXT    NOT NULL DEFAULT '',
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel, id);

CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
)SQL";

AgentError sqlite_err(sqlite3* db, int rc, const char* op)
{
    const char* msg = (db != nullptr) ? sqlite3_errmsg(db) : sqlite3_errstr(rc);
    if (msg == nullptr) msg = "(no message)";
    return AgentError::sqlite(std::string(op) + ": " + msg, rc);
}

} // namespace

struct Memory::Impl {
    sqlite3* db = nullptr;

    ~Impl()
    {
        if (db != nullptr) sqlite3_close(db);
    }

    Impl()                       = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
};

Memory::Memory()                                = default;
Memory::~Memory()                               = default;
Memory::Memory(Memory&&) noexcept               = default;
Memory& Memory::operator=(Memory&&) noexcept    = default;

std::string Memory::sqlite_version() noexcept
{
    return sqlite3_libversion();
}

std::expected<Memory, AgentError>
Memory::open(const std::filesystem::path& path)
{
    Memory m;
    m.p_ = std::make_unique<Impl>();
    int rc = sqlite3_open_v2(
        path.string().c_str(), &m.p_->db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        nullptr);
    if (rc != SQLITE_OK) {
        return std::unexpected(sqlite_err(m.p_->db, rc, ("open " + path.string()).c_str()));
    }
    // Enable foreign keys + WAL for write throughput on real disk.
    char* err = nullptr;
    rc = sqlite3_exec(m.p_->db,
                      "PRAGMA journal_mode = WAL;"
                      "PRAGMA synchronous = NORMAL;"
                      "PRAGMA foreign_keys = ON;",
                      nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string msg = err != nullptr ? err : "pragma failed";
        sqlite3_free(err);
        return std::unexpected(AgentError::sqlite(std::move(msg), rc));
    }
    rc = sqlite3_exec(m.p_->db, kSchemaSql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string msg = err != nullptr ? err : "schema apply failed";
        sqlite3_free(err);
        return std::unexpected(AgentError::sqlite(std::move(msg), rc));
    }
    return m;
}

// ---- helpers (statement RAII) ------------------------------------------

namespace {

class Stmt {
public:
    Stmt(sqlite3* db, std::string_view sql)
        : db_(db)
    {
        rc_ = sqlite3_prepare_v2(db_, sql.data(),
                                 static_cast<int>(sql.size()),
                                 &s_, nullptr);
    }
    ~Stmt() { if (s_ != nullptr) sqlite3_finalize(s_); }
    Stmt(const Stmt&)            = delete;
    Stmt& operator=(const Stmt&) = delete;

    [[nodiscard]] int rc()  const noexcept { return rc_; }
    [[nodiscard]] sqlite3_stmt* get() const noexcept { return s_; }
    [[nodiscard]] sqlite3* db() const noexcept { return db_; }

    int bind_text(int i, std::string_view v)
    {
        // sqlite3_bind_text(nullptr, 0) binds NULL, not empty-string.
        // We want columns NOT NULL on schema, so substitute "" when
        // the caller passes a default-constructed string_view.
        const char* p = v.data();
        const char  empty[] = "";
        if (p == nullptr) p = empty;
        return sqlite3_bind_text(s_, i, p,
                                 static_cast<int>(v.size()),
                                 SQLITE_TRANSIENT);
    }
    int bind_int64(int i, std::int64_t v)
    {
        return sqlite3_bind_int64(s_, i, v);
    }

private:
    sqlite3*      db_ = nullptr;
    sqlite3_stmt* s_  = nullptr;
    int           rc_ = SQLITE_OK;
};

} // namespace

std::expected<std::int64_t, AgentError>
Memory::append_message(std::string_view channel,
                       std::string_view user_id,
                       std::string_view role,
                       std::string_view content,
                       std::string_view tool_calls_json,
                       std::int64_t     created_at)
{
    Stmt st(p_->db,
            "INSERT INTO messages "
            "(channel, user_id, role, content, tool_calls_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?);");
    if (st.rc() != SQLITE_OK) {
        return std::unexpected(sqlite_err(p_->db, st.rc(), "prepare append_message"));
    }
    st.bind_text(1, channel);
    st.bind_text(2, user_id);
    st.bind_text(3, role);
    st.bind_text(4, content);
    st.bind_text(5, tool_calls_json);
    st.bind_int64(6, created_at);
    int rc = sqlite3_step(st.get());
    if (rc != SQLITE_DONE) {
        return std::unexpected(sqlite_err(p_->db, rc, "step append_message"));
    }
    return sqlite3_last_insert_rowid(p_->db);
}

std::expected<std::vector<StoredMessage>, AgentError>
Memory::recent_messages(std::string_view channel, int n) const
{
    std::vector<StoredMessage> out;
    if (n <= 0) return out;

    // Newest N (DESC), then we reverse on the way out so callers get
    // chronological order — matches the OpenAI history convention.
    Stmt st(p_->db,
            "SELECT id, channel, user_id, role, content, tool_calls_json, created_at "
            "FROM messages WHERE channel = ? "
            "ORDER BY id DESC LIMIT ?;");
    if (st.rc() != SQLITE_OK) {
        return std::unexpected(sqlite_err(p_->db, st.rc(), "prepare recent_messages"));
    }
    st.bind_text(1, channel);
    st.bind_int64(2, n);
    out.reserve(static_cast<std::size_t>(n));
    for (;;) {
        int rc = sqlite3_step(st.get());
        if (rc == SQLITE_DONE) break;
        if (rc != SQLITE_ROW) {
            return std::unexpected(sqlite_err(p_->db, rc, "step recent_messages"));
        }
        StoredMessage m;
        m.id              = sqlite3_column_int64(st.get(), 0);
        if (auto* t = sqlite3_column_text(st.get(), 1); t != nullptr) {
            m.channel = reinterpret_cast<const char*>(t);
        }
        if (auto* t = sqlite3_column_text(st.get(), 2); t != nullptr) {
            m.user_id = reinterpret_cast<const char*>(t);
        }
        if (auto* t = sqlite3_column_text(st.get(), 3); t != nullptr) {
            m.role = reinterpret_cast<const char*>(t);
        }
        if (auto* t = sqlite3_column_text(st.get(), 4); t != nullptr) {
            m.content = reinterpret_cast<const char*>(t);
        }
        if (auto* t = sqlite3_column_text(st.get(), 5); t != nullptr) {
            m.tool_calls_json = reinterpret_cast<const char*>(t);
        }
        m.created_at = sqlite3_column_int64(st.get(), 6);
        out.push_back(std::move(m));
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::expected<std::int64_t, AgentError>
Memory::trim_messages(std::int64_t keep)
{
    if (keep <= 0) return std::int64_t{0};
    Stmt st(p_->db,
            "DELETE FROM messages WHERE id <= "
            "(SELECT MAX(id) FROM messages) - ?;");
    if (st.rc() != SQLITE_OK) {
        return std::unexpected(sqlite_err(p_->db, st.rc(), "prepare trim_messages"));
    }
    st.bind_int64(1, keep);
    int rc = sqlite3_step(st.get());
    if (rc != SQLITE_DONE) {
        return std::unexpected(sqlite_err(p_->db, rc, "step trim_messages"));
    }
    return static_cast<std::int64_t>(sqlite3_changes(p_->db));
}

std::expected<void, AgentError>
Memory::upsert_fact(std::string_view key,
                    std::string_view value,
                    std::int64_t     updated_at)
{
    Stmt st(p_->db,
            "INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at;");
    if (st.rc() != SQLITE_OK) {
        return std::unexpected(sqlite_err(p_->db, st.rc(), "prepare upsert_fact"));
    }
    st.bind_text(1, key);
    st.bind_text(2, value);
    st.bind_int64(3, updated_at);
    int rc = sqlite3_step(st.get());
    if (rc != SQLITE_DONE) {
        return std::unexpected(sqlite_err(p_->db, rc, "step upsert_fact"));
    }
    return {};
}

std::expected<std::optional<std::string>, AgentError>
Memory::get_fact(std::string_view key) const
{
    Stmt st(p_->db, "SELECT value FROM facts WHERE key = ?;");
    if (st.rc() != SQLITE_OK) {
        return std::unexpected(sqlite_err(p_->db, st.rc(), "prepare get_fact"));
    }
    st.bind_text(1, key);
    int rc = sqlite3_step(st.get());
    if (rc == SQLITE_DONE) return std::optional<std::string>{};
    if (rc != SQLITE_ROW) {
        return std::unexpected(sqlite_err(p_->db, rc, "step get_fact"));
    }
    std::string out;
    if (auto* t = sqlite3_column_text(st.get(), 0); t != nullptr) {
        out = reinterpret_cast<const char*>(t);
    }
    return std::optional<std::string>{std::move(out)};
}

std::expected<std::int64_t, AgentError>
Memory::fact_count() const
{
    Stmt st(p_->db, "SELECT COUNT(*) FROM facts;");
    if (st.rc() != SQLITE_OK) {
        return std::unexpected(sqlite_err(p_->db, st.rc(), "prepare fact_count"));
    }
    int rc = sqlite3_step(st.get());
    if (rc != SQLITE_ROW) {
        return std::unexpected(sqlite_err(p_->db, rc, "step fact_count"));
    }
    return sqlite3_column_int64(st.get(), 0);
}

} // namespace onebit::agent
