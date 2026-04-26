// Spawns scripts/discord-shim.ts under bun, exchanges JSON-per-line on
// stdin/stdout. Outbound writes lock a mutex around the FILE*; inbound
// reads run on a jthread that fills a queue surfaced via recv().
#include "onebit/agent/adapter_discord_shim.hpp"
#include "onebit/agent/event.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <queue>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern "C" char** environ;

namespace onebit::agent {

namespace {

constexpr std::string_view kDefaultShim =
    "/home/bcloud/repos/1bit-systems/cpp/agent/scripts/discord-shim.ts";

[[nodiscard]] std::vector<std::string>
build_child_env(const DiscordShimAdapterConfig& cfg)
{
    std::vector<std::string> env;
    for (char** p = environ; *p != nullptr; ++p) {
        std::string_view s(*p);
        // Strip the parent's DISCORD_* tokens — we set DISCORD_TOKEN
        // explicitly to the per-instance value below so each instance
        // doesn't accidentally pick up the wrong env var.
        if (s.starts_with("DISCORD_TOKEN=") ||
            s.starts_with("DISCORD_INTENTS=")) {
            continue;
        }
        env.emplace_back(*p);
    }
    env.push_back("DISCORD_TOKEN=" + cfg.token);
    if (cfg.intents != 0) {
        env.push_back("DISCORD_INTENTS=" + std::to_string(cfg.intents));
    }
    return env;
}

} // namespace

struct DiscordShimAdapter::Impl {
    DiscordShimAdapterConfig cfg;

    std::atomic<bool> started{false};
    std::atomic<bool> stop_flag{false};
    pid_t             child_pid    = -1;
    int               child_stdin  = -1;
    int               child_stdout = -1;
    std::string       self_user_id;

    std::jthread reader;
    std::mutex   q_mu;
    std::condition_variable q_cv;
    std::deque<IncomingMessage> queue;

    std::mutex   write_mu;

    ~Impl() { stop(); }

    void stop() noexcept {
        if (stop_flag.exchange(true)) return;
        if (child_pid > 0) ::kill(child_pid, SIGTERM);
        if (child_stdin  >= 0) { ::close(child_stdin);  child_stdin  = -1; }
        if (child_stdout >= 0) { ::close(child_stdout); child_stdout = -1; }
        if (child_pid > 0) {
            int status = 0;
            ::waitpid(child_pid, &status, 0);
            child_pid = -1;
        }
        if (reader.joinable()) reader.request_stop();
        q_cv.notify_all();
    }

    [[nodiscard]] bool spawn_shim() {
        int in_pipe[2]{-1, -1};
        int out_pipe[2]{-1, -1};
        if (::pipe2(in_pipe,  O_CLOEXEC) < 0) return false;
        if (::pipe2(out_pipe, O_CLOEXEC) < 0) {
            ::close(in_pipe[0]);  ::close(in_pipe[1]);
            return false;
        }
        // Child reads from in_pipe[0], writes to out_pipe[1].
        posix_spawn_file_actions_t fa;
        ::posix_spawn_file_actions_init(&fa);
        ::posix_spawn_file_actions_adddup2(&fa, in_pipe[0],  STDIN_FILENO);
        ::posix_spawn_file_actions_adddup2(&fa, out_pipe[1], STDOUT_FILENO);
        ::posix_spawn_file_actions_addclose(&fa, in_pipe[0]);
        ::posix_spawn_file_actions_addclose(&fa, in_pipe[1]);
        ::posix_spawn_file_actions_addclose(&fa, out_pipe[0]);
        ::posix_spawn_file_actions_addclose(&fa, out_pipe[1]);

        const std::string shim_path = cfg.shim_path.empty()
            ? std::string(kDefaultShim) : cfg.shim_path;
        const char* argv[] = { "bun", shim_path.c_str(), nullptr };
        auto envv = build_child_env(cfg);
        std::vector<char*> envp;
        envp.reserve(envv.size() + 1);
        for (auto& s : envv) envp.push_back(s.data());
        envp.push_back(nullptr);

        pid_t pid = -1;
        const int rc = ::posix_spawnp(&pid, "bun", &fa, nullptr,
                                      const_cast<char* const*>(argv),
                                      envp.data());
        ::posix_spawn_file_actions_destroy(&fa);
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);
        if (rc != 0) {
            ::close(in_pipe[1]);  ::close(out_pipe[0]);
            return false;
        }
        child_pid    = pid;
        child_stdin  = in_pipe[1];
        child_stdout = out_pipe[0];
        return true;
    }

    void reader_loop(std::stop_token tok) {
        std::string buf;
        char        chunk[4096];
        while (!tok.stop_requested() && !stop_flag.load()) {
            const ssize_t n = ::read(child_stdout, chunk, sizeof(chunk));
            if (n <= 0) {
                if (n < 0 && (errno == EINTR)) continue;
                break;
            }
            buf.append(chunk, static_cast<std::size_t>(n));
            for (;;) {
                const auto nl = buf.find('\n');
                if (nl == std::string::npos) break;
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                if (line.empty()) continue;
                handle_line(line);
            }
        }
        q_cv.notify_all();
    }

    void handle_line(const std::string& line) {
        nlohmann::json j;
        try { j = nlohmann::json::parse(line); }
        catch (...) {
            std::fprintf(stderr, "[discord-shim] bad json: %s\n", line.c_str());
            return;
        }
        const std::string op = j.value("op", std::string{});
        if (op == "ready") {
            self_user_id = j["user"].value("id", std::string{});
            std::fprintf(stderr,
                "[discord-shim] ready as %s (id=%s)\n",
                j["user"].value("username", std::string{}).c_str(),
                self_user_id.c_str());
        } else if (op == "message") {
            IncomingMessage m;
            m.channel   = j.value("channel", std::string{});
            m.user_id   = j.value("user", std::string{});
            m.user_name = j.value("username", std::string{});
            m.text      = j.value("text", std::string{});
            const bool is_dm = j.value("guild", nlohmann::json(nullptr)).is_null();
            // DM gating: drop unless sender is on allowlist (or list empty
            // means lockdown). Guild mentions always pass — the shim
            // already filters server messages to @-mentions only.
            if (is_dm) {
                bool allowed = false;
                for (const auto& id : cfg.dm_allowlist) {
                    if (id == m.user_id) { allowed = true; break; }
                }
                if (!allowed) return;
            }
            {
                std::lock_guard lk(q_mu);
                queue.push_back(std::move(m));
            }
            q_cv.notify_one();
        } else if (op == "error") {
            std::fprintf(stderr, "[discord-shim] err: %s\n",
                         j.value("msg", std::string{}).c_str());
        }
    }

    [[nodiscard]] bool write_line(const std::string& s) {
        std::lock_guard lk(write_mu);
        if (child_stdin < 0) return false;
        std::string out = s + "\n";
        std::size_t off = 0;
        while (off < out.size()) {
            const ssize_t n = ::write(child_stdin, out.data() + off, out.size() - off);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                return false;
            }
            off += static_cast<std::size_t>(n);
        }
        return true;
    }
};

DiscordShimAdapter::DiscordShimAdapter(DiscordShimAdapterConfig cfg)
    : p_(std::make_unique<Impl>())
{
    p_->cfg = std::move(cfg);
}

DiscordShimAdapter::~DiscordShimAdapter() { if (p_) p_->stop(); }

std::expected<void, AgentError> DiscordShimAdapter::start()
{
    if (p_->cfg.token.empty()) {
        return std::unexpected(AgentError::adapter("discord-shim: no token"));
    }
    if (p_->started.exchange(true)) return {};
    if (!p_->spawn_shim()) {
        return std::unexpected(AgentError::adapter(
            "discord-shim: posix_spawnp failed (bun missing on PATH?)"));
    }
    p_->reader = std::jthread(
        [impl = p_.get()](std::stop_token tok) { impl->reader_loop(tok); });
    return {};
}

std::expected<IncomingMessage, AgentError>
DiscordShimAdapter::recv(std::chrono::milliseconds timeout)
{
    std::unique_lock lk(p_->q_mu);
    if (!p_->q_cv.wait_for(lk, timeout, [&] {
            return !p_->queue.empty() ||
                   p_->stop_flag.load(std::memory_order_acquire);
        })) {
        return std::unexpected(AgentError::adapter_timeout());
    }
    if (p_->queue.empty()) {
        return std::unexpected(AgentError::adapter_closed());
    }
    IncomingMessage m = std::move(p_->queue.front());
    p_->queue.pop_front();
    return m;
}

std::expected<void, AgentError>
DiscordShimAdapter::send(const std::string& channel, std::string_view text)
{
    nlohmann::json j;
    j["op"]      = "send";
    j["channel"] = channel;
    j["text"]    = std::string(text);
    if (!p_->write_line(j.dump())) {
        return std::unexpected(AgentError::adapter("discord-shim: write failed"));
    }
    return {};
}

void DiscordShimAdapter::stop() noexcept { if (p_) p_->stop(); }

} // namespace onebit::agent
