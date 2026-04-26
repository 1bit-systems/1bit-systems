// 1bit-agent — default adapter + tool factory.
//
// This translation unit exists so the core agent library + halo-agent
// binary link cleanly *before* the sibling agents land their concrete
// adapter_*.cpp and tools/*.cpp files. The sibling work replaces this
// file (or wires CMake to skip it) once their factories are real.
//
// The defaults are deliberately minimal:
//   * adapter "stdin"  -> reads lines off std::cin, writes to std::cout.
//   * adapter anything else -> std::unexpected so misconfig surfaces.
//   * tool registry    -> empty (no tools).

#include "onebit/agent/factories.hpp"
#include "onebit/agent/tools/registry.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace onebit::agent {

namespace {

// ---- stdin loopback adapter -------------------------------------------
class StdinAdapter final : public IAdapter {
public:
    ~StdinAdapter() override { stop(); }

    std::expected<void, AgentError> start() override
    {
        if (started_.exchange(true)) return {};
        reader_ = std::thread([this] {
            std::string line;
            while (!stop_.load()) {
                if (!std::getline(std::cin, line)) {
                    closed_.store(true);
                    {
                        std::lock_guard lk(m_);
                        cv_.notify_all();
                    }
                    return;
                }
                IncomingMessage msg;
                msg.channel   = "stdin";
                msg.user_id   = "local";
                msg.user_name = "local";
                msg.text      = line;
                {
                    std::lock_guard lk(m_);
                    queue_.push(std::move(msg));
                }
                cv_.notify_one();
            }
        });
        return {};
    }

    std::expected<IncomingMessage, AgentError>
    recv(std::chrono::milliseconds timeout) override
    {
        std::unique_lock lk(m_);
        if (!cv_.wait_for(lk, timeout, [&] {
                return !queue_.empty() || closed_.load() || stop_.load();
            })) {
            return std::unexpected(AgentError::adapter_timeout());
        }
        if (!queue_.empty()) {
            auto m = std::move(queue_.front());
            queue_.pop();
            return m;
        }
        if (closed_.load() || stop_.load()) {
            return std::unexpected(AgentError::adapter_closed());
        }
        return std::unexpected(AgentError::adapter_timeout());
    }

    std::expected<void, AgentError>
    send(const std::string& /*channel*/, std::string_view text) override
    {
        std::cout << text << '\n' << std::flush;
        return {};
    }

    void stop() noexcept override
    {
        if (stop_.exchange(true)) {
            // already stopping; still ensure we join.
        }
        cv_.notify_all();
        if (reader_.joinable()) {
            try {
                reader_.join();
            } catch (...) { /* swallow per noexcept */ }
        }
    }

private:
    std::atomic<bool>  started_{false};
    std::atomic<bool>  stop_{false};
    std::atomic<bool>  closed_{false};
    std::mutex         m_;
    std::condition_variable cv_;
    std::queue<IncomingMessage> queue_;
    std::thread        reader_;
};

// ---- empty tool registry ----------------------------------------------
class EmptyToolRegistry final : public IToolRegistry {
public:
    [[nodiscard]] std::vector<nlohmann::json>
    list_tools_openai_format() const override { return {}; }

    [[nodiscard]] std::expected<ToolResult, AgentError>
    call(const ToolCall& c) override
    {
        return std::unexpected(AgentError::tool(
            c.name, "no tool registry installed"));
    }
};

} // namespace

[[nodiscard]] std::expected<std::unique_ptr<IAdapter>, AgentError>
make_adapter(const Config& cfg)
{
    if (cfg.adapter.kind == "stdin") {
        return std::unique_ptr<IAdapter>(new StdinAdapter());
    }
    return std::unexpected(AgentError::config(
        "adapter.kind=\"" + cfg.adapter.kind +
        "\" not registered (sibling agent must provide make_adapter)"));
}

[[nodiscard]] std::expected<std::unique_ptr<IToolRegistry>, AgentError>
make_tool_registry(const Config& cfg)
{
    if (cfg.tools.enabled.empty()) {
        return std::unique_ptr<IToolRegistry>(new EmptyToolRegistry());
    }
    auto reg = std::make_unique<ToolRegistry>();

    ToolRegistry::BuildOptions opts;
    opts.self_name              = cfg.agent.name;
    opts.consult_peer_name      = cfg.tools.agent_consult.peer_name;
    opts.consult_peer_brain_url = cfg.tools.agent_consult.peer_brain_url;
    opts.consult_peer_model     = cfg.tools.agent_consult.peer_model;
    opts.echo_url               = cfg.tools.speak_to_echo.echo_url;
    opts.echo_auto_speak        = cfg.tools.speak_to_echo.auto_speak;

    auto outcome = reg->build(cfg.tools.enabled, opts);
    for (const auto& w : outcome.warnings) {
        spdlog::warn("tool registry: {}", w);
    }
    return std::unique_ptr<IToolRegistry>(reg.release());
}

} // namespace onebit::agent
