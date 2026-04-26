#pragma once

// 1bit-agent — event loop.
//
// Single-threaded dispatch:
//
//   for each IncomingMessage from adapter:
//     1. persist user message
//     2. build BrainRequest from history + system_prompt + tool list
//     3. brain.chat(req)  -> BrainReply
//     4. if reply.tool_calls non-empty: run each tool, persist
//        tool-result messages, re-enter step 2 (cap at max_tool_iters)
//     5. else: persist assistant text, adapter.send(text)
//
// Threading: brain HTTP is sync per turn (no threads here). Adapters
// run their own internal threads if they need them; recv(timeout) is
// the only synchronization point. Stop is signalled via a
// std::stop_source -> jthread token; the loop polls between recv
// timeouts and exits cleanly on SIGTERM.

#include "onebit/agent/adapter.hpp"
#include "onebit/agent/brain.hpp"
#include "onebit/agent/config.hpp"
#include "onebit/agent/error.hpp"
#include "onebit/agent/memory.hpp"
#include "onebit/agent/tools.hpp"

#include <chrono>
#include <expected>
#include <memory>

namespace onebit::agent {

struct LoopStats {
    std::int64_t messages_in        = 0;
    std::int64_t messages_out       = 0;
    std::int64_t tool_calls         = 0;
    std::int64_t brain_calls        = 0;
    std::int64_t adapter_timeouts   = 0;
    std::int64_t hard_errors        = 0;
};

class AgentLoop {
public:
    // Borrows by raw pointer; lifetime is the caller's job (main owns
    // them). nullptr in any slot is a precondition violation.
    AgentLoop(Config         cfg,
              IAdapter*      adapter,
              IBrain*        brain,
              Memory*        memory,
              IToolRegistry* tools);

    ~AgentLoop();
    AgentLoop(const AgentLoop&)            = delete;
    AgentLoop& operator=(const AgentLoop&) = delete;
    AgentLoop(AgentLoop&&) noexcept;
    AgentLoop& operator=(AgentLoop&&) noexcept;

    [[nodiscard]] const Config&    config() const noexcept;
    [[nodiscard]] const LoopStats& stats()  const noexcept;

    // Blocks until stop() is called or the adapter reports closed.
    // Returns the terminal error if the loop bailed out early; returns
    // void on graceful shutdown.
    [[nodiscard]] std::expected<void, AgentError> run_forever();

    // Single-iteration variant used by tests + the main loop. Pumps
    // exactly one inbound message (or one timeout). Returns ok on
    // timeout, ok on normal turn, error on hard failure.
    [[nodiscard]] std::expected<void, AgentError>
    pump_once(std::chrono::milliseconds timeout);

    // Stop hook. Safe from any thread; idempotent.
    void stop() noexcept;
    [[nodiscard]] bool stopping() const noexcept;

    // Test seam: lets a test push a synthesized IncomingMessage
    // through the same dispatch path used by run_forever.
    [[nodiscard]] std::expected<void, AgentError>
    handle_for_tests(const IncomingMessage& msg);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::agent
