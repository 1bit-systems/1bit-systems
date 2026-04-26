#pragma once

// 1bit-agent — abstract adapter interface.
//
// Concrete implementations live in cpp/agent/src/adapter_*.cpp (Discord,
// Telegram, web, stdin) and are owned by sibling agents. This header
// only declares the contract the loop talks to; do not pull in any
// transport-specific dependency.
//
// recv() must support a timeout so the loop can poll for stop flags
// without blocking forever. Returning AgentError::adapter_timeout()
// (not a fatal error) signals "no message yet, try again". Returning
// AgentError::adapter_closed() signals "transport gone, exit cleanly".

#include "onebit/agent/error.hpp"
#include "onebit/agent/event.hpp"

#include <chrono>
#include <expected>
#include <string>
#include <string_view>

namespace onebit::agent {

class IAdapter {
public:
    virtual ~IAdapter() = default;

    // Boots the adapter (open socket, log in, etc.). Called once
    // before the first recv().
    [[nodiscard]] virtual std::expected<void, AgentError> start() = 0;

    // Pulls the next inbound message, waiting at most `timeout`. On
    // expiry returns AgentError::adapter_timeout(). On graceful
    // shutdown returns AgentError::adapter_closed().
    [[nodiscard]] virtual std::expected<IncomingMessage, AgentError>
    recv(std::chrono::milliseconds timeout) = 0;

    // Sends a reply to `channel`. Adapter is free to chunk long
    // messages internally.
    [[nodiscard]] virtual std::expected<void, AgentError>
    send(const std::string& channel, std::string_view text) = 0;

    // Asynchronous stop. Must be safe to call from any thread (the
    // loop's signal handler lives off the main thread). Idempotent.
    virtual void stop() noexcept = 0;
};

} // namespace onebit::agent
