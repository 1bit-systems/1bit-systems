#pragma once

// 1bit-agent — factory entry points for adapters and tool registries.
//
// Implementations live in cpp/agent/src/adapter_*.cpp and
// cpp/agent/src/tools/*.cpp (sibling agents own those files). main.cpp
// resolves a concrete IAdapter / IToolRegistry from the parsed config
// via these two functions; the core loop never knows which adapter
// kind it's talking to.
//
// Implementing the factory is the sibling agent's responsibility. To
// keep the link clean while sibling work is in flight, a default
// implementation is provided in cpp/agent/src/factories_default.cpp
// that returns a stdin-loopback adapter + an empty tool registry.
// Sibling agents replace that .cpp with the real factory.

#include "onebit/agent/adapter.hpp"
#include "onebit/agent/config.hpp"
#include "onebit/agent/error.hpp"
#include "onebit/agent/tools.hpp"

#include <expected>
#include <memory>

namespace onebit::agent {

[[nodiscard]] std::expected<std::unique_ptr<IAdapter>, AgentError>
make_adapter(const Config& cfg);

[[nodiscard]] std::expected<std::unique_ptr<IToolRegistry>, AgentError>
make_tool_registry(const Config& cfg);

} // namespace onebit::agent
