#pragma once

// Stdio MCP client. Spawns a child process whose stdin/stdout speak
// newline-delimited JSON-RPC 2.0 (the local MCP transport contract).
// Stderr is inherited so server-side logs reach the parent terminal.
//
// Implementation uses pipe(2) + fork(2) + execvp(2) directly. We avoid
// popen because we need both input and output pipes plus the ability
// to wait/kill the child.
//
// The transport is pluggable via StdioTransport so tests can verify
// framing without spawning a real subprocess.

#include "onebit/mcp_clients/error.hpp"
#include "onebit/mcp_clients/protocol.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::mcp_clients {

// Pluggable stdio transport. Production: ProcessStdioTransport (fork/exec).
// Tests: an in-memory fake.
class StdioTransport {
public:
    virtual ~StdioTransport() = default;

    // Write one newline-terminated JSON line to the child stdin.
    virtual void write_line(std::string_view line) = 0;

    // Read one line from the child stdout. Returns empty string on EOF.
    virtual std::string read_line() = 0;

    // Best-effort shutdown.
    virtual void shutdown() {}
};

class ProcessStdioTransport : public StdioTransport {
public:
    ProcessStdioTransport(const std::string& program,
                          const std::vector<std::string>& args);
    ~ProcessStdioTransport() override;

    ProcessStdioTransport(const ProcessStdioTransport&)            = delete;
    ProcessStdioTransport& operator=(const ProcessStdioTransport&) = delete;

    void        write_line(std::string_view line) override;
    std::string read_line() override;
    void        shutdown() override;

private:
    int  stdin_fd_  = -1;
    int  stdout_fd_ = -1;
    int  pid_       = -1;
    std::string read_buf_;
};

class StdioClient {
public:
    explicit StdioClient(std::shared_ptr<StdioTransport> transport);

    // Convenience: spawn `program args...` and wrap with a Process transport.
    static StdioClient spawn(const std::string& program,
                             const std::vector<std::string>& args);

    json initialize(std::string_view client_name, std::string_view client_version);
    [[nodiscard]] std::vector<Tool> list_tools();
    ToolCallResult call_tool(std::string_view name, json arguments);

    void shutdown();

private:
    std::uint64_t next_id();
    json          send_recv(std::string_view method, const std::optional<json>& params);

    std::shared_ptr<StdioTransport> transport_;
    std::atomic<std::uint64_t>      next_id_{1};
};

} // namespace onebit::mcp_clients
