#pragma once

#include <stdexcept>
#include <string>

namespace onebit::mcp_clients {

enum class ErrorKind {
    Io,
    Http,
    Json,
    Rpc,
    Protocol,
    Closed,
};

class McpError : public std::runtime_error {
public:
    McpError(ErrorKind kind, std::string message, int code = 0)
        : std::runtime_error(message),
          kind_(kind),
          message_(std::move(message)),
          code_(code)
    {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }
    [[nodiscard]] int       code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    static McpError io(std::string m)        { return {ErrorKind::Io, std::move(m)}; }
    static McpError http(std::string m)      { return {ErrorKind::Http, std::move(m)}; }
    static McpError json(std::string m)      { return {ErrorKind::Json, std::move(m)}; }
    static McpError rpc(int code, std::string m) {
        return {ErrorKind::Rpc, std::move(m), code};
    }
    static McpError protocol(std::string m)  { return {ErrorKind::Protocol, std::move(m)}; }
    static McpError closed()                 { return {ErrorKind::Closed, "transport closed before response arrived"}; }

private:
    ErrorKind   kind_;
    std::string message_;
    int         code_;
};

} // namespace onebit::mcp_clients
