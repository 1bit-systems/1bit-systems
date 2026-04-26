#include "onebit/mcp_clients/stdio_client.hpp"

#include "onebit/mcp/jsonrpc.hpp"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

#include <utility>
#include <vector>

namespace onebit::mcp_clients {

namespace jr = onebit::mcp::jsonrpc;

// --- ProcessStdioTransport ------------------------------------------------

ProcessStdioTransport::ProcessStdioTransport(const std::string& program,
                                             const std::vector<std::string>& args)
{
    int in_pipe[2]  = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0) {
        throw McpError::io(std::string("pipe(): ") + std::strerror(errno));
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe[0]);  ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        throw McpError::io(std::string("fork(): ") + std::strerror(errno));
    }
    if (pid == 0) {
        // child: stdin <- in_pipe[0], stdout -> out_pipe[1]
        ::dup2(in_pipe[0],  STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[0]);  ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        // stderr inherited intentionally.

        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(program.c_str()));
        for (const auto& a : args) {
            argv.push_back(const_cast<char*>(a.c_str()));
        }
        argv.push_back(nullptr);
        ::execvp(program.c_str(), argv.data());
        std::_Exit(127);
    }
    // parent
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    stdin_fd_  = in_pipe[1];
    stdout_fd_ = out_pipe[0];
    pid_       = pid;
}

ProcessStdioTransport::~ProcessStdioTransport()
{
    shutdown();
}

void ProcessStdioTransport::write_line(std::string_view line)
{
    if (stdin_fd_ < 0) {
        throw McpError::closed();
    }
    const char* p     = line.data();
    std::size_t left  = line.size();
    while (left > 0) {
        ssize_t n = ::write(stdin_fd_, p, left);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EPIPE) throw McpError::closed();
            throw McpError::io(std::string("write(): ") + std::strerror(errno));
        }
        p    += n;
        left -= static_cast<std::size_t>(n);
    }
}

std::string ProcessStdioTransport::read_line()
{
    if (stdout_fd_ < 0) {
        return {};
    }
    while (true) {
        auto nl = read_buf_.find('\n');
        if (nl != std::string::npos) {
            std::string line = read_buf_.substr(0, nl);
            read_buf_.erase(0, nl + 1);
            return line;
        }
        char buf[1024];
        ssize_t n = ::read(stdout_fd_, buf, sizeof(buf));
        if (n == 0) {
            if (read_buf_.empty()) return {};
            std::string remaining = std::move(read_buf_);
            read_buf_.clear();
            return remaining;
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            throw McpError::io(std::string("read(): ") + std::strerror(errno));
        }
        read_buf_.append(buf, static_cast<std::size_t>(n));
    }
}

void ProcessStdioTransport::shutdown()
{
    if (stdin_fd_ >= 0)  { ::close(stdin_fd_);  stdin_fd_  = -1; }
    if (stdout_fd_ >= 0) { ::close(stdout_fd_); stdout_fd_ = -1; }
    if (pid_ > 0) {
        ::kill(pid_, SIGTERM);
        int status = 0;
        ::waitpid(pid_, &status, 0);
        pid_ = -1;
    }
}

// --- StdioClient ----------------------------------------------------------

StdioClient::StdioClient(std::shared_ptr<StdioTransport> transport)
    : transport_(std::move(transport))
{}

StdioClient StdioClient::spawn(const std::string& program,
                               const std::vector<std::string>& args)
{
    return StdioClient(std::make_shared<ProcessStdioTransport>(program, args));
}

std::uint64_t StdioClient::next_id()
{
    return next_id_.fetch_add(1, std::memory_order_relaxed);
}

json StdioClient::send_recv(std::string_view method, const std::optional<json>& params)
{
    const auto id   = next_id();
    json        frame = build_request(id, method, params);
    std::string line  = frame.dump();
    line.push_back('\n');
    transport_->write_line(line);

    std::string reply = transport_->read_line();
    if (reply.empty()) {
        throw McpError::closed();
    }
    json body;
    try {
        body = json::parse(reply);
    } catch (const json::parse_error& e) {
        throw McpError::json(std::string("response parse: ") + e.what());
    }
    auto parsed = jr::parse_response(body);
    if (parsed.has_error()) {
        throw McpError::rpc(*parsed.error_code,
                            parsed.error_message.value_or(std::string{}));
    }
    if (!parsed.result) {
        throw McpError::protocol("no result");
    }
    return *parsed.result;
}

json StdioClient::initialize(std::string_view client_name, std::string_view client_version)
{
    return send_recv("initialize", initialize_params(client_name, client_version));
}

std::vector<Tool> StdioClient::list_tools()
{
    json result = send_recv("tools/list", json::object());
    std::vector<Tool> out;
    if (result.contains("tools") && result.at("tools").is_array()) {
        for (const auto& v : result.at("tools")) {
            out.push_back(v.get<Tool>());
        }
    }
    return out;
}

ToolCallResult StdioClient::call_tool(std::string_view name, json arguments)
{
    json p = {{"name", std::string(name)}, {"arguments", std::move(arguments)}};
    json result = send_recv("tools/call", p);
    return result.get<ToolCallResult>();
}

void StdioClient::shutdown()
{
    if (transport_) transport_->shutdown();
}

} // namespace onebit::mcp_clients
