#include "onebit/mcp_linuxgsm/server.hpp"

#include "onebit/mcp/jsonrpc.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#include <algorithm>
#include <istream>
#include <ostream>
#include <sstream>
#include <utility>

namespace onebit::mcp_linuxgsm {

namespace fs = std::filesystem;
namespace jr = onebit::mcp::jsonrpc;

bool is_allowed_subcommand(std::string_view sub) noexcept
{
    return std::any_of(ALLOWED_SUBCOMMANDS.begin(), ALLOWED_SUBCOMMANDS.end(),
                       [&](std::string_view s) { return s == sub; });
}

bool is_safe_server_name(std::string_view name) noexcept
{
    if (name.empty()) return false;
    for (unsigned char c : name) {
        const bool ok = (c >= 'a' && c <= 'z')
                     || (c >= '0' && c <= '9')
                     || c == '-' || c == '_';
        if (!ok) return false;
    }
    return true;
}

fs::path gsm_root()
{
    if (const char* p = std::getenv("HALO_LINUXGSM_ROOT")) {
        if (*p) return fs::path(p);
    }
    if (const char* h = std::getenv("HOME")) {
        if (*h) return fs::path(h) / "linuxgsm";
    }
    return fs::path("/var/lib/linuxgsm");
}

json tools()
{
    json allowed_arr = json::array();
    for (auto s : ALLOWED_SUBCOMMANDS) allowed_arr.push_back(std::string(s));

    return json::array({
        {
            {"name",        "linuxgsm_list"},
            {"description", "List detected LinuxGSM servers under HALO_LINUXGSM_ROOT (one dir per server)."},
            {"inputSchema", { {"type", "object"}, {"properties", json::object()} }},
        },
        {
            {"name",        "linuxgsm_run"},
            {"description", "Run an allowlisted <game>server subcommand. Returns stdout. "
                            "Allowed: details, status, start, stop, restart, update, backup."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"server",     { {"type", "string"},
                                     {"description", "<game>server driver name (e.g. mcserver)"} }},
                    {"subcommand", { {"type", "string"}, {"enum", allowed_arr} }},
                }},
                {"required", json::array({"server", "subcommand"})},
            }},
        },
    });
}

json text_result(std::string_view text, bool is_error)
{
    return {
        {"content", json::array({
            { {"type", "text"}, {"text", std::string(text)} },
        })},
        {"isError", is_error},
    };
}

json list_servers(const fs::path& root)
{
    std::vector<std::string> found;
    std::error_code ec;
    if (fs::exists(root, ec) && fs::is_directory(root, ec)) {
        for (const auto& entry : fs::directory_iterator(root, ec)) {
            if (ec) break;
            if (!entry.is_directory()) continue;
            const std::string name = entry.path().filename().string();
            const fs::path driver  = entry.path() / (name + "server");
            if (fs::exists(driver, ec)) {
                found.push_back(name);
            }
        }
    }
    std::sort(found.begin(), found.end());

    std::string joined;
    for (std::size_t i = 0; i < found.size(); ++i) {
        if (i) joined.push_back('\n');
        joined += found[i];
    }
    return text_result(joined, false);
}

json run_driver(const fs::path& root,
                std::string_view server,
                std::string_view subcommand,
                const DriverRunner& runner)
{
    if (!is_allowed_subcommand(subcommand)) {
        std::ostringstream os;
        os << "subcommand not allowed: " << subcommand;
        return text_result(os.str(), true);
    }
    if (!is_safe_server_name(server)) {
        std::ostringstream os;
        os << "invalid server name: " << server;
        return text_result(os.str(), true);
    }
    const std::string s_str{server};
    const fs::path driver = root / s_str / (s_str + "server");
    std::error_code ec;
    if (!fs::exists(driver, ec)) {
        std::ostringstream os;
        os << "driver missing: " << driver.string();
        return text_result(os.str(), true);
    }
    DriverOutput o = runner(driver, subcommand);
    return text_result(o.text, !o.success);
}

DriverOutput run_driver_process(const fs::path& driver, std::string_view subcommand)
{
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0) {
        return {std::string("pipe failed: ") + std::strerror(errno), false};
    }
    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        return {std::string("fork failed: ") + std::strerror(errno), false};
    }
    if (pid == 0) {
        // child
        int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) ::dup2(devnull, STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        if (devnull >= 0) ::close(devnull);

        std::string sub{subcommand};
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(driver.c_str()));
        argv.push_back(const_cast<char*>(sub.c_str()));
        argv.push_back(nullptr);
        ::execv(driver.c_str(), argv.data());
        std::_Exit(127);
    }
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    auto drain = [](int fd) {
        std::string s;
        char buf[1024];
        while (true) {
            ssize_t n = ::read(fd, buf, sizeof(buf));
            if (n > 0) s.append(buf, static_cast<std::size_t>(n));
            else if (n == 0) break;
            else if (errno == EINTR) continue;
            else break;
        }
        ::close(fd);
        return s;
    };

    std::string out = drain(out_pipe[0]);
    std::string err = drain(err_pipe[0]);

    int status = 0;
    ::waitpid(pid, &status, 0);
    const bool success = WIFEXITED(status) && WEXITSTATUS(status) == 0;

    std::string combined = std::move(out);
    if (!err.empty()) {
        combined.append("\n---stderr---\n");
        combined.append(err);
    }
    return {std::move(combined), success};
}

json handle(const json& request, const fs::path& root, const DriverRunner& runner)
{
    json id = request.contains("id") ? request.at("id") : json(nullptr);
    const std::string method =
        request.contains("method") && request.at("method").is_string()
            ? request.at("method").get<std::string>()
            : std::string{};

    if (method == "initialize") {
        json result = {
            {"protocolVersion", std::string(PROTOCOL_VERSION)},
            {"capabilities",    { {"tools", { {"listChanged", false} }} }},
            {"serverInfo",      { {"name",    std::string(SERVER_NAME)},
                                  {"version", std::string(SERVER_VERSION)} }},
        };
        return jr::build_result(std::move(id), std::move(result));
    }
    if (method == "tools/list") {
        return jr::build_result(std::move(id), { {"tools", tools()} });
    }
    if (method == "tools/call") {
        const json* params = request.contains("params") ? &request.at("params") : nullptr;
        std::string name;
        json arguments = json::object();
        if (params && params->is_object()) {
            if (params->contains("name") && params->at("name").is_string()) {
                name = params->at("name").get<std::string>();
            }
            if (params->contains("arguments")) {
                arguments = params->at("arguments");
            }
        }
        json result;
        if (name == "linuxgsm_list") {
            result = list_servers(root);
        } else if (name == "linuxgsm_run") {
            const std::string server =
                arguments.value("server", std::string{});
            const std::string sub =
                arguments.value("subcommand", std::string{});
            result = run_driver(root, server, sub, runner);
        } else {
            std::ostringstream os;
            os << "unknown tool: " << name;
            result = text_result(os.str(), true);
        }
        return jr::build_result(std::move(id), std::move(result));
    }
    std::ostringstream os;
    os << "method not found: " << method;
    return jr::build_error(std::move(id), jr::kMethodNotFound, os.str());
}

void run(std::istream& in, std::ostream& out,
         const fs::path& root, const DriverRunner& runner)
{
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json req;
        try {
            req = json::parse(line);
        } catch (const json::parse_error& e) {
            json err = jr::build_error(json(nullptr), jr::kParseError,
                                       std::string("parse error: ") + e.what());
            jr::write_line(out, err);
            continue;
        }
        json resp = handle(req, root, runner);
        jr::write_line(out, resp);
    }
}

} // namespace onebit::mcp_linuxgsm
