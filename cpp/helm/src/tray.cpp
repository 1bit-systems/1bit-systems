#include "onebit/helm/tray.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace onebit::helm::tray {

std::string_view action_label(Action a) noexcept
{
    switch (a) {
        case Action::Status:        return "Status";
        case Action::StartAll:      return "Start All";
        case Action::StopAll:       return "Stop All";
        case Action::RestartServer: return "Restart 1bit-server";
        case Action::OpenSite:      return "Open 1bit.systems";
        case Action::Quit:          return "Quit";
    }
    return "";
}

std::string_view service_state_str(ServiceState s) noexcept
{
    switch (s) {
        case ServiceState::Active:   return "active";
        case ServiceState::Inactive: return "inactive";
        case ServiceState::Unknown:  return "unknown";
    }
    return "unknown";
}

ServiceState parse_service_state(std::string_view s) noexcept
{
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
        s.remove_suffix(1);
    }
    if (s == "active" || s == "activating" || s == "reloading") {
        return ServiceState::Active;
    }
    if (s == "inactive" || s == "failed" || s == "deactivating") {
        return ServiceState::Inactive;
    }
    return ServiceState::Unknown;
}

std::string build_status_line(const std::vector<ServiceStatus>& rows)
{
    if (rows.empty()) return "no services";
    std::string out;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i != 0) out += " · ";
        out += rows[i].name;
        out += ": ";
        out += service_state_str(rows[i].state);
    }
    return out;
}

namespace {

// Run `argv` to completion, capturing stdout into `out_text`.
// Returns the exit code, or -1 on spawn failure.
int run_capture(const std::vector<std::string>& argv, std::string& out_text)
{
    int pipe_fd[2] = {-1, -1};
    if (::pipe(pipe_fd) < 0) return -1;
    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipe_fd[0]); ::close(pipe_fd[1]);
        return -1;
    }
    if (pid == 0) {
        ::dup2(pipe_fd[1], STDOUT_FILENO);
        ::close(pipe_fd[0]); ::close(pipe_fd[1]);
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        ::execvp(cargv[0], cargv.data());
        std::_Exit(127);
    }
    ::close(pipe_fd[1]);
    char buf[1024];
    while (true) {
        ssize_t n = ::read(pipe_fd[0], buf, sizeof(buf));
        if (n > 0) out_text.append(buf, static_cast<std::size_t>(n));
        else if (n == 0) break;
        else if (errno == EINTR) continue;
        else break;
    }
    ::close(pipe_fd[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

} // namespace

std::vector<ServiceStatus> probe_services()
{
    std::vector<ServiceStatus> out;
    out.reserve(SERVICES.size());
    for (auto unit : SERVICES) {
        std::string text;
        run_capture({"systemctl", "--user", "is-active", std::string(unit)},
                    text);
        out.push_back(ServiceStatus{std::string(unit),
                                    parse_service_state(text)});
    }
    return out;
}

int systemctl(std::string_view verb, std::vector<std::string_view> units)
{
    std::vector<std::string> argv;
    argv.reserve(units.size() + 3);
    argv.emplace_back("systemctl");
    argv.emplace_back("--user");
    argv.emplace_back(std::string(verb));
    for (auto u : units) argv.emplace_back(std::string(u));
    std::string sink;
    return run_capture(argv, sink);
}

bool open_site()
{
    pid_t pid = ::fork();
    if (pid < 0) return false;
    if (pid == 0) {
        const char* argv[] = {"xdg-open", "https://1bit.systems", nullptr};
        ::execvp(argv[0], const_cast<char* const*>(argv));
        std::_Exit(127);
    }
    return true;
}

} // namespace onebit::helm::tray
