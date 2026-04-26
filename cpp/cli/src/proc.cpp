#include "onebit/cli/proc.hpp"

#include "onebit/cli/package.hpp"
#include "onebit/cli/paths.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace onebit::cli {

namespace {

[[nodiscard]] std::vector<char*> argv_to_cstr(const std::vector<std::string>& argv)
{
    std::vector<char*> out;
    out.reserve(argv.size() + 1);
    for (const auto& s : argv) {
        out.push_back(const_cast<char*>(s.c_str()));
    }
    out.push_back(nullptr);
    return out;
}

[[nodiscard]] std::expected<std::pair<int, int>, Error> make_pipe()
{
    int fds[2];
    if (::pipe(fds) != 0) {
        return std::unexpected(Error::subprocess(
            std::string("pipe failed: ") + std::strerror(errno)));
    }
    return std::make_pair(fds[0], fds[1]);
}

[[nodiscard]] std::string drain_fd(int fd)
{
    std::string out;
    std::array<char, 4096> buf{};
    while (true) {
        const ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n > 0) {
            out.append(buf.data(), static_cast<std::size_t>(n));
            continue;
        }
        if (n == 0) break;
        if (errno == EINTR) continue;
        break;
    }
    return out;
}

}  // namespace

std::expected<CommandResult, Error>
run_capture(const std::vector<std::string>& argv,
            const std::filesystem::path& cwd)
{
    if (argv.empty()) {
        return std::unexpected(Error::invalid("run_capture: empty argv"));
    }

    auto out_pipe = make_pipe();
    if (!out_pipe) return std::unexpected(out_pipe.error());
    auto err_pipe = make_pipe();
    if (!err_pipe) {
        ::close(out_pipe->first);
        ::close(out_pipe->second);
        return std::unexpected(err_pipe.error());
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe->first);
        ::close(out_pipe->second);
        ::close(err_pipe->first);
        ::close(err_pipe->second);
        return std::unexpected(Error::subprocess(
            std::string("fork failed: ") + std::strerror(errno)));
    }
    if (pid == 0) {
        // Child.
        ::close(out_pipe->first);
        ::close(err_pipe->first);
        ::dup2(out_pipe->second, STDOUT_FILENO);
        ::dup2(err_pipe->second, STDERR_FILENO);
        ::close(out_pipe->second);
        ::close(err_pipe->second);

        if (!cwd.empty()) {
            (void)::chdir(cwd.c_str());
        }
        auto cv = argv_to_cstr(argv);
        ::execvp(cv[0], cv.data());
        ::_exit(127);  // exec failure
    }

    ::close(out_pipe->second);
    ::close(err_pipe->second);

    CommandResult res;
    res.stdout_text = drain_fd(out_pipe->first);
    res.stderr_text = drain_fd(err_pipe->first);
    ::close(out_pipe->first);
    ::close(err_pipe->first);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return std::unexpected(Error::subprocess(
                std::string("waitpid failed: ") + std::strerror(errno)));
        }
    }
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        res.signaled  = true;
        res.exit_code = -1;
    }
    return res;
}

std::expected<int, Error>
run_inherit(const std::vector<std::string>& argv,
            const std::filesystem::path& cwd)
{
    if (argv.empty()) {
        return std::unexpected(Error::invalid("run_inherit: empty argv"));
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        return std::unexpected(Error::subprocess(
            std::string("fork failed: ") + std::strerror(errno)));
    }
    if (pid == 0) {
        if (!cwd.empty()) (void)::chdir(cwd.c_str());
        auto cv = argv_to_cstr(argv);
        ::execvp(cv[0], cv.data());
        ::_exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return std::unexpected(Error::subprocess(
                std::string("waitpid failed: ") + std::strerror(errno)));
        }
    }
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -1;
    return -1;
}

std::expected<int, Error>
run_with_stdin(const std::vector<std::string>& argv,
               std::string_view stdin_bytes,
               const std::filesystem::path& cwd)
{
    if (argv.empty()) {
        return std::unexpected(Error::invalid("run_with_stdin: empty argv"));
    }
    auto in_pipe = make_pipe();
    if (!in_pipe) return std::unexpected(in_pipe.error());

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe->first);
        ::close(in_pipe->second);
        return std::unexpected(Error::subprocess(
            std::string("fork failed: ") + std::strerror(errno)));
    }
    if (pid == 0) {
        // Child — wire pipe read end to stdin, inherit stdout / stderr.
        ::close(in_pipe->second);
        ::dup2(in_pipe->first, STDIN_FILENO);
        ::close(in_pipe->first);
        if (!cwd.empty()) (void)::chdir(cwd.c_str());
        auto cv = argv_to_cstr(argv);
        ::execvp(cv[0], cv.data());
        ::_exit(127);
    }
    ::close(in_pipe->first);

    // Parent — write all bytes, ignoring SIGPIPE so a misbehaving child
    // exiting early returns a clean exit code rather than killing us.
    struct sigaction sa_old{};
    struct sigaction sa_ign{};
    sa_ign.sa_handler = SIG_IGN;
    sigemptyset(&sa_ign.sa_mask);
    ::sigaction(SIGPIPE, &sa_ign, &sa_old);

    const char* p = stdin_bytes.data();
    std::size_t remaining = stdin_bytes.size();
    bool write_failed = false;
    while (remaining > 0) {
        const ssize_t n = ::write(in_pipe->second, p, remaining);
        if (n > 0) {
            p         += n;
            remaining -= static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        write_failed = true;
        break;
    }
    ::close(in_pipe->second);
    ::sigaction(SIGPIPE, &sa_old, nullptr);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return std::unexpected(Error::subprocess(
                std::string("waitpid failed: ") + std::strerror(errno)));
        }
    }
    if (write_failed) {
        return std::unexpected(Error::subprocess(
            "stdin write to child failed (broken pipe / short write)"));
    }
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return -1;
    return -1;
}

bool which(std::string_view bin)
{
    auto res = run_capture({"which", std::string(bin)});
    return res.has_value() && res->exit_code == 0;
}

std::string expand_tilde(std::string_view raw)
{
    // Two-stage expansion to cover every shape that lands in `argv` on the
    // way out of `packages.toml`:
    //
    //   1. Leading-`~` is shell-only (we don't ever run argv through a
    //      shell), so the CLI does it itself for both the bare `~` and the
    //      `~/foo` prefix forms.
    //   2. Substring placeholder expansion (`${HOME}`, `$HOME`, `${USER}`,
    //      `$USER`, `${XDG_CONFIG_HOME}`, `${XDG_DATA_HOME}`) — strict, no
    //      shell-style `${VAR:-default}` / `$(cmd)` / glob, see
    //      `package.cpp`'s comment block. The audit
    //      (cpp/cli/src/package.cpp:13-28) flagged the missing
    //      `${HOME}/.local/bin/...` rewrite as a P0 fresh-box bug — every
    //      install recipe was creating a literal `${HOME}` directory under
    //      cwd because this pass only knew bare-`~`.
    std::string s;
    if (raw == "~") {
        s = home_dir().string();
    } else if (raw.starts_with("~/")) {
        s = (home_dir() / std::filesystem::path(raw.substr(2))).string();
    } else {
        s.assign(raw);
    }
    return expand_placeholder(s);
}

}  // namespace onebit::cli
