#include "onebit/halo_ralph/ralph.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace onebit::halo_ralph {

namespace {

using json = nlohmann::json;

// Drain a fd into `out` until EOF.
void drain_fd(int fd, std::string& out)
{
    std::array<char, 4096> buf{};
    while (true) {
        ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n > 0) {
            out.append(buf.data(), static_cast<std::size_t>(n));
        } else if (n == 0) {
            break;
        } else {
            if (errno == EINTR) continue;
            break;
        }
    }
}

std::string trim_trailing_slash(std::string s)
{
    while (!s.empty() && s.back() == '/') s.pop_back();
    return s;
}

} // namespace

std::string serialize_chat_request(std::string_view             model,
                                   const std::vector<Message>&  messages,
                                   float                        temperature,
                                   bool                         stream)
{
    json msgs = json::array();
    for (const auto& m : messages) {
        msgs.push_back({{"role", m.role}, {"content", m.content}});
    }
    json req = {
        {"model",       std::string(model)},
        {"messages",    std::move(msgs)},
        {"temperature", temperature},
        {"stream",      stream},
    };
    return req.dump();
}

std::optional<std::string>
parse_first_choice_content(std::string_view body)
{
    json v;
    try {
        v = json::parse(body);
    } catch (const json::parse_error&) {
        return std::nullopt;
    }
    if (!v.contains("choices") || !v["choices"].is_array()
        || v["choices"].empty()) {
        return std::string{};
    }
    const auto& choice = v["choices"][0];
    if (!choice.contains("message") || !choice["message"].is_object()) {
        return std::string{};
    }
    const auto& msg = choice["message"];
    if (!msg.contains("content") || !msg["content"].is_string()) {
        return std::string{};
    }
    return msg["content"].get<std::string>();
}

std::optional<ParsedUrl> parse_base_url(std::string_view url)
{
    ParsedUrl out;
    constexpr std::string_view kHttp  = "http://";
    constexpr std::string_view kHttps = "https://";
    std::string_view rest;
    if (url.size() > kHttps.size() &&
        url.substr(0, kHttps.size()) == kHttps) {
        out.is_https = true;
        out.port = 443;
        rest = url.substr(kHttps.size());
    } else if (url.size() > kHttp.size() &&
               url.substr(0, kHttp.size()) == kHttp) {
        out.is_https = false;
        out.port = 80;
        rest = url.substr(kHttp.size());
    } else {
        return std::nullopt;
    }

    auto slash_pos = rest.find('/');
    std::string_view authority = rest.substr(0, slash_pos);
    std::string_view path =
        (slash_pos == std::string_view::npos) ? std::string_view{}
                                              : rest.substr(slash_pos);

    auto colon_pos = authority.find(':');
    if (colon_pos == std::string_view::npos) {
        out.host = std::string(authority);
    } else {
        out.host = std::string(authority.substr(0, colon_pos));
        try {
            out.port = std::stoi(std::string(authority.substr(colon_pos + 1)));
        } catch (...) {
            return std::nullopt;
        }
    }
    out.base_path = trim_trailing_slash(std::string(path));
    return out;
}

std::optional<TestCmdResult> run_test_cmd(std::string_view cmd)
{
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) < 0) return std::nullopt;
    if (::pipe(err_pipe) < 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        return std::nullopt;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        return std::nullopt;
    }
    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        std::string s(cmd);
        const char* argv[] = {"sh", "-c", s.c_str(), nullptr};
        ::execvp(argv[0], const_cast<char* const*>(argv));
        std::_Exit(127);
    }

    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    TestCmdResult r;
    drain_fd(out_pipe[0], r.stdout_text);
    drain_fd(err_pipe[0], r.stderr_text);
    ::close(out_pipe[0]);
    ::close(err_pipe[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return std::nullopt;
    }
    if (WIFEXITED(status))        r.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) r.exit_code = 128 + WTERMSIG(status);
    else                          r.exit_code = -1;
    return r;
}

std::string format_test_failure_feedback(std::string_view cmd,
                                         const TestCmdResult& r)
{
    std::string s = "Previous action failed. Test command `";
    s.append(cmd);
    s += "` exited with code ";
    s += std::to_string(r.exit_code);
    s += ".\n\nSTDOUT:\n";
    s += r.stdout_text;
    s += "\n\nSTDERR:\n";
    s += r.stderr_text;
    return s;
}

namespace {

// httplib::Client and httplib::SSLClient share no common base we can use
// portably, so we keep both alive in an owning struct and dispatch per-call.
struct HttpClient {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    std::unique_ptr<httplib::SSLClient> ssl;
#endif
    std::unique_ptr<httplib::Client>    plain;

    httplib::Result Post(const char*             path,
                         const httplib::Headers& headers,
                         const std::string&      body,
                         const char*             content_type)
    {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        if (ssl) return ssl->Post(path, headers, body, content_type);
#endif
        return plain->Post(path, headers, body, content_type);
    }
};

HttpClient make_http_client(const ParsedUrl& u)
{
    HttpClient c;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (u.is_https) {
        c.ssl = std::make_unique<httplib::SSLClient>(u.host, u.port);
        c.ssl->enable_server_certificate_verification(true);
        c.ssl->set_connection_timeout(60, 0);
        c.ssl->set_read_timeout(600, 0);
        return c;
    }
#endif
    c.plain = std::make_unique<httplib::Client>(u.host, u.port);
    c.plain->set_connection_timeout(60, 0);
    c.plain->set_read_timeout(600, 0);
    return c;
}

} // namespace

RunStatus run_loop(const Args& args)
{
    auto parsed = parse_base_url(args.base_url);
    if (!parsed) {
        std::fprintf(stderr, "ralph: malformed --base-url %s\n",
                     args.base_url.c_str());
        return RunStatus::HttpError;
    }
    const std::string endpoint_path = parsed->base_path + "/chat/completions";

    auto cli = make_http_client(*parsed);
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (parsed->is_https) {
        std::fprintf(stderr,
                     "ralph: built without OpenSSL, https:// base_url unsupported\n");
        return RunStatus::HttpError;
    }
#endif

    const std::string system =
        args.system ? *args.system : std::string(DEFAULT_SYSTEM);

    std::vector<Message> history;
    history.push_back({"system", system});
    history.push_back({"user",   args.task});

    for (std::uint32_t iter = 1; iter <= args.max_iter; ++iter) {
        std::printf("── ralph iter %u/%u ────────────────────\n",
                    iter, args.max_iter);
        std::fflush(stdout);

        const std::string body = serialize_chat_request(
            args.model, history, args.temperature, /*stream=*/false);

        httplib::Headers headers{
            {"Content-Type", "application/json"},
        };
        if (args.api_key) {
            headers.emplace("Authorization", std::string("Bearer ") + *args.api_key);
        }

        auto res = cli.Post(endpoint_path.c_str(), headers,
                            body, "application/json");
        if (!res) {
            std::fprintf(stderr, "ralph: POST %s failed (network)\n",
                         endpoint_path.c_str());
            return RunStatus::HttpError;
        }
        if (res->status < 200 || res->status >= 300) {
            std::fprintf(stderr, "ralph: POST %s failed (status %d)\n",
                         endpoint_path.c_str(), res->status);
            return RunStatus::HttpError;
        }

        auto content = parse_first_choice_content(res->body);
        if (!content) {
            std::fprintf(stderr, "ralph: malformed chat response\n");
            return RunStatus::HttpError;
        }
        const std::string text = *content;
        std::printf("%s\n\n", text.c_str());
        std::fflush(stdout);
        history.push_back({"assistant", text});

        if (!args.test_cmd) {
            return RunStatus::NoTestCmd;
        }

        std::printf("── test · %s\n", args.test_cmd->c_str());
        std::fflush(stdout);

        auto t = run_test_cmd(*args.test_cmd);
        if (!t) {
            std::fprintf(stderr, "ralph: failed to spawn test-cmd: %s\n",
                         args.test_cmd->c_str());
            return RunStatus::HttpError;
        }
        if (t->exit_code == 0) {
            std::printf("✓ tests passed at iter %u\n", iter);
            return RunStatus::TestsPassed;
        }

        std::fprintf(stderr, "✗ tests failed at iter %u\n", iter);
        history.push_back({"user",
                          format_test_failure_feedback(*args.test_cmd, *t)});
    }

    std::fprintf(stderr, "── ralph gave up after %u iterations\n",
                 args.max_iter);
    return RunStatus::GaveUp;
}

} // namespace onebit::halo_ralph
