#include "onebit/watchdog/poller.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace onebit::watchdog {

namespace {

using json = nlohmann::json;

constexpr const char* kGithubHost      = "api.github.com";
constexpr const char* kHuggingfaceHost = "huggingface.co";

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
using HttpsClient = httplib::SSLClient;
#else
using HttpsClient = httplib::Client;
#endif

std::optional<std::string> https_get_json(const char*               host,
                                          const std::string&        path,
                                          const httplib::Headers&   headers,
                                          PollError*                err)
{
    HttpsClient cli(host);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    cli.enable_server_certificate_verification(true);
#endif
    cli.set_connection_timeout(15, 0);
    cli.set_read_timeout(30, 0);
    cli.set_follow_location(true);

    auto res = cli.Get(path.c_str(), headers);
    if (!res) {
        if (err) *err = PollError::Network;
        return std::nullopt;
    }
    if (res->status < 200 || res->status >= 300) {
        if (err) *err = PollError::HttpStatus;
        return std::nullopt;
    }
    return res->body;
}

std::optional<std::string> extract_sha(const std::string& body, PollError* err)
{
    json v;
    try {
        v = json::parse(body);
    } catch (const json::parse_error&) {
        if (err) *err = PollError::BadJson;
        return std::nullopt;
    }
    if (!v.contains("sha") || !v["sha"].is_string()) {
        if (err) *err = PollError::MissingSha;
        return std::nullopt;
    }
    return v["sha"].get<std::string>();
}

} // namespace

std::optional<std::string> poll_github(std::string_view repo,
                                       std::string_view branch_or_empty,
                                       PollError*       err)
{
    const std::string branch =
        branch_or_empty.empty() ? "main" : std::string(branch_or_empty);
    const std::string path =
        "/repos/" + std::string(repo) + "/commits/" + branch;

    httplib::Headers hdrs{
        {"Accept",     "application/vnd.github+json"},
        {"User-Agent", "1bit-watchdog/0.1"},
    };
    if (const char* tok = std::getenv("GH_TOKEN"); tok && *tok) {
        hdrs.emplace("Authorization", std::string("Bearer ") + tok);
    }

    auto body = https_get_json(kGithubHost, path, hdrs, err);
    if (!body) return std::nullopt;
    return extract_sha(*body, err);
}

std::optional<std::string> poll_huggingface(std::string_view repo,
                                            PollError*       err)
{
    const std::string path = "/api/models/" + std::string(repo);
    httplib::Headers hdrs{{"User-Agent", "1bit-watchdog/0.1"}};
    auto body = https_get_json(kHuggingfaceHost, path, hdrs, err);
    if (!body) return std::nullopt;
    return extract_sha(*body, err);
}

namespace {

// Run argv with no shell, return exit code (-1 on spawn failure).
int spawn_argv(const std::vector<std::string>& argv)
{
    if (argv.empty()) return 0;
    std::vector<char*> cargs;
    cargs.reserve(argv.size() + 1);
    for (const auto& a : argv) {
        cargs.push_back(const_cast<char*>(a.c_str()));
    }
    cargs.push_back(nullptr);

    pid_t pid = ::fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        ::execvp(cargs[0], cargs.data());
        std::_Exit(127);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

std::string argv_repr(const std::vector<std::string>& argv)
{
    std::string s = "[";
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i) s += ", ";
        s += '"';
        s += argv[i];
        s += '"';
    }
    s += "]";
    return s;
}

} // namespace

bool run_hooks(const WatchEntry& entry, bool dry_run)
{
    const auto& hooks = (entry.kind == WatchKind::Github)
                            ? entry.on_merge
                            : entry.on_bump;
    for (const auto& argv : hooks) {
        if (argv.empty()) continue;
        if (dry_run) {
            spdlog::info("dry-run: would run hook argv={}", argv_repr(argv));
            continue;
        }
        const int code = spawn_argv(argv);
        if (code != 0) {
            spdlog::error("hook {} exit {}", argv_repr(argv), code);
            return false;
        }
    }
    return true;
}

namespace {

void notify(const WatchEntry& entry, std::string_view msg)
{
    spdlog::info("[{}] notify={} {}", entry.id, entry.notify, msg);
}

} // namespace

bool poll_entry(const WatchEntry& entry, State& state, bool dry_run)
{
    PollError perr{};
    std::optional<std::string> latest;
    if (entry.kind == WatchKind::Github) {
        latest = poll_github(entry.repo,
                             entry.branch ? *entry.branch : std::string_view{},
                             &perr);
    } else {
        latest = poll_huggingface(entry.repo, &perr);
    }
    if (!latest) {
        spdlog::warn("[{}] poll failed (err={})", entry.id,
                     static_cast<int>(perr));
        return false;
    }

    const auto t = state.observe(entry.id, *latest, entry.soak_hours);
    spdlog::info("[{}] latest={} transition={}",
                 entry.id, *latest, to_string(t.kind));

    switch (t.kind) {
        case Transition::Kind::NoChange:
            break;
        case Transition::Kind::SeenNew: {
            std::string m = "new upstream ref " + *latest +
                            " — dwell " + std::to_string(entry.soak_hours) + "h";
            notify(entry, m);
            break;
        }
        case Transition::Kind::Soaking:
            spdlog::info("[{}] still dwelling, remaining_hours={}",
                         entry.id, t.remaining_hours);
            break;
        case Transition::Kind::SoakComplete: {
            std::string m = "soak clean — triggering on_merge/on_bump for " + *latest;
            notify(entry, m);
            if (!dry_run) {
                if (!run_hooks(entry, dry_run)) {
                    return false;
                }
            }
            state.mark_merged(entry.id, Clock::now());
            break;
        }
    }
    return true;
}

} // namespace onebit::watchdog
