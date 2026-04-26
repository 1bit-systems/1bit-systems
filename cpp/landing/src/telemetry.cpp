#include "onebit/landing/telemetry.hpp"

#include <httplib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace onebit::landing {

namespace {

using json = nlohmann::json;

// ---- subprocess helper -----------------------------------------------------
// popen() captures stdout into a string and returns the exit status.
// Stderr is silenced (`2>/dev/null`) to match the Rust port's
// `Stdio::null()` pipes.
struct ExecResult {
    int         status{-1};
    std::string stdout_;
};

[[nodiscard]] ExecResult exec_capture(const std::string& cmd)
{
    ExecResult r;
    std::FILE* f = popen(cmd.c_str(), "r");
    if (!f) {
        return r;
    }
    char buf[4096];
    while (auto n = std::fread(buf, 1, sizeof(buf), f)) {
        r.stdout_.append(buf, n);
    }
    int rc = pclose(f);
    if (rc == -1) {
        r.status = -1;
    } else {
        // WEXITSTATUS portable enough on Linux.
        r.status = (rc >> 8) & 0xff;
    }
    return r;
}

// Returns true if `path` exists at the filesystem level (no perms check).
[[nodiscard]] bool path_exists(const std::filesystem::path& p) noexcept
{
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

[[nodiscard]] std::string shell_quote(const std::string& s)
{
    std::string out{"'"};
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// Split an http(s) URL into host+port for httplib::Client.
struct UrlParts {
    std::string scheme;
    std::string host;
    int         port{-1};
    bool        ok{false};
};
[[nodiscard]] UrlParts split_url(std::string_view url)
{
    UrlParts u;
    static const std::regex re(R"(^(https?)://([^:/]+)(?::(\d+))?(?:/.*)?$)");
    std::cmatch m;
    if (!std::regex_match(url.data(), url.data() + url.size(), m, re)) {
        return u;
    }
    u.scheme = m[1].str();
    u.host   = m[2].str();
    u.port   = m[3].matched ? std::stoi(m[3].str())
                            : (u.scheme == "https" ? 443 : 80);
    u.ok = true;
    return u;
}

[[nodiscard]] std::optional<json> http_get_json(const std::string& base,
                                                std::string_view   path,
                                                std::chrono::milliseconds timeout)
{
    auto u = split_url(base);
    if (!u.ok) {
        return std::nullopt;
    }
    httplib::Client cli(u.scheme + "://" + u.host + ":" + std::to_string(u.port));
    cli.set_connection_timeout(timeout);
    cli.set_read_timeout(timeout);
    cli.set_write_timeout(timeout);
    auto res = cli.Get(std::string{path});
    if (!res || res->status < 200 || res->status >= 300) {
        return std::nullopt;
    }
    try {
        return json::parse(res->body);
    } catch (const json::parse_error&) {
        return std::nullopt;
    }
}

} // namespace

// ---- Sources ---------------------------------------------------------------

Sources Sources::defaults()
{
    Sources s;
    // ~/claude output/shadow-burnin.jsonl (folder name has a literal space —
    // project convention from CLAUDE.md "Benchmark output folder").
    const char* home = std::getenv("HOME");
    std::filesystem::path home_p = home ? home : "/home/bcloud";
    s.shadow_burnin_jsonl = home_p / "claude output" / "shadow-burnin.jsonl";
    s.services.reserve(TRACKED_SERVICES.size());
    for (const auto& sv : TRACKED_SERVICES) {
        s.services.emplace_back(sv);
    }
    return s;
}

// ---- service_delta ---------------------------------------------------------

std::optional<std::vector<ServiceState>>
service_delta(const std::vector<ServiceState>& prev,
              const std::vector<ServiceState>& next)
{
    std::vector<ServiceState> changes;
    for (const auto& n : next) {
        auto it = std::find_if(prev.begin(), prev.end(),
                               [&](const ServiceState& p) { return p.name == n.name; });
        const bool was_known    = (it != prev.end());
        const bool prev_active  = was_known ? it->active : !n.active; // sentinel: anything != n.active
        if (!was_known || prev_active != n.active) {
            changes.push_back(n);
        }
    }
    if (changes.empty()) {
        return std::nullopt;
    }
    return changes;
}

// ---- parse_rocm_smi_json ---------------------------------------------------

std::pair<float, std::uint8_t> parse_rocm_smi_json(const json& v)
{
    if (!v.is_object()) {
        return {0.0F, 0};
    }
    // Find the first "card*" key.
    const json* card = nullptr;
    for (auto it = v.begin(); it != v.end(); ++it) {
        const auto& key = it.key();
        if (key.rfind("card", 0) == 0) {
            card = &it.value();
            break;
        }
    }
    if (!card || !card->is_object()) {
        return {0.0F, 0};
    }

    float        temp = 0.0F;
    std::uint8_t util = 0;

    for (auto it = card->begin(); it != card->end(); ++it) {
        const std::string& k = it.key();
        if (!it.value().is_string()) {
            continue;
        }
        const std::string vstr = it.value().get<std::string>();
        if (k.find("Temperature") != std::string::npos
            && k.find("edge") != std::string::npos)
        {
            try { temp = std::stof(vstr); }
            catch (...) { temp = 0.0F; }
        } else if (k.find("GPU use") != std::string::npos) {
            // Trim whitespace + trailing '%' then parse.
            std::string trimmed = vstr;
            while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) {
                trimmed.erase(trimmed.begin());
            }
            while (!trimmed.empty() && (std::isspace(static_cast<unsigned char>(trimmed.back()))
                                         || trimmed.back() == '%'))
            {
                trimmed.pop_back();
            }
            try {
                int n = std::stoi(trimmed);
                if (n < 0)        { util = 0; }
                else if (n > 255) { util = 255; }
                else              { util = static_cast<std::uint8_t>(n); }
            } catch (...) { util = 0; }
        }
    }
    return {temp, util};
}

// ---- HTTP probes -----------------------------------------------------------

std::optional<std::string> probe_model(const Sources& s)
{
    auto v = http_get_json(s.onebit_server_base, "/v1/models",
                           std::chrono::milliseconds{1500});
    if (!v) { return std::nullopt; }
    auto data = v->find("data");
    if (data == v->end() || !data->is_array() || data->empty()) {
        return std::nullopt;
    }
    const auto& first = (*data)[0];
    auto id = first.find("id");
    if (id == first.end() || !id->is_string()) {
        return std::nullopt;
    }
    return id->get<std::string>();
}

std::optional<float> probe_tokps(const Sources& s)
{
    auto v = http_get_json(s.onebit_server_base, "/metrics",
                           std::chrono::milliseconds{1500});
    if (!v) { return std::nullopt; }
    auto t = v->find("tokps_recent");
    if (t == v->end() || !t->is_number()) { return std::nullopt; }
    return static_cast<float>(t->get<double>());
}

// ---- subprocess + filesystem probes ---------------------------------------

std::pair<float, std::uint8_t> probe_rocm_smi(const Sources& s)
{
    if (!std::filesystem::exists(s.rocm_smi_bin)
        && s.rocm_smi_bin.is_absolute())
    {
        return {0.0F, 0};
    }
    const std::string cmd =
        shell_quote(s.rocm_smi_bin.string())
        + " --showtemp --showuse --json 2>/dev/null </dev/null";
    auto r = exec_capture(cmd);
    if (r.status != 0 || r.stdout_.empty()) {
        return {0.0F, 0};
    }
    try {
        auto v = json::parse(r.stdout_);
        return parse_rocm_smi_json(v);
    } catch (const json::parse_error&) {
        return {0.0F, 0};
    }
}

bool probe_npu(const Sources& s)
{
    if (s.xrt_smi_bin.is_absolute() && !path_exists(s.xrt_smi_bin)) {
        return path_exists(s.accel_dev);
    }
    const std::string cmd =
        shell_quote(s.xrt_smi_bin.string())
        + " examine >/dev/null 2>/dev/null </dev/null";
    auto r = exec_capture(cmd);
    if (r.status == 0) {
        return true;
    }
    // Fallback to /dev/accel/accel0 presence (matches Rust path).
    return path_exists(s.accel_dev);
}

float probe_shadow_burn(const std::filesystem::path& p)
{
    constexpr std::size_t WINDOW = 200;
    std::ifstream in(p);
    if (!in.is_open()) {
        return 0.0F;
    }
    std::deque<std::string> ring;
    std::string line;
    while (std::getline(in, line)) {
        if (ring.size() == WINDOW) {
            ring.pop_front();
        }
        ring.push_back(line);
    }
    if (ring.empty()) {
        return 0.0F;
    }
    std::uint32_t matches = 0;
    std::uint32_t total   = 0;
    for (const auto& l : ring) {
        if (l.empty()) { continue; }
        try {
            auto v = json::parse(l);
            ++total;
            auto it = v.find("full_match");
            if (it != v.end() && it->is_boolean() && it->get<bool>()) {
                ++matches;
            }
        } catch (const json::parse_error&) {
            // skip — matches Rust's `continue`.
        }
    }
    if (total == 0) {
        return 0.0F;
    }
    return static_cast<float>(matches) * 100.0F / static_cast<float>(total);
}

std::vector<ServiceState> probe_services(const Sources& s)
{
    std::vector<ServiceState> out;
    out.reserve(s.services.size());
    const bool sysctl_missing =
        s.systemctl_bin.is_absolute() && !path_exists(s.systemctl_bin);
    for (const auto& name : s.services) {
        bool active = false;
        if (!sysctl_missing) {
            const std::string cmd =
                shell_quote(s.systemctl_bin.string())
                + " --user is-active " + shell_quote(name)
                + " >/dev/null 2>/dev/null </dev/null";
            auto r = exec_capture(cmd);
            active = (r.status == 0);
        }
        out.push_back(ServiceState{name, active});
    }
    return out;
}

// ---- Telemetry -------------------------------------------------------------

Telemetry::Telemetry(Sources sources, std::chrono::milliseconds ttl)
    : sources_(std::move(sources)), ttl_(ttl) {}

Stats Telemetry::snapshot()
{
    {
        std::lock_guard lk(mu_);
        if (fetched_at_.has_value()) {
            const auto age = std::chrono::steady_clock::now() - *fetched_at_;
            if (age < ttl_) {
                return cached_;
            }
        }
    }

    Stats fresh = collect();

    std::lock_guard lk(mu_);
    cached_     = fresh;
    fetched_at_ = std::chrono::steady_clock::now();
    return fresh;
}

Stats Telemetry::collect()
{
    std::string prev_model;
    {
        std::lock_guard lk(mu_);
        prev_model = cached_.loaded_model;
    }

    auto model_opt = probe_model(sources_);
    std::string model = model_opt.value_or(prev_model);
    bool stale        = !model_opt.has_value();

    float tok_s_decode  = probe_tokps(sources_).value_or(0.0F);
    auto [temp, util]   = probe_rocm_smi(sources_);
    bool npu_up         = probe_npu(sources_);
    float burn          = probe_shadow_burn(sources_.shadow_burnin_jsonl);
    auto services       = probe_services(sources_);

    Stats s;
    s.loaded_model          = std::move(model);
    s.tok_s_decode          = tok_s_decode;
    s.gpu_temp_c            = temp;
    s.gpu_util_pct          = util;
    s.npu_up                = npu_up;
    s.shadow_burn_exact_pct = burn;
    s.services              = std::move(services);
    s.stale                 = stale;
    return s;
}

} // namespace onebit::landing
