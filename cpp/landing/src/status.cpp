#include "onebit/landing/status.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <regex>
#include <string>

namespace onebit::landing {

namespace {

using json = nlohmann::json;

// Split an "http(s)://host[:port]" URL into pieces httplib::Client likes.
struct UrlParts {
    std::string scheme; // "http" or "https"
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
    if (m[3].matched) {
        u.port = std::stoi(m[3].str());
    } else {
        u.port = (u.scheme == "https") ? 443 : 80;
    }
    u.ok = true;
    return u;
}

// HTTP GET with a 2 s timeout. Returns the parsed JSON body on 2xx, or
// std::nullopt on any failure (timeout, refused, non-2xx, malformed).
[[nodiscard]] std::optional<json> http_get_json(const std::string& base_url,
                                                std::string_view   path)
{
    auto u = split_url(base_url);
    if (!u.ok) {
        return std::nullopt;
    }
    httplib::Client cli(u.scheme + "://" + u.host + ":" + std::to_string(u.port));
    cli.set_connection_timeout(std::chrono::seconds{2});
    cli.set_read_timeout(std::chrono::seconds{2});
    cli.set_write_timeout(std::chrono::seconds{2});

    auto res = cli.Get(std::string{path});
    if (!res) {
        return std::nullopt;
    }
    if (res->status < 200 || res->status >= 300) {
        return std::nullopt;
    }
    try {
        return json::parse(res->body);
    } catch (const json::parse_error&) {
        return std::nullopt;
    }
}

} // namespace

LiveStatus HttpLemondProbe::probe() const
{
    auto models = http_get_json(base_url_, "/v1/models");
    if (!models) {
        return LiveStatus::offline();
    }

    // Extract first model's "id" — `{"object":"list","data":[{"id":...}]}`.
    std::string model = "unknown";
    if (auto it = models->find("data");
        it != models->end() && it->is_array() && !it->empty())
    {
        const auto& first = (*it)[0];
        if (auto id = first.find("id"); id != first.end() && id->is_string()) {
            model = id->get<std::string>();
        }
    }

    LiveStatus s;
    s.v2_up = true;
    s.v1_up = false; // gen-1 check is future work, mirroring Rust
    s.model = std::move(model);

    if (auto m = http_get_json(base_url_, "/metrics")) {
        auto get_f64 = [&](const char* k) -> double {
            if (auto it = m->find(k); it != m->end() && it->is_number()) {
                return it->get<double>();
            }
            return 0.0;
        };
        auto get_u64 = [&](const char* k) -> std::uint64_t {
            if (auto it = m->find(k); it != m->end() && it->is_number_unsigned()) {
                return it->get<std::uint64_t>();
            }
            if (auto it = m->find(k); it != m->end() && it->is_number_integer()) {
                auto v = it->get<long long>();
                return v < 0 ? 0 : static_cast<std::uint64_t>(v);
            }
            return 0;
        };
        s.tokps            = get_f64("tokps_recent");
        s.p50_ms           = get_f64("p50_ms");
        s.p95_ms           = get_f64("p95_ms");
        s.requests         = get_u64("requests");
        s.generated_tokens = get_u64("generated_tokens");
    }
    return s;
}

} // namespace onebit::landing
