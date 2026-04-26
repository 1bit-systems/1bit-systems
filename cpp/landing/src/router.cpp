#include "onebit/landing/router.hpp"

#include "onebit/landing/assets.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

namespace onebit::landing {

namespace {

[[nodiscard]] std::string ieq_lower(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

[[nodiscard]] Response not_found()
{
    return Response{404, "text/plain; charset=utf-8", "not found"};
}

[[nodiscard]] Response method_not_allowed()
{
    return Response{405, "text/plain; charset=utf-8", "method not allowed"};
}

} // namespace

Router::Router(std::shared_ptr<Telemetry>   telemetry,
               std::shared_ptr<LemondProbe> lemond)
    : telemetry_(std::move(telemetry)), lemond_(std::move(lemond))
{
}

Router Router::make_default(const std::string& lemond_url)
{
    auto sources = Sources::defaults();
    sources.onebit_server_base = lemond_url;
    auto telem = std::make_shared<Telemetry>(std::move(sources));
    auto probe = std::make_shared<HttpLemondProbe>(lemond_url);
    return Router{std::move(telem), std::move(probe)};
}

Response Router::live_status() const
{
    LiveStatus s = lemond_->probe();
    return Response{200, "application/json", s.to_json().dump()};
}

Response Router::live_stats_first_frame() const
{
    Stats snap = telemetry_->snapshot();
    std::string body = "data: " + snap.to_json().dump() + "\n\n";
    return Response{200, "text/event-stream", std::move(body)};
}

Response Router::live_services_first_frame() const
{
    Stats snap = telemetry_->snapshot();
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& s : snap.services) {
        arr.push_back(nlohmann::json{ {"name", s.name}, {"active", s.active} });
    }
    nlohmann::json payload{
        {"kind",     "snapshot"},
        {"services", std::move(arr)},
    };
    std::string body = "data: " + payload.dump() + "\n\n";
    return Response{200, "text/event-stream", std::move(body)};
}

Response Router::handle(std::string_view method,
                        std::string_view path,
                        std::string_view /*body*/) const
{
    const std::string m = ieq_lower(method);
    if (m != "get") {
        return method_not_allowed();
    }

    if (path == "/") {
        return Response{200, "text/html; charset=utf-8",
                        std::string{assets::INDEX_HTML}};
    }
    if (path == "/style.css") {
        return Response{200, "text/css; charset=utf-8",
                        std::string{assets::STYLE_CSS}};
    }
    if (path == "/logo.svg") {
        return Response{200, "image/svg+xml; charset=utf-8",
                        std::string{assets::LOGO_SVG}};
    }
    if (path == "/_health") {
        return Response{200, "text/plain; charset=utf-8", "ok"};
    }
    if (path == "/_live/status") {
        return live_status();
    }
    if (path == "/_live/stats") {
        return live_stats_first_frame();
    }
    if (path == "/_live/services") {
        return live_services_first_frame();
    }
    return not_found();
}

} // namespace onebit::landing
