#pragma once

// HTTP router for 1bit-landing (port of crates/1bit-landing/src/main.rs).
//
// Routes (must stay byte-stable with the Rust port):
//   GET /                    — embedded index.html
//   GET /style.css           — embedded css
//   GET /logo.svg            — embedded svg
//   GET /_live/status        — one-shot LiveStatus JSON (legacy)
//   GET /_live/stats         — SSE, full Stats snapshot every ~1.5 s
//   GET /_live/services      — SSE, service-flip deltas only
//   GET /_health             — text/plain "ok"
//
// `Router::handle(method, path, body)` is exposed for unit tests — it
// runs handler logic without opening a TCP port. The SSE endpoints
// return only their first frame in this synchronous form (sufficient
// for shape assertions). The streaming loop lives in Server.

#include "onebit/landing/status.hpp"
#include "onebit/landing/telemetry.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

namespace onebit::landing {

struct Response {
    int                                    status{200};
    std::string                            content_type{"text/plain; charset=utf-8"};
    std::string                            body{};
};

// SSE cadence for /_live/stats and /_live/services. Must stay >= the
// telemetry cache TTL so repeated ticks don't always collide with the
// cache boundary and miss a refresh.
inline constexpr std::chrono::milliseconds SSE_INTERVAL{1500};

class Router {
public:
    Router(std::shared_ptr<Telemetry> telemetry,
           std::shared_ptr<LemondProbe> lemond);

    [[nodiscard]] Response handle(std::string_view method,
                                  std::string_view path,
                                  std::string_view body = {}) const;

    [[nodiscard]] const Telemetry&   telemetry() const noexcept { return *telemetry_; }
    [[nodiscard]] Telemetry&         telemetry()       noexcept { return *telemetry_; }
    [[nodiscard]] const LemondProbe& lemond()    const noexcept { return *lemond_; }

    // Build a Router with default production wiring: HttpLemondProbe at
    // `lemond_url`, Telemetry::defaults() with onebit_server_base set.
    [[nodiscard]] static Router make_default(const std::string& lemond_url);

private:
    [[nodiscard]] Response live_status() const;
    [[nodiscard]] Response live_stats_first_frame() const;
    [[nodiscard]] Response live_services_first_frame() const;

    std::shared_ptr<Telemetry>   telemetry_;
    std::shared_ptr<LemondProbe> lemond_;
};

} // namespace onebit::landing
