#include "onebit/landing/server.hpp"
#include "onebit/landing/assets.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

namespace onebit::landing {

namespace {

// Helper — set headers + body from a Response onto httplib::Response.
void apply(const Response& src, httplib::Response& dst)
{
    dst.status = src.status;
    dst.set_content(src.body, src.content_type.c_str());
}

} // namespace

int run_server(const Config& cfg)
{
    auto router = std::make_shared<Router>(Router::make_default(cfg.lemond_url));

    httplib::Server server;

    // ---- static + JSON routes (delegate to Router::handle) ---------------
    auto bind_simple = [&](const char* path) {
        server.Get(path, [router, path](const httplib::Request&,
                                        httplib::Response& res) {
            apply(router->handle("GET", path), res);
        });
    };
    bind_simple("/");
    bind_simple("/style.css");
    bind_simple("/logo.svg");
    bind_simple("/_health");
    bind_simple("/_live/status");

    // ---- SSE: /_live/stats — full snapshot every SSE_INTERVAL ----------
    server.Get("/_live/stats", [router](const httplib::Request&,
                                        httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [router](std::size_t /*offset*/, httplib::DataSink& sink) {
                while (sink.is_writable()) {
                    Stats snap = router->telemetry().snapshot();
                    std::string frame = "data: " + snap.to_json().dump() + "\n\n";
                    if (!sink.write(frame.data(), frame.size())) {
                        break;
                    }
                    std::this_thread::sleep_for(SSE_INTERVAL);
                }
                sink.done();
                return true;
            });
    });

    // ---- SSE: /_live/services — first snapshot, then deltas only -------
    server.Get("/_live/services", [router](const httplib::Request&,
                                           httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [router](std::size_t /*offset*/, httplib::DataSink& sink) {
                if (!sink.is_writable()) {
                    sink.done();
                    return true;
                }

                Stats first = router->telemetry().snapshot();
                {
                    nlohmann::json arr = nlohmann::json::array();
                    for (const auto& s : first.services) {
                        arr.push_back(nlohmann::json{
                            {"name", s.name}, {"active", s.active}});
                    }
                    nlohmann::json payload{
                        {"kind",     "snapshot"},
                        {"services", std::move(arr)},
                    };
                    std::string frame = "data: " + payload.dump() + "\n\n";
                    if (!sink.write(frame.data(), frame.size())) {
                        sink.done();
                        return true;
                    }
                }

                std::vector<ServiceState> prev = first.services;
                while (sink.is_writable()) {
                    std::this_thread::sleep_for(SSE_INTERVAL);
                    Stats next = router->telemetry().snapshot();
                    if (auto delta = service_delta(prev, next.services)) {
                        nlohmann::json arr = nlohmann::json::array();
                        for (const auto& s : *delta) {
                            arr.push_back(nlohmann::json{
                                {"name", s.name}, {"active", s.active}});
                        }
                        nlohmann::json payload{
                            {"kind",     "delta"},
                            {"services", std::move(arr)},
                        };
                        std::string frame = "data: " + payload.dump() + "\n\n";
                        if (!sink.write(frame.data(), frame.size())) {
                            break;
                        }
                        prev = next.services;
                    }
                }
                sink.done();
                return true;
            });
    });

    // ---- catch-all 404 ---------------------------------------------------
    server.set_error_handler([](const httplib::Request&,
                                httplib::Response& res) {
        if (res.status == -1) {
            res.status = 404;
        }
    });

    std::fprintf(stderr,
                 "1bit-landing listening on %s:%d (lemond=%s)\n",
                 cfg.bind.c_str(), cfg.port, cfg.lemond_url.c_str());

    if (!server.listen(cfg.bind.c_str(), cfg.port)) {
        std::fprintf(stderr,
                     "1bit-landing: bind %s:%d failed\n",
                     cfg.bind.c_str(), cfg.port);
        return 1;
    }
    return 0;
}

} // namespace onebit::landing
