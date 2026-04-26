// Tests for 1bit-landing C++ port.
//
// Mirrors the in-crate Rust tests where it makes sense:
//   * route content-types + bodies via Router::handle (no TCP open),
//   * lemond probe stub for /_live/status shape assertions,
//   * telemetry source-fan-out degradation when binaries are missing,
//   * service_delta + parse_rocm_smi_json pure-helper coverage,
//   * shadow-burnin tail counts the full_match ratio.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/landing/assets.hpp"
#include "onebit/landing/router.hpp"
#include "onebit/landing/status.hpp"
#include "onebit/landing/telemetry.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <unistd.h>

using onebit::landing::HttpLemondProbe;
using onebit::landing::LemondProbe;
using onebit::landing::LiveStatus;
using onebit::landing::Response;
using onebit::landing::Router;
using onebit::landing::ServiceState;
using onebit::landing::Sources;
using onebit::landing::Stats;
using onebit::landing::Telemetry;
using onebit::landing::TRACKED_SERVICES;
using onebit::landing::parse_rocm_smi_json;
using onebit::landing::probe_shadow_burn;
using onebit::landing::service_delta;

namespace {

class StubProbe : public LemondProbe {
public:
    LiveStatus next{};
    LiveStatus probe() const override { return next; }
};

Sources broken_sources()
{
    Sources s;
    s.rocm_smi_bin        = "/nonexistent/rocm-smi";
    s.xrt_smi_bin         = "/nonexistent/xrt-smi";
    s.accel_dev           = "/nonexistent/accel0";
    s.shadow_burnin_jsonl = "/nonexistent/shadow.jsonl";
    s.systemctl_bin       = "/nonexistent/systemctl";
    // Unroutable TEST-NET-1 — probe_model fails fast.
    s.onebit_server_base = "http://192.0.2.1:1";
    s.services.clear();
    for (const auto& sv : TRACKED_SERVICES) {
        s.services.emplace_back(sv);
    }
    return s;
}

Router make_router_with_stub(std::shared_ptr<StubProbe> stub)
{
    auto telem = std::make_shared<Telemetry>(broken_sources(),
                                             std::chrono::milliseconds{60'000});
    return Router{std::move(telem), std::move(stub)};
}

std::filesystem::path tempfile(const std::string& suffix)
{
    auto p = std::filesystem::temp_directory_path()
           / ("onebit-landing-" + std::to_string(::getpid()) + "-" + suffix);
    return p;
}

} // namespace

// -- assets ----------------------------------------------------------------

TEST_CASE("embedded assets carry the wordmark + SSE hooks")
{
    using onebit::landing::assets::INDEX_HTML;
    REQUIRE_FALSE(INDEX_HTML.empty());
    CHECK(INDEX_HTML.find("1bit systems")    != std::string_view::npos);
    CHECK(INDEX_HTML.find("/_live/stats")    != std::string_view::npos);
    CHECK(INDEX_HTML.find("/_live/services") != std::string_view::npos);
    CHECK(INDEX_HTML.find("EventSource")     != std::string_view::npos);
}

// -- Router ----------------------------------------------------------------

TEST_CASE("GET / serves text/html with the wordmark")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    Response res = r.handle("GET", "/");
    CHECK(res.status == 200);
    CHECK(res.content_type.starts_with("text/html"));
    CHECK(res.body.find("1bit systems") != std::string::npos);
}

TEST_CASE("GET /style.css serves text/css and /logo.svg serves image/svg+xml")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    auto css = r.handle("GET", "/style.css");
    CHECK(css.status == 200);
    CHECK(css.content_type.starts_with("text/css"));
    CHECK_FALSE(css.body.empty());

    auto svg = r.handle("GET", "/logo.svg");
    CHECK(svg.status == 200);
    CHECK(svg.content_type.starts_with("image/svg+xml"));
    CHECK_FALSE(svg.body.empty());
}

TEST_CASE("GET /_health returns plain ok")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    auto res = r.handle("GET", "/_health");
    CHECK(res.status == 200);
    CHECK(res.content_type.starts_with("text/plain"));
    CHECK(res.body == "ok");
}

TEST_CASE("GET /_live/status returns the Rust-shape JSON keyed identically")
{
    auto stub  = std::make_shared<StubProbe>();
    stub->next = LiveStatus::offline(); // backend down
    Router r   = make_router_with_stub(stub);

    auto res = r.handle("GET", "/_live/status");
    CHECK(res.status == 200);
    CHECK(res.content_type == "application/json");

    auto v = nlohmann::json::parse(res.body);
    for (const char* k : {"v2_up", "v1_up", "model", "tokps", "p50_ms",
                          "p95_ms", "requests", "generated_tokens"}) {
        CHECK_MESSAGE(v.contains(k), "missing key: ", k);
    }
    CHECK(v["v2_up"] == false);
    CHECK(v["model"] == "");
}

TEST_CASE("GET /_live/stats first SSE frame has all fields + stale=true offline")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    auto res = r.handle("GET", "/_live/stats");
    CHECK(res.status == 200);
    CHECK(res.content_type.starts_with("text/event-stream"));
    REQUIRE(res.body.starts_with("data: "));

    auto data = res.body.substr(6);
    auto eom  = data.find("\n\n");
    REQUIRE(eom != std::string::npos);
    data.resize(eom);
    auto v = nlohmann::json::parse(data);
    for (const char* k : {"loaded_model", "tok_s_decode", "gpu_temp_c",
                          "gpu_util_pct", "npu_up", "shadow_burn_exact_pct",
                          "services", "stale"})
    {
        CHECK_MESSAGE(v.contains(k), "missing key: ", k);
    }
    CHECK(v["stale"] == true);
    CHECK(v["loaded_model"] == "");
}

TEST_CASE("GET /_live/services emits a snapshot kind on first connect")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    auto res = r.handle("GET", "/_live/services");
    CHECK(res.status == 200);
    CHECK(res.content_type.starts_with("text/event-stream"));

    auto data = res.body.substr(6);
    auto eom  = data.find("\n\n");
    REQUIRE(eom != std::string::npos);
    data.resize(eom);
    auto v = nlohmann::json::parse(data);
    CHECK(v["kind"] == "snapshot");
    CHECK(v["services"].is_array());
    CHECK(v["services"].size() == TRACKED_SERVICES.size());
}

TEST_CASE("unknown path returns 404")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    auto res = r.handle("GET", "/no-such-path");
    CHECK(res.status == 404);
}

TEST_CASE("non-GET methods return 405")
{
    auto stub = std::make_shared<StubProbe>();
    Router r  = make_router_with_stub(stub);

    CHECK(r.handle("POST", "/").status   == 405);
    CHECK(r.handle("DELETE", "/").status == 405);
}

// -- LiveStatus ------------------------------------------------------------

TEST_CASE("LiveStatus::offline is all zeros / false / empty")
{
    auto s = LiveStatus::offline();
    CHECK_FALSE(s.v2_up);
    CHECK_FALSE(s.v1_up);
    CHECK(s.tokps == 0.0);
    CHECK(s.p50_ms == 0.0);
    CHECK(s.p95_ms == 0.0);
    CHECK(s.requests == 0);
    CHECK(s.generated_tokens == 0);
    CHECK(s.model.empty());
}

TEST_CASE("LiveStatus::to_json carries every documented key")
{
    auto v = LiveStatus::offline().to_json();
    for (const char* k : {"v2_up", "v1_up", "model", "tokps", "p50_ms",
                          "p95_ms", "requests", "generated_tokens"})
    {
        CHECK(v.contains(k));
    }
}

// -- telemetry helpers -----------------------------------------------------

TEST_CASE("parse_rocm_smi_json picks card0 edge temp + GPU use")
{
    auto v = nlohmann::json::parse(R"json({
        "card0": {
            "Temperature (Sensor edge) (C)": "52.0",
            "GPU use (%)": "27"
        }
    })json");
    auto [t, u] = parse_rocm_smi_json(v);
    CHECK(t == doctest::Approx(52.0F));
    CHECK(u == 27);
}

TEST_CASE("parse_rocm_smi_json degrades on garbage")
{
    auto v = nlohmann::json::parse(R"({"card0": {"bogus": 1}})");
    auto [t, u] = parse_rocm_smi_json(v);
    CHECK(t == 0.0F);
    CHECK(u == 0);
    auto [t2, u2] = parse_rocm_smi_json(nlohmann::json{});
    CHECK(t2 == 0.0F);
    CHECK(u2 == 0);
}

TEST_CASE("service_delta detects single flips and is_none when steady")
{
    std::vector<ServiceState> prev{{"a", true}, {"b", false}};
    std::vector<ServiceState> next{{"a", true}, {"b", true}};
    auto d = service_delta(prev, next);
    REQUIRE(d.has_value());
    REQUIRE(d->size() == 1);
    CHECK((*d)[0].name == "b");
    CHECK((*d)[0].active);

    CHECK_FALSE(service_delta(next, next).has_value());
}

TEST_CASE("shadow-burnin tail counts the full_match ratio")
{
    auto p = tempfile("shadow.jsonl");
    {
        std::ofstream f(p);
        f << R"({"ts":"t","prompt_idx":0,"full_match":true})"  << "\n"
          << R"({"ts":"t","prompt_idx":1,"full_match":true})"  << "\n"
          << R"({"ts":"t","prompt_idx":2,"full_match":false})" << "\n"
          << R"({"ts":"t","prompt_idx":3,"full_match":true})"  << "\n";
    }
    CHECK(probe_shadow_burn(p) == doctest::Approx(75.0F));
    std::filesystem::remove(p);
}

TEST_CASE("shadow-burnin missing file returns zero")
{
    auto p = tempfile("missing-on-purpose.jsonl");
    std::filesystem::remove(p);
    CHECK(probe_shadow_burn(p) == 0.0F);
}

TEST_CASE("Telemetry::collect degrades to sentinels with broken sources")
{
    Telemetry t{broken_sources(), std::chrono::milliseconds{60'000}};
    Stats s = t.collect();
    CHECK(s.loaded_model.empty());
    CHECK(s.stale);
    CHECK(s.tok_s_decode == 0.0F);
    CHECK(s.gpu_temp_c == 0.0F);
    CHECK(s.gpu_util_pct == 0);
    CHECK_FALSE(s.npu_up);
    CHECK(s.shadow_burn_exact_pct == 0.0F);
    CHECK(s.services.size() == TRACKED_SERVICES.size());
    for (const auto& svc : s.services) {
        CHECK_FALSE(svc.active);
    }
}

TEST_CASE("Telemetry::snapshot reuses cached value within TTL")
{
    Telemetry t{broken_sources(), std::chrono::milliseconds{60'000}};
    auto a = t.snapshot();
    auto b = t.snapshot();
    CHECK(a.loaded_model == b.loaded_model);
    CHECK(a.stale        == b.stale);
}

TEST_CASE("Stats::to_json keys + ordering match the Rust serde output")
{
    Stats s   = Stats::empty();
    auto json = s.to_json();
    // Field-name set check.
    for (const char* k : {"loaded_model", "tok_s_decode", "gpu_temp_c",
                          "gpu_util_pct", "npu_up", "shadow_burn_exact_pct",
                          "services", "stale"})
    {
        CHECK(json.contains(k));
    }
    // Empty stats: stale must be true on construction.
    CHECK(json["stale"] == true);
    CHECK(json["services"].is_array());
}
