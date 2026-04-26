#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/stream/handlers.hpp"

#include "onebit/ingest/ingest.hpp"
#include "onebit/stream/jwt.hpp"

#include <httplib.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

[[nodiscard]] fs::path mktemp_dir()
{
    auto                                         base = fs::temp_directory_path();
    std::random_device                           rd;
    std::uniform_int_distribution<std::uint64_t> dist;
    for (int i = 0; i < 16; ++i) {
        auto candidate = base / ("onebit-stream-h-" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
    }
    return base / "onebit-stream-h-fallback";
}

void write_file(const fs::path& p, std::string_view body)
{
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(body.data(), static_cast<std::streamsize>(body.size()));
}

[[nodiscard]] std::string sample_toml(std::string_view cat)
{
    std::string out;
    out += "catalog = \"";
    out.append(cat);
    out +=
        "\"\ntitle = \"T\"\nartist = \"T\"\nlicense = \"CC0-1.0\"\n"
        "created = \"2026-04-23T00:00:00Z\"\ntier = \"both\"\n"
        "license_txt = \"pd\"\n"
        "[codec]\naudio = \"mimi-12hz\"\nsample_rate = 24000\nchannels = 2\n"
        "[model]\narch = \"bitnet-1p58\"\nparams = 1\nbpw = 1.58\n";
    return out;
}

void write_test_catalog(const fs::path& dir, std::string_view slug)
{
    const auto gguf = dir / (std::string{slug} + ".gguf");
    write_file(gguf, "GGUF fake bytes");
    const auto tomlp = dir / (std::string{slug} + ".toml");
    write_file(tomlp, sample_toml(slug));
    const auto out = dir / (std::string{slug} + ".1bl");
    auto       p   = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE(p.has_value());
}

[[nodiscard]] std::span<const std::uint8_t> as_bytes(std::string_view s) noexcept
{
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

class Server {
public:
    Server(onebit::stream::AppState& state, int port) : port_{port}
    {
        onebit::stream::build(http_, state);
        thread_ = std::thread([this]() { http_.listen("127.0.0.1", port_); });
        // Wait for bind.
        for (int i = 0; i < 40; ++i) {
            if (http_.is_running()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    ~Server()
    {
        http_.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    [[nodiscard]] int port() const noexcept { return port_; }

private:
    httplib::Server http_;
    std::thread     thread_;
    int             port_;
};

[[nodiscard]] int random_port() noexcept
{
    std::random_device              rd;
    std::uniform_int_distribution<> d(20000, 30000);
    return d(rd);
}

} // namespace

TEST_CASE("empty catalog dir lists empty array")
{
    const auto tmp = mktemp_dir();
    onebit::stream::AppState state(tmp, onebit::stream::AuthConfig{});
    const auto rep = state.reindex();
    CHECK(rep.loaded == 0);

    Server         srv(state, random_port());
    httplib::Client cli("127.0.0.1", srv.port());
    auto           res = cli.Get("/v1/catalogs");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("\"data\":[]") != std::string::npos);
    fs::remove_all(tmp);
}

TEST_CASE("health endpoint returns ok")
{
    const auto tmp = mktemp_dir();
    onebit::stream::AppState state(tmp, onebit::stream::AuthConfig{});
    Server                   srv(state, random_port());
    httplib::Client          cli("127.0.0.1", srv.port());
    auto                     res = cli.Get("/v1/health");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body == "ok");
    fs::remove_all(tmp);
}

TEST_CASE("missing catalog returns 404")
{
    const auto tmp = mktemp_dir();
    onebit::stream::AppState state(tmp, onebit::stream::AuthConfig{});
    Server                   srv(state, random_port());
    httplib::Client          cli("127.0.0.1", srv.port());
    auto                     res = cli.Get("/v1/catalogs/does-not-exist");
    REQUIRE(res);
    CHECK(res->status == 404);
    fs::remove_all(tmp);
}

TEST_CASE("reindex endpoint fails closed when admin bearer not configured")
{
    // Regression: previously this returned 200 (fail-OPEN) when the admin
    // secret was empty, exposing /internal/reindex to the world. Now the
    // auth layer returns ServerMisconfigured (HTTP 503) until an operator
    // sets HALO_STREAM_ADMIN_BEARER.
    const auto tmp = mktemp_dir();
    onebit::stream::AppState state(tmp, onebit::stream::AuthConfig{});
    Server                   srv(state, random_port());
    httplib::Client          cli("127.0.0.1", srv.port());
    auto                     res = cli.Post("/internal/reindex", "", "text/plain");
    REQUIRE(res);
    CHECK(res->status == 503);
    fs::remove_all(tmp);
}

TEST_CASE("populated catalog list shows the catalog")
{
    const auto tmp = mktemp_dir();
    write_test_catalog(tmp, "kevin-mini");
    onebit::stream::AppState state(tmp, onebit::stream::AuthConfig{});
    state.reindex();
    Server          srv(state, random_port());
    httplib::Client cli("127.0.0.1", srv.port());
    auto            res = cli.Get("/v1/catalogs");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->body.find("kevin-mini") != std::string::npos);
    fs::remove_all(tmp);
}

TEST_CASE("lossless endpoint requires premium token")
{
    const auto tmp = mktemp_dir();
    write_test_catalog(tmp, "premium-cat");

    const std::string secret = "test-stream-secret-32-bytes-long-x";
    onebit::stream::AppState state(
        tmp, onebit::stream::AuthConfig::make({secret.begin(), secret.end()}, ""));
    state.reindex();
    Server          srv(state, random_port());
    httplib::Client cli("127.0.0.1", srv.port());

    // Without token => 401.
    {
        auto res = cli.Get("/v1/catalogs/premium-cat/lossless");
        REQUIRE(res);
        CHECK(res->status == 401);
    }

    // With premium token => 200.
    {
        const auto token = onebit::stream::jwt::mint_hs256(
            as_bytes(secret),
            R"({"sub":"u","tier":"premium","iss":"1bit.systems"})");
        httplib::Headers h{{"Authorization", "Bearer " + token}};
        auto             res = cli.Get("/v1/catalogs/premium-cat/lossless", h);
        REQUIRE(res);
        CHECK(res->status == 200);
        CHECK(!res->body.empty());
    }
    fs::remove_all(tmp);
}
