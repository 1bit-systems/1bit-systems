#include <doctest/doctest.h>

#include "onebit/power/ec.hpp"
#include "onebit/power/metrics.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>

using onebit::power::collect_sample;
using onebit::power::EcBackend;
using onebit::power::sample_to_json;

namespace fs = std::filesystem;

TEST_CASE("collect_sample emits parseable JSON with expected keys")
{
    // Use an empty tempdir as EC root so EC fields all become null.
    auto tmp = fs::temp_directory_path() /
               ("onebit-power-metrics-" + std::to_string(::getpid()));
    fs::create_directories(tmp);
    EcBackend ec{tmp};

    auto r = collect_sample(ec);
    fs::remove_all(tmp);
    REQUIRE(r.ok());

    const auto json_text = sample_to_json(r.value());
    auto v = nlohmann::json::parse(json_text);
    CHECK(v.contains("ts_unix"));
    CHECK(v.contains("host"));
    CHECK(v.contains("tctl_c"));
    CHECK(v.contains("edge_c"));
    CHECK(v.contains("pkg_power_w"));
    CHECK(v.contains("ec_temp_c"));
    CHECK(v.contains("ec_power_mode"));
    CHECK(v.contains("ec_fan1_rpm"));
    CHECK(v.contains("ec_fan2_rpm"));
    CHECK(v.contains("ec_fan3_rpm"));

    // EC unavailable → all EC fields null.
    CHECK(v["ec_temp_c"].is_null());
    CHECK(v["ec_power_mode"].is_null());
    CHECK(v["ec_fan1_rpm"].is_null());
}

TEST_CASE("read_hostname returns non-empty string")
{
    auto h = onebit::power::read_hostname();
    CHECK_FALSE(h.empty());
}

TEST_CASE("sample_to_json round-trips an Optional<std::string>")
{
    onebit::power::Sample s;
    s.ts_unix = 1234567890;
    s.host = "test-host";
    s.ec_power_mode = std::string{"balanced"};
    auto j = nlohmann::json::parse(sample_to_json(s));
    CHECK(j["host"] == "test-host");
    CHECK(j["ts_unix"] == 1234567890);
    CHECK(j["ec_power_mode"] == "balanced");
    CHECK(j["tctl_c"].is_null());
}

TEST_CASE("sample_to_json emits null when Optional is empty")
{
    onebit::power::Sample s;
    s.ts_unix = 0;
    s.host    = "h";
    auto j = nlohmann::json::parse(sample_to_json(s));
    CHECK(j["tctl_c"].is_null());
    CHECK(j["pkg_power_w"].is_null());
    CHECK(j["ec_fan2_rpm"].is_null());
}

TEST_CASE("read_hwmon returns nullopt when looking for impossible name")
{
    auto v = onebit::power::read_hwmon("definitely-not-a-real-hwmon-name-xyz",
                                       "temp1_input");
    CHECK_FALSE(v.has_value());
}
