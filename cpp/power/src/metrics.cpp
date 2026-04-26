#include "onebit/power/metrics.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace onebit::power {

namespace {

[[nodiscard]] std::string trim_copy(std::string s)
{
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r'
                       || s.back() == ' '  || s.back() == '\t')) s.pop_back();
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

[[nodiscard]] std::optional<std::string> read_file(const std::filesystem::path& p)
{
    std::ifstream f{p};
    if (!f) return std::nullopt;
    std::ostringstream buf; buf << f.rdbuf();
    return trim_copy(buf.str());
}

} // namespace

std::string read_hostname()
{
    auto v = read_file("/proc/sys/kernel/hostname");
    return v.value_or("unknown");
}

std::optional<float> read_hwmon(std::string_view want, std::string_view file)
{
    std::error_code ec;
    const std::filesystem::path root{"/sys/class/hwmon"};
    if (!std::filesystem::is_directory(root, ec)) return std::nullopt;

    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) return std::nullopt;
        auto name = read_file(entry.path() / "name");
        if (!name || *name != want) continue;
        auto raw = read_file(entry.path() / file);
        if (!raw) return std::nullopt;
        try {
            return std::stof(*raw);
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

Result<Sample> collect_sample(const EcBackend& ec)
{
    Sample s;
    s.ts_unix = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    s.host = read_hostname();

    if (auto v = read_hwmon("k10temp", "temp1_input"))     s.tctl_c     = *v / 1000.0f;
    if (auto v = read_hwmon("amdgpu",  "temp1_input"))     s.edge_c     = *v / 1000.0f;
    if (auto v = read_hwmon("amdgpu",  "power1_average"))  s.pkg_power_w = *v / 1'000'000.0f;

    if (ec.available()) {
        if (auto r = ec.temp_c();     r) s.ec_temp_c     = r.value();
        if (auto r = ec.power_mode(); r) s.ec_power_mode = r.value();
        if (auto r = ec.fan(1);       r) s.ec_fan1_rpm   = r.value().rpm;
        if (auto r = ec.fan(2);       r) s.ec_fan2_rpm   = r.value().rpm;
        if (auto r = ec.fan(3);       r) s.ec_fan3_rpm   = r.value().rpm;
    }
    return s;
}

std::string sample_to_json(const Sample& s)
{
    // NB: must use parens, not braces — `nlohmann::json{x}` invokes the
    // initializer-list ctor and produces a 1-element ARRAY, not a scalar.
    auto opt_s = [](const std::optional<std::string>& v) -> nlohmann::json {
        return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
    };
    auto opt_f = [](const std::optional<float>& v) -> nlohmann::json {
        return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
    };
    auto opt_i = [](const std::optional<std::int32_t>& v) -> nlohmann::json {
        return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
    };
    auto opt_u = [](const std::optional<std::uint32_t>& v) -> nlohmann::json {
        return v ? nlohmann::json(*v) : nlohmann::json(nullptr);
    };

    // Match Rust serde key order.
    nlohmann::ordered_json j;
    j["ts_unix"]       = s.ts_unix;
    j["host"]          = s.host;
    j["tctl_c"]        = opt_f(s.tctl_c);
    j["edge_c"]        = opt_f(s.edge_c);
    j["pkg_power_w"]   = opt_f(s.pkg_power_w);
    j["ec_temp_c"]     = opt_i(s.ec_temp_c);
    j["ec_power_mode"] = opt_s(s.ec_power_mode);
    j["ec_fan1_rpm"]   = opt_u(s.ec_fan1_rpm);
    j["ec_fan2_rpm"]   = opt_u(s.ec_fan2_rpm);
    j["ec_fan3_rpm"]   = opt_u(s.ec_fan3_rpm);
    return j.dump();
}

} // namespace onebit::power
