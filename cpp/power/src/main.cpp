// 1bit-power — Linux power + thermal control for Strix Halo.
//
// C++20 port of crates/1bit-power. Mirrors the Rust crate's CLI surface
// bit-for-bit (subcommands, exit codes, stdout shape) so existing
// scripts continue to work. Two backends are available:
//
//   --backend shellout (default) : forks /usr/bin/ryzenadj
//   --backend lib                : dlopen()s libryzenadj.so directly
//
// Subcommands:
//   1bit-power status                    — JSON snapshot
//   1bit-power profile <name>            — apply profile from profiles.toml
//   1bit-power set <key> <value>         — one-shot knob override
//   1bit-power log                       — emit one JSON metric line
//   1bit-power board <mode>              — set EC apu/power_mode
//   1bit-power fan <id> mode <mode>
//   1bit-power fan <id> level <0..5>
//   1bit-power fan <id> curve <up|down> <a,b,c,d,e>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "onebit/power/ec.hpp"
#include "onebit/power/metrics.hpp"
#include "onebit/power/profile.hpp"
#include "onebit/power/result.hpp"
#include "onebit/power/ryzen.hpp"

namespace op = onebit::power;

namespace {

constexpr const char* kDefaultProfilesPath = "/etc/halo-power/profiles.toml";

void install_logger()
{
    // stderr only — never corrupt stdout (some commands emit JSON to stdout).
    auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();  // header is stdout_color_sinks.h despite name
    auto log  = std::make_shared<spdlog::logger>("1bit-power", sink);
    log->set_pattern("%^[%l]%$ %v");
    const char* env = std::getenv("RUST_LOG");
    if (!env) env = std::getenv("ONEBIT_LOG");
    if (env && std::string_view{env}.find("debug") != std::string_view::npos) {
        log->set_level(spdlog::level::debug);
    } else {
        log->set_level(spdlog::level::info);
    }
    spdlog::set_default_logger(log);
}

// Build the configured PowerBackend. Returns nullptr + prints to stderr
// on failure; callers should exit 1 in that case.
std::unique_ptr<op::PowerBackend> make_backend(std::string_view kind, bool dry_run)
{
    if (kind == "shellout") {
        return std::make_unique<op::ShelloutBackend>(dry_run);
    }
    if (kind == "lib") {
        auto r = op::LibBackend::open(dry_run);
        if (!r) {
            std::fprintf(stderr,
                "1bit-power: libryzenadj backend unavailable: %s\n"
                "  Falling back is not automatic; pass --backend shellout to use the fork/exec path.\n",
                r.status().message.c_str());
            return nullptr;
        }
        return std::move(r).value();
    }
    std::fprintf(stderr, "1bit-power: --backend must be `shellout` or `lib`, got `%.*s`\n",
                 static_cast<int>(kind.size()), kind.data());
    return nullptr;
}

int do_status(const op::PowerBackend& backend,
              std::string_view profiles_path,
              const op::EcBackend& ec,
              bool dry_run)
{
    op::Profiles profiles;
    if (auto r = op::Profiles::load(profiles_path); r) {
        profiles = std::move(r).value();
    } else {
        spdlog::warn("could not load profiles.toml: {}; showing defaults",
                     r.status().message);
    }

    nlohmann::ordered_json j;
    j["profiles_path"]   = std::string{profiles_path};
    j["known_profiles"]  = profiles.names();
    j["backend"]         = std::string{backend.name()};
    j["dry_run"]         = dry_run;
    if (ec.available()) {
        auto snap = ec.snapshot();
        nlohmann::ordered_json e;
        e["temp_c"]     = snap.temp_c.has_value()
                            ? nlohmann::json{*snap.temp_c}
                            : nlohmann::json{nullptr};
        e["power_mode"] = snap.power_mode.has_value()
                            ? nlohmann::json{*snap.power_mode}
                            : nlohmann::json{nullptr};
        nlohmann::json fans = nlohmann::json::array();
        for (const auto& f : snap.fans) {
            nlohmann::ordered_json fj;
            fj["id"]       = static_cast<int>(f.id);
            fj["rpm"]      = f.rpm;
            fj["mode"]     = f.mode;
            fj["level"]    = static_cast<int>(f.level);
            fj["rampup"]   = f.rampup;
            fj["rampdown"] = f.rampdown;
            fans.emplace_back(std::move(fj));
        }
        e["fans"] = fans;
        j["ec"] = e;
    } else {
        j["ec"] = nullptr;
    }
    std::cout << j.dump(2) << '\n';
    return 0;
}

int do_profile(op::PowerBackend& backend,
               std::string_view profiles_path,
               const std::string& name)
{
    auto loaded = op::Profiles::load(profiles_path);
    if (!loaded) {
        std::fprintf(stderr, "1bit-power: %s: %s\n",
            std::string{profiles_path}.c_str(),
            loaded.status().message.c_str());
        return 1;
    }
    const op::Profile* p = loaded.value().get(name);
    if (!p) {
        std::fprintf(stderr, "1bit-power: profile `%s` not in %s\n",
            name.c_str(), std::string{profiles_path}.c_str());
        return 1;
    }
    spdlog::info("applying profile {}", name);
    auto s = backend.apply_profile(*p);
    if (!s) {
        std::fprintf(stderr, "1bit-power: apply_profile failed: %s\n",
            s.message.c_str());
        return 1;
    }
    return 0;
}

int do_set(op::PowerBackend& backend, const std::string& key, std::uint32_t value)
{
    spdlog::info("one-shot set {}={}", key, value);
    auto s = backend.set_one(key, value);
    if (!s) {
        std::fprintf(stderr, "1bit-power: set %s=%u failed: %s\n",
            key.c_str(), value, s.message.c_str());
        return 1;
    }
    return 0;
}

int do_log(const op::EcBackend& ec)
{
    auto r = op::collect_sample(ec);
    if (!r) {
        std::fprintf(stderr, "1bit-power: collecting metrics: %s\n",
            r.status().message.c_str());
        return 1;
    }
    std::cout << op::sample_to_json(r.value()) << '\n';
    return 0;
}

int do_board(const op::EcBackend& ec, const std::string& mode, bool dry_run)
{
    if (dry_run) {
        spdlog::info("dry-run: would set apu/power_mode={}", mode);
        return 0;
    }
    auto s = ec.set_power_mode(mode);
    if (!s) {
        std::fprintf(stderr, "1bit-power: set_power_mode: %s\n", s.message.c_str());
        return 1;
    }
    spdlog::info("board power_mode set to {}", mode);
    return 0;
}

int do_fan_mode(const op::EcBackend& ec, std::uint8_t id, const std::string& mode, bool dry_run)
{
    if (dry_run) {
        spdlog::info("dry-run: would set fan{} mode={}", static_cast<int>(id), mode);
        return 0;
    }
    auto s = ec.set_fan_mode(id, mode);
    if (!s) {
        std::fprintf(stderr, "1bit-power: set_fan_mode: %s\n", s.message.c_str());
        return 1;
    }
    spdlog::info("fan{} mode set to {}", static_cast<int>(id), mode);
    return 0;
}

int do_fan_level(const op::EcBackend& ec, std::uint8_t id, std::uint8_t level, bool dry_run)
{
    if (dry_run) {
        spdlog::info("dry-run: would set fan{} level={}",
                     static_cast<int>(id), static_cast<int>(level));
        return 0;
    }
    if (auto s = ec.set_fan_mode(id, "fixed"); !s) {
        std::fprintf(stderr, "1bit-power: set_fan_mode(fixed): %s\n", s.message.c_str());
        return 1;
    }
    if (auto s = ec.set_fan_level(id, level); !s) {
        std::fprintf(stderr, "1bit-power: set_fan_level: %s\n", s.message.c_str());
        return 1;
    }
    spdlog::info("fan{} level set to {}", static_cast<int>(id), static_cast<int>(level));
    return 0;
}

int do_fan_curve(const op::EcBackend& ec,
                 std::uint8_t id,
                 const std::string& direction,
                 const std::string& csv,
                 bool dry_run)
{
    op::CurveDir dir;
    if (direction == "up" || direction == "rampup") dir = op::CurveDir::Rampup;
    else if (direction == "down" || direction == "rampdown") dir = op::CurveDir::Rampdown;
    else {
        std::fprintf(stderr, "1bit-power: direction must be up|down, got `%s`\n",
                     direction.c_str());
        return 1;
    }
    auto vals = op::parse_curve_csv(csv);
    if (vals.size() != 5) {
        std::fprintf(stderr, "1bit-power: need exactly 5 values, got %zu\n", vals.size());
        return 1;
    }
    std::array<std::uint8_t, 5> arr{vals[0], vals[1], vals[2], vals[3], vals[4]};
    if (dry_run) {
        spdlog::info("dry-run: would set fan{} curve dir={} = [{},{},{},{},{}]",
                     static_cast<int>(id), direction,
                     arr[0], arr[1], arr[2], arr[3], arr[4]);
        return 0;
    }
    auto s = ec.set_fan_curve(id, dir, arr);
    if (!s) {
        std::fprintf(stderr, "1bit-power: set_fan_curve: %s\n", s.message.c_str());
        return 1;
    }
    spdlog::info("fan{} curve set", static_cast<int>(id));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    install_logger();

    CLI::App app{"Strix Halo power/thermal control (RyzenAdj wrapper)"};
    app.name("1bit-power");
    app.require_subcommand(1);

    std::string profiles_path = kDefaultProfilesPath;
    bool        dry_run       = false;
    std::string backend_kind  = "shellout";

    app.add_option("--profiles", profiles_path,
        "Alternate path to profiles.toml")->capture_default_str();
    app.add_flag("--dry-run", dry_run,
        "Print invocations but do not execute");
    app.add_option("--backend", backend_kind,
        "RyzenAdj backend: shellout|lib")
        ->check(CLI::IsMember({"shellout", "lib"}))
        ->capture_default_str();

    auto* status_cmd = app.add_subcommand("status",
        "Print current profile + last-applied knobs as JSON");

    auto* profile_cmd = app.add_subcommand("profile",
        "Apply a named profile from profiles.toml");
    std::string profile_name;
    profile_cmd->add_option("name", profile_name, "Profile name")->required();

    auto* set_cmd = app.add_subcommand("set",
        "Override a single RyzenAdj knob without switching profiles");
    std::string set_key;
    std::uint32_t set_value{0};
    set_cmd->add_option("key", set_key, "Knob name (e.g. stapm-limit)")->required();
    set_cmd->add_option("value", set_value, "Integer value (mW/mA/°C)")->required();

    auto* log_cmd = app.add_subcommand("log",
        "Emit one line of JSON metrics on stdout and exit");

    auto* board_cmd = app.add_subcommand("board",
        "Set AXB35 EC board power mode (quiet|balanced|performance)");
    std::string board_mode;
    board_cmd->add_option("mode", board_mode, "Mode")->required();

    auto* fan_cmd = app.add_subcommand("fan",
        "Fan control via EC sysfs (ec_su_axb35)");
    std::uint8_t fan_id{0};
    fan_cmd->add_option("id", fan_id, "Fan id: 1|2|3")->required();
    fan_cmd->require_subcommand(1);

    auto* fan_mode_cmd = fan_cmd->add_subcommand("mode", "Switch mode: auto|fixed|curve");
    std::string fan_mode_arg;
    fan_mode_cmd->add_option("mode", fan_mode_arg, "Mode")->required();

    auto* fan_level_cmd = fan_cmd->add_subcommand("level", "Set fixed level 0..=5");
    std::uint8_t fan_level_arg{0};
    fan_level_cmd->add_option("level", fan_level_arg, "Level 0..5")->required();

    auto* fan_curve_cmd = fan_cmd->add_subcommand("curve",
        "Write 5 °C thresholds for rampup or rampdown");
    std::string fan_curve_dir;
    std::string fan_curve_csv;
    fan_curve_cmd->add_option("direction", fan_curve_dir, "up|down")->required();
    fan_curve_cmd->add_option("thresholds", fan_curve_csv,
        "Five comma-separated °C values, e.g. 60,70,83,95,97")->required();

    CLI11_PARSE(app, argc, argv);

    op::EcBackend ec{};

    if (status_cmd->parsed()) {
        auto backend = make_backend(backend_kind, dry_run);
        if (!backend) return 1;
        return do_status(*backend, profiles_path, ec, dry_run);
    }
    if (profile_cmd->parsed()) {
        auto backend = make_backend(backend_kind, dry_run);
        if (!backend) return 1;
        return do_profile(*backend, profiles_path, profile_name);
    }
    if (set_cmd->parsed()) {
        auto backend = make_backend(backend_kind, dry_run);
        if (!backend) return 1;
        return do_set(*backend, set_key, set_value);
    }
    if (log_cmd->parsed()) {
        return do_log(ec);
    }
    if (board_cmd->parsed()) {
        return do_board(ec, board_mode, dry_run);
    }
    if (fan_cmd->parsed()) {
        if (fan_mode_cmd->parsed())  return do_fan_mode (ec, fan_id, fan_mode_arg,  dry_run);
        if (fan_level_cmd->parsed()) return do_fan_level(ec, fan_id, fan_level_arg, dry_run);
        if (fan_curve_cmd->parsed()) return do_fan_curve(ec, fan_id, fan_curve_dir, fan_curve_csv, dry_run);
    }
    // Should be unreachable: app.require_subcommand(1) above.
    std::fprintf(stderr, "1bit-power: no subcommand?\n");
    return 2;
}
