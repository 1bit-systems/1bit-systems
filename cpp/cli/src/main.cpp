// 1bit — unified operator CLI entry point. CLI11 dispatches to the static
// onebit::cli library; the dispatch path itself is exception-free.

#include "onebit/cli/budget.hpp"
#include "onebit/cli/burnin.hpp"
#include "onebit/cli/doctor.hpp"
#include "onebit/cli/error.hpp"
#include "onebit/cli/http.hpp"
#include "onebit/cli/install.hpp"
#include "onebit/cli/install_model.hpp"
#include "onebit/cli/oobe_error.hpp"
#include "onebit/cli/paths.hpp"
#include "onebit/cli/power.hpp"
#include "onebit/cli/preflight.hpp"
#include "onebit/cli/proc.hpp"
#include "onebit/cli/registry.hpp"
#include "onebit/cli/rollback.hpp"
#include "onebit/cli/service.hpp"
#include "onebit/cli/update.hpp"
#include "onebit/cli/version.hpp"

#include <CLI/CLI.hpp>
#include <fmt/core.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace cli = onebit::cli;

namespace {

[[nodiscard]] cli::InstallContext make_ctx()
{
    cli::InstallContext c;
    std::error_code ec;
    c.workspace_root = std::filesystem::current_path(ec);
    c.config_root    = cli::xdg_config_home();
    return c;
}

[[nodiscard]] int run_install_subcmd(const cli::Manifest& m,
                                     std::optional<std::string> component,
                                     bool list, bool oobe,
                                     bool skip_build, bool yes, bool doctor_skip)
{
    if (list || (!oobe && !component)) {
        for (const auto& row : cli::render_install_list(m)) {
            std::cout << row << '\n';
        }
        return 0;
    }

    auto ctx = make_ctx();
    auto host = cli::make_real_executor();

    if (oobe) {
        cli::RealProbe probe;
        struct Tally : cli::DoctorProbe {
            std::pair<std::uint32_t, std::uint32_t> run() override {
                return cli::tally_for_oobe();
            }
        } doctor;
        cli::OobeDefaults d;
        d.component   = component.value_or("core");
        d.skip_build  = skip_build;
        d.yes         = yes;
        d.doctor_skip = doctor_skip;
        auto rc = cli::run_oobe_full(*host, probe, doctor, m, d, ctx);
        if (!rc) {
            std::cerr << "1bit install --oobe: " << rc.error().message << '\n';
            return 2;
        }
        return 0;
    }

    // Plain install of a named component or model.
    const std::string& name = *component;

    if (auto it = m.models.find(name); it != m.models.end()) {
        std::vector<std::string> units;
        for (const auto& engine : it->second.requires_) {
            cli::InstallTracker tracker;
            auto rc = cli::run_install(*host, m, engine, tracker, ctx);
            if (!rc) {
                std::cerr << "engine install failed: " << rc.error().message << '\n';
                return 2;
            }
            if (auto eng = m.components.find(engine); eng != m.components.end()) {
                for (const auto& u : eng->second.units) units.push_back(u);
            }
        }
        auto rc = cli::install_model(it->second, units);
        if (!rc) {
            std::cerr << "model install failed: " << rc.error().message << '\n';
            return 2;
        }
        return 0;
    }

    cli::InstallTracker tracker;
    auto rc = cli::run_install(*host, m, name, tracker, ctx);
    if (!rc) {
        std::cerr << "install failed: " << rc.error().message << '\n';
        return 2;
    }
    return 0;
}

[[nodiscard]] int run_registry_subcmd(const cli::Manifest& m,
                                      bool list_all,
                                      std::optional<std::string> add_name,
                                      std::optional<std::string> add_url,
                                      std::vector<std::string> add_units,
                                      std::optional<std::string> add_desc)
{
    if (add_name) {
        cli::OverlayAddRequest req;
        req.name        = *add_name;
        req.description = add_desc.value_or(add_url.value_or(""));
        req.units       = std::move(add_units);
        req.check       = add_url.value_or("");
        auto rc = cli::overlay_add(cli::overlay_path(), req);
        if (!rc) {
            std::cerr << "registry add: " << rc.error().message << '\n';
            return 2;
        }
        std::cout << "registered '" << req.name << "' in "
                  << cli::overlay_path().string() << '\n';
        return 0;
    }
    if (list_all) {
        for (const auto& row : cli::render_registry_list(m)) {
            std::cout << row << '\n';
        }
        return 0;
    }
    // Default = list.
    for (const auto& row : cli::render_registry_list(m)) {
        std::cout << row << '\n';
    }
    return 0;
}

[[nodiscard]] int run_burnin_default()
{
    const auto path = cli::home_dir() / "claude output" / "shadow-burnin.jsonl";
    auto rows = cli::load_rows(path);
    if (!rows) {
        std::cerr << rows.error().message << '\n';
        return 2;
    }
    auto stats = cli::compute_stats(*rows);
    if (stats.total == 0) {
        std::cout << "shadow-burnin: 0 rounds logged at " << path.string() << '\n';
        return 1;
    }
    std::cout << fmt::format(
        "shadow-burnin: {}/{} byte-exact = {:.2f}% (threshold {:.0f}%)\n",
        stats.pass, stats.total, stats.pct, cli::PASS_THRESHOLD_PCT);
    return stats.pct >= cli::PASS_THRESHOLD_PCT ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    CLI::App app{
        "1bit — unified CLI for the 1bit-systems Strix Halo stack",
        "1bit",
    };
    app.set_version_flag("--version",
        std::string("1bit ") + std::string(cli::ONEBIT_CLI_VERSION));
    app.require_subcommand(0, 1);

    // status / logs / restart / doctor / version --------------------------
    app.add_subcommand("status",  "One-line-per-service state snapshot");
    auto* logs = app.add_subcommand("logs", "Tail systemd journal for a 1bit service");
    std::string logs_service;
    bool logs_follow = false;
    std::uint32_t logs_lines = 50;
    logs->add_option("service", logs_service, "Service short name")->required();
    logs->add_flag("-f,--follow", logs_follow, "Follow");
    logs->add_option("-n,--lines", logs_lines, "Last N lines")->capture_default_str();

    auto* restart = app.add_subcommand("restart", "Restart a 1bit service");
    std::string restart_service;
    restart->add_option("service", restart_service)->required();

    app.add_subcommand("doctor", "Comprehensive health check across the stack");
    app.add_subcommand("version", "1bit stack version + component SHAs");

    // update --------------------------------------------------------------
    auto* upd = app.add_subcommand("update", "Check for / install a signed release");
    bool upd_check = false, upd_install = false;
    bool upd_no_build = false, upd_no_restart = false, upd_legacy = false;
    upd->add_flag("--check",          upd_check,    "Probe release feed and report");
    upd->add_flag("--install",        upd_install,  "Download + verify the latest artifact");
    upd->add_flag("--no-build",       upd_no_build, "Legacy: skip cargo build phase");
    upd->add_flag("--no-restart",     upd_no_restart,"Legacy: skip systemctl restart");
    upd->add_flag("--legacy-rebuild", upd_legacy,   "Legacy git-rebuild path");

    // install / registry --------------------------------------------------
    auto* ins = app.add_subcommand("install",
        "Install a component from packages.toml (core, agents, voice, sd, ...)");
    std::optional<std::string> ins_component;
    bool ins_list = false, ins_oobe = false, ins_skip_build = false;
    bool ins_yes = false, ins_doctor_skip = false;
    ins->add_option("component", ins_component, "Component or model name");
    ins->add_flag("--list",       ins_list);
    ins->add_flag("--oobe",       ins_oobe);
    ins->add_flag("--skip-build", ins_skip_build);
    ins->add_flag("--yes",        ins_yes);
    ins->add_flag("--doctor-skip",ins_doctor_skip);

    auto* reg = app.add_subcommand("registry",
        "List components / register an overlay component");
    auto* reg_list = reg->add_subcommand("list",
        "Print the merged canonical+overlay registry");
    auto* reg_add = reg->add_subcommand("add",
        "Register a new component into packages.local.toml");
    std::string                add_name;
    std::optional<std::string> add_url;
    std::optional<std::string> add_desc;
    std::vector<std::string>   add_units;
    reg_add->add_option("name", add_name, "Component name")->required();
    reg_add->add_option("--url", add_url, "Healthcheck URL");
    reg_add->add_option("--description", add_desc, "Description");
    reg_add->add_option("--systemd", add_units, "systemd unit(s)");

    // rollback / say / chat / bench / ppl / power / npu / burnin / budget
    auto* rb = app.add_subcommand("rollback", "Rollback to a snapper snapshot");
    std::optional<std::uint32_t> rb_snapshot;
    bool rb_yes = false;
    rb->add_option("snapshot", rb_snapshot, "Snapshot number");
    rb->add_flag("--yes", rb_yes);

    auto* say = app.add_subcommand("say", "Speak text via 1bit-halo-kokoro");
    std::vector<std::string> say_text;
    std::optional<std::string> say_voice;
    float say_speed = 1.0f;
    say->add_option("text", say_text, "Text to synthesize");
    say->add_option("-v,--voice", say_voice);
    say->add_option("-s,--speed", say_speed)->capture_default_str();

    auto* chat = app.add_subcommand("chat", "One-shot REPL against 1bit-server :8180");
    std::optional<std::string> chat_url, chat_model;
    std::uint32_t chat_max = 128;
    chat->add_option("--url",        chat_url);
    chat->add_option("--model",      chat_model);
    chat->add_option("--max-tokens", chat_max)->capture_default_str();

    auto* bench = app.add_subcommand("bench", "Shadow-burnin summary");
    std::optional<std::uint32_t> bench_rounds;
    std::optional<std::string>   bench_since;
    bench->add_option("--rounds", bench_rounds);
    bench->add_option("--since",  bench_since);

    auto* ppl = app.add_subcommand("ppl", "Perplexity vs gen-1 baseline");
    std::optional<std::string> ppl_url;
    std::uint32_t ppl_stride = 1024, ppl_max = 1024;
    std::size_t   ppl_bytes  = 6000;
    ppl->add_option("--url",        ppl_url);
    ppl->add_option("--stride",     ppl_stride)->capture_default_str();
    ppl->add_option("--max-tokens", ppl_max)->capture_default_str();
    ppl->add_option("--bytes",      ppl_bytes)->capture_default_str();

    auto* power = app.add_subcommand("power", "Apply / query Ryzen APU power profile");
    std::optional<std::string> power_profile;
    bool power_dry = false, power_list = false;
    power->add_option("profile", power_profile);
    power->add_flag("--dry-run", power_dry);
    power->add_flag("--list",    power_list);

    auto* npu = app.add_subcommand("npu", "XDNA 2 NPU diagnostics");
    npu->add_subcommand("status",   "Quick status");
    npu->add_subcommand("examine",  "Full xrt-smi examine dump");
    npu->add_subcommand("validate", "xrt-smi validate");
    auto* npu_fw = npu->add_subcommand("firmware", "List firmware blobs");
    bool npu_fw_check = false;
    npu_fw->add_flag("--check-remote", npu_fw_check);
    npu->add_subcommand("snapshot",  "NPU boot probe snapshot path");

    auto* bn = app.add_subcommand("burnin", "Shadow-burnin log analyzer");
    auto* bn_stats  = bn->add_subcommand("stats",  "Overall byte-exact rate");
    auto* bn_drift  = bn->add_subcommand("drift",  "Top-N drift buckets");
    auto* bn_recent = bn->add_subcommand("recent", "Last N entries");
    auto* bn_since  = bn->add_subcommand("since",  "Slice by timestamp");
    std::optional<std::string> bn_log;
    std::size_t bn_top = 10, bn_tail = 20;
    std::string bn_since_ts;
    for (auto* sub : {bn_stats, bn_drift, bn_recent, bn_since}) {
        sub->add_option("--log", bn_log);
    }
    bn_drift->add_option("--top", bn_top)->capture_default_str();
    bn_recent->add_option("-n,--tail", bn_tail)->capture_default_str();
    bn_since->add_option("timestamp", bn_since_ts)->required();

    app.add_subcommand("budget", "GTT + RAM budget audit");

    CLI11_PARSE(app, argc, argv);

    // ---- registry-bound subcommands need a parsed manifest -----------
    auto require_manifest = [&]() -> std::optional<cli::Manifest> {
        auto m = cli::load_default();
        if (!m) {
            std::cerr << "could not load packages.toml: " << m.error().message << '\n';
            return std::nullopt;
        }
        return *m;
    };

    if (app.got_subcommand("status"))    return cli::run_status();
    if (app.got_subcommand("logs"))      return cli::run_logs(logs_service, logs_follow, logs_lines);
    if (app.got_subcommand("restart"))   return cli::run_restart(restart_service);
    if (app.got_subcommand("doctor"))    return cli::run_doctor();
    if (app.got_subcommand("version")) {
        std::cout << "1bit " << cli::ONEBIT_CLI_VERSION
                  << " — strix-ai gen 2 (C++23)\n";
        return 0;
    }

    if (app.got_subcommand("update")) {
        const std::string url = cli::env_or("HALO_RELEASE_FEED", "https://1bit.systems/releases.json");
        if (upd_check && upd_install) {
            std::cerr << "--check and --install are mutually exclusive\n";
            return 2;
        }
        if (upd_legacy || upd_no_build || upd_no_restart) {
            std::cerr << "legacy git-rebuild path not yet ported to C++; "
                         "use the Rust build for now or `--check`/`--install`\n";
            return 2;
        }
        auto resp = cli::default_http_client().get(url, 5000);
        if (!resp) {
            std::cerr << "feed unreachable: " << resp.error().message << '\n';
            return 2;
        }
        if (resp->status < 200 || resp->status >= 300) {
            std::cerr << "feed " << url << " returned HTTP " << resp->status << '\n';
            return 2;
        }
        auto feed = cli::parse_feed(resp->body);
        if (!feed) {
            std::cerr << "feed parse: " << feed.error().message << '\n';
            return 2;
        }
        auto outcome = cli::classify_check(*feed, cli::ONEBIT_CLI_VERSION);
        if (auto* up = std::get_if<cli::CheckUpToDate>(&outcome)) {
            std::cout << "1bit " << up->current
                      << " — up to date (feed latest: " << up->latest << ")\n";
        } else if (auto* av = std::get_if<cli::CheckAvailable>(&outcome)) {
            std::cout << "update available: " << av->current << " → "
                      << av->picked.release.version << "\n"
                      << "  artifact: " << av->picked.artifact.url << "\n"
                      << "  sha256:   " << av->picked.artifact.sha256 << "\n";
        }
        return cli::exit_code_for(outcome);
    }

    if (app.got_subcommand("install")) {
        auto m = require_manifest();
        if (!m) return 2;
        return run_install_subcmd(*m, ins_component, ins_list, ins_oobe,
                                   ins_skip_build, ins_yes, ins_doctor_skip);
    }

    if (app.got_subcommand("registry")) {
        auto m = require_manifest();
        if (!m) return 2;
        if (reg->got_subcommand("add")) {
            return run_registry_subcmd(*m, false, add_name, add_url,
                                        add_units, add_desc);
        }
        return run_registry_subcmd(*m, true, std::nullopt, std::nullopt, {}, std::nullopt);
    }

    if (app.got_subcommand("rollback")) {
        std::cerr << "1bit rollback: pure-C++ snapper driver pending; "
                     "use `1bit-rollback` shim or rerun with the Rust build.\n";
        return 2;
    }

    if (app.got_subcommand("say"))   { std::cerr << "say: TTS HTTP path not yet ported (kokoro :8083)\n"; return 2; }
    if (app.got_subcommand("chat"))  { std::cerr << "chat: SSE REPL not yet ported\n"; return 2; }
    if (app.got_subcommand("bench")) { std::cerr << "bench: shell-out shim not yet ported\n"; return 2; }
    if (app.got_subcommand("ppl"))   { std::cerr << "ppl: HTTP shim not yet ported\n"; return 2; }

    if (app.got_subcommand("power")) {
        if (power_list) {
            std::cout << "1bit power — available profiles:\n\n";
            for (auto p : cli::list_profiles()) {
                const auto e = cli::envelope_of(p);
                std::cout << fmt::format("  {:<10} {}\n",
                                          cli::name_of(p), cli::description_of(p));
                std::cout << fmt::format("  {:<10}   stapm={} W  fast={} W  slow={} W  tctl={} °C\n",
                    "", e.stapm_mw / 1000, e.fast_mw / 1000,
                    e.slow_mw / 1000, e.tctl_c);
            }
            return 0;
        }
        if (power_profile) {
            auto p = cli::parse_profile(*power_profile);
            if (!p) { std::cerr << p.error().message << '\n'; return 2; }
            const auto argv_v = cli::ryzenadj_argv(*p);
            if (power_dry) {
                std::cout << "1bit power " << cli::name_of(*p) << " --dry-run\n";
                std::cout << "    would exec: sudo ryzenadj";
                for (const auto& a : argv_v) std::cout << ' ' << a;
                std::cout << '\n';
                return 0;
            }
            if (!cli::which("ryzenadj")) {
                std::cerr << "ryzenadj not found on PATH (sudo pacman -S ryzenadj)\n";
                return 2;
            }
            std::vector<std::string> full = {"sudo", "ryzenadj"};
            for (const auto& a : argv_v) full.push_back(a);
            auto rc = cli::run_inherit(full);
            if (!rc) { std::cerr << rc.error().message << '\n'; return 2; }
            return *rc;
        }
        // No profile, no list: print summarized info.
        if (cli::which("ryzenadj")) {
            auto out = cli::run_capture({"ryzenadj", "--info"});
            if (out && out->exit_code == 0) {
                std::cout << cli::summarize_info(out->stdout_text) << '\n';
                return 0;
            }
        }
        std::cerr << "warning: ryzenadj not on PATH (sudo pacman -S ryzenadj)\n";
        return 0;
    }

    if (app.got_subcommand("npu")) {
        if (npu->got_subcommand("examine")) {
            auto rc = cli::run_inherit({"xrt-smi", "examine"});
            return rc ? *rc : 2;
        }
        if (npu->got_subcommand("validate")) {
            auto rc = cli::run_inherit({"xrt-smi", "validate"});
            return rc ? *rc : 2;
        }
        if (npu->got_subcommand("snapshot")) {
            std::cout << cli::env_or("HALO_NPU_SNAPSHOT",
                "/home/bcloud/claude output/npu-boot-2026-04-20") << '\n';
            return 0;
        }
        if (npu->got_subcommand("firmware")) {
            std::cout << "/usr/lib/firmware/amdnpu/\n";
            if (npu_fw_check) std::cout << "  (check_remote: not ported to C++ yet)\n";
            return 0;
        }
        // status is the default + explicit subcommand.
        auto rc = cli::run_inherit({"xrt-smi", "examine"});
        return rc ? *rc : 2;
    }

    if (app.got_subcommand("burnin")) {
        const bool any_sub = bn->got_subcommand("stats")
                          || bn->got_subcommand("drift")
                          || bn->got_subcommand("recent")
                          || bn->got_subcommand("since");
        if (!any_sub) return run_burnin_default();
        const auto path = bn_log.has_value()
            ? std::filesystem::path(*bn_log)
            : (cli::home_dir() / "claude output" / "shadow-burnin.jsonl");
        auto rows = cli::load_rows(path);
        if (!rows) { std::cerr << rows.error().message << '\n'; return 2; }
        if (bn->got_subcommand("stats")) {
            auto s = cli::compute_stats(*rows);
            std::cout << fmt::format(
                "log:            {}\nrounds:         {}\nbyte-exact:     {} ({:.2f}%)\n"
                "divergent:      {} ({:.2f}%)\nmean v1 ms:     {:.1f}\nmean v2 ms:     {:.1f}\n",
                path.string(), s.total, s.pass, s.pct, s.fail, 100.0 - s.pct,
                s.mean_v1_ms, s.mean_v2_ms);
            return 0;
        }
        if (bn->got_subcommand("drift")) {
            auto d = cli::compute_drift(*rows, bn_top);
            for (std::size_t i = 0; i < d.size(); ++i) {
                std::cout << fmt::format(
                    "{:>2}. idx={:<3}  fails={:<5}  offset={}\n",
                    i + 1, d[i].prompt_idx, d[i].fail_count, d[i].typical_offset);
            }
            return 0;
        }
        if (bn->got_subcommand("recent")) {
            for (const auto& r : cli::tail_rows(*rows, bn_tail)) {
                std::cout << (r.full_match ? "✓ " : "✗ ") << r.ts
                          << " idx=" << r.prompt_idx << '\n';
            }
            return 0;
        }
        if (bn->got_subcommand("since")) {
            auto s = cli::compute_stats(cli::filter_since(*rows, bn_since_ts));
            std::cout << fmt::format(
                "since:          {}\nrounds:         {}\nbyte-exact:     {} ({:.2f}%)\n",
                bn_since_ts, s.total, s.pass, s.pct);
            return 0;
        }
    }

    if (app.got_subcommand("budget")) {
        std::cerr << "budget: sysfs scan not yet wired (depends on /sys/class/drm)\n";
        return 2;
    }

    // No subcommand: print help-equivalent.
    std::cout << app.help();
    return 0;
}
