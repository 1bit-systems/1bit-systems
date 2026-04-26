// 1bit-watchdog — 24h soak upstream tracker for the 1bit-systems C++ media
// stack. Polls GitHub + HuggingFace, dwells N hours on new commits/releases,
// then triggers fork-merge + rebuild + redeploy hooks declared in
// packages.toml. Subcommands mirror the Rust crate verbatim:
//   1bit-watchdog check       — one poll cycle (called by the timer)
//   1bit-watchdog status      — print current state table as JSON
//   1bit-watchdog force <id>  — bypass dwell, fire merge/bump now
//   1bit-watchdog reset <id>  — forget the seen-new SHA, re-arm dwell

#include "onebit/watchdog/config.hpp"
#include "onebit/watchdog/poller.hpp"
#include "onebit/watchdog/state.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace onebit::watchdog;

namespace {

int cmd_status(const std::string& manifest_path,
               const std::string& state_path,
               const State&       state)
{
    nlohmann::json out = nlohmann::json::object();
    out["manifest_path"] = manifest_path;
    out["state_path"]    = state_path;

    nlohmann::json entries = nlohmann::json::object();
    for (const auto& [id, e] : state.entries()) {
        nlohmann::json ej = nlohmann::json::object();
        ej["last_seen_sha"]   = e.last_seen_sha   ? nlohmann::json(*e.last_seen_sha)
                                                  : nlohmann::json(nullptr);
        ej["first_seen_at"]   = e.first_seen_at
                                    ? nlohmann::json(to_iso8601(*e.first_seen_at))
                                    : nlohmann::json(nullptr);
        ej["last_merged_sha"] = e.last_merged_sha ? nlohmann::json(*e.last_merged_sha)
                                                  : nlohmann::json(nullptr);
        ej["last_merged_at"]  = e.last_merged_at
                                    ? nlohmann::json(to_iso8601(*e.last_merged_at))
                                    : nlohmann::json(nullptr);
        entries[id] = std::move(ej);
    }
    out["entries"] = std::move(entries);
    std::printf("%s\n", out.dump(2).c_str());
    return 0;
}

int cmd_check(const Manifest&    manifest,
              State&             state,
              const std::string& state_path,
              bool               dry_run)
{
    for (const auto& [_, entry] : manifest.watch) {
        if (!poll_entry(entry, state, dry_run)) {
            spdlog::warn("poll failed for {}", entry.id);
        }
    }
    StateError serr{};
    if (!state.save(state_path, &serr)) {
        spdlog::error("failed to save state to {}", state_path);
        return 1;
    }
    return 0;
}

int cmd_force(const Manifest&    manifest,
              State&             state,
              const std::string& state_path,
              const std::string& id,
              bool               dry_run)
{
    auto it = manifest.watch.find(id);
    if (it == manifest.watch.end()) {
        spdlog::error("no watch entry `{}` in manifest", id);
        return 1;
    }
    spdlog::info("[{}] forcing merge/bump (dwell bypassed)", id);
    if (!run_hooks(it->second, dry_run)) {
        return 1;
    }
    state.mark_merged(id, Clock::now());
    StateError serr{};
    if (!state.save(state_path, &serr)) {
        spdlog::error("failed to save state to {}", state_path);
        return 1;
    }
    return 0;
}

int cmd_reset(State&             state,
              const std::string& state_path,
              const std::string& id)
{
    state.reset(id);
    StateError serr{};
    if (!state.save(state_path, &serr)) {
        spdlog::error("failed to save state to {}", state_path);
        return 1;
    }
    spdlog::info("[{}] state cleared; dwell re-armed on next check", id);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app{"24h soak upstream tracker for 1bit-systems C++ media stack",
                 "1bit-watchdog"};
    app.set_version_flag("--version", "1bit-watchdog 0.1.0");

    std::string manifest_path;
    std::string state_path;
    bool        dry_run = false;
    app.add_option("--manifest",  manifest_path, "Alternate packages.toml path");
    app.add_option("--state-file", state_path,   "Alternate state file");
    app.add_flag("--dry-run", dry_run,
                 "Dry-run: poll + log, don't run on_merge/on_bump");

    auto* sub_check = app.add_subcommand("check",  "One poll cycle");
    auto* sub_status = app.add_subcommand("status", "Print current state table as JSON");

    auto* sub_force = app.add_subcommand("force",
                                         "Bypass the dwell timer, fire on_merge/on_bump now");
    std::string force_id;
    sub_force->add_option("id", force_id, "Watch-entry id")->required();

    auto* sub_reset = app.add_subcommand("reset",
                                         "Forget the seen-new SHA, re-arm dwell");
    std::string reset_id;
    sub_reset->add_option("id", reset_id, "Watch-entry id")->required();

    app.require_subcommand(1);

    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("%Y-%m-%dT%H:%M:%S.%e%z %^%l%$ %v");

    if (manifest_path.empty()) manifest_path = default_manifest_path();
    if (state_path.empty())    state_path    = default_state_path();

    ManifestError merr{};
    auto m = Manifest::load(manifest_path, &merr);
    if (!m) {
        spdlog::error("loading watch entries from {} failed", manifest_path);
        return 1;
    }

    auto loaded = State::load(state_path);
    State state = loaded ? std::move(*loaded) : State{};

    if (sub_status->parsed()) {
        return cmd_status(manifest_path, state_path, state);
    }
    if (sub_check->parsed()) {
        return cmd_check(*m, state, state_path, dry_run);
    }
    if (sub_force->parsed()) {
        return cmd_force(*m, state, state_path, force_id, dry_run);
    }
    if (sub_reset->parsed()) {
        return cmd_reset(state, state_path, reset_id);
    }
    return 0;
}
