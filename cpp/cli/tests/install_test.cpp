#include <doctest/doctest.h>

#include "onebit/cli/install.hpp"
#include "onebit/cli/proc.hpp"
#include "onebit/cli/registry.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace onebit::cli;

namespace {

// FakeExecutor — records every host-side call without touching the box.
// Used to assert run_install's policy decisions (path-traversal rejection
// before any subprocess fires) and to capture argv for inspection.
class FakeExecutor final : public HostExecutor {
public:
    struct ArgvCall {
        std::filesystem::path                cwd;
        std::vector<std::string>             argv;
    };
    struct CopyCall {
        std::filesystem::path                src;
        std::filesystem::path                dest;
        std::map<std::string, std::string>   subs;
    };

    std::vector<ArgvCall> argv_calls;
    std::vector<CopyCall> copy_calls;
    std::vector<std::string> systemctl_calls;

    std::expected<void, Error>
    run_argv(const std::filesystem::path& cwd,
             const std::vector<std::string>& argv) override
    {
        argv_calls.push_back({cwd, argv});
        return {};
    }

    std::expected<void, Error>
    systemctl_enable_now(std::string_view unit) override
    {
        systemctl_calls.emplace_back(unit);
        return {};
    }

    std::expected<void, Error>
    systemctl_restart(std::string_view) override { return {}; }

    std::expected<bool, Error>
    copy_tracked_file(const std::filesystem::path& src,
                      const std::filesystem::path& dest,
                      const std::map<std::string, std::string>& subs) override
    {
        copy_calls.push_back({src, dest, subs});
        return true;  // pretend a write happened
    }

    bool healthcheck(std::string_view) override { return true; }
};

}  // namespace

// --- Bug A: ${HOME} substring expansion in argv -----------------------------

TEST_CASE("Bug A — expand_placeholder substitutes ${HOME} substring")
{
    ::setenv("HOME", "/home/halo-test", 1);
    CHECK(expand_placeholder("${HOME}/foo/bar") == "/home/halo-test/foo/bar");
}

TEST_CASE("Bug A — expand_placeholder substitutes $HOME substring")
{
    ::setenv("HOME", "/home/halo-test", 1);
    CHECK(expand_placeholder("$HOME/.local/bin/1bit") ==
          "/home/halo-test/.local/bin/1bit");
}

TEST_CASE("Bug A — expand_placeholder substitutes ${USER} and ${HOME} together")
{
    ::setenv("HOME", "/home/halo-test", 1);
    ::setenv("USER", "halo-test",      1);
    CHECK(expand_placeholder("/var/${USER}${HOME}") ==
          "/var/halo-test/home/halo-test");
}

TEST_CASE("Bug A — expand_placeholder leaves ${VAR:-default} untouched")
{
    // Strict semantics — no shell-style fallback. The literal `:-default`
    // is part of a non-matching tail, so the whole token is passthrough.
    ::setenv("HOME", "/home/halo-test", 1);
    const auto out = expand_placeholder("${HOME:-/tmp}/x");
    // The substring `${HOME}` does NOT appear (the literal has `${HOME:-`
    // with the colon-dash inside the braces). Whole string preserved.
    CHECK(out == "${HOME:-/tmp}/x");
}

TEST_CASE("Bug A — expand_placeholder ignores command substitution")
{
    // No `$(cmd)` interpretation. The literal must survive verbatim.
    const auto out = expand_placeholder("/etc/$(whoami)/foo");
    CHECK(out == "/etc/$(whoami)/foo");
}

TEST_CASE("Bug A — expand_placeholder handles ${XDG_CONFIG_HOME} with fallback")
{
    ::setenv("HOME", "/home/halo-test", 1);
    ::unsetenv("XDG_CONFIG_HOME");
    CHECK(expand_placeholder("${XDG_CONFIG_HOME}/1bit") ==
          "/home/halo-test/.config/1bit");
}

TEST_CASE("Bug A — expand_tilde chains placeholder expansion")
{
    ::setenv("HOME", "/home/halo-test", 1);
    CHECK(expand_tilde("~/.local/bin/1bit") ==
          "/home/halo-test/.local/bin/1bit");
    CHECK(expand_tilde("${HOME}/.local/bin/1bit") ==
          "/home/halo-test/.local/bin/1bit");
}

// --- Bug C: argv form, no shell composition ---------------------------------

TEST_CASE("Bug C — run_with_stdin pipes bytes via argv (no shell)")
{
    const std::string payload = "hello halo\n";
    auto out = run_capture({"cat", "-"});
    // sanity: cat exists on PATH (precondition for the next assertion)
    REQUIRE(out.has_value());

    // Now exercise stdin pipe through run_with_stdin. We can't capture
    // stdout from run_with_stdin (it inherits parent stdout); we just
    // assert the child exits 0 — combined with the literal-preservation
    // test below, that proves argv form.
    auto rc = run_with_stdin({"cat", "-"}, payload);
    REQUIRE(rc.has_value());
    CHECK(*rc == 0);
}

TEST_CASE("Bug C — argv preserves shell-metacharacters as literal arguments")
{
    // The whole point of argv form: shell metachars never reach a
    // shell. printf '%s\n' "$@" emits one line per argv slot. Pass a
    // string with `;` and `$()` and verify it round-trips intact.
    const std::string evil = "; rm -rf /tmp/fake-$(whoami)";
    auto rc = run_capture({"printf", "%s\n", evil});
    REQUIRE(rc.has_value());
    CHECK(rc->exit_code == 0);
    // Trim trailing newline.
    std::string got = rc->stdout_text;
    if (!got.empty() && got.back() == '\n') got.pop_back();
    CHECK(got == evil);
}

// --- Bug D: overlay dst path traversal --------------------------------------

namespace {

// Hand-build a manifest with a single component that has one file entry —
// avoids round-tripping through the TOML parser so the test pins the
// exact dst we want to attack with.
[[nodiscard]] Manifest make_manifest_with_dst(const std::string& dst)
{
    Manifest m;
    Component c;
    c.name        = "evil";
    c.description = "test component";
    FileEntry f;
    f.src = "src.tmpl";
    f.dst = dst;
    c.files.push_back(std::move(f));
    m.components.emplace("evil", std::move(c));
    return m;
}

[[nodiscard]] std::filesystem::path test_scratch(const char* tag)
{
    namespace fs = std::filesystem;
    auto td = fs::temp_directory_path() /
        ("onebit_cli_install_" + std::string(tag) +
         "_" + std::to_string(::getpid()));
    fs::create_directories(td);
    fs::create_directories(td / "config");
    fs::create_directories(td / "workspace");
    // Create the src so copy_tracked_file would otherwise succeed.
    std::ofstream(td / "workspace" / "src.tmpl") << "payload\n";
    return td;
}

}  // namespace

TEST_CASE("Bug D — overlay dst with .. traversal is rejected")
{
    const auto td = test_scratch("traversal");
    auto m = make_manifest_with_dst("../../etc/passwd");

    InstallContext ctx;
    ctx.workspace_root = td / "workspace";
    ctx.config_root    = td / "config";

    FakeExecutor fake;
    InstallTracker tracker;
    auto rc = run_install(fake, m, "evil", tracker, ctx);
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == ErrorKind::PreconditionFailed);
    CHECK(rc.error().message.find("escapes") != std::string::npos);
    // No copy must have happened — rejection fires before
    // copy_tracked_file is called.
    CHECK(fake.copy_calls.empty());

    std::filesystem::remove_all(td);
}

TEST_CASE("Bug D — overlay absolute dst outside allowlist is rejected")
{
    const auto td = test_scratch("outside_root");
    // /opt is not in the allowlist (allowed: /etc, /usr/local,
    // /var/lib/1bit, ${HOME}/.local, ${HOME}/.config, /tmp).
    auto m = make_manifest_with_dst("/opt/evil/sudoers");

    InstallContext ctx;
    ctx.workspace_root = td / "workspace";
    ctx.config_root    = td / "config";

    FakeExecutor fake;
    InstallTracker tracker;
    auto rc = run_install(fake, m, "evil", tracker, ctx);
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == ErrorKind::PreconditionFailed);
    CHECK(rc.error().message.find("/opt/evil/sudoers") != std::string::npos);
    CHECK(fake.copy_calls.empty());

    std::filesystem::remove_all(td);
}

TEST_CASE("Bug D — relative dst that lexically escapes config_root is rejected")
{
    const auto td = test_scratch("relesc");
    auto m = make_manifest_with_dst("../../../etc/cron.d/evil");

    InstallContext ctx;
    ctx.workspace_root = td / "workspace";
    ctx.config_root    = td / "config";

    FakeExecutor fake;
    InstallTracker tracker;
    auto rc = run_install(fake, m, "evil", tracker, ctx);
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == ErrorKind::PreconditionFailed);
    CHECK(fake.copy_calls.empty());

    std::filesystem::remove_all(td);
}

TEST_CASE("Bug D — canonical dst under /etc passes containment")
{
    const auto td = test_scratch("etc_ok");
    // Mirrors the canonical packages.toml `npu` component dst.
    auto m = make_manifest_with_dst("/etc/security/limits.d/99-npu-memlock.conf");

    InstallContext ctx;
    ctx.workspace_root = td / "workspace";
    ctx.config_root    = td / "config";

    FakeExecutor fake;
    InstallTracker tracker;
    auto rc = run_install(fake, m, "evil", tracker, ctx);
    REQUIRE(rc.has_value());
    REQUIRE(fake.copy_calls.size() == 1);
    CHECK(fake.copy_calls.front().dest ==
          std::filesystem::path("/etc/security/limits.d/99-npu-memlock.conf"));

    std::filesystem::remove_all(td);
}

TEST_CASE("Bug D — relative dst staying inside config_root is allowed")
{
    const auto td = test_scratch("relok");
    auto m = make_manifest_with_dst("systemd/user/strix-cloudflared.service");

    InstallContext ctx;
    ctx.workspace_root = td / "workspace";
    ctx.config_root    = td / "config";

    FakeExecutor fake;
    InstallTracker tracker;
    auto rc = run_install(fake, m, "evil", tracker, ctx);
    REQUIRE(rc.has_value());
    REQUIRE(fake.copy_calls.size() == 1);
    const auto expected = std::filesystem::weakly_canonical(
        td / "config" / "systemd/user/strix-cloudflared.service");
    CHECK(fake.copy_calls.front().dest == expected);

    std::filesystem::remove_all(td);
}
