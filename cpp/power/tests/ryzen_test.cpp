#include <doctest/doctest.h>

#include "onebit/power/ryzen.hpp"

#include <algorithm>
#include <string>
#include <vector>

using onebit::power::is_known_knob;
using onebit::power::LibBackend;
using onebit::power::Profile;
using onebit::power::ShelloutBackend;

TEST_CASE("is_known_knob accepts canonical names")
{
    CHECK(is_known_knob("stapm-limit"));
    CHECK(is_known_knob("fast-limit"));
    CHECK(is_known_knob("vrmsocmax-current"));
    CHECK_FALSE(is_known_knob("not-a-knob"));
    CHECK_FALSE(is_known_knob(""));
    CHECK_FALSE(is_known_knob("STAPM-LIMIT"));
}

TEST_CASE("ShelloutBackend::profile_to_args emits only set fields, in fixed order")
{
    Profile p;
    p.stapm_limit = 55000;
    p.fast_limit  = 80000;
    auto args = ShelloutBackend::profile_to_args(p);
    REQUIRE(args.size() == 2);
    CHECK(args[0] == "--stapm-limit=55000");
    CHECK(args[1] == "--fast-limit=80000");
}

TEST_CASE("ShelloutBackend::profile_to_args is empty when profile is empty")
{
    Profile p;
    CHECK(ShelloutBackend::profile_to_args(p).empty());
}

TEST_CASE("ShelloutBackend dry-run apply succeeds without spawning")
{
    ShelloutBackend b{/*dry_run=*/true, "/nonexistent/ryzenadj"};
    Profile p;
    p.stapm_limit = 55000;
    auto s = b.apply_profile(p);
    CHECK(s.ok());
}

TEST_CASE("ShelloutBackend dry-run set_one validates knob name")
{
    ShelloutBackend b{/*dry_run=*/true};
    auto bad = b.set_one("not-a-knob", 1);
    CHECK_FALSE(bad.ok());
    CHECK(bad.code == onebit::power::Error::UnknownKnob);

    auto ok = b.set_one("stapm-limit", 55000);
    CHECK(ok.ok()); // dry-run: no fork, no failure
}

TEST_CASE("ShelloutBackend live spawn against missing binary returns BackendError")
{
    // Live (non-dry) run with a path that does not exist. We stay
    // synchronous and check the error class — not auto-skipped because
    // it does not need root or a real ryzenadj.
    ShelloutBackend b{/*dry_run=*/false, "/nonexistent/ryzenadj"};
    auto s = b.set_one("stapm-limit", 55000);
    CHECK_FALSE(s.ok());
    CHECK(s.code == onebit::power::Error::BackendError);
}

TEST_CASE("LibBackend::open returns NotAvailable when soname is bogus")
{
    auto r = LibBackend::open(/*dry_run=*/false,
        "this_is_not_a_real_lib_libryzenadj_xyz.so");
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code == onebit::power::Error::NotAvailable);
    CHECK(r.status().message.find("dlopen") != std::string::npos);
}

TEST_CASE("LibBackend::open against real libryzenadj.so (skipped if absent or non-root)")
{
    // We deliberately do NOT pull in libryzenadj.so at build time; this
    // test only exercises the dlopen path opportunistically. The actual
    // init_ryzenadj() needs root + the kernel module loaded, so we run
    // dry-run mode to get past init. Even dlopen of the real .so usually
    // fails on a dev box, in which case we record SKIP.
    auto r = LibBackend::open(/*dry_run=*/true, "libryzenadj.so");
    if (!r.ok()) {
        // Expected on dev boxes / CI without FlyGoat/RyzenAdj installed.
        DOCTEST_WARN_MESSAGE(true,
            "libryzenadj.so not present on this host; dlopen path skipped: "
            << r.status().message);
        return;
    }
    REQUIRE(r.value() != nullptr);
    CHECK(r.value()->name() == "libryzenadj");
    // dry_run apply_profile must succeed without touching the access handle.
    Profile p;
    p.stapm_limit = 55000;
    CHECK(r.value()->apply_profile(p).ok());
    CHECK(r.value()->set_one("fast-limit", 80000).ok());
}

TEST_CASE("LibBackend::open requires root for live writes (skipped, no real path)")
{
    // Live LibBackend.set_one would need root + libryzenadj.so + the
    // kernel module. We mark this as intentionally skipped to avoid
    // touching MSRs in CI / dev. Documented for completeness.
    DOCTEST_WARN_MESSAGE(true,
        "Skipped: live MSR write needs root, libryzenadj.so, ec_su_axb35 module.");
}
