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

constexpr const char* kOverlay = R"toml(
[component.core]
description = "overlay-overridden core"
units       = ["strix-overridden.service"]

[component.thirdparty]
description = "registered by a third party"
check       = "http://127.0.0.1:9999/health"
units       = ["thirdparty.service"]
)toml";

}  // namespace

TEST_CASE("merge — overlay overrides matching component description")
{
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlay,   PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    merge_into(*base, std::move(*over));

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
    merge_into(*base, std::move(*over));

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
    merge_into(*base, std::move(*over));

    REQUIRE(base->components.contains("voice"));
    CHECK(base->components.at("voice").origin == PackageOrigin::Canonical);
}

TEST_CASE("registry list rows include both canonical and overlay tags")
{
    auto base = parse_manifest_str(kCanonical, PackageOrigin::Canonical);
    auto over = parse_manifest_str(kOverlay,   PackageOrigin::Overlay);
    REQUIRE(base.has_value());
    REQUIRE(over.has_value());
    merge_into(*base, std::move(*over));

    const auto rows = render_registry_list(*base);
    bool saw_canon = false, saw_overlay = false;
    for (const auto& r : rows) {
        if (r.find("[canonical]") != std::string::npos) saw_canon = true;
        if (r.find("[overlay  ]") != std::string::npos) saw_overlay = true;
    }
    CHECK(saw_canon);
    CHECK(saw_overlay);
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
