#include <doctest/doctest.h>

#include "onebit/cli/registry.hpp"

using onebit::cli::PackageOrigin;
using onebit::cli::parse_manifest_str;

namespace {

constexpr const char* kSampleManifest = R"toml(
[component.core]
description = "core stack"
deps        = ["voice", "echo"]
build       = [["echo", "core"]]
units       = ["strix-landing.service"]
check       = "http://127.0.0.1:8190/_health"

[component.voice]
description = "voice cli"
deps        = []
build       = [["cargo", "install", "--path", "crates/1bit-voice"]]
units       = []
check       = ""

[component.echo]
description = "echo ws server"
deps        = ["voice"]
build       = [["cargo", "install", "--path", "crates/1bit-echo"]]
units       = ["strix-echo.service"]

[component.npu]
description = "XDNA 2 NPU userspace"
packages    = ["xrt", "xrt-plugin-amdxdna"]
files = [
  { src = "strixhalo/security/99-npu-memlock.conf.tmpl", dst = "/etc/security/limits.d/99-npu-memlock.conf", substitute = { USER = "$USER" } },
]

[component.tunnel]
description = "cloudflared tunnel"
files = [
  ["strixhalo/systemd/strix-cloudflared.service",
   "systemd/user/strix-cloudflared.service"],
]

[model.qwen3-tts-0p6b-base]
description = "Qwen3-TTS 0.6B base"
hf_repo     = "khimaros/Qwen3-TTS-0.6B-Base-GGUF"
hf_file     = "model.gguf"
size_mb     = 650
license     = "Qwen-permissive"
requires    = ["tts-engine"]
sha256      = "UPSTREAM"
)toml";

}  // namespace

TEST_CASE("manifest parses every required component")
{
    auto m = parse_manifest_str(kSampleManifest, PackageOrigin::Canonical);
    REQUIRE(m.has_value());
    CHECK(m->components.contains("core"));
    CHECK(m->components.contains("voice"));
    CHECK(m->components.contains("echo"));
    CHECK(m->components.contains("npu"));
    CHECK(m->components.contains("tunnel"));
}

TEST_CASE("model parses size_mb + sha256 sentinel")
{
    auto m = parse_manifest_str(kSampleManifest, PackageOrigin::Canonical);
    REQUIRE(m.has_value());
    REQUIRE(m->models.contains("qwen3-tts-0p6b-base"));
    const auto& mm = m->models.at("qwen3-tts-0p6b-base");
    CHECK(mm.size_mb == 650);
    CHECK(mm.sha256 == "UPSTREAM");
    CHECK(mm.requires_.size() == 1);
    CHECK(mm.requires_.front() == "tts-engine");
}

TEST_CASE("files entry — pair shape parses + carries empty substitute")
{
    auto m = parse_manifest_str(kSampleManifest, PackageOrigin::Canonical);
    REQUIRE(m.has_value());
    const auto& tun = m->components.at("tunnel");
    REQUIRE(tun.files.size() == 1);
    CHECK(tun.files.front().src.find("strix-cloudflared.service") != std::string::npos);
    CHECK(tun.files.front().dst.find("systemd/user/") != std::string::npos);
    CHECK(tun.files.front().substitute.empty());
}

TEST_CASE("files entry — table shape parses substitute = { USER = \"$USER\" }")
{
    auto m = parse_manifest_str(kSampleManifest, PackageOrigin::Canonical);
    REQUIRE(m.has_value());
    const auto& npu = m->components.at("npu");
    REQUIRE(npu.files.size() == 1);
    const auto& f = npu.files.front();
    CHECK(f.dst == "/etc/security/limits.d/99-npu-memlock.conf");
    REQUIRE(f.substitute.contains("USER"));
    CHECK(f.substitute.at("USER") == "$USER");
}

TEST_CASE("origin tag is stamped onto every parsed component")
{
    auto m = parse_manifest_str(kSampleManifest, PackageOrigin::Canonical);
    REQUIRE(m.has_value());
    for (const auto& [_, c] : m->components) {
        CHECK(c.origin == PackageOrigin::Canonical);
    }
}

TEST_CASE("origin tag tracks Overlay when caller passes Overlay")
{
    auto m = parse_manifest_str(kSampleManifest, PackageOrigin::Overlay);
    REQUIRE(m.has_value());
    for (const auto& [_, c] : m->components) {
        CHECK(c.origin == PackageOrigin::Overlay);
    }
}

TEST_CASE("malformed TOML returns Error::parse")
{
    const char* bad = "not = a [valid toml file";
    auto m = parse_manifest_str(bad, PackageOrigin::Canonical);
    REQUIRE_FALSE(m.has_value());
    CHECK(m.error().kind == onebit::cli::ErrorKind::Parse);
}
