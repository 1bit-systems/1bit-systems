#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/retrieval/wiki_index.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

using onebit::retrieval::format_for_system_prompt;
using onebit::retrieval::slugify;
using onebit::retrieval::tokenize;
using onebit::retrieval::WikiIndex;

namespace fs = std::filesystem;

namespace {

[[nodiscard]] fs::path mktemp_dir()
{
    auto base = fs::temp_directory_path();
    std::random_device                          rd;
    std::uniform_int_distribution<std::uint64_t> dist;
    for (int i = 0; i < 16; ++i) {
        auto          candidate = base / ("onebit-retrieval-" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
    }
    return base / "onebit-retrieval-fallback";
}

void write_file(const fs::path& p, std::string_view body)
{
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
}

[[nodiscard]] fs::path build_fixture_wiki()
{
    const auto root = mktemp_dir();
    write_file(root / "troubleshooting.md",
               "# Troubleshooting\n\n"
               "Common failure modes.\n\n"
               "## amdgpu OPTC hang\n\n"
               "Kernel 7.0 on gfx1151 hits an OPTC CRTC hang signature.\n"
               "The amdgpu driver freezes Wayland and needs a power-cycle.\n");
    write_file(root / "installation.md",
               "# Installation\n\n"
               "## Distro policy\n\n"
               "We target CachyOS first.\n");
    write_file(root / "clients.md",
               "# Clients\n\n"
               "Open WebUI is the blessed client.\n");
    return root;
}

} // namespace

TEST_CASE("slugify matches gh style")
{
    CHECK(slugify("Distro policy") == "Distro-policy");
    CHECK(slugify("amdgpu OPTC hang") == "amdgpu-OPTC-hang");
    CHECK(slugify("").empty());
    CHECK(slugify("   ").empty());
}

TEST_CASE("tokenize keeps hyphens and ids and strips stopwords")
{
    auto toks = tokenize("Strix Halo gfx1151 fish-shell amdgpu OPTC.");
    bool has_gfx = false;
    bool has_fish = false;
    bool has_optc = false;
    bool has_the  = false;
    for (auto& t : toks) {
        if (t == "gfx1151") {
            has_gfx = true;
        }
        if (t == "fish-shell") {
            has_fish = true;
        }
        if (t == "optc") {
            has_optc = true;
        }
        if (t == "the") {
            has_the = true;
        }
    }
    CHECK(has_gfx);
    CHECK(has_fish);
    CHECK(has_optc);
    CHECK_FALSE(has_the);
}

TEST_CASE("load fixture produces chunks")
{
    const auto wiki = build_fixture_wiki();
    auto       res  = WikiIndex::load(wiki);
    REQUIRE(res.has_value());
    CHECK(res->len() >= 3);
    fs::remove_all(wiki);
}

TEST_CASE("known term finds right file top-1")
{
    const auto wiki = build_fixture_wiki();
    auto       res  = WikiIndex::load(wiki);
    REQUIRE(res.has_value());
    auto hits = res->top_k("OPTC hang amdgpu", 3);
    REQUIRE_FALSE(hits.empty());
    CHECK(hits[0].file == "troubleshooting.md");
    CHECK(hits[0].score > 0.0F);
    fs::remove_all(wiki);
}

TEST_CASE("absent term returns empty")
{
    const auto wiki = build_fixture_wiki();
    auto       res  = WikiIndex::load(wiki);
    REQUIRE(res.has_value());
    auto hits = res->top_k("xyzzy_does_not_exist_in_any_fixture", 5);
    CHECK(hits.empty());
    fs::remove_all(wiki);
}

TEST_CASE("missing wiki dir returns error")
{
    auto res = WikiIndex::load(fs::path{"/nonexistent/wiki/path/xxx-zzz"});
    REQUIRE_FALSE(res.has_value());
    CHECK(res.error().what().find("wiki directory not found") != std::string::npos);
}

TEST_CASE("format for prompt shapes output")
{
    const auto wiki = build_fixture_wiki();
    auto       res  = WikiIndex::load(wiki);
    REQUIRE(res.has_value());
    auto hits = res->top_k("OPTC", 2);
    auto out  = format_for_system_prompt(hits);
    CHECK(out.find("RELEVANT DOCS") == 0);
    CHECK(out.find("troubleshooting.md") != std::string::npos);
    fs::remove_all(wiki);
}

TEST_CASE("empty input to format gives empty output")
{
    CHECK(format_for_system_prompt({}).empty());
}
