#include <doctest/doctest.h>

#include "onebit/stream/container.hpp"

#include "onebit/ingest/ingest.hpp"

#include <filesystem>
#include <fstream>
#include <random>

namespace fs = std::filesystem;

namespace {

[[nodiscard]] fs::path mktemp_dir()
{
    auto                                         base = fs::temp_directory_path();
    std::random_device                           rd;
    std::uniform_int_distribution<std::uint64_t> dist;
    for (int i = 0; i < 16; ++i) {
        auto candidate = base / ("onebit-stream-" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
    }
    return base / "onebit-stream-fallback";
}

void write_file(const fs::path& p, std::string_view body)
{
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(body.data(), static_cast<std::streamsize>(body.size()));
}

[[nodiscard]] std::string sample_toml(std::string_view cat)
{
    std::string out;
    out += "catalog = \"";
    out.append(cat);
    out +=
        "\"\ntitle = \"T\"\nartist = \"T\"\nlicense = \"CC0-1.0\"\n"
        "created = \"2026-04-23T00:00:00Z\"\ntier = \"both\"\n"
        "license_txt = \"public domain\"\n"
        "[codec]\naudio = \"mimi-12hz\"\nsample_rate = 24000\nchannels = 2\n"
        "[model]\narch = \"bitnet-1p58\"\nparams = 1\nbpw = 1.58\n";
    return out;
}

[[nodiscard]] fs::path build_test_catalog(const fs::path& dir,
                                           std::string_view slug,
                                           bool             with_residual)
{
    const auto gguf = dir / (std::string{slug} + ".gguf");
    write_file(gguf, "GGUF fake bytes for hashing");
    const auto tomlp = dir / (std::string{slug} + ".toml");
    write_file(tomlp, sample_toml(slug));
    const auto out = dir / (std::string{slug} + ".1bl");
    auto packed    = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE(packed.has_value());
    if (with_residual) {
        const auto residual = dir / (std::string{slug} + ".residual");
        std::vector<std::uint8_t> rb(64, 0xCC);
        {
            std::ofstream o(residual, std::ios::binary | std::ios::trunc);
            o.write(reinterpret_cast<const char*>(rb.data()),
                    static_cast<std::streamsize>(rb.size()));
        }
        const auto                idx = dir / (std::string{slug} + ".idx");
        std::vector<std::uint8_t> ib{0xA0};
        {
            std::ofstream o(idx, std::ios::binary | std::ios::trunc);
            o.write(reinterpret_cast<const char*>(ib.data()),
                    static_cast<std::streamsize>(ib.size()));
        }
        const auto upgraded = dir / (std::string{slug} + ".premium.1bl");
        auto       res = onebit::ingest::add_residual(out, residual, idx, upgraded);
        REQUIRE(res.has_value());
        return upgraded;
    }
    return out;
}

} // namespace

TEST_CASE("open_catalog reads sections and slug")
{
    const auto tmp = mktemp_dir();
    const auto p   = build_test_catalog(tmp, "test-cat", /*residual=*/false);
    auto cat = onebit::stream::open_catalog(p);
    REQUIRE(cat.has_value());
    CHECK(cat->slug() == "test-cat");
    CHECK(cat->total_bytes >= 32);
    CHECK_FALSE(cat->sections.empty());
    fs::remove_all(tmp);
}

TEST_CASE("Section::is_lossy_tier excludes residual tags")
{
    onebit::stream::Section a{onebit::stream::tag::MODEL_GGUF, 0, 0};
    onebit::stream::Section b{onebit::stream::tag::ATTRIBUTION_TXT, 0, 0};
    onebit::stream::Section c{onebit::stream::tag::RESIDUAL_BLOB, 0, 0};
    onebit::stream::Section d{onebit::stream::tag::RESIDUAL_INDEX, 0, 0};
    CHECK(a.is_lossy_tier());
    CHECK(b.is_lossy_tier());
    CHECK_FALSE(c.is_lossy_tier());
    CHECK_FALSE(d.is_lossy_tier());
}

TEST_CASE("build_lossy_bytes drops residual sections and validates")
{
    const auto tmp      = mktemp_dir();
    const auto premium  = build_test_catalog(tmp, "lossy-strip", /*residual=*/true);
    auto       cat_full = onebit::stream::open_catalog(premium);
    REQUIRE(cat_full.has_value());

    auto bytes = onebit::stream::build_lossy_bytes(*cat_full);
    REQUIRE(bytes.has_value());

    auto rep = onebit::ingest::parse_bytes(*bytes);
    REQUIRE(rep.has_value());
    CHECK(rep->footer_ok);
    for (const auto& s : rep->sections) {
        CHECK(s.tag != onebit::stream::tag::RESIDUAL_BLOB);
        CHECK(s.tag != onebit::stream::tag::RESIDUAL_INDEX);
    }
    fs::remove_all(tmp);
}

TEST_CASE("open_catalog returns error on bad magic")
{
    const auto tmp = mktemp_dir();
    const auto p   = tmp / "bad.1bl";
    write_file(p, "NOPE");
    auto cat = onebit::stream::open_catalog(p);
    CHECK_FALSE(cat.has_value());
    fs::remove_all(tmp);
}

TEST_CASE("open_catalog returns error on missing file")
{
    auto cat = onebit::stream::open_catalog("/nonexistent/x.1bl");
    CHECK_FALSE(cat.has_value());
}
