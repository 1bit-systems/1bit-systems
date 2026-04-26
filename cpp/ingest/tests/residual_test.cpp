#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

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
        auto candidate = base / ("onebit-residual-" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
    }
    return base / "onebit-residual-fallback";
}

void write_file(const fs::path& p, std::string_view body)
{
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(body.data(), static_cast<std::streamsize>(body.size()));
}

void write_bytes(const fs::path& p, std::span<const std::uint8_t> bytes)
{
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

[[nodiscard]] std::string sample_toml(std::string_view cat)
{
    std::string out;
    out += "catalog = \"";
    out.append(cat);
    out +=
        "\"\ntitle = \"T\"\nartist = \"T\"\nlicense = \"CC0-1.0\"\n"
        "created = \"2026-04-23T00:00:00Z\"\ntier = \"lossy\"\n"
        "license_txt = \"pd\"\n"
        "[codec]\naudio = \"mimi-12hz\"\nsample_rate = 24000\nchannels = 2\n"
        "[model]\narch = \"bitnet-1p58\"\nparams = 1\nbpw = 1.58\n";
    return out;
}

} // namespace

TEST_CASE("add-residual roundtrip")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, "GGUF fake");
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp, sample_toml("premium"));
    const auto lossy = tmp / "lossy.1bl";
    {
        auto r = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, lossy);
        REQUIRE(r.has_value());
    }

    const auto                residual = tmp / "residual.bin";
    std::vector<std::uint8_t> rbytes(1024, 0xAA);
    write_bytes(residual, rbytes);

    const auto                index = tmp / "index.cbor";
    std::vector<std::uint8_t> ibytes{0xA0}; // CBOR empty-map
    write_bytes(index, ibytes);

    const auto premium = tmp / "premium.1bl";
    auto       s = onebit::ingest::add_residual(lossy, residual, index, premium);
    REQUIRE(s.has_value());
    CHECK(s->residual_bytes == 1024);
    CHECK(s->index_bytes == 1);

    // Upgraded file must validate end-to-end with two extra sections.
    auto rep = onebit::ingest::validate(premium);
    REQUIRE(rep.has_value());
    CHECK(rep->footer_ok);
    REQUIRE(rep->sections.size() == 5);
    bool has_blob  = false;
    bool has_index = false;
    for (const auto& sec : rep->sections) {
        if (sec.tag == onebit::ingest::tag::RESIDUAL_BLOB) {
            has_blob = true;
        }
        if (sec.tag == onebit::ingest::tag::RESIDUAL_INDEX) {
            has_index = true;
        }
    }
    CHECK(has_blob);
    CHECK(has_index);
    fs::remove_all(tmp);
}

TEST_CASE("add-residual rejects bad input footer")
{
    const auto tmp = mktemp_dir();
    const auto bad = tmp / "bad.1bl";
    write_file(bad, std::string{"1BL\x01", 4});
    const auto out = tmp / "out.1bl";
    auto       r = onebit::ingest::add_residual(bad, bad, bad, out);
    CHECK_FALSE(r.has_value());
    fs::remove_all(tmp);
}
