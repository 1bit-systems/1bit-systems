#include <doctest/doctest.h>

#include "onebit/ingest/ingest.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

[[nodiscard]] fs::path mktemp_dir()
{
    auto                                         base = fs::temp_directory_path();
    std::random_device                           rd;
    std::uniform_int_distribution<std::uint64_t> dist;
    for (int i = 0; i < 16; ++i) {
        auto candidate = base / ("onebit-ingest-" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
    }
    return base / "onebit-ingest-fallback";
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
        "\"\n"
        "title = \"Test Catalog\"\n"
        "artist = \"Test Artist\"\n"
        "license = \"CC0-1.0\"\n"
        "license_url = \"https://creativecommons.org/publicdomain/zero/1.0/\"\n"
        "attribution = \"Test Artist\"\n"
        "created = \"2026-04-23T00:00:00Z\"\n"
        "tier = \"lossy\"\n"
        "license_txt = \"CC0 — public domain dedication.\"\n"
        "\n"
        "[codec]\n"
        "audio = \"mimi-12hz\"\n"
        "sample_rate = 24000\n"
        "channels = 2\n"
        "\n"
        "[model]\n"
        "arch = \"bitnet-1p58\"\n"
        "params = 1048576\n"
        "bpw = 1.58\n"
        "\n"
        "[[tracks]]\n"
        "id = \"01\"\n"
        "title = \"Track One\"\n"
        "length_ms = 60000\n";
    return out;
}

} // namespace

TEST_CASE("toml parser pulls required fields")
{
    auto cat = onebit::ingest::parse_catalog_toml(sample_toml("toml-test"));
    REQUIRE(cat.has_value());
    CHECK(cat->catalog == "toml-test");
    CHECK(cat->codec.sample_rate == 24000);
    CHECK(cat->model.bpw == doctest::Approx(1.58));
    CHECK(cat->tracks.size() == 1);
    CHECK(cat->tracks[0].id == "01");
}

TEST_CASE("pack writes magic header and sections")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, std::string{"GGUF\0\0\0\0fake weights bytes here", 28});
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp, sample_toml("test-pack"));
    const auto out = tmp / "test.1bl";

    auto r = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE(r.has_value());
    CHECK(r->section_count == 3); // MODEL_GGUF + ATTRIBUTION + LICENSE
    CHECK(r->total_bytes > 32);

    std::ifstream  in(out, std::ios::binary);
    std::array<char, 4> magic{};
    in.read(magic.data(), 4);
    CHECK(magic[0] == '1');
    CHECK(magic[1] == 'B');
    CHECK(magic[2] == 'L');
    CHECK(magic[3] == 0x01);

    fs::remove_all(tmp);
}

TEST_CASE("pack rejects toml without license_txt or path")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, "GGUF fake");
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp,
               "catalog = \"x\"\ntitle = \"x\"\nartist = \"x\"\n"
               "license = \"x\"\ncreated = \"x\"\n"
               "[codec]\naudio = \"x\"\nsample_rate = 1\nchannels = 1\n"
               "[model]\narch = \"x\"\nparams = 1\nbpw = 1.0\n");
    const auto out = tmp / "x.1bl";
    auto       r   = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("license_txt") != std::string::npos);
    fs::remove_all(tmp);
}

TEST_CASE("pack handles cover and lyrics sidecars")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, "GGUF fake bytes for hashing only");
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp, sample_toml("optsec"));
    const auto cover = tmp / "cover.webp";
    write_file(cover, std::string{"RIFF\0\0WEBPVP8 fake", 18});
    const auto lyrics = tmp / "lyrics.txt";
    write_file(lyrics, "[01] la la la\n");
    const auto out = tmp / "o.1bl";

    auto r = onebit::ingest::pack(gguf, tomlp, cover, lyrics, out);
    REQUIRE(r.has_value());
    CHECK(r->section_count == 5);
    fs::remove_all(tmp);
}
