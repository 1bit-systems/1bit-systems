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
        auto candidate = base / ("onebit-validate-" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
    }
    return base / "onebit-validate-fallback";
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
        "\"\ntitle = \"Test\"\nartist = \"T\"\nlicense = \"CC0-1.0\"\n"
        "created = \"2026-04-23T00:00:00Z\"\ntier = \"lossy\"\n"
        "license_txt = \"public domain\"\n"
        "[codec]\naudio = \"mimi-12hz\"\nsample_rate = 24000\nchannels = 2\n"
        "[model]\narch = \"bitnet-1p58\"\nparams = 1000\nbpw = 1.58\n"
        "[[tracks]]\nid = \"01\"\ntitle = \"A\"\nlength_ms = 1000\n";
    return out;
}

} // namespace

TEST_CASE("pack -> validate roundtrip")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, "GGUF fake weights");
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp, sample_toml("roundtrip"));
    const auto out = tmp / "rt.1bl";

    auto p = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE(p.has_value());
    auto r = onebit::ingest::validate(out);
    REQUIRE(r.has_value());
    CHECK(r->version == 0x01);
    CHECK(r->footer_ok);
    CHECK(r->manifest.catalog == "roundtrip");
    REQUIRE(r->sections.size() == 3);
    CHECK(r->sections[0].tag == onebit::ingest::tag::MODEL_GGUF);
    CHECK(r->manifest.tracks.size() == 1);
    fs::remove_all(tmp);
}

TEST_CASE("corrupted footer is detected")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, "GGUF fake");
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp, sample_toml("corrupt"));
    const auto out = tmp / "c.1bl";
    auto       p   = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE(p.has_value());

    // Flip a byte just before the footer; lands inside a section payload.
    std::vector<std::uint8_t> bytes;
    {
        std::ifstream in(out, std::ios::binary);
        in.seekg(0, std::ios::end);
        const auto sz = in.tellg();
        in.seekg(0, std::ios::beg);
        bytes.resize(static_cast<std::size_t>(sz));
        in.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(sz));
    }
    bytes[bytes.size() - 33] ^= 0xFF;
    {
        std::ofstream o(out, std::ios::binary | std::ios::trunc);
        o.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    }

    auto r = onebit::ingest::validate(out);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what().find("footer hash") != std::string::npos);
    fs::remove_all(tmp);
}

TEST_CASE("bad magic rejected")
{
    const auto tmp = mktemp_dir();
    const auto p   = tmp / "x.1bl";
    write_file(p, std::string{"NOPE\0\0\0\0", 8});
    auto r = onebit::ingest::validate(p);
    CHECK_FALSE(r.has_value());
    fs::remove_all(tmp);
}

TEST_CASE("truncated input rejected")
{
    const auto tmp = mktemp_dir();
    const auto p   = tmp / "x.1bl";
    write_file(p, "1BL");
    auto r = onebit::ingest::validate(p);
    CHECK_FALSE(r.has_value());
    fs::remove_all(tmp);
}

TEST_CASE("format_report writes expected lines")
{
    const auto tmp = mktemp_dir();
    const auto gguf = tmp / "m.gguf";
    write_file(gguf, "GGUF fake");
    const auto tomlp = tmp / "catalog.toml";
    write_file(tomlp, sample_toml("fmt"));
    const auto out = tmp / "f.1bl";
    auto p = onebit::ingest::pack(gguf, tomlp, std::nullopt, std::nullopt, out);
    REQUIRE(p.has_value());
    auto r = onebit::ingest::validate(out);
    REQUIRE(r.has_value());
    auto s = onebit::ingest::format_report(*r);
    CHECK(s.find("1bl container") != std::string::npos);
    CHECK(s.find("catalog") != std::string::npos);
    CHECK(s.find("OK") != std::string::npos);
    fs::remove_all(tmp);
}
