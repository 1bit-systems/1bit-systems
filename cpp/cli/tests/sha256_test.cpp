#include <doctest/doctest.h>

#include "onebit/cli/update.hpp"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>

using onebit::cli::sha256_file;
using onebit::cli::verify_sha256;

namespace {

std::atomic<std::uint64_t> g_sha_seq{0};

std::filesystem::path tmpfile_with(std::string_view content)
{
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() /
        ("onebit_cli_sha_" + std::to_string(::getpid()) + "_" +
         std::to_string(g_sha_seq.fetch_add(1)));
    std::ofstream out(path, std::ios::binary);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return path;
}

}  // namespace

TEST_CASE("sha256_file — known vector for 'hello world'")
{
    auto p = tmpfile_with("hello world");
    auto h = sha256_file(p);
    REQUIRE(h.has_value());
    CHECK(*h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    std::filesystem::remove(p);
}

TEST_CASE("sha256_file — empty file maps to RFC 6234 vector")
{
    auto p = tmpfile_with("");
    auto h = sha256_file(p);
    REQUIRE(h.has_value());
    CHECK(*h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    std::filesystem::remove(p);
}

TEST_CASE("verify_sha256 — accepts the matching pin")
{
    auto p = tmpfile_with("hello world");
    auto v = verify_sha256(p,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    CHECK(v.has_value());
    std::filesystem::remove(p);
}

TEST_CASE("verify_sha256 — rejects mismatched pin")
{
    auto p = tmpfile_with("hello world");
    auto v = verify_sha256(p,
        "0000000000000000000000000000000000000000000000000000000000000000");
    REQUIRE_FALSE(v.has_value());
    CHECK(v.error().kind == onebit::cli::ErrorKind::Hash);
    std::filesystem::remove(p);
}
