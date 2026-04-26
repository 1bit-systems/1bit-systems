#include <doctest/doctest.h>

#include "onebit/cli/push.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace onebit::cli::push;

namespace {

std::filesystem::path mk_tmp_root()
{
    auto p = std::filesystem::temp_directory_path()
             / ("push-test-" + std::to_string(::getpid()) + "-"
                + std::to_string(std::rand()));
    std::filesystem::create_directories(p);
    return p;
}

void write_file(const std::filesystem::path& p, std::string_view body)
{
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
}

}  // namespace

TEST_CASE("R2Config from_env_file: missing keys → Error")
{
    auto p = std::filesystem::temp_directory_path() / "r2-empty.env";
    std::ofstream(p) << "# nothing\n";
    auto r = R2Config::from_env_file(p);
    CHECK_FALSE(r.has_value());
}

TEST_CASE("R2Config from_env_file: full set + region default")
{
    auto p = std::filesystem::temp_directory_path() / "r2-full.env";
    {
        std::ofstream f(p);
        f << "R2_ACCOUNT_ID=acc123\n"
             "R2_ACCESS_KEY_ID=ak\n"
             "R2_SECRET_ACCESS_KEY=sk\n"
             "R2_BUCKET=halo\n";
    }
    auto r = R2Config::from_env_file(p);
    REQUIRE(r.has_value());
    CHECK(r->account_id        == "acc123");
    CHECK(r->access_key_id     == "ak");
    CHECK(r->secret_access_key == "sk");
    CHECK(r->bucket            == "halo");
    CHECK(r->region            == "auto");
    CHECK(r->endpoint_host     == "acc123.r2.cloudflarestorage.com");
}

TEST_CASE("sigv4_sign_put produces an Authorization header with all expected pieces")
{
    // Inputs are arbitrary; we verify shape, not against an AWS test
    // vector (we'd need GET, not PUT, for the canonical AWS test set).
    auto auth = sigv4_sign_put(
        "halo.r2.cloudflarestorage.com", "halo",
        "lossy/downsiders-ep1.1bl", "auto",
        "AKIATEST", "supersecretkey",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  // sha256("")
        "20260426T120000Z", "20260426");
    CHECK(auth.starts_with("AWS4-HMAC-SHA256 "));
    CHECK(auth.find("Credential=AKIATEST/20260426/auto/s3/aws4_request")
          != std::string::npos);
    CHECK(auth.find("SignedHeaders=host;x-amz-content-sha256;x-amz-date")
          != std::string::npos);
    CHECK(auth.find("Signature=") != std::string::npos);
}

TEST_CASE("dry-run scans + counts but does no I/O")
{
    auto root  = mk_tmp_root();
    auto state = root / "state.json";
    write_file(root / "downsiders" / "ep1.1bl",     std::string(2048, 'a'));
    write_file(root / "downsiders" / "ep1.flac",    std::string(8192, 'b'));
    write_file(root / "kevinmacleod" / "intro.1bl", std::string(512,  'c'));

    R2Config cfg;
    cfg.account_id        = "x";
    cfg.access_key_id     = "x";
    cfg.secret_access_key = "x";
    cfg.bucket            = "halo";
    cfg.endpoint_host     = "x.r2.cloudflarestorage.com";

    PushOptions opts;
    opts.catalogs_root = root;
    opts.state_file    = state;
    opts.tier          = Tier::Lossy;
    opts.dry_run       = true;
    opts.verbose       = false;

    auto r = run(cfg, opts);
    REQUIRE(r.has_value());
    CHECK(r->scanned  >= 2);
    CHECK(r->uploaded == 2);          // 2 .1bl
    CHECK(r->skipped  == 0);
    CHECK(r->failed   == 0);
    CHECK_FALSE(std::filesystem::exists(state));   // dry-run wrote nothing

    std::filesystem::remove_all(root);
}

TEST_CASE("tier filter respects Lossless")
{
    auto root  = mk_tmp_root();
    write_file(root / "a.1bl",  std::string(64, 'x'));
    write_file(root / "a.flac", std::string(64, 'y'));

    R2Config cfg;
    cfg.account_id    = "x"; cfg.access_key_id = "x";
    cfg.secret_access_key = "x"; cfg.bucket = "h";
    cfg.endpoint_host = "x.r2.cloudflarestorage.com";

    PushOptions opts;
    opts.catalogs_root = root;
    opts.state_file    = root / "state.json";
    opts.tier          = Tier::Lossless;
    opts.dry_run       = true;
    opts.verbose       = false;

    auto r = run(cfg, opts);
    REQUIRE(r.has_value());
    CHECK(r->uploaded == 1);          // only the .flac

    std::filesystem::remove_all(root);
}

TEST_CASE("only_slug narrows the upload set")
{
    auto root  = mk_tmp_root();
    write_file(root / "alpha" / "x.1bl", "1");
    write_file(root / "beta"  / "x.1bl", "2");
    write_file(root / "gamma" / "x.1bl", "3");

    R2Config cfg;
    cfg.account_id    = "x"; cfg.access_key_id = "x";
    cfg.secret_access_key = "x"; cfg.bucket = "h";
    cfg.endpoint_host = "x.r2.cloudflarestorage.com";

    PushOptions opts;
    opts.catalogs_root = root;
    opts.state_file    = root / "state.json";
    opts.tier          = Tier::Both;
    opts.only_slug     = "beta";
    opts.dry_run       = true;
    opts.verbose       = false;

    auto r = run(cfg, opts);
    REQUIRE(r.has_value());
    CHECK(r->uploaded == 1);

    std::filesystem::remove_all(root);
}
