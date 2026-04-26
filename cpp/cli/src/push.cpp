#include "onebit/cli/push.hpp"

#include "onebit/ingest/sha256.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace onebit::cli::push {

namespace {

constexpr std::string_view SERVICE = "s3";

[[nodiscard]] std::string to_hex_lower(std::span<const std::uint8_t> bytes)
{
    std::string out;
    out.reserve(bytes.size() * 2);
    constexpr char hex[] = "0123456789abcdef";
    for (auto b : bytes) {
        out.push_back(hex[b >> 4]);
        out.push_back(hex[b & 0xF]);
    }
    return out;
}

[[nodiscard]] std::array<std::uint8_t, 32>
hmac_sha256(std::span<const std::uint8_t> key, std::span<const std::uint8_t> msg)
{
    using onebit::ingest::detail::Sha256;
    constexpr std::size_t B = 64;

    std::array<std::uint8_t, B> kpad{};
    if (key.size() > B) {
        Sha256 h;
        h.update({reinterpret_cast<const char*>(key.data()), key.size()});
        auto digest = h.finalize();
        std::copy(digest.begin(), digest.end(), kpad.begin());
    } else {
        std::copy(key.begin(), key.end(), kpad.begin());
    }

    std::array<std::uint8_t, B> ipad, opad;
    for (std::size_t i = 0; i < B; ++i) {
        ipad[i] = kpad[i] ^ 0x36;
        opad[i] = kpad[i] ^ 0x5C;
    }

    Sha256 inner;
    inner.update({reinterpret_cast<const char*>(ipad.data()), B});
    inner.update({reinterpret_cast<const char*>(msg.data()),  msg.size()});
    auto inner_digest = inner.finalize();

    Sha256 outer;
    outer.update({reinterpret_cast<const char*>(opad.data()), B});
    outer.update({reinterpret_cast<const char*>(inner_digest.data()), 32});
    auto out = outer.finalize();
    std::array<std::uint8_t, 32> r;
    std::copy(out.begin(), out.end(), r.begin());
    return r;
}

[[nodiscard]] std::string sha256_hex_of(std::span<const std::uint8_t> bytes)
{
    onebit::ingest::detail::Sha256 h;
    h.update({reinterpret_cast<const char*>(bytes.data()), bytes.size()});
    auto d = h.finalize();
    return to_hex_lower({d.data(), 32});
}

[[nodiscard]] std::string sha256_hex_of_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    onebit::ingest::detail::Sha256 h;
    constexpr std::size_t CHUNK = 64 * 1024;
    std::vector<char> buf(CHUNK);
    while (f.read(buf.data(), CHUNK) || f.gcount() > 0) {
        h.update({buf.data(), static_cast<std::size_t>(f.gcount())});
    }
    auto d = h.finalize();
    return to_hex_lower({d.data(), 32});
}

[[nodiscard]] std::string strip_kv_quotes(std::string s)
{
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'')
        && s.front() == s.back()) {
        s = s.substr(1, s.size() - 2);
    }
    return s;
}

}  // namespace

std::expected<R2Config, Error>
R2Config::from_env_file(const std::filesystem::path& env_path)
{
    std::ifstream f(env_path);
    if (!f) {
        return std::unexpected(Error::io(
            "cannot open R2 env file: " + env_path.string()));
    }
    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv.emplace(line.substr(0, eq), strip_kv_quotes(line.substr(eq + 1)));
    }

    auto must = [&](const char* key) -> std::expected<std::string, Error> {
        auto it = kv.find(key);
        if (it == kv.end() || it->second.empty()) {
            return std::unexpected(Error::invalid(
                std::string{"R2 env file missing required key: "} + key));
        }
        return it->second;
    };

    R2Config c;
    auto v_acc = must("R2_ACCOUNT_ID");        if (!v_acc) return std::unexpected(v_acc.error());
    auto v_ak  = must("R2_ACCESS_KEY_ID");     if (!v_ak)  return std::unexpected(v_ak.error());
    auto v_sk  = must("R2_SECRET_ACCESS_KEY"); if (!v_sk)  return std::unexpected(v_sk.error());
    auto v_b   = must("R2_BUCKET");            if (!v_b)   return std::unexpected(v_b.error());
    c.account_id        = std::move(*v_acc);
    c.access_key_id     = std::move(*v_ak);
    c.secret_access_key = std::move(*v_sk);
    c.bucket            = std::move(*v_b);
    c.endpoint_host     = c.account_id + ".r2.cloudflarestorage.com";
    if (auto it = kv.find("R2_REGION"); it != kv.end()) c.region = it->second;
    return c;
}

std::string
sigv4_sign_put(std::string_view host,
               std::string_view bucket,
               std::string_view key,
               std::string_view region,
               std::string_view access_key_id,
               std::string_view secret_access_key,
               std::string_view payload_sha256_hex,
               std::string_view amz_date,
               std::string_view date_stamp)
{
    // Canonical request.
    const std::string canonical_uri =
        "/" + std::string(bucket) + "/" + std::string(key);
    const std::string canonical_headers =
        "host:" + std::string(host) + "\n"
        "x-amz-content-sha256:" + std::string(payload_sha256_hex) + "\n"
        "x-amz-date:" + std::string(amz_date) + "\n";
    const std::string signed_headers =
        "host;x-amz-content-sha256;x-amz-date";
    const std::string canonical_request =
        "PUT\n" + canonical_uri + "\n\n"
        + canonical_headers + "\n"
        + signed_headers + "\n"
        + std::string(payload_sha256_hex);

    const std::string canonical_request_hash = sha256_hex_of(
        {reinterpret_cast<const std::uint8_t*>(canonical_request.data()),
         canonical_request.size()});

    // String to sign.
    const std::string credential_scope =
        std::string(date_stamp) + "/" + std::string(region) + "/"
        + std::string(SERVICE) + "/aws4_request";
    const std::string string_to_sign =
        std::string("AWS4-HMAC-SHA256\n")
        + std::string(amz_date) + "\n"
        + credential_scope + "\n"
        + canonical_request_hash;

    // Derive signing key.
    auto as_bytes = [](std::string_view s) {
        return std::span<const std::uint8_t>{
            reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
    };
    const std::string k_secret_str = "AWS4" + std::string(secret_access_key);
    auto k_secret = std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(k_secret_str.data()),
        k_secret_str.size()};
    auto k_date    = hmac_sha256(k_secret, as_bytes(date_stamp));
    auto k_region  = hmac_sha256({k_date.data(), 32}, as_bytes(region));
    auto k_service = hmac_sha256({k_region.data(), 32}, as_bytes(SERVICE));
    auto k_signing = hmac_sha256({k_service.data(), 32}, as_bytes("aws4_request"));

    auto sig_bytes = hmac_sha256({k_signing.data(), 32}, as_bytes(string_to_sign));
    const std::string signature = to_hex_lower({sig_bytes.data(), 32});

    return std::string("AWS4-HMAC-SHA256 ")
           + "Credential=" + std::string(access_key_id) + "/" + credential_scope
           + ",SignedHeaders=" + signed_headers
           + ",Signature=" + signature;
}

namespace {

[[nodiscard]] std::pair<std::string, std::string> utc_amz_now()
{
    const auto t  = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm{};
    gmtime_r(&t, &tm);
    char date[16], full[32];
    std::strftime(date, sizeof(date), "%Y%m%d",         &tm);
    std::strftime(full, sizeof(full), "%Y%m%dT%H%M%SZ", &tm);
    return {date, full};
}

[[nodiscard]] std::expected<bool, Error>
upload_one(httplib::Client& cli, const R2Config& cfg, std::string_view key,
           const std::vector<std::uint8_t>& payload,
           std::string_view payload_sha_hex)
{
    const auto [date_stamp, amz_date] = utc_amz_now();
    const std::string auth = sigv4_sign_put(
        cfg.endpoint_host, cfg.bucket, key, cfg.region,
        cfg.access_key_id, cfg.secret_access_key,
        payload_sha_hex, amz_date, date_stamp);

    httplib::Headers h{
        {"Host",                 cfg.endpoint_host},
        {"x-amz-content-sha256", std::string(payload_sha_hex)},
        {"x-amz-date",           amz_date},
        {"Authorization",        auth},
    };

    const std::string path = "/" + cfg.bucket + "/" + std::string(key);
    auto res = cli.Put(path.c_str(), h,
                       reinterpret_cast<const char*>(payload.data()),
                       payload.size(),
                       "application/octet-stream");
    if (!res) {
        return std::unexpected(Error::network(
            "PUT " + std::string(key) + ": "
            + httplib::to_string(res.error())));
    }
    if (res->status < 200 || res->status >= 300) {
        return std::unexpected(Error::network(
            "PUT " + std::string(key) + " returned "
            + std::to_string(res->status) + ": " + res->body));
    }
    return true;
}

[[nodiscard]] std::vector<std::uint8_t> read_all(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    std::vector<std::uint8_t> out(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return out;
}

}  // namespace

std::expected<PushReport, Error>
run(const R2Config& cfg, const PushOptions& opts)
{
    PushReport rpt;

    nlohmann::json state = nlohmann::json::object();
    if (std::filesystem::exists(opts.state_file)) {
        std::ifstream f(opts.state_file);
        try { state = nlohmann::json::parse(f); }
        catch (const nlohmann::json::parse_error&) {
            state = nlohmann::json::object();
        }
    }

    if (!std::filesystem::exists(opts.catalogs_root)) {
        return std::unexpected(Error::io(
            "catalogs root not found: " + opts.catalogs_root.string()));
    }

    httplib::Client cli(("https://" + cfg.endpoint_host).c_str());
    cli.set_connection_timeout(cfg.timeout);
    cli.set_read_timeout(cfg.timeout);
    cli.set_write_timeout(cfg.timeout);

    auto consider = [&](const std::filesystem::path& file,
                        const std::string& key) {
        ++rpt.scanned;
        const auto sha = sha256_hex_of_file(file);
        if (sha.empty()) return;
        if (state.value(key, std::string{}) == sha) {
            ++rpt.skipped;
            return;
        }
        if (opts.dry_run) {
            ++rpt.uploaded;  // count what would-have-uploaded
            if (opts.verbose) std::printf("[dry] would PUT %s (%s)\n",
                                          key.c_str(), sha.substr(0, 12).c_str());
            return;
        }
        const auto bytes = read_all(file);
        auto ok = upload_one(cli, cfg, key, bytes, sha);
        if (!ok) {
            ++rpt.failed;
            rpt.failures.push_back(key + ": " + ok.error().message);
            return;
        }
        state[key] = sha;
        ++rpt.uploaded;
        rpt.bytes_uploaded += bytes.size();
        if (opts.verbose) std::printf("PUT %s (%zu bytes)\n", key.c_str(), bytes.size());
    };

    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(opts.catalogs_root))
    {
        if (!entry.is_regular_file()) continue;
        const auto rel = std::filesystem::relative(entry.path(),
                                                   opts.catalogs_root).string();
        const std::string lower = [&]() {
            std::string s = rel;
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            return s;
        }();

        if (opts.tier == Tier::Lossy
            && lower.find(".1bl") == std::string::npos)  continue;
        if (opts.tier == Tier::Lossless
            && lower.find(".flac") == std::string::npos) continue;

        if (opts.only_slug && rel.find(*opts.only_slug) == std::string::npos)
            continue;

        consider(entry.path(), rel);
    }

    if (!opts.dry_run) {
        std::filesystem::create_directories(opts.state_file.parent_path());
        std::ofstream f(opts.state_file);
        f << state.dump(2);
    }
    return rpt;
}

}  // namespace onebit::cli::push
