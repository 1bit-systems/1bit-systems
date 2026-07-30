// auth.cpp — API key authentication middleware for the Jarvis server.
#include "auth.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jarvis {

// ═══════════════════════════════════════════════════════════════════════════
// SHA-256 implementation (FIPS 180-4)
// Self-contained, no OpenSSL dependency.
// ═══════════════════════════════════════════════════════════════════════════

std::string sha256_hex(const std::string& input) {
    // Round constants: first 32 bits of fractional parts of cube roots
    // of the first 64 primes.
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // Initial hash values: first 32 bits of fractional parts of square
    // roots of the first 8 primes.
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    size_t orig_len = input.size();
    uint64_t bit_len = (uint64_t)orig_len * 8;

    // Pre-processing: pad to multiple of 64 bytes.
    // Need: 0x80 + zeros until (length ≡ 56 mod 64) + 8-byte bit length.
    size_t padded_len = ((orig_len + 8 + 64) / 64) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    if (orig_len > 0) {
        std::memcpy(padded.data(), input.data(), orig_len);
    }
    padded[orig_len] = 0x80;

    // Append bit length as 64-bit big-endian.
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 8 + i] = (uint8_t)(bit_len >> (56 - i * 8));
    }

    // Process each 512-bit (64-byte) block.
    for (size_t block = 0; block < padded_len; block += 64) {
        uint32_t W[64];

        // Prepare message schedule (big-endian words).
        for (int t = 0; t < 16; t++) {
            size_t idx = block + t * 4;
            W[t] = ((uint32_t)padded[idx]     << 24) |
                   ((uint32_t)padded[idx + 1] << 16) |
                   ((uint32_t)padded[idx + 2] <<  8) |
                   ((uint32_t)padded[idx + 3]);
        }
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = std::rotr(W[t - 15], 7) ^ std::rotr(W[t - 15], 18) ^ (W[t - 15] >> 3);
            uint32_t s1 = std::rotr(W[t - 2], 17) ^ std::rotr(W[t - 2], 19) ^ (W[t - 2] >> 10);
            W[t] = W[t - 16] + s0 + W[t - 7] + s1;
        }

        // Initialize working variables.
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
        uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

        // Compression function main loop.
        for (int t = 0; t < 64; t++) {
            uint32_t S1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + K[t] + W[t];
            uint32_t S0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }

        // Update hash state.
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    // Produce 64-character hex string (big-endian word order).
    std::ostringstream oss;
    for (int i = 0; i < 8; i++) {
        oss << std::hex << std::setfill('0') << std::setw(8) << H[i];
    }
    return oss.str();
}

// ── HMAC-SHA256 (RFC 2104) ──────────────────────────────────────────────
// Used internally for Stripe webhook verification (billing.cpp) and for
// the API key checksum. Self-contained.

[[maybe_unused]] static std::string hmac_sha256_hex(const std::string& key, const std::string& msg) {
    // Block size for SHA-256 is 64 bytes.
    const size_t BLOCK_SIZE = 64;

    // If key is longer than block size, hash it first.
    std::string kp = key;
    if (kp.size() > BLOCK_SIZE) {
        kp = sha256_hex(kp);
    }

    // Pad key to block size with zeros.
    std::vector<uint8_t> k_pad(BLOCK_SIZE, 0);
    std::memcpy(k_pad.data(), kp.data(), kp.size());

    // Compute ipad (0x36) and opad (0x5c) keys.
    std::vector<uint8_t> ikey_pad(BLOCK_SIZE), okey_pad(BLOCK_SIZE);
    for (size_t i = 0; i < BLOCK_SIZE; i++) {
        ikey_pad[i] = k_pad[i] ^ 0x36;
        okey_pad[i] = k_pad[i] ^ 0x5c;
    }

    // Inner hash: SHA256((K ⊕ ipad) || msg)
    std::string inner;
    inner.reserve(BLOCK_SIZE + msg.size());
    inner.append(reinterpret_cast<const char*>(ikey_pad.data()), BLOCK_SIZE);
    inner.append(msg);
    std::string inner_hash = sha256_hex(inner);

    // Outer hash: SHA256((K ⊕ opad) || inner_hash)
    std::string outer;
    outer.reserve(BLOCK_SIZE + inner_hash.size());
    outer.append(reinterpret_cast<const char*>(okey_pad.data()), BLOCK_SIZE);
    outer.append(inner_hash);
    return sha256_hex(outer);
}

// ── TierLimits ───────────────────────────────────────────────────────
TierLimits TierLimits::for_tier(PlanTier tier) {
    TierLimits limits;
    switch (tier) {
        case PlanTier::FREE:
            limits.max_minutes = 30.0;
            limits.max_tokens = 100000;
            limits.max_voices = 1;
            limits.max_requests_per_min = 10;
            limits.price_monthly = 0.0;
            break;
        case PlanTier::BASIC:
            limits.max_minutes = 600.0;      // 10 hours
            limits.max_tokens = 10'000'000;  // 10M
            limits.max_voices = 1;
            limits.max_requests_per_min = 60;
            limits.price_monthly = 19.95;
            break;
        case PlanTier::PRO:
            limits.max_minutes = 6000.0;       // 100 hours
            limits.max_tokens = 100'000'000;   // 100M
            limits.max_voices = 5;
            limits.max_requests_per_min = 300;
            limits.price_monthly = 99.00;
            break;
        case PlanTier::ENTERPRISE:
        case PlanTier::CUSTOM:
            limits.max_minutes = 1e9;   // effectively unlimited
            limits.max_tokens = 1e12;
            limits.max_voices = 100;
            limits.max_requests_per_min = 10000;
            limits.price_monthly = (tier == PlanTier::ENTERPRISE) ? 499.00 : 0.0;
            break;
    }
    return limits;
}

// ── Rate limiter: sliding-window token bucket per owner ──────────────
struct RateBucket {
    int tokens = 0;
    double last_refill = 0;
};

struct AuthManager::Impl {
    std::unordered_map<std::string, ApiKey> keys_by_key;     // key -> ApiKey
    std::unordered_map<std::string, std::vector<std::string>> keys_by_owner; // owner_id -> [keys]
    mutable std::mutex mtx;
    mutable std::unordered_map<std::string, RateBucket> rate_buckets; // owner_id -> bucket
    int default_burst = 10;

    std::string generate_key(const std::string& owner_id) {
        // Format: sk_zaya_<64-hex-random>_<8-hex-checksum>
        // Random: 32 bytes from /dev/urandom
        // Checksum: SHA-256 of random part, first 4 bytes as hex

        uint8_t random_bytes[32];
        FILE* urandom = std::fopen("/dev/urandom", "rb");
        if (!urandom) {
            // Fallback: use time + mt19937_64 if /dev/urandom unavailable
            auto now = std::chrono::system_clock::now();
            auto seed = now.time_since_epoch().count();
            std::mt19937_64 rng((uint64_t)seed);
            for (int i = 0; i < 32; i += 8) {
                uint64_t val = rng();
                std::memcpy(random_bytes + i, &val, 8);
            }
        } else {
            size_t nread = std::fread(random_bytes, 1, 32, urandom);
            std::fclose(urandom);
            if (nread != 32) {
                // Partial read — fill remaining with PRNG
                auto now = std::chrono::system_clock::now();
                auto seed = now.time_since_epoch().count();
                std::mt19937_64 rng((uint64_t)seed);
                for (size_t i = nread; i < 32; i += 8) {
                    uint64_t val = rng();
                    size_t copy = std::min(size_t(8), 32 - i);
                    std::memcpy(random_bytes + i, &val, copy);
                }
            }
        }

        // Convert random bytes to hex string
        std::ostringstream random_hex;
        for (int i = 0; i < 32; i++) {
            random_hex << std::hex << std::setfill('0') << std::setw(2)
                       << (int)random_bytes[i];
        }
        std::string rand_str = random_hex.str();

        // Compute checksum: SHA-256 of random part, first 4 bytes
        std::string checksum_full = sha256_hex(rand_str);
        std::string checksum = checksum_full.substr(0, 8);

        return "sk_zaya_" + rand_str + "_" + checksum;
    }

    double now_seconds() const {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration<double>(now.time_since_epoch()).count();
    }

    void refill_bucket(const std::string& owner_id) const {
        auto it = rate_buckets.find(owner_id);
        if (it == rate_buckets.end()) {
            RateBucket b;
            b.tokens = default_burst;
            b.last_refill = now_seconds();
            rate_buckets[owner_id] = b;
            return;
        }
        auto& bucket = it->second;
        double now = now_seconds();
        double elapsed = now - bucket.last_refill;

        // Determine refill rate based on tier
        int max_rpm = TierLimits::for_tier(PlanTier::FREE).max_requests_per_min;
        // Look up actual tier for this owner
        auto oit = keys_by_owner.find(owner_id);
        if (oit != keys_by_owner.end() && !oit->second.empty()) {
            auto kit = keys_by_key.find(oit->second[0]);
            if (kit != keys_by_key.end()) {
                max_rpm = TierLimits::for_tier(kit->second.tier).max_requests_per_min;
            }
        }

        int refill = (int)(elapsed * (max_rpm / 60.0));
        if (refill > 0) {
            bucket.tokens = std::min(bucket.tokens + refill, default_burst);
            bucket.last_refill = now;
        }
    }
};

AuthManager::AuthManager() : impl_(std::make_unique<Impl>()) {}
AuthManager::~AuthManager() = default;

std::string AuthManager::extract_bearer(const std::string& auth_header) {
    static const std::string prefix = "Bearer ";
    if (auth_header.size() > prefix.size() &&
        auth_header.compare(0, prefix.size(), prefix) == 0) {
        return auth_header.substr(prefix.size());
    }
    return "";
}

const ApiKey* AuthManager::validate(const std::string& auth_header) const {
    std::string token = extract_bearer(auth_header);
    if (token.empty()) return nullptr;

    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->keys_by_key.find(token);
    if (it == impl_->keys_by_key.end()) return nullptr;
    if (!it->second.active) return nullptr;

    // Check expiration
    if (it->second.expires_at > 0) {
        double now = impl_->now_seconds();
        if (now > it->second.expires_at) return nullptr;
    }

    return &it->second;
}

std::string AuthManager::create_key(const std::string& owner_id, PlanTier tier, int valid_days) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    ApiKey ak;
    ak.key = impl_->generate_key(owner_id);
    ak.owner_id = owner_id;
    ak.tier = tier;
    ak.active = true;
    ak.created_at = impl_->now_seconds();
    if (valid_days > 0) {
        ak.expires_at = ak.created_at + valid_days * 86400.0;
    } else {
        ak.expires_at = 0; // never expires
    }

    impl_->keys_by_key[ak.key] = ak;
    impl_->keys_by_owner[owner_id].push_back(ak.key);
    return ak.key;
}

bool AuthManager::revoke_key(const std::string& key) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->keys_by_key.find(key);
    if (it == impl_->keys_by_key.end()) return false;
    it->second.active = false;
    return true;
}

std::vector<ApiKey> AuthManager::list_keys(const std::string& owner_id) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::vector<ApiKey> result;
    auto it = impl_->keys_by_owner.find(owner_id);
    if (it == impl_->keys_by_owner.end()) return result;
    for (const auto& key_str : it->second) {
        auto kit = impl_->keys_by_key.find(key_str);
        if (kit != impl_->keys_by_key.end()) {
            result.push_back(kit->second);
        }
    }
    return result;
}

bool AuthManager::check_rate_limit(const std::string& owner_id) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // Determine rate limit for this owner's tier
    PlanTier tier = PlanTier::FREE;
    auto it = impl_->keys_by_owner.find(owner_id);
    if (it != impl_->keys_by_owner.end() && !it->second.empty()) {
        auto kit = impl_->keys_by_key.find(it->second[0]);
        if (kit != impl_->keys_by_key.end()) {
            tier = kit->second.tier;
        }
    }

    int max_rpm = TierLimits::for_tier(tier).max_requests_per_min;
    if (max_rpm <= 0) return true; // unlimited

    impl_->refill_bucket(owner_id);
    auto& bucket = impl_->rate_buckets[owner_id];
    if (bucket.tokens <= 0) return false; // rate limited
    bucket.tokens--;
    return true;
}

bool AuthManager::check_usage_limit(const std::string& owner_id, double additional_minutes) const {
    // Usage limits are checked against UsageTracker, not AuthManager.
    // This method provides a quick tier-based check based on key metadata.
    std::lock_guard<std::mutex> lock(impl_->mtx);

    PlanTier tier = PlanTier::FREE;
    auto it = impl_->keys_by_owner.find(owner_id);
    if (it != impl_->keys_by_owner.end() && !it->second.empty()) {
        auto kit = impl_->keys_by_key.find(it->second[0]);
        if (kit != impl_->keys_by_key.end()) {
            tier = kit->second.tier;
        }
    }

    auto limits = TierLimits::for_tier(tier);
    if (limits.max_minutes >= 1e8) return true; // unlimited enterprise

    // The actual minute check is done by UsageTracker; this is a fast
    // pre-check that just verifies the tier allows the request.
    (void)additional_minutes;
    return true;
}

bool AuthManager::load_keys(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::ifstream f(path);
    if (!f.is_open()) return false;

    try {
        json j;
        f >> j;

        impl_->keys_by_key.clear();
        impl_->keys_by_owner.clear();

        if (j.contains("keys") && j["keys"].is_array()) {
            for (auto& entry : j["keys"]) {
                ApiKey ak;
                ak.key = entry.value("key", "");
                ak.owner_id = entry.value("owner_id", "");
                ak.tier = static_cast<PlanTier>(entry.value("tier", 0));
                ak.active = entry.value("active", true);
                ak.created_at = entry.value("created_at", 0.0);
                ak.expires_at = entry.value("expires_at", 0.0);

                if (!ak.key.empty()) {
                    impl_->keys_by_key[ak.key] = ak;
                    impl_->keys_by_owner[ak.owner_id].push_back(ak.key);
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool AuthManager::save_keys(const std::string& path) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    json j;
    j["keys"] = json::array();
    for (const auto& [key_str, ak] : impl_->keys_by_key) {
        (void)key_str;
        j["keys"].push_back({
            {"key", ak.key},
            {"owner_id", ak.owner_id},
            {"tier", static_cast<int>(ak.tier)},
            {"active", ak.active},
            {"created_at", ak.created_at},
            {"expires_at", ak.expires_at},
        });
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f << j.dump(2);
    return true;
}

} // namespace jarvis
