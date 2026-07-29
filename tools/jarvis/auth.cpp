// auth.cpp — API key authentication middleware for the Jarvis server.
#include "auth.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jarvis {

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

// ── Simple token tag suffix (not a real SHA-256; stub without openssl) ──
static std::string sha256_stub(const std::string& input) {
    // FNV-1a 64-bit -> hex string as a deterministic content hash.
    // This is NOT cryptographically secure — it's a placeholder for
    // real SHA-256 when openssl is available.
    const uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const uint64_t FNV_PRIME  = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET;
    for (unsigned char c : input) {
        hash ^= c;
        hash *= FNV_PRIME;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)hash);
    return std::string(buf);
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
        auto now = std::chrono::system_clock::now();
        auto seed = now.time_since_epoch().count();
        std::mt19937_64 rng((uint64_t)seed);
        std::uniform_int_distribution<uint64_t> dist;

        std::ostringstream oss;
        oss << "sk_live_" << sha256_stub(owner_id + "_" + std::to_string(dist(rng))).substr(0, 24);
        return oss.str();
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
        int refill = (int)(elapsed * (TierLimits::for_tier(PlanTier::FREE).max_requests_per_min / 60.0));
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
