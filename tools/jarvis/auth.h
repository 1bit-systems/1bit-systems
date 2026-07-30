// auth.h — API key authentication middleware for the Jarvis server.
// Phase 2.3: Commercial API (SaaS Layer).
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace jarvis {

enum class PlanTier {
    FREE,       // 30 min demo
    BASIC,      // $19.95/mo - 1 voice, 10 hours
    PRO,        // $99/mo - 5 voices, 100 hours
    ENTERPRISE, // $499/mo - unlimited
    CUSTOM      // Negotiated
};

struct ApiKey {
    std::string key;          // sk_live_xxx / sk_test_xxx
    std::string owner_id;     // user identifier
    PlanTier tier = PlanTier::FREE;
    bool active = true;
    double created_at = 0;
    double expires_at = 0;    // 0 = never
};

struct UsageRecord {
    std::string owner_id;
    double minutes_used = 0;       // voice conversation minutes
    int64_t tokens_processed = 0;  // LLM tokens
    int64_t requests_count = 0;    // total API requests
    double period_start = 0;       // billing period start
    double period_end = 0;         // billing period end
};

struct TierLimits {
    double max_minutes = 30.0;
    int64_t max_tokens = 100000;
    int max_voices = 1;
    int max_requests_per_min = 10;
    double price_monthly = 0.0;

    static TierLimits for_tier(PlanTier tier);
};

class AuthManager {
public:
    AuthManager();
    ~AuthManager();

    // Validate API key from Authorization header
    // Returns nullptr if invalid
    const ApiKey* validate(const std::string& auth_header) const;

    // Create a new API key
    std::string create_key(const std::string& owner_id, PlanTier tier, int valid_days = 365);

    // Revoke an API key
    bool revoke_key(const std::string& key);

    // List keys for an owner
    std::vector<ApiKey> list_keys(const std::string& owner_id) const;

    // Check rate limit
    bool check_rate_limit(const std::string& owner_id) const;

    // Check if owner has reached their tier limit
    bool check_usage_limit(const std::string& owner_id, double additional_minutes = 0) const;

    // Extract bearer token from Authorization header
    static std::string extract_bearer(const std::string& auth_header);

    // Load/save API keys from file
    bool load_keys(const std::string& path);
    bool save_keys(const std::string& path) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// SHA-256 hex digest (FIPS 180-4, no external dependencies)
std::string sha256_hex(const std::string& input);

} // namespace jarvis
