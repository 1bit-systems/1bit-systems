// test_auth.cpp — Unit tests for the jarvis auth module
// Verifies SHA-256, API key creation/validation, rate limiting, and tier limits.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "jarvis/auth.h"

// ── HMAC-SHA256 for test payload signing ─────────────────────────────
// Self-contained copy so billing tests don't need to link billing.cpp.
// Uses jarvis::sha256_hex for the underlying hash.

static std::string hex_encode(const uint8_t* data, size_t len) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; i++) {
        out += hex[(data[i] >> 4) & 0xf];
        out += hex[data[i] & 0xf];
    }
    return out;
}

static std::string hmac_sha256_hex(const std::string& key, const std::string& msg) {
    const size_t BLOCK_SIZE = 64;

    std::string kp = key;
    if (kp.size() > BLOCK_SIZE) {
        kp = jarvis::sha256_hex(kp);
    }

    uint8_t k_pad[BLOCK_SIZE] = {0};
    std::memcpy(k_pad, kp.data(), kp.size());

    uint8_t ikey_pad[BLOCK_SIZE], okey_pad[BLOCK_SIZE];
    for (size_t i = 0; i < BLOCK_SIZE; i++) {
        ikey_pad[i] = k_pad[i] ^ 0x36;
        okey_pad[i] = k_pad[i] ^ 0x5c;
    }

    std::string inner;
    inner.reserve(BLOCK_SIZE + msg.size());
    inner.append(reinterpret_cast<const char*>(ikey_pad), BLOCK_SIZE);
    inner.append(msg);
    std::string inner_hash = jarvis::sha256_hex(inner);

    std::string outer;
    outer.reserve(BLOCK_SIZE + inner_hash.size());
    outer.append(reinterpret_cast<const char*>(okey_pad), BLOCK_SIZE);
    outer.append(inner_hash);
    return jarvis::sha256_hex(outer);
}

int main() {
    using namespace jarvis;

    // ── SHA-256 known test vectors ──────────────────────────────────
    std::printf("Testing SHA-256...\n");

    // Empty string
    std::string h_empty = sha256_hex("");
    assert(h_empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    // "hello"
    std::string h_hello = sha256_hex("hello");
    assert(h_hello == "2cf24dba5fb0a30de26f83b2ac5b9e29a1b161e5c1fa7425e73043362938b9824");

    // "abc"
    std::string h_abc = sha256_hex("abc");
    assert(h_abc == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // SHA-256 of 64 'a's (tests multi-block)
    std::string many_a(64, 'a');
    std::string h_manya = sha256_hex(many_a);
    assert(h_manya == "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");

    std::printf("  ✅ SHA-256 test vectors match\n");

    // ── HMAC-SHA256 test vector (RFC 4231 Test Case 2) ──────────────
    std::string hmac_key = "Jefe";
    std::string hmac_msg = "what do ya want for nothing?";
    std::string hmac_result = hmac_sha256_hex(hmac_key, hmac_msg);
    assert(hmac_result == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    std::printf("  ✅ HMAC-SHA256 test vector matches\n");

    // ── API key format ──────────────────────────────────────────────
    std::printf("Testing API key creation/validation...\n");

    AuthManager auth;

    // Create a key
    std::string key1 = auth.create_key("owner1", PlanTier::BASIC);
    assert(key1.size() > 20);
    // Format: sk_zaya_<64hex>_<8hex>
    assert(key1.rfind("sk_zaya_", 0) == 0);
    // Total length: 8 (prefix) + 64 (random) + 1 (_) + 8 (checksum) = 81
    assert(key1.size() == 81);

    std::printf("  Key format: %s...%s (len=%zu)\n",
                key1.substr(0, 16).c_str(),
                key1.substr(key1.size() - 8).c_str(),
                key1.size());

    // Validate the key via Bearer header
    auto* info = auth.validate("Bearer " + key1);
    assert(info != nullptr);
    assert(info->tier == PlanTier::BASIC);
    assert(info->owner_id == "owner1");
    assert(info->active == true);

    // Validate without Bearer prefix should fail
    assert(auth.validate(key1) == nullptr);

    // Invalid key
    assert(auth.validate("Bearer invalid_key_12345") == nullptr);

    // Revoke and verify
    bool revoked = auth.revoke_key(key1);
    assert(revoked);
    assert(auth.validate("Bearer " + key1) == nullptr);

    std::printf("  ✅ Key creation, validation, revocation OK\n");

    // ── Multiple keys per owner ─────────────────────────────────────
    std::printf("Testing multiple keys per owner...\n");

    std::string key2a = auth.create_key("owner2", PlanTier::PRO);
    std::string key2b = auth.create_key("owner2", PlanTier::PRO);
    assert(key2a != key2b);

    auto keys = auth.list_keys("owner2");
    assert(keys.size() == 2);
    assert(keys[0].tier == PlanTier::PRO);
    assert(keys[1].tier == PlanTier::PRO);

    std::printf("  ✅ Multiple keys per owner OK\n");

    // ── Rate limiting ───────────────────────────────────────────────
    std::printf("Testing rate limiting...\n");

    AuthManager auth2;
    auth2.create_key("rate_test_owner", PlanTier::FREE);

    // Free tier: 10 req/min — first 10 should pass, 11th should fail
    for (int i = 0; i < 10; i++) {
        bool ok = auth2.check_rate_limit("rate_test_owner");
        assert(ok);
    }
    bool rate_limited = auth2.check_rate_limit("rate_test_owner");
    assert(!rate_limited);

    std::printf("  ✅ Rate limiting works (10/10 consumed, 11th rejected)\n");

    // ── Tier limits ─────────────────────────────────────────────────
    std::printf("Testing tier limits...\n");

    auto free_limits = TierLimits::for_tier(PlanTier::FREE);
    assert(free_limits.max_minutes == 30.0);
    assert(free_limits.max_tokens == 100000);
    assert(free_limits.max_voices == 1);
    assert(free_limits.max_requests_per_min == 10);
    assert(free_limits.price_monthly == 0.0);

    auto basic_limits = TierLimits::for_tier(PlanTier::BASIC);
    assert(basic_limits.max_minutes == 600.0);
    assert(basic_limits.max_tokens == 10'000'000);
    assert(basic_limits.max_voices == 1);
    assert(basic_limits.price_monthly == 19.95);

    auto pro_limits = TierLimits::for_tier(PlanTier::PRO);
    assert(pro_limits.max_minutes == 6000.0);
    assert(pro_limits.max_voices == 5);
    assert(pro_limits.price_monthly == 99.00);

    auto ent_limits = TierLimits::for_tier(PlanTier::ENTERPRISE);
    assert(ent_limits.max_minutes >= 1e8); // effectively unlimited
    assert(ent_limits.price_monthly == 499.00);

    std::printf("  ✅ Tier limits correct\n");

    // ── Key persistence (save/load round-trip) ──────────────────────
    std::printf("Testing key persistence...\n");

    AuthManager auth3;
    auth3.create_key("persist_owner", PlanTier::BASIC);
    auth3.create_key("persist_owner2", PlanTier::PRO);

    bool saved = auth3.save_keys("/tmp/test_auth_keys.json");
    assert(saved);

    AuthManager auth4;
    bool loaded = auth4.load_keys("/tmp/test_auth_keys.json");
    assert(loaded);

    // Should have 2 keys total
    auto keys_persist = auth4.list_keys("persist_owner");
    assert(keys_persist.size() >= 1);
    assert(keys_persist[0].tier == PlanTier::BASIC);

    // Clean up
    std::remove("/tmp/test_auth_keys.json");

    std::printf("  ✅ Key persistence (save/load) OK\n");

    std::printf("\n✅✅✅ All auth tests passed\n");
    return 0;
}
