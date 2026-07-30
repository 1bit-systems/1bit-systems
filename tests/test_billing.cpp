// test_billing.cpp — Unit tests for the jarvis billing module
// Tests Stripe webhook signature verification, subscription state
// management, and pricing.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

#include "jarvis/auth.h"    // for sha256_hex
#include "jarvis/billing.h"

// ── HMAC-SHA256 for test payload signing ─────────────────────────────
// Self-contained so billing tests don't need extra dependencies.
// Uses jarvis::sha256_hex for the underlying hash.

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

// Build a Stripe-signed webhook payload.
// Returns the signature header string.
static std::string sign_webhook(const std::string& payload,
                                const std::string& secret,
                                const std::string& timestamp_str,
                                std::string& out_sig_header) {
    std::string signed_payload = timestamp_str + "." + payload;
    std::string sig = hmac_sha256_hex(secret, signed_payload);
    out_sig_header = "t=" + timestamp_str + ",v1=" + sig;
    return signed_payload;
}

int main() {
    using namespace jarvis;

    std::printf("Testing billing module...\n");

    // ── Pricing ─────────────────────────────────────────────────────
    std::printf("  Testing pricing...\n");
    {
        BillingManager billing;
        auto pricing = billing.get_pricing();
        assert(pricing.basic_monthly == 19.95);
        assert(pricing.pro_monthly == 99.00);
        assert(pricing.enterprise_monthly == 499.00);
        assert(pricing.voice_clone_fee == 29.95);
        std::printf("    ✅ Default pricing OK\n");

        // Test env override
        setenv("STRIPE_PRICE_BASIC_AMOUNT", "24.95", 1);
        pricing = billing.get_pricing();
        assert(pricing.basic_monthly == 24.95);
        unsetenv("STRIPE_PRICE_BASIC_AMOUNT");
        std::printf("    ✅ Pricing env override OK\n");
    }

    // ── Customer linking ────────────────────────────────────────────
    std::printf("  Testing customer-owner linking...\n");
    {
        BillingManager billing;
        billing.link_customer("cus_test123", "owner1");
        assert(billing.get_owner_for_customer("cus_test123") == "owner1");
        assert(billing.get_customer_for_owner("owner1") == "cus_test123");

        // Non-existent lookups
        assert(billing.get_owner_for_customer("nonexistent").empty());
        assert(billing.get_customer_for_owner("nonexistent").empty());
        std::printf("    ✅ Customer-owner linking OK\n");
    }

    // ── Webhook signature verification ──────────────────────────────
    std::printf("  Testing webhook signature verification...\n");
    {
        std::string secret = "whsec_test_secret_key_for_hmac_test_12345";
        setenv("STRIPE_WEBHOOK_SECRET", secret.c_str(), 1);

        BillingManager billing;

        // Test with valid signature — subscription created event
        std::string payload = R"({
            "type": "customer.subscription.created",
            "data": {
                "object": {
                    "id": "sub_test_abc123",
                    "customer": "cus_test_valid",
                    "status": "active",
                    "current_period_start": 1700000000,
                    "current_period_end": 1702592000,
                    "items": {
                        "data": [{
                            "price": {
                                "id": "price_basic_monthly",
                                "product": "prod_basic"
                            }
                        }]
                    }
                }
            }
        })";

        std::string sig_header;
        sign_webhook(payload, secret, "1700000000", sig_header);

        bool processed = billing.process_webhook(payload, sig_header);
        assert(processed);
        assert(billing.is_subscription_active("cus_test_valid"));
        std::printf("    ✅ Valid webhook processed, subscription active\n");

        // Test with tampered payload (wrong timestamp in sig)
        std::string sig_header_wrong;
        sign_webhook(payload + " ", secret, "1700000000", sig_header_wrong);
        assert(sig_header_wrong != sig_header);
        // This should fail because signed payload doesn't match
        // (payload content differs)
        bool tampered = billing.process_webhook(payload + " ", sig_header);
        assert(!tampered);
        std::printf("    ✅ Tampered payload rejected\n");

        // Test with completely invalid signature
        bool bad_sig = billing.process_webhook(payload, "t=0,v1=0000000000000000000000000000000000000000000000000000000000000000");
        assert(!bad_sig);
        std::printf("    ✅ Invalid signature rejected\n");

        // Test with empty signature
        bool empty_sig = billing.process_webhook(payload, "");
        assert(!empty_sig);
        std::printf("    ✅ Empty signature rejected\n");

        // Test dev mode (no secret)
        unsetenv("STRIPE_WEBHOOK_SECRET");
        // In dev mode, process_webhook should accept any signature
        // that has valid format "t=..."
        bool dev_mode = billing.process_webhook(payload, "t=12345,v1=anything");
        assert(dev_mode);
        std::printf("    ✅ Dev mode (no secret) passes\n");

        // Re-set the secret for remaining tests
        setenv("STRIPE_WEBHOOK_SECRET", secret.c_str(), 1);
    }

    // ── Subscription state transitions ──────────────────────────────
    std::printf("  Testing subscription state transitions...\n");
    {
        std::string secret = "whsec_test_transitions";
        setenv("STRIPE_WEBHOOK_SECRET", secret.c_str(), 1);

        BillingManager billing;
        billing.link_customer("cus_transition", "owner_transition");

        // Start with FREE (no subscription)
        assert(!billing.is_subscription_active("cus_transition"));

        // Subscribe to BASIC
        std::string create_payload = R"({
            "type": "customer.subscription.created",
            "data": {
                "object": {
                    "id": "sub_trans_1",
                    "customer": "cus_transition",
                    "status": "active",
                    "current_period_start": 1700000000,
                    "current_period_end": 1702592000,
                    "items": {
                        "data": [{
                            "price": {
                                "id": "price_basic_monthly",
                                "product": "prod_basic"
                            }
                        }]
                    }
                }
            }
        })";

        std::string sig;
        sign_webhook(create_payload, secret, "1700000000", sig);
        assert(billing.process_webhook(create_payload, sig));
        assert(billing.is_subscription_active("cus_transition"));
        std::printf("    ✅ Subscription created -> active\n");

        // Cancel subscription
        std::string cancel_payload = R"({
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_trans_1",
                    "customer": "cus_transition",
                    "status": "canceled"
                }
            }
        })";

        sign_webhook(cancel_payload, secret, "1700000001", sig);
        assert(billing.process_webhook(cancel_payload, sig));
        assert(!billing.is_subscription_active("cus_transition"));
        std::printf("    ✅ Subscription deleted -> inactive\n");

        unsetenv("STRIPE_WEBHOOK_SECRET");
    }

    // ── Persistence (save/load round-trip) ──────────────────────────
    std::printf("  Testing billing persistence...\n");
    {
        std::string secret = "whsec_persist_test";
        setenv("STRIPE_WEBHOOK_SECRET", secret.c_str(), 1);

        BillingManager billing1;
        billing1.link_customer("cus_persist", "owner_persist");

        // Create a subscription
        std::string create_payload = R"({
            "type": "customer.subscription.created",
            "data": {
                "object": {
                    "id": "sub_persist_1",
                    "customer": "cus_persist",
                    "status": "active",
                    "current_period_start": 1700000000,
                    "current_period_end": 1702592000,
                    "items": {
                        "data": [{
                            "price": {
                                "id": "price_pro_monthly",
                                "product": "prod_pro"
                            }
                        }]
                    }
                }
            }
        })";

        std::string sig;
        sign_webhook(create_payload, secret, "1700000000", sig);
        assert(billing1.process_webhook(create_payload, sig));

        // Save
        assert(billing1.save("/tmp/test_billing.json"));

        // Load into new instance
        BillingManager billing2;
        assert(billing2.load("/tmp/test_billing.json"));

        // Verify mapping survived
        assert(billing2.get_owner_for_customer("cus_persist") == "owner_persist");

        // Verify subscription survived
        assert(billing2.is_subscription_active("cus_persist"));

        // Clean up
        std::remove("/tmp/test_billing.json");
        unsetenv("STRIPE_WEBHOOK_SECRET");
        std::printf("    ✅ Billing persistence (save/load) OK\n");
    }

    // ── Unknown event types pass through ────────────────────────────
    std::printf("  Testing unknown event types...\n");
    {
        std::string secret = "whsec_unknown";
        setenv("STRIPE_WEBHOOK_SECRET", secret.c_str(), 1);

        BillingManager billing;

        // Unknown type but correctly signed should return true
        std::string payload = R"({"type": "ping", "data": {"object": {}}})";
        std::string sig;
        sign_webhook(payload, secret, "1700000000", sig);
        assert(billing.process_webhook(payload, sig));

        std::printf("    ✅ Unknown event types pass through\n");
        unsetenv("STRIPE_WEBHOOK_SECRET");
    }

    std::printf("\n✅✅✅ All billing tests passed\n");
    return 0;
}
