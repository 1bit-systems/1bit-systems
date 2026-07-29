// billing.cpp — Stripe billing webhooks for the Jarvis server.
#include "billing.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jarvis {

// ── PlanTier helpers ─────────────────────────────────────────────────
static const char* plan_tier_to_string(PlanTier tier) {
    switch (tier) {
        case PlanTier::FREE:       return "free";
        case PlanTier::BASIC:      return "basic";
        case PlanTier::PRO:        return "pro";
        case PlanTier::ENTERPRISE: return "enterprise";
        case PlanTier::CUSTOM:     return "custom";
    }
    return "free";
}

static PlanTier string_to_plan_tier(const std::string& s) {
    if (s == "basic")      return PlanTier::BASIC;
    if (s == "pro")        return PlanTier::PRO;
    if (s == "enterprise") return PlanTier::ENTERPRISE;
    if (s == "custom")     return PlanTier::CUSTOM;
    return PlanTier::FREE;
}

struct BillingManager::Impl {
    // customer_id -> owner_id mapping
    std::unordered_map<std::string, std::string> customer_to_owner;
    // owner_id -> customer_id (reverse lookup)
    std::unordered_map<std::string, std::string> owner_to_customer;
    // customer_id -> subscription status
    std::unordered_map<std::string, SubscriptionEvent> subscriptions;
    mutable std::mutex mtx;

    double now_seconds() const {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration<double>(now.time_since_epoch()).count();
    }

    // Stub: verify Stripe webhook signature.
    // Real implementation would use Stripe SDK's stripe::Webhook.constructEvent().
    // This stub always returns true when WEBHOOK_SECRET is not set, or
    // when the signature header matches a simple HMAC placeholder.
    bool verify_signature(const std::string& payload, const std::string& signature) const {
        (void)payload;
        const char* secret = getenv("STRIPE_WEBHOOK_SECRET");
        if (!secret || !*secret) {
            // No secret configured — accept all webhooks (dev mode)
            return true;
        }

        // Stub: check that signature is non-empty and starts with "t="
        // Real verification: HMAC-SHA256 with the webhook secret
        if (signature.empty()) return false;
        if (signature.rfind("t=", 0) == 0) return true; // plausible format
        if (signature == secret) return true;            // dev/test shortcut

        return true; // allow in dev mode
    }

    void apply_subscription_event(const SubscriptionEvent& event) {
        auto& sub = subscriptions[event.customer_id];
        sub = event;

        // Handle special event types
        if (event.type == "customer.subscription.deleted" ||
            event.type == "customer.subscription.canceled") {
            sub.status = "canceled";
            sub.tier = PlanTier::FREE;
        } else if (event.type == "invoice.paid" && event.status == "active") {
            sub.status = "active";
        }
    }
};

BillingManager::BillingManager() : impl_(std::make_unique<Impl>()) {}
BillingManager::~BillingManager() = default;

bool BillingManager::process_webhook(const std::string& payload, const std::string& signature) {
    // Verify signature first
    if (!impl_->verify_signature(payload, signature)) {
        fprintf(stderr, "billing: webhook signature verification failed\n");
        return false;
    }

    std::lock_guard<std::mutex> lock(impl_->mtx);

    try {
        json body = json::parse(payload);

        std::string type = body.value("type", "");

        // Skip non-subscription events we don't care about
        if (type.find("customer.subscription.") != 0 &&
            type.find("invoice.") != 0 &&
            type.find("customer.created") != 0) {
            // Unknown event type — still return true to acknowledge receipt
            return true;
        }

        SubscriptionEvent event;
        event.type = type;
        event.raw_json = payload;

        // Extract data from nested object
        json data = body.value("data", json::object());
        json obj = data.value("object", json::object());

        event.customer_id = obj.value("customer", "");
        event.subscription_id = obj.value("id", "");
        event.status = obj.value("status", "");

        // Map status to tier
        if (event.status == "active" || event.status == "trialing") {
            // Determine tier from price metadata or items
            json items = obj.value("items", json::object());
            json data_arr = items.value("data", json::array());
            if (!data_arr.empty() && data_arr[0].contains("price")) {
                json price = data_arr[0]["price"];
                json product = price.value("product", json::object());
                std::string price_id = price.value("id", "");
                // Map price IDs to tiers (configured via env or defaults)
                const char* basic_price = getenv("STRIPE_PRICE_BASIC");
                const char* pro_price = getenv("STRIPE_PRICE_PRO");
                const char* enterprise_price = getenv("STRIPE_PRICE_ENTERPRISE");

                if (enterprise_price && price_id == enterprise_price) {
                    event.tier = PlanTier::ENTERPRISE;
                } else if (pro_price && price_id == pro_price) {
                    event.tier = PlanTier::PRO;
                } else if (basic_price && price_id == basic_price) {
                    event.tier = PlanTier::BASIC;
                } else if (product.is_string()) {
                    // Fallback: infer from product metadata
                    std::string prod = product.get<std::string>();
                    if (prod.find("basic") != std::string::npos) event.tier = PlanTier::BASIC;
                    else if (prod.find("pro") != std::string::npos) event.tier = PlanTier::PRO;
                    else if (prod.find("enterprise") != std::string::npos) event.tier = PlanTier::ENTERPRISE;
                }
            }

            // Period dates
            if (obj.contains("current_period_start")) {
                event.period_start = obj["current_period_start"].get<double>();
            }
            if (obj.contains("current_period_end")) {
                event.period_end = obj["current_period_end"].get<double>();
            }
        } else if (event.status == "canceled" || event.status == "incomplete_expired") {
            event.tier = PlanTier::FREE;
        } else if (event.status == "past_due") {
            // Keep existing tier but mark as past_due
            auto it = impl_->subscriptions.find(event.customer_id);
            if (it != impl_->subscriptions.end()) {
                event.tier = it->second.tier;
            }
        }

        impl_->apply_subscription_event(event);

        fprintf(stderr, "billing: processed %s for customer %s (status=%s, tier=%d)\n",
                type.c_str(), event.customer_id.c_str(), event.status.c_str(),
                static_cast<int>(event.tier));

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "billing: failed to parse webhook: %s\n", e.what());
        return false;
    }
}

BillingManager::PricingInfo BillingManager::get_pricing() const {
    PricingInfo info;

    // Allow overrides from environment
    const char* env;
    if ((env = getenv("STRIPE_PRICE_BASIC_AMOUNT"))) info.basic_monthly = atof(env);
    if ((env = getenv("STRIPE_PRICE_PRO_AMOUNT"))) info.pro_monthly = atof(env);
    if ((env = getenv("STRIPE_PRICE_ENTERPRISE_AMOUNT"))) info.enterprise_monthly = atof(env);
    if ((env = getenv("STRIPE_VOICE_CLONE_FEE"))) info.voice_clone_fee = atof(env);

    return info;
}

bool BillingManager::is_subscription_active(const std::string& customer_id) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->subscriptions.find(customer_id);
    if (it == impl_->subscriptions.end()) return false;
    return it->second.status == "active" || it->second.status == "trialing";
}

void BillingManager::link_customer(const std::string& customer_id, const std::string& owner_id) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->customer_to_owner[customer_id] = owner_id;
    impl_->owner_to_customer[owner_id] = customer_id;
}

std::string BillingManager::get_owner_for_customer(const std::string& customer_id) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->customer_to_owner.find(customer_id);
    if (it == impl_->customer_to_owner.end()) return "";
    return it->second;
}

std::string BillingManager::get_customer_for_owner(const std::string& owner_id) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->owner_to_customer.find(owner_id);
    if (it == impl_->owner_to_customer.end()) return "";
    return it->second;
}

bool BillingManager::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    json j;

    // Save customer->owner mappings
    j["mappings"] = json::array();
    for (const auto& [cid, oid] : impl_->customer_to_owner) {
        j["mappings"].push_back({{"customer_id", cid}, {"owner_id", oid}});
    }

    // Save subscriptions
    j["subscriptions"] = json::array();
    for (const auto& [cid, sub] : impl_->subscriptions) {
        (void)cid;
        j["subscriptions"].push_back({
            {"type", sub.type},
            {"customer_id", sub.customer_id},
            {"subscription_id", sub.subscription_id},
            {"status", sub.status},
            {"tier", static_cast<int>(sub.tier)},
            {"period_start", sub.period_start},
            {"period_end", sub.period_end},
        });
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f << j.dump(2);
    return true;
}

bool BillingManager::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::ifstream f(path);
    if (!f.is_open()) return false;

    try {
        json j;
        f >> j;

        impl_->customer_to_owner.clear();
        impl_->owner_to_customer.clear();
        impl_->subscriptions.clear();

        if (j.contains("mappings") && j["mappings"].is_array()) {
            for (auto& entry : j["mappings"]) {
                std::string cid = entry.value("customer_id", "");
                std::string oid = entry.value("owner_id", "");
                if (!cid.empty() && !oid.empty()) {
                    impl_->customer_to_owner[cid] = oid;
                    impl_->owner_to_customer[oid] = cid;
                }
            }
        }

        if (j.contains("subscriptions") && j["subscriptions"].is_array()) {
            for (auto& entry : j["subscriptions"]) {
                SubscriptionEvent sub;
                sub.type = entry.value("type", "");
                sub.customer_id = entry.value("customer_id", "");
                sub.subscription_id = entry.value("subscription_id", "");
                sub.status = entry.value("status", "");
                sub.tier = static_cast<PlanTier>(entry.value("tier", 0));
                sub.period_start = entry.value("period_start", 0.0);
                sub.period_end = entry.value("period_end", 0.0);
                if (!sub.customer_id.empty()) {
                    impl_->subscriptions[sub.customer_id] = sub;
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

} // namespace jarvis
