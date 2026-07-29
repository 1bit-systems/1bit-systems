// billing.h — Stripe billing webhooks for the Jarvis server.
// Phase 2.3: Commercial API (SaaS Layer).
#pragma once

#include "auth.h"
#include <string>
#include <functional>
#include <memory>

namespace jarvis {

struct SubscriptionEvent {
    std::string type;              // "customer.subscription.created", "updated", "deleted", "invoice.paid"
    std::string customer_id;       // Stripe customer ID
    std::string subscription_id;   // Stripe subscription ID
    std::string status;            // "active", "past_due", "canceled", "incomplete"
    PlanTier tier = PlanTier::FREE;
    double period_start = 0;
    double period_end = 0;
    std::string raw_json;          // original webhook body for verification
};

class BillingManager {
public:
    BillingManager();
    ~BillingManager();

    // Process a Stripe webhook event
    // Returns true if event was processed successfully
    bool process_webhook(const std::string& payload, const std::string& signature);

    // Get pricing info
    struct PricingInfo {
        double basic_monthly = 19.95;
        double pro_monthly = 99.00;
        double enterprise_monthly = 499.00;
        double voice_clone_fee = 29.95;  // one-time
    };
    PricingInfo get_pricing() const;

    // Check if a subscription is active
    bool is_subscription_active(const std::string& customer_id) const;

    // Map customer to owner
    void link_customer(const std::string& customer_id, const std::string& owner_id);
    std::string get_owner_for_customer(const std::string& customer_id) const;

    // Get customer Stripe ID for an owner (for portal link generation)
    std::string get_customer_for_owner(const std::string& owner_id) const;

    // Save/load billing state
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
