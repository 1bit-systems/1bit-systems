// usage.h — Usage tracking for the Jarvis server.
// Phase 2.3: Commercial API (SaaS Layer).
#pragma once

#include "auth.h"
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

namespace jarvis {

class UsageTracker {
public:
    UsageTracker();
    ~UsageTracker();

    // Record usage for an owner
    void record_usage(const std::string& owner_id, double minutes, int64_t tokens);

    // Get current usage this period
    UsageRecord get_usage(const std::string& owner_id) const;

    // Get all usage records (for admin)
    std::vector<UsageRecord> get_all_usage() const;

    // Reset a usage period (called when subscription renews)
    void reset_period(const std::string& owner_id, double period_start, double period_end);

    // Check if owner has exceeded their tier limits
    // Returns empty string if ok, error message if exceeded
    std::string check_limits(const std::string& owner_id, PlanTier tier) const;

    // Save/load usage to file (periodic persistence)
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // Auto-save every N seconds (call from timer)
    void set_auto_save(const std::string& path, int interval_seconds = 300);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
