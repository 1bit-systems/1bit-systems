// usage.cpp — Usage tracking for the Jarvis server.
#include "usage.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace jarvis {

struct UsageTracker::Impl {
    std::unordered_map<std::string, UsageRecord> records;
    mutable std::mutex mtx;
    std::thread auto_save_thread;
    bool auto_save_running = false;
    std::string auto_save_path;
    int auto_save_interval = 300;

    double now_seconds() const {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration<double>(now.time_since_epoch()).count();
    }

    void auto_save_loop() {
        while (auto_save_running) {
            std::this_thread::sleep_for(std::chrono::seconds(auto_save_interval));
            if (auto_save_running && !auto_save_path.empty()) {
                std::lock_guard<std::mutex> lock(mtx);
                json j;
                j["records"] = json::array();
                for (const auto& [oid, rec] : records) {
                    (void)oid;
                    j["records"].push_back({
                        {"owner_id", rec.owner_id},
                        {"minutes_used", rec.minutes_used},
                        {"tokens_processed", rec.tokens_processed},
                        {"requests_count", rec.requests_count},
                        {"period_start", rec.period_start},
                        {"period_end", rec.period_end},
                    });
                }
                std::ofstream f(auto_save_path, std::ios::binary | std::ios::trunc);
                if (f.is_open()) {
                    f << j.dump(2);
                }
            }
        }
    }
};

UsageTracker::UsageTracker() : impl_(std::make_unique<Impl>()) {}
UsageTracker::~UsageTracker() {
    if (impl_->auto_save_running) {
        impl_->auto_save_running = false;
        if (impl_->auto_save_thread.joinable()) {
            impl_->auto_save_thread.join();
        }
    }
}

void UsageTracker::record_usage(const std::string& owner_id, double minutes, int64_t tokens) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto& rec = impl_->records[owner_id];
    rec.owner_id = owner_id;
    rec.minutes_used += minutes;
    rec.tokens_processed += tokens;
    rec.requests_count++;
    // Set period if not set
    if (rec.period_start == 0) {
        rec.period_start = impl_->now_seconds();
        rec.period_end = rec.period_start + 30 * 86400; // ~30 days
    }
}

UsageRecord UsageTracker::get_usage(const std::string& owner_id) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(owner_id);
    if (it == impl_->records.end()) {
        UsageRecord empty;
        empty.owner_id = owner_id;
        return empty;
    }
    return it->second;
}

std::vector<UsageRecord> UsageTracker::get_all_usage() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::vector<UsageRecord> result;
    for (const auto& [oid, rec] : impl_->records) {
        (void)oid;
        result.push_back(rec);
    }
    return result;
}

void UsageTracker::reset_period(const std::string& owner_id, double period_start, double period_end) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto& rec = impl_->records[owner_id];
    rec.owner_id = owner_id;
    rec.minutes_used = 0;
    rec.tokens_processed = 0;
    rec.requests_count = 0;
    rec.period_start = period_start;
    rec.period_end = period_end;
}

std::string UsageTracker::check_limits(const std::string& owner_id, PlanTier tier) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(owner_id);

    auto limits = TierLimits::for_tier(tier);

    // Enterprise/Custom: no limits
    if (limits.max_minutes >= 1e8) return "";

    if (it == impl_->records.end()) return ""; // no usage yet

    const auto& rec = it->second;

    if (rec.minutes_used >= limits.max_minutes) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Monthly voice minute limit exceeded (%.1f / %.1f min)",
                 rec.minutes_used, limits.max_minutes);
        return std::string(buf);
    }

    if (rec.tokens_processed >= limits.max_tokens) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Monthly token limit exceeded (%lld / %lld tokens)",
                 (long long)rec.tokens_processed, (long long)limits.max_tokens);
        return std::string(buf);
    }

    return "";
}

bool UsageTracker::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    json j;
    j["records"] = json::array();
    for (const auto& [oid, rec] : impl_->records) {
        (void)oid;
        j["records"].push_back({
            {"owner_id", rec.owner_id},
            {"minutes_used", rec.minutes_used},
            {"tokens_processed", rec.tokens_processed},
            {"requests_count", rec.requests_count},
            {"period_start", rec.period_start},
            {"period_end", rec.period_end},
        });
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f << j.dump(2);
    return true;
}

bool UsageTracker::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::ifstream f(path);
    if (!f.is_open()) return false;

    try {
        json j;
        f >> j;

        impl_->records.clear();
        if (j.contains("records") && j["records"].is_array()) {
            for (auto& entry : j["records"]) {
                UsageRecord rec;
                rec.owner_id = entry.value("owner_id", "");
                rec.minutes_used = entry.value("minutes_used", 0.0);
                rec.tokens_processed = entry.value("tokens_processed", (int64_t)0);
                rec.requests_count = entry.value("requests_count", (int64_t)0);
                rec.period_start = entry.value("period_start", 0.0);
                rec.period_end = entry.value("period_end", 0.0);
                if (!rec.owner_id.empty()) {
                    impl_->records[rec.owner_id] = rec;
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

void UsageTracker::set_auto_save(const std::string& path, int interval_seconds) {
    if (impl_->auto_save_running) {
        impl_->auto_save_running = false;
        if (impl_->auto_save_thread.joinable()) {
            impl_->auto_save_thread.join();
        }
    }
    impl_->auto_save_path = path;
    impl_->auto_save_interval = interval_seconds;
    impl_->auto_save_running = true;
    impl_->auto_save_thread = std::thread(&Impl::auto_save_loop, impl_.get());
}

} // namespace jarvis
