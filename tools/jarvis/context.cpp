// context.cpp — ContextMemory implementation.
// Conversation history management with summarization and persistence.

#include "jarvis/context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

using json = nlohmann::json;

namespace jarvis {

// ── Implementation struct ──────────────────────────────────────────────

struct ContextMemory::Impl {
    size_t max_turns_;
    std::deque<Turn> turns_;
    mutable std::mutex mutex_;

    explicit Impl(size_t max_turns) : max_turns_(max_turns) {}

    static double now_seconds() {
        return std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

// ── Public API ─────────────────────────────────────────────────────────

ContextMemory::ContextMemory(size_t max_turns)
    : impl_(std::make_unique<Impl>(max_turns)) {}
ContextMemory::~ContextMemory() = default;

void ContextMemory::add_turn(const std::string& role, const std::string& text, float duration_ms) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    Turn turn;
    turn.role = role;
    turn.text = text;
    turn.timestamp = Impl::now_seconds();
    turn.duration_ms = duration_ms;

    impl_->turns_.push_back(std::move(turn));

    // Summarize if exceeded max_turns
    while (impl_->turns_.size() > impl_->max_turns_) {
        // Pop oldest turns and summarize them
        std::string summary_text;
        int batch_size = std::min((size_t)5, impl_->turns_.size() - impl_->max_turns_ + 1);
        for (int i = 0; i < batch_size && !impl_->turns_.empty(); i++) {
            const auto& old = impl_->turns_.front();
            if (!summary_text.empty()) summary_text += " ";
            summary_text += old.role + ": " + old.text;
            impl_->turns_.pop_front();
        }

        // Insert summarized version
        if (!summary_text.empty()) {
            Turn summary_turn;
            summary_turn.role = "system";
            summary_turn.text = "[Summarized earlier conversation: " + summarize(summary_text) + "]";
            summary_turn.timestamp = Impl::now_seconds();
            summary_turn.duration_ms = 0;
            impl_->turns_.push_front(std::move(summary_turn));
        }
    }
}

std::vector<Turn> ContextMemory::get_turns() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    std::vector<Turn> result;
    result.reserve(impl_->turns_.size());
    for (const auto& t : impl_->turns_) {
        result.push_back(t);
    }
    return result;
}

std::vector<Turn> ContextMemory::get_recent(int n) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    std::vector<Turn> result;
    int count = std::min(n, (int)impl_->turns_.size());
    result.reserve(count);
    auto start = impl_->turns_.end() - count;
    for (auto it = start; it != impl_->turns_.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

std::string ContextMemory::build_context(int max_recent_turns) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    std::vector<Turn> recent = get_recent(max_recent_turns);
    if (recent.empty()) return "";

    std::ostringstream ss;
    ss << "## Conversation History\n\n";

    for (const auto& turn : recent) {
        std::string label = (turn.role == "assistant") ? "Assistant" : "User";
        ss << label << ": " << turn.text << "\n";
    }

    return ss.str();
}

std::string ContextMemory::summarize(const std::string& text) {
    if (text.empty()) return "";

    // Simple extractive summarizer: keep first 200 chars + important keywords.
    // This avoids needing an external summarization model.
    const size_t max_summary = 200;
    if (text.size() <= max_summary) return text;

    // Take first 150 chars as a prefix
    std::string prefix = text.substr(0, 150);

    // Find sentence boundary
    size_t last_period = prefix.rfind('.');
    size_t last_excl = prefix.rfind('!');
    size_t last_q = prefix.rfind('?');
    size_t boundary = std::max({last_period, last_excl, last_q});

    if (boundary != std::string::npos && boundary > 30) {
        prefix = prefix.substr(0, boundary + 1);
    }

    // Extract key terms (words that appear multiple times)
    std::string lower = text;
    for (auto& c : lower) c = (char)std::tolower((unsigned char)c);

    std::unordered_set<std::string> stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "and", "but", "or", "if", "while", "about", "up", "it", "its",
        "i", "you", "he", "she", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "our", "their", "this",
        "that", "these", "those"
    };

    // Count word frequencies
    std::unordered_set<std::string> keywords;
    std::istringstream iss(lower);
    std::string word;
    while (iss >> word) {
        // Clean word
        std::string clean;
        for (char c : word) {
            if (std::isalnum((unsigned char)c)) clean += c;
        }
        if (clean.size() >= 4 && stop_words.find(clean) == stop_words.end()) {
            keywords.insert(clean);
        }
    }

    // Build summary: prefix + key terms
    std::ostringstream summary;
    summary << prefix;

    // Add key terms as context hints
    if (keywords.size() > 3) {
        summary << " [Topics: ";
        int count = 0;
        for (const auto& kw : keywords) {
            if (count >= 5) break;
            if (count > 0) summary << ", ";
            summary << kw;
            count++;
        }
        summary << "]";
    }

    return summary.str();
}

void ContextMemory::clear() {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->turns_.clear();
}

bool ContextMemory::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    json j = json::array();
    for (const auto& turn : impl_->turns_) {
        j.push_back({
            {"role", turn.role},
            {"text", turn.text},
            {"timestamp", turn.timestamp},
            {"duration_ms", turn.duration_ms}
        });
    }

    try {
        // Ensure directory exists
        std::filesystem::path p(path);
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << j.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

bool ContextMemory::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    json j;
    try {
        f >> j;
    } catch (...) {
        return false;
    }

    if (!j.is_array()) return false;

    impl_->turns_.clear();
    for (const auto& entry : j) {
        Turn turn;
        turn.role = entry.value("role", "user");
        turn.text = entry.value("text", "");
        turn.timestamp = entry.value("timestamp", 0.0);
        turn.duration_ms = entry.value("duration_ms", 0.0f);
        impl_->turns_.push_back(std::move(turn));
    }

    return true;
}

size_t ContextMemory::size() const {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    return impl_->turns_.size();
}

size_t ContextMemory::max_turns() const {
    return impl_->max_turns_;
}

} // namespace jarvis
