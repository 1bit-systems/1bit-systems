// context.h — Conversation context memory with summarization.
// Maintains a turn-based conversation history for LLM prompt building.
#pragma once

#include <string>
#include <vector>
#include <memory>

namespace jarvis {

struct Turn {
    std::string role;        // "user" or "assistant"
    std::string text;
    double timestamp;        // unix seconds
    float duration_ms;       // how long the utterance took (0 if unknown)
};

class ContextMemory {
public:
    explicit ContextMemory(size_t max_turns = 50);
    ~ContextMemory();

    // Add a turn to the conversation
    void add_turn(const std::string& role, const std::string& text, float duration_ms = 0);

    // Get all turns
    std::vector<Turn> get_turns() const;

    // Get recent N turns
    std::vector<Turn> get_recent(int n) const;

    // Build context string for LLM prompt
    std::string build_context(int max_recent_turns = 10) const;

    // Summarize old turns when exceeding max_turns.
    // Simple truncation-based summarizer that extracts key phrases.
    std::string summarize(const std::string& text);

    // Clear conversation
    void clear();

    // Save/load conversation to/from JSON file
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // Get current turn count
    size_t size() const;

    // Get max turns setting
    size_t max_turns() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
