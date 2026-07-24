// rag.h — local markdown knowledge base + session memory.
// C++ port of jarvis/rag.py (deleted upstream by c252174aa, recovered from
// git history at c252174aa~1). Keyword-substring search over plain .md
// files, no embeddings, no database — matches the original exactly.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace jarvis {

struct SearchResult {
    std::string path;      // relative to KB root
    std::string title;     // first "# " line, or the filename
    int score;
    std::string snippet;
};

struct ConversationTurn {
    std::string role;
    std::string content;
};

// UTC timestamp, e.g. "2026-07-24T13:45:00Z" — matches Python's
// datetime.utcnow().isoformat()+"Z" used throughout the original.
std::string iso8601_now();

class KnowledgeBase {
public:
    // Root defaults to $JARVIS_KNOWLEDGE_DIR, else ~/jarvis/data/knowledge.
    KnowledgeBase();
    explicit KnowledgeBase(std::filesystem::path root);

    std::vector<std::filesystem::path> all_files() const;
    std::vector<SearchResult> search(const std::string& query, int max_results = 5) const;

    // Sanitizes filename, stores under documents/, wraps with frontmatter
    // if not already a .md file. Returns path relative to root.
    std::string add_document(const std::string& filename, const std::string& content);

    // Empty string if no results.
    std::string get_knowledge_context(const std::string& query, int max_results = 3) const;

    void save_turn(const std::string& session_id, const std::string& role, const std::string& content);
    std::vector<ConversationTurn> get_recent_conversation(const std::string& session_id, int max_turns = 10) const;
    std::vector<std::string> list_sessions() const;

    const std::filesystem::path& root() const { return root_; }

private:
    std::filesystem::path root_;
    std::filesystem::path resolve_safe(const std::vector<std::string>& components) const;
    std::filesystem::path session_path(const std::string& session_id) const;
};

} // namespace jarvis
