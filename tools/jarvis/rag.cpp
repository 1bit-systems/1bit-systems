#include "rag.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

namespace jarvis {

std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

static std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
    return out;
}

static int count_occurrences(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return 0;
    int count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        count++;
        pos += needle.size();
    }
    return count;
}

static std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

static fs::path default_root() {
    if (const char* env = getenv("JARVIS_KNOWLEDGE_DIR")) return fs::path(env);
    const char* home = getenv("HOME");
    return fs::path(home ? home : "/tmp") / "jarvis" / "data" / "knowledge";
}

KnowledgeBase::KnowledgeBase() : KnowledgeBase(default_root()) {}

KnowledgeBase::KnowledgeBase(fs::path root) : root_(std::move(root)) {
    std::error_code ec;
    for (const char* sub : {"facts", "documents", "conversations", "tools"}) {
        fs::create_directories(root_ / sub, ec);
    }
}

fs::path KnowledgeBase::resolve_safe(const std::vector<std::string>& components) const {
    fs::path p = root_;
    for (auto& c : components) p /= c;
    fs::path resolved = fs::weakly_canonical(p);
    fs::path root_resolved = fs::weakly_canonical(root_);
    auto rs = root_resolved.string();
    auto ps = resolved.string();
    if (ps.compare(0, rs.size(), rs) != 0) {
        throw std::runtime_error("path escapes knowledge base root: " + p.string());
    }
    return resolved;
}

static std::string sanitize_component(const std::string& s, size_t max_len, const std::string& allowed_extra) {
    std::string out;
    for (char c : s) {
        if (std::isalnum((unsigned char)c) || allowed_extra.find(c) != std::string::npos) out += c;
        else out += '_';
    }
    if (out.size() > max_len) out.resize(max_len);
    if (out.empty()) out = "default";
    return out;
}

fs::path KnowledgeBase::session_path(const std::string& session_id) const {
    std::string safe = sanitize_component(session_id, 128, "_-");
    return root_ / "conversations" / (safe + ".md");
}

std::vector<fs::path> KnowledgeBase::all_files() const {
    std::vector<fs::path> out;
    std::error_code ec;
    if (!fs::exists(root_, ec)) return out;
    for (auto it = fs::recursive_directory_iterator(root_, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (it->is_regular_file(ec) && it->path().extension() == ".md") out.push_back(it->path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<SearchResult> KnowledgeBase::search(const std::string& query, int max_results) const {
    std::vector<std::string> terms = split_ws(to_lower(query));
    std::vector<SearchResult> results;

    for (const auto& path : all_files()) {
        std::ifstream f(path, std::ios::binary);
        if (!f) continue;
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        std::string text_lower = to_lower(text);

        int score = 0;
        for (const auto& term : terms) score += count_occurrences(text_lower, term);
        if (score == 0) continue;

        // Title: first line starting with "# "
        std::string title;
        {
            std::istringstream lines(text);
            std::string line;
            while (std::getline(lines, line)) {
                if (line.rfind("# ", 0) == 0) { title = line.substr(2); break; }
            }
            if (title.empty()) title = path.filename().string();
        }

        // Snippet: window around the first occurring term.
        std::string snippet;
        size_t hit_pos = std::string::npos;
        for (const auto& term : terms) {
            size_t p = text_lower.find(term);
            if (p != std::string::npos && (hit_pos == std::string::npos || p < hit_pos)) hit_pos = p;
        }
        if (hit_pos != std::string::npos) {
            size_t start = (hit_pos > 60) ? hit_pos - 60 : 0;
            size_t end = std::min(text.size(), hit_pos + 120);
            snippet = text.substr(start, end - start);
            for (auto& c : snippet) if (c == '\n' || c == '\r') c = ' ';
            if (snippet.size() > 200) snippet.resize(200);
        }

        std::error_code ec;
        std::string rel = fs::relative(path, root_, ec).string();
        results.push_back({rel, title, score, snippet});
    }

    std::stable_sort(results.begin(), results.end(),
                      [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    if ((int)results.size() > max_results) results.resize(max_results);
    return results;
}

std::string KnowledgeBase::add_document(const std::string& filename, const std::string& content) {
    std::string safe = sanitize_component(filename, 4096, "_.-");
    bool is_md = safe.size() >= 3 && safe.compare(safe.size() - 3, 3, ".md") == 0;

    fs::path target = resolve_safe({"documents", is_md ? safe : safe + ".md"});
    std::ofstream f(target, std::ios::binary | std::ios::trunc);
    if (!is_md) {
        f << "---\n"
          << "type: document\n"
          << "created: " << iso8601_now() << "\n"
          << "source: upload\n"
          << "---\n\n"
          << "# " << filename << "\n\n";
    }
    f << content;
    std::error_code ec;
    return fs::relative(target, root_, ec).string();
}

std::string KnowledgeBase::get_knowledge_context(const std::string& query, int max_results) const {
    auto results = search(query, max_results);
    if (results.empty()) return "";
    std::ostringstream out;
    out << "Here is relevant information from the knowledge base:\n\n";
    for (const auto& r : results) {
        out << "--- " << r.title << " ---\n" << r.snippet << "\n\n";
    }
    std::string s = out.str();
    while (!s.empty() && (s.back() == '\n')) s.pop_back();
    return s;
}

void KnowledgeBase::save_turn(const std::string& session_id, const std::string& role, const std::string& content) {
    fs::path path = session_path(session_id);
    bool is_new = !fs::exists(path);
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (is_new) {
        std::string safe = sanitize_component(session_id, 128, "_-");
        f << "---\n"
          << "type: conversation\n"
          << "session: " << safe << "\n"
          << "created: " << iso8601_now() << "\n"
          << "---\n\n"
          << "# Session " << safe << "\n\n";
    }
    f << "## " << iso8601_now() << " \xE2\x80\x94 " << role << "\n\n" << content << "\n";
}

std::vector<ConversationTurn> KnowledgeBase::get_recent_conversation(const std::string& session_id, int max_turns) const {
    fs::path path = session_path(session_id);
    std::vector<ConversationTurn> turns;
    std::ifstream f(path, std::ios::binary);
    if (!f) return turns;
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    // Split on "\n## "
    static const std::string delim = "\n## ";
    std::vector<std::string> blocks;
    size_t pos = 0;
    while (true) {
        size_t next = text.find(delim, pos);
        if (next == std::string::npos) {
            blocks.push_back(text.substr(pos));
            break;
        }
        blocks.push_back(text.substr(pos, next - pos));
        pos = next + delim.size();
    }

    // Header line pattern: "<timestamp> — <role>" (role = \w+)
    static const std::regex role_re(R"(.*?\xE2\x80\x94\s*(\w+))");
    for (const auto& block : blocks) {
        if (block.empty()) continue;
        size_t nl = block.find('\n');
        std::string header = (nl == std::string::npos) ? block : block.substr(0, nl);
        std::smatch m;
        if (!std::regex_search(header, m, role_re)) continue;
        std::string role = m[1].str();
        std::string content = (nl == std::string::npos) ? "" : block.substr(nl + 1);
        while (!content.empty() && content.front() == '\n') content.erase(content.begin());
        while (!content.empty() && (content.back() == '\n' || content.back() == ' ')) content.pop_back();
        turns.push_back({role, content});
    }

    if ((int)turns.size() > max_turns) {
        turns.erase(turns.begin(), turns.begin() + (turns.size() - max_turns));
    }
    return turns;
}

std::vector<std::string> KnowledgeBase::list_sessions() const {
    std::vector<std::string> out;
    std::error_code ec;
    fs::path dir = root_ / "conversations";
    if (!fs::exists(dir, ec)) return out;
    for (auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".md")
            out.push_back(entry.path().stem().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace jarvis
