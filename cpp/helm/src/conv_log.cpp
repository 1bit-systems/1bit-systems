#include "onebit/helm/conv_log.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace onebit::helm {

namespace {

std::filesystem::path home_dir()
{
    if (const char* h = std::getenv("HOME"); h && *h) {
        return std::filesystem::path(h);
    }
    return std::filesystem::path{"."};
}

} // namespace

std::filesystem::path default_log_root()
{
    return home_dir() / ".halo" / "helm" / "conversations";
}

std::expected<std::filesystem::path, std::string>
write_session(const std::filesystem::path& root, const Conversation& conv)
{
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        return std::unexpected("mkdir " + root.string() + ": " + ec.message());
    }
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto sec = std::chrono::duration_cast<std::chrono::seconds>(now);
    auto path = root / (std::to_string(sec.count()) + ".jsonl");
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected("open " + path.string());
    for (const auto& t : conv.turns()) {
        nlohmann::json row = {
            {"role",    std::string(role_to_string(t.role))},
            {"content", t.content},
            {"ts",      t.ts},
        };
        f << row.dump() << "\n";
    }
    if (!f) return std::unexpected("write " + path.string());
    return path;
}

std::expected<std::vector<LogEntry>, std::string>
read_session(const std::filesystem::path& path)
{
    std::ifstream f(path);
    if (!f) return std::unexpected("open " + path.string());
    std::vector<LogEntry> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            LogEntry e;
            e.role    = j.value("role",    std::string{});
            e.content = j.value("content", std::string{});
            e.ts      = j.value("ts",      std::uint64_t{0});
            out.push_back(std::move(e));
        } catch (const nlohmann::json::parse_error&) {
            // Skip malformed lines — same lenient policy the Rust crate
            // ships ("never gate user's last session on strict parse").
        }
    }
    return out;
}

} // namespace onebit::helm
