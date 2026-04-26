#include "onebit/helm_tui/layout.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <system_error>

namespace onebit::helm_tui {

namespace {

std::expected<Node, std::string>
try_load(const std::filesystem::path& path)
{
    std::ifstream f(path);
    if (!f) {
        std::ostringstream oss;
        oss << "open " << path;
        return std::unexpected(oss.str());
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(buf.str());
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(std::string{"parse: "} + e.what());
    }
    try {
        return node_from_json(j);
    } catch (const std::exception& e) {
        return std::unexpected(std::string{"shape: "} + e.what());
    }
}

} // namespace

Node load_or_default(const std::filesystem::path& path)
{
    auto r = try_load(path);
    if (r) return std::move(*r);
    return Node::default_layout();
}

std::expected<void, std::string>
save(const std::filesystem::path& path, const Node& node)
{
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            return std::unexpected("mkdir " + path.parent_path().string()
                                   + ": " + ec.message());
        }
    }
    const auto j   = node_to_json(node);
    const auto txt = j.dump(2);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return std::unexpected("open " + path.string());
    }
    f << txt;
    if (!f) {
        return std::unexpected("write " + path.string());
    }
    return {};
}

} // namespace onebit::helm_tui
