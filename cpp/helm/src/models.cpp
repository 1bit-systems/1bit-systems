#include "onebit/helm/models.hpp"

#include <nlohmann/json.hpp>

namespace onebit::helm {

std::expected<std::vector<ModelCard>, std::string>
parse_models(std::string_view body)
{
    if (body.empty()) {
        return std::unexpected("empty body");
    }
    nlohmann::json v;
    try {
        v = nlohmann::json::parse(body);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(std::string{"json parse: "} + e.what());
    }
    if (!v.is_object()) {
        return std::unexpected("top-level not an object");
    }
    std::vector<ModelCard> out;
    if (!v.contains("data") || !v["data"].is_array()) {
        return out;
    }
    for (const auto& row : v["data"]) {
        if (!row.is_object() || !row.contains("id")) continue;
        ModelCard c;
        c.id       = row["id"].get<std::string>();
        c.owned_by = row.value("owned_by", std::string{});
        c.created  = row.value("created", std::uint64_t{0});
        out.push_back(std::move(c));
    }
    return out;
}

} // namespace onebit::helm
