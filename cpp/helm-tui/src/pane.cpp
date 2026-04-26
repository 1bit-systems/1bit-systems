#include "onebit/helm_tui/pane.hpp"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <utility>

namespace onebit::helm_tui {

Node Node::leaf(std::string widget)
{
    return Node{LeafNode{std::move(widget)}};
}

Node Node::vsplit(float ratio, Node top, Node bottom)
{
    return Node{VerticalSplit{
        ratio,
        std::make_unique<Node>(std::move(top)),
        std::make_unique<Node>(std::move(bottom)),
    }};
}

Node Node::hsplit(float ratio, Node left, Node right)
{
    return Node{HorizontalSplit{
        ratio,
        std::make_unique<Node>(std::move(left)),
        std::make_unique<Node>(std::move(right)),
    }};
}

Node Node::default_layout()
{
    return vsplit(
        0.65F,
        hsplit(0.60F,
               leaf("status"),
               leaf("gpu")),
        leaf("logs"));
}

std::size_t Node::leaf_count() const noexcept
{
    if (const auto* l = std::get_if<LeafNode>(&v_)) {
        (void)l;
        return 1;
    }
    if (const auto* vs = std::get_if<VerticalSplit>(&v_)) {
        return vs->top->leaf_count() + vs->bottom->leaf_count();
    }
    if (const auto* hs = std::get_if<HorizontalSplit>(&v_)) {
        return hs->left->leaf_count() + hs->right->leaf_count();
    }
    return 0;
}

nlohmann::json node_to_json(const Node& n)
{
    using nlohmann::json;
    return std::visit(
        [](const auto& alt) -> json {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, LeafNode>) {
                return json{{"kind", "leaf"}, {"widget", alt.widget}};
            } else if constexpr (std::is_same_v<T, VerticalSplit>) {
                return json{{"kind", "vertical_split"},
                            {"ratio", alt.ratio},
                            {"top", node_to_json(*alt.top)},
                            {"bottom", node_to_json(*alt.bottom)}};
            } else {
                return json{{"kind", "horizontal_split"},
                            {"ratio", alt.ratio},
                            {"left", node_to_json(*alt.left)},
                            {"right", node_to_json(*alt.right)}};
            }
        },
        n.variant());
}

Node node_from_json(const nlohmann::json& j)
{
    if (!j.is_object() || !j.contains("kind") || !j["kind"].is_string()) {
        throw std::runtime_error("layout: missing 'kind' tag");
    }
    const std::string kind = j["kind"].get<std::string>();
    if (kind == "leaf") {
        return Node::leaf(j.value("widget", std::string{}));
    }
    if (kind == "vertical_split") {
        return Node::vsplit(j.value("ratio", 0.5F),
                            node_from_json(j.at("top")),
                            node_from_json(j.at("bottom")));
    }
    if (kind == "horizontal_split") {
        return Node::hsplit(j.value("ratio", 0.5F),
                            node_from_json(j.at("left")),
                            node_from_json(j.at("right")));
    }
    throw std::runtime_error("layout: unknown kind '" + kind + "'");
}

} // namespace onebit::helm_tui
