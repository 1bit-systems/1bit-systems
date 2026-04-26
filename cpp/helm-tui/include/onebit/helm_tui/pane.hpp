// 1bit-helm-tui — pane tree, recursive splits with per-node ratio.
//
// Mirrors crates/1bit-helm-tui/src/pane.rs. The runtime renderer in
// `widgets.cpp` walks this tree; persistence lives in `layout.cpp`.

#pragma once

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <variant>

namespace onebit::helm_tui {

class Node;

// Leaf pane — renders one widget by string key. Keys are validated
// against `WIDGET_KEYS` in widgets.hpp.
struct LeafNode {
    std::string widget;
};

struct VerticalSplit {
    float ratio{0.5F};
    std::unique_ptr<Node> top;
    std::unique_ptr<Node> bottom;
};

struct HorizontalSplit {
    float ratio{0.5F};
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

// Recursive pane tree. The `kind` discriminator on the wire is one of
// {leaf, vertical_split, horizontal_split}; matches Rust serde
// `#[serde(tag="kind", rename_all="snake_case")]`.
class Node {
public:
    using Variant = std::variant<LeafNode, VerticalSplit, HorizontalSplit>;

    explicit Node(Variant v) : v_(std::move(v)) {}

    [[nodiscard]] const Variant& variant() const noexcept { return v_; }

    // Built-in default: 65/35 vertical split, top row is 60/40 status |
    // gpu, bottom row is the logs pane. Same as the Rust default.
    [[nodiscard]] static Node default_layout();

    // Count leaf panes recursively.
    [[nodiscard]] std::size_t leaf_count() const noexcept;

    // Convenience constructors.
    [[nodiscard]] static Node leaf(std::string widget);
    [[nodiscard]] static Node vsplit(float ratio, Node top, Node bottom);
    [[nodiscard]] static Node hsplit(float ratio, Node left, Node right);

private:
    Variant v_;
};

// Serde-compatible JSON round-trip.
[[nodiscard]] nlohmann::json node_to_json(const Node& n);
[[nodiscard]] Node           node_from_json(const nlohmann::json& j);

} // namespace onebit::helm_tui
