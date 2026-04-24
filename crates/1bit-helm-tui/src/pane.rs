//! Pane tree — recursive splits with per-node ratio.
//!
//! Skeleton only; the v1 layout rendered by `widgets::draw_root_panes`
//! hardcodes three panes (status / logs / repl). This type will replace
//! that once resize + rearrange bindings land.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Root of the pane tree. `serde` round-trips through the layout JSON
/// at `~/.config/1bit/tui-layout.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Node {
    /// Leaf — renders one named widget.
    Leaf {
        /// Widget key — must match one of [`crate::widgets::WIDGET_KEYS`].
        widget: String,
    },
    /// Vertical split — two children stacked top/bottom.
    VerticalSplit {
        /// 0.0–1.0 top child height ratio. Default 0.5.
        ratio: f32,
        /// Top child.
        top: Box<Node>,
        /// Bottom child.
        bottom: Box<Node>,
    },
    /// Horizontal split — two children side by side.
    HorizontalSplit {
        /// 0.0–1.0 left child width ratio. Default 0.5.
        ratio: f32,
        /// Left child.
        left: Box<Node>,
        /// Right child.
        right: Box<Node>,
    },
}

impl Node {
    /// Built-in default: status pane top-left, gpu top-right, logs full-bottom.
    pub fn default_layout() -> Self {
        Node::VerticalSplit {
            ratio: 0.65,
            top: Box::new(Node::HorizontalSplit {
                ratio: 0.60,
                left: Box::new(Node::Leaf { widget: "status".into() }),
                right: Box::new(Node::Leaf { widget: "gpu".into() }),
            }),
            bottom: Box::new(Node::Leaf { widget: "logs".into() }),
        }
    }

    /// Count leaf panes recursively. Used by the pane-cycle binding.
    pub fn leaf_count(&self) -> usize {
        match self {
            Node::Leaf { .. } => 1,
            Node::VerticalSplit { top, bottom, .. } => top.leaf_count() + bottom.leaf_count(),
            Node::HorizontalSplit { left, right, .. } => left.leaf_count() + right.leaf_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_has_three_leaves() {
        assert_eq!(Node::default_layout().leaf_count(), 3);
    }

    #[test]
    fn round_trip_default_layout_json() {
        let n = Node::default_layout();
        let j = serde_json::to_string(&n).unwrap();
        let n2: Node = serde_json::from_str(&j).unwrap();
        assert_eq!(n.leaf_count(), n2.leaf_count());
    }
}
