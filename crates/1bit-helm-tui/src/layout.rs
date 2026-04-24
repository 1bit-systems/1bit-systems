//! Layout persistence — load/save the pane tree to disk.

// v1 skeleton doesn't call these from main yet; the pane-tree walker
// lands in the next commit pass. Keep the module warning-clean under
// CI's -D warnings gate.
#![allow(dead_code)]

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::pane::Node;

/// Load a layout from `path`. Falls back to [`Node::default_layout`] on
/// any error (missing file, parse error, etc.) — the TUI always shows
/// something.
pub fn load_or_default(path: &Path) -> Node {
    try_load(path).unwrap_or_else(|_| Node::default_layout())
}

fn try_load(path: &Path) -> Result<Node> {
    let text = fs::read_to_string(path).with_context(|| format!("read {path:?}"))?;
    let node: Node = serde_json::from_str(&text).with_context(|| format!("parse {path:?}"))?;
    Ok(node)
}

/// Persist a layout to `path`, pretty-printed JSON. Creates parent dirs.
pub fn save(path: &Path, node: &Node) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("mkdir {parent:?}"))?;
    }
    let json = serde_json::to_string_pretty(node).context("serialize layout")?;
    fs::write(path, json).with_context(|| format!("write {path:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_or_default_on_missing_returns_builtin() {
        let n = load_or_default(Path::new("/nonexistent/1bit/tui-layout.json"));
        assert_eq!(n.leaf_count(), 3);
    }

    #[test]
    fn save_round_trip() {
        // Write to a deterministic path under /tmp — no tempfile dep.
        let path = std::env::temp_dir().join("1bit-helm-tui-layout-test.json");
        let node = Node::default_layout();
        save(&path, &node).unwrap();
        let loaded = load_or_default(&path);
        assert_eq!(loaded.leaf_count(), node.leaf_count());
        let _ = std::fs::remove_file(&path);
    }
}
