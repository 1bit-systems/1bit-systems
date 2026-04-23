//! On-disk install layout.
//!
//! All paths hang off `~/.local/share/halo-pkg/`:
//!
//! ```text
//! ~/.local/share/halo-pkg/
//! ├── registry.toml           — one-file plugin index (registry.rs)
//! ├── state.toml              — enabled/disabled + install times
//! ├── cache/                  — tarball cache, gc'd on update
//! └── plugins/
//!     └── <name>/
//!         ├── current -> @<version>   — atomic pointer to active version
//!         └── @<version>/             — extracted plugin root
//!             ├── plugin.toml
//!             └── …
//! ```
//!
//! Every mutation is atomic: extract-to-tmp, `rename` into place, update
//! `current` symlink, write `state.toml` via write-to-tmp + rename.

use std::path::{Path, PathBuf};

/// Handle to the on-disk store root.
pub struct Store {
    /// Usually `~/.local/share/halo-pkg`.
    pub root: PathBuf,
}

impl Store {
    /// Open the default store under the user's XDG data dir.
    pub fn open_default() -> anyhow::Result<Self> {
        let root = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("no XDG data dir"))?
            .join("halo-pkg");
        Ok(Self { root })
    }

    /// Ensure the root tree exists (`plugins/`, `cache/`).
    pub fn ensure_layout(&self) -> anyhow::Result<()> {
        todo!("mkdir -p plugins/ cache/ logs/; create empty state.toml if absent")
    }

    /// `~/.local/share/halo-pkg/plugins/<name>/`
    pub fn plugin_dir(&self, name: &str) -> PathBuf {
        self.root.join("plugins").join(name)
    }

    /// `~/.local/share/halo-pkg/plugins/<name>/current`
    pub fn current_symlink(&self, name: &str) -> PathBuf {
        self.plugin_dir(name).join("current")
    }

    /// `~/.local/share/halo-pkg/plugins/<name>/@<version>/`
    pub fn versioned_dir(&self, name: &str, version: &str) -> PathBuf {
        self.plugin_dir(name).join(format!("@{version}"))
    }

    /// `~/.local/share/halo-pkg/state.toml`
    pub fn state_path(&self) -> PathBuf {
        self.root.join("state.toml")
    }

    /// `~/.local/share/halo-pkg/cache/`
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// Extract a verified tarball into a new versioned dir, atomically.
    pub fn install(&self, _name: &str, _version: &str, _tarball: &Path) -> anyhow::Result<PathBuf> {
        // extract to cache/tmp-XXXX/, rename into versioned_dir,
        // flip `current` symlink.
        todo!("atomic tarball extract + symlink flip")
    }

    /// Remove a plugin entirely. Caller must have already disabled it.
    pub fn remove(&self, _name: &str) -> anyhow::Result<()> {
        todo!("rm -rf plugin_dir(name), remove state entry")
    }

    /// Enumerate installed plugins (one entry per `plugins/<name>/`).
    pub fn installed(&self) -> anyhow::Result<Vec<InstalledPlugin>> {
        todo!("scandir plugins/, read each current/plugin.toml, cross-reference state.toml")
    }
}

/// One row of `halo-pkg list`.
#[derive(Debug, Clone)]
pub struct InstalledPlugin {
    pub name: String,
    pub version: String,
    pub kind: crate::manifest::Kind,
    pub enabled: bool,
}

/// State file living at `~/.local/share/halo-pkg/state.toml`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct State {
    #[serde(default)]
    pub plugins: std::collections::BTreeMap<String, PluginState>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PluginState {
    pub enabled: bool,
    /// RFC3339 timestamp of install. Used for collision-resolution order
    /// (see spec § 2.4).
    pub installed_at: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_store_root_is_under_data_dir() {
        let s = Store::open_default().expect("dirs");
        assert!(s.root.ends_with("halo-pkg"));
    }

    #[test]
    fn versioned_dir_uses_at_prefix() {
        let s = Store {
            root: PathBuf::from("/tmp/halo-pkg-test"),
        };
        let p = s.versioned_dir("claude-context", "0.3.1");
        assert!(p.ends_with("plugins/claude-context/@0.3.1"));
    }
}
