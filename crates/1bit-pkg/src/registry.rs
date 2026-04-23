//! Plugin registry — the thing halo-pkg asks "who publishes a plugin
//! called X, and where's the tarball?"
//!
//! In v0.1 the registry is a single TOML file at
//! `~/.local/share/halo-pkg/registry.toml` maintained by
//! [`halo-pkg update`](crate). The file is a flat list of entries; no
//! curation, no auth, no signatures yet (see spec § 4 for why).
//!
//! The [`Registry`] trait lets us swap in an HTTP-backed registry later
//! without changing the install / search paths. v0.1 ships only
//! [`LocalFileRegistry`].

use crate::manifest::Manifest;

/// A published plugin entry — what the registry returns.
///
/// Not the full manifest. The manifest lives inside the tarball at
/// `<root>/plugin.toml` and is only parsed after download + sha256
/// verification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub license: String,
    /// HTTP(S) URL of the plugin tarball.
    pub tarball: String,
    /// Expected sha256 of the tarball, 64 hex chars.
    pub sha256: String,
}

/// Registry abstraction. Local-file impl is [`LocalFileRegistry`]; an
/// HTTP-backed registry will slot in behind this trait in v0.2.
pub trait Registry {
    /// Refresh the local snapshot from upstream (no-op for the local-file
    /// impl).
    fn refresh(&mut self) -> anyhow::Result<()>;

    /// Look up a single plugin by name. Returns the latest version
    /// known to the registry.
    fn resolve(&self, name: &str) -> anyhow::Result<Option<RegistryEntry>>;

    /// Resolve a specific version of a plugin.
    fn resolve_version(&self, name: &str, version: &str) -> anyhow::Result<Option<RegistryEntry>>;

    /// Substring / fuzzy search across `name` + `description`.
    fn search(&self, query: &str) -> anyhow::Result<Vec<RegistryEntry>>;

    /// Fetch a tarball and verify its sha256. Returns the local path.
    fn fetch(&self, entry: &RegistryEntry) -> anyhow::Result<std::path::PathBuf>;

    /// Parse the manifest out of a fetched tarball (post-sha256 check).
    fn read_manifest(&self, tarball: &std::path::Path) -> anyhow::Result<Manifest>;
}

/// Local-file registry: reads `~/.local/share/halo-pkg/registry.toml`.
///
/// Format:
///
/// ```toml
/// [[plugin]]
/// name        = "claude-context"
/// version     = "0.3.1"
/// description = "Semantic code search over your project."
/// license     = "MIT"
/// tarball     = "https://example.com/claude-context-0.3.1.tar.gz"
/// sha256      = "…"
///
/// [[plugin]]
/// name = "…"
/// # …
/// ```
pub struct LocalFileRegistry {
    /// Path to `registry.toml`. Usually
    /// `~/.local/share/halo-pkg/registry.toml`.
    pub path: std::path::PathBuf,
}

impl LocalFileRegistry {
    pub fn open_default() -> anyhow::Result<Self> {
        let data = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("no XDG data dir"))?
            .join("halo-pkg")
            .join("registry.toml");
        Ok(Self { path: data })
    }
}

impl Registry for LocalFileRegistry {
    fn refresh(&mut self) -> anyhow::Result<()> {
        // Local-file registry has nothing to refresh from; an HTTP-backed
        // impl would pull a signed snapshot here.
        todo!("http-backed registry refresh — v0.2")
    }

    fn resolve(&self, _name: &str) -> anyhow::Result<Option<RegistryEntry>> {
        todo!("load self.path as toml, pick highest semver matching name")
    }

    fn resolve_version(
        &self,
        _name: &str,
        _version: &str,
    ) -> anyhow::Result<Option<RegistryEntry>> {
        todo!("exact-match lookup by (name, version)")
    }

    fn search(&self, _query: &str) -> anyhow::Result<Vec<RegistryEntry>> {
        todo!("case-insensitive substring over name + description")
    }

    fn fetch(&self, _entry: &RegistryEntry) -> anyhow::Result<std::path::PathBuf> {
        // reqwest blocking GET, sha2::Sha256 stream-hash to temp file,
        // rename into ~/.local/share/halo-pkg/cache/.
        todo!("tarball download + sha256 verification")
    }

    fn read_manifest(&self, _tarball: &std::path::Path) -> anyhow::Result<Manifest> {
        todo!("open tarball, locate plugin.toml, hand to Manifest::from_toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_path_is_under_data_dir() {
        let r = LocalFileRegistry::open_default().expect("dirs");
        assert!(r.path.ends_with("halo-pkg/registry.toml"));
    }
}
