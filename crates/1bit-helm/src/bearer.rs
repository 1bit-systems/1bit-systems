//! Bearer-token storage for lemond's `/v1/*` endpoints.
//!
//! Strategy (spec: Crate-halo-helm.md invariant 4):
//!   1. Try the system keyring via the `keyring` crate (secret-service on
//!      Linux, keychain on macOS). No plaintext on disk.
//!   2. If the keyring is not reachable (headless box, no running
//!      secret-service, sandbox), fall back to `~/.config/1bit-helm/bearer.txt`
//!      with 0600 perms.
//!
//! Both backends expose the same [`Bearer`] surface so the UI doesn't
//! branch. Which backend was actually used is available via
//! [`Bearer::backend`] for the Settings pane to surface.
//!
//! Keyring 3.x pins itself to a single backend per `cfg(target_os)`. On
//! Linux we use `dbus-secret-service` (sync, pure Rust). If that crate
//! can't reach a secret-service daemon the first `set_password` call
//! fails with `PlatformFailure` — we swallow it and persist via the XDG
//! file instead.

use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Which on-disk surface ended up holding the bearer for this session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BearerBackend {
    /// System keyring (Linux secret-service, macOS keychain, etc.).
    Keyring,
    /// XDG-config file fallback. Readable only by the current user.
    XdgFile,
    /// No bearer stored yet.
    None,
}

impl BearerBackend {
    pub fn label(self) -> &'static str {
        match self {
            BearerBackend::Keyring => "system keyring",
            BearerBackend::XdgFile => "~/.config/1bit-helm/bearer.txt (0600)",
            BearerBackend::None => "unset",
        }
    }
}

/// Keyring "service" name. One-off product-scoped namespace so we don't
/// collide with other apps using `default` or `keyring`.
const KEYRING_SERVICE: &str = "1bit-helm";
/// Keyring "user" name. Kept constant — this is a single-user desktop
/// client, we don't juggle multiple identities.
const KEYRING_USER: &str = "api-bearer";

/// File fallback path. Relative to the XDG_CONFIG_HOME root (or `~/.config`).
const FALLBACK_DIR: &str = "1bit-helm";
const FALLBACK_FILE: &str = "bearer.txt";

/// Bearer-token handle. Hold one per `HelmApp`.
#[derive(Debug, Clone)]
pub struct Bearer {
    /// Root of the XDG-config fallback. Tests override to a tempdir so
    /// we don't write into the user's actual `~/.config`.
    fallback_root: PathBuf,
    /// `true` if the keyring should be tried at all. Disabled in tests.
    try_keyring: bool,
    /// Last backend used — set after `load` / `store`.
    backend: BearerBackend,
    /// In-memory copy so the UI isn't hitting DBus every frame.
    cached: Option<String>,
}

impl Bearer {
    /// Production bearer. Uses the real XDG config dir + tries the keyring.
    pub fn new() -> Self {
        Self {
            fallback_root: default_fallback_root(),
            try_keyring: true,
            backend: BearerBackend::None,
            cached: None,
        }
    }

    /// Test-visible constructor: force the fallback root + skip keyring.
    pub fn with_file_only(fallback_root: impl Into<PathBuf>) -> Self {
        Self {
            fallback_root: fallback_root.into(),
            try_keyring: false,
            backend: BearerBackend::None,
            cached: None,
        }
    }

    pub fn backend(&self) -> BearerBackend {
        self.backend
    }

    pub fn get(&self) -> Option<&str> {
        self.cached.as_deref()
    }

    /// Best-effort load. Prefer keyring → fall back to file → give up.
    /// Never errors; a missing bearer is a first-run condition, not a bug.
    pub fn load(&mut self) {
        self.cached = None;
        self.backend = BearerBackend::None;

        if self.try_keyring
            && let Some(v) = load_keyring()
        {
            self.cached = Some(v);
            self.backend = BearerBackend::Keyring;
            return;
        }
        if let Some(v) = load_file(&self.fallback_root) {
            self.cached = Some(v);
            self.backend = BearerBackend::XdgFile;
        }
    }

    /// Write `token`. Keyring first; on failure persist to the XDG file
    /// with 0600. Returns which backend we ended up on.
    pub fn store(&mut self, token: &str) -> Result<BearerBackend> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            anyhow::bail!("bearer token is empty");
        }

        if self.try_keyring && store_keyring(trimmed).is_ok() {
            self.cached = Some(trimmed.to_string());
            self.backend = BearerBackend::Keyring;
            // Best-effort: nuke any stale file fallback so we don't have
            // two stores drifting. Ignore errors — the file may not exist.
            let _ = delete_file(&self.fallback_root);
            return Ok(BearerBackend::Keyring);
        }

        store_file(&self.fallback_root, trimmed)?;
        self.cached = Some(trimmed.to_string());
        self.backend = BearerBackend::XdgFile;
        Ok(BearerBackend::XdgFile)
    }

    /// Forget the bearer everywhere. Clears both backends; safe to call
    /// when no bearer is set.
    pub fn clear(&mut self) -> Result<()> {
        self.cached = None;
        self.backend = BearerBackend::None;
        if self.try_keyring {
            let _ = clear_keyring();
        }
        delete_file(&self.fallback_root)
    }
}

impl Default for Bearer {
    fn default() -> Self {
        Self::new()
    }
}

fn default_fallback_root() -> PathBuf {
    if let Some(base) = dirs::config_dir() {
        base.join(FALLBACK_DIR)
    } else {
        // Last-ditch: `$HOME/.1bit-helm`. Desktop machines always have
        // HOME set; this branch exists so we never panic.
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".1bit-helm")
    }
}

fn fallback_path(root: &Path) -> PathBuf {
    root.join(FALLBACK_FILE)
}

fn load_keyring() -> Option<String> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER).ok()?;
    entry.get_password().ok()
}

fn store_keyring(token: &str) -> Result<()> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER).context("keyring entry")?;
    entry.set_password(token).context("keyring set_password")?;
    Ok(())
}

fn clear_keyring() -> Result<()> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER).context("keyring entry")?;
    // `delete_credential` is the 3.x API; older 2.x was `delete_password`.
    entry
        .delete_credential()
        .context("keyring delete_credential")
}

fn load_file(root: &Path) -> Option<String> {
    let p = fallback_path(root);
    let raw = fs::read_to_string(p).ok()?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn store_file(root: &Path, token: &str) -> Result<()> {
    fs::create_dir_all(root).with_context(|| format!("mkdir {}", root.display()))?;
    let p = fallback_path(root);
    // Create-or-truncate with 0600 perms. `File::create` doesn't honour
    // mode directly on Linux — we chmod after write.
    {
        let mut f = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&p)
            .with_context(|| format!("open {}", p.display()))?;
        f.write_all(token.as_bytes())
            .with_context(|| format!("write {}", p.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        fs::set_permissions(&p, perms).with_context(|| format!("chmod 600 {}", p.display()))?;
    }
    Ok(())
}

fn delete_file(root: &Path) -> Result<()> {
    let p = fallback_path(root);
    if p.exists() {
        fs::remove_file(&p).with_context(|| format!("rm {}", p.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn file_fallback_roundtrip_creates_0600_and_reloads() {
        // Tests pin try_keyring = false so we exercise the file path
        // deterministically — CI runners rarely have secret-service up.
        let td = TempDir::new().unwrap();
        let mut b = Bearer::with_file_only(td.path().to_path_buf());
        assert_eq!(b.backend(), BearerBackend::None);
        assert!(b.get().is_none());

        let backend = b.store("sk-test-123").unwrap();
        assert_eq!(backend, BearerBackend::XdgFile);
        assert_eq!(b.backend(), BearerBackend::XdgFile);
        assert_eq!(b.get(), Some("sk-test-123"));

        // On Unix the file must be 0600 — Rule A-adjacent: secrets on
        // disk should at minimum be user-private.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = std::fs::metadata(fallback_path(td.path())).unwrap();
            let mode = meta.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "bearer.txt must be 0600, got {mode:o}");
        }

        // A fresh Bearer pointed at the same root picks the token back up.
        let mut b2 = Bearer::with_file_only(td.path().to_path_buf());
        b2.load();
        assert_eq!(b2.backend(), BearerBackend::XdgFile);
        assert_eq!(b2.get(), Some("sk-test-123"));
    }

    #[test]
    fn clear_removes_on_disk_copy() {
        let td = TempDir::new().unwrap();
        let mut b = Bearer::with_file_only(td.path().to_path_buf());
        b.store("sk-test-456").unwrap();
        assert!(fallback_path(td.path()).exists());

        b.clear().unwrap();
        assert_eq!(b.backend(), BearerBackend::None);
        assert!(b.get().is_none());
        assert!(!fallback_path(td.path()).exists());
    }

    #[test]
    fn empty_token_is_rejected() {
        let td = TempDir::new().unwrap();
        let mut b = Bearer::with_file_only(td.path().to_path_buf());
        assert!(b.store("   ").is_err());
        assert!(b.store("").is_err());
        assert_eq!(b.backend(), BearerBackend::None);
    }

    #[test]
    fn load_on_fresh_root_reports_none() {
        let td = TempDir::new().unwrap();
        let mut b = Bearer::with_file_only(td.path().to_path_buf());
        b.load();
        assert_eq!(b.backend(), BearerBackend::None);
        assert!(b.get().is_none());
    }

    #[test]
    fn backend_labels_are_stable_strings() {
        // The Settings pane shows these verbatim; lock them here so a
        // rename surfaces as a test-diff rather than a silent UI change.
        assert_eq!(BearerBackend::Keyring.label(), "system keyring");
        assert_eq!(
            BearerBackend::XdgFile.label(),
            "~/.config/1bit-helm/bearer.txt (0600)"
        );
        assert_eq!(BearerBackend::None.label(), "unset");
    }
}
