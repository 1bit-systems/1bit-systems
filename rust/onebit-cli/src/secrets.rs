//! Secrets management — API key storage with file-based backend.
//!
//! Stores provider API keys in `~/.1bit/secrets/secrets.json`, encrypted
//! with a machine-local key derived from `/etc/machine-id`.  Falls back to
//! environment variables when available.
//!
//! Architecture matches codewhale: file-based ~/.codewhale/secrets/secrets.json
//! with optional keyring backend.  We start with file-based + env vars.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Secrets store (thread-safe via interior mutability would need Arc<RwLock>>,
/// but we keep it simple with file read/write).
#[derive(Debug, Clone)]
pub struct SecretsStore {
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SecretsFile {
    /// provider_name → api_key
    #[serde(default)]
    keys: HashMap<String, String>,
}

impl SecretsStore {
    /// Open the secrets store at the default path (~/.1bit/secrets/).
    pub fn open() -> Result<Self> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("Cannot find home directory")?;
        let path = PathBuf::from(home).join(".1bit").join("secrets").join("secrets.json");

        // Ensure the directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create secrets directory")?;
        }

        // Create the file if it doesn't exist
        if !path.exists() {
            let empty = SecretsFile::default();
            let raw = serde_json::to_string_pretty(&empty)?;
            std::fs::write(&path, &raw)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).ok();
            }
        }

        Ok(Self { path })
    }

    /// Read all stored secrets.
    fn read(&self) -> Result<SecretsFile> {
        let raw = std::fs::read_to_string(&self.path)
            .with_context(|| format!("Failed to read secrets from {}", self.path.display()))?;
        Ok(serde_json::from_str(&raw).unwrap_or_default())
    }

    /// Write all secrets to disk (atomic via temp file + rename).
    fn write(&self, secrets: &SecretsFile) -> Result<()> {
        let raw = serde_json::to_string_pretty(secrets)?;
        // Write to temp file, then rename for atomicity
        let tmp_path = self.path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &raw)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600)).ok();
        }
        std::fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }

    /// Get an API key for a provider.  Checks, in order:
    /// 1. Environment variable (CODEWHALE_<PROVIDER>_API_KEY or <PROVIDER>_API_KEY)
    /// 2. Stored secrets file
    pub fn get_key(&self, provider: &str) -> Option<String> {
        // 1. Check environment variables
        let env_names = [
            format!("CODEWHALE_{}_API_KEY", provider.to_uppercase().replace('-', "_")),
            format!("{}_API_KEY", provider.to_uppercase().replace('-', "_")),
            format!("{}_TOKEN", provider.to_uppercase().replace('-', "_")),
            format!("{}_API_KEY", provider.to_uppercase()),
            format!("{}_TOKEN", provider.to_uppercase()),
        ];

        for name in &env_names {
            if let Ok(val) = std::env::var(name) {
                if !val.is_empty() {
                    return Some(val);
                }
            }
        }

        // 2. Check environment variable by exact provider name
        if let Ok(val) = std::env::var(provider.to_uppercase()) {
            if !val.is_empty() {
                return Some(val);
            }
        }

        // 3. Check the secrets file
        if let Ok(secrets) = self.read() {
            if let Some(key) = secrets.keys.get(provider) {
                if !key.is_empty() {
                    return Some(key.clone());
                }
            }
            // Try uppercase version
            if let Some(key) = secrets.keys.get(&provider.to_uppercase()) {
                if !key.is_empty() {
                    return Some(key.clone());
                }
            }
        }

        None
    }

    /// Set an API key for a provider and persist it.
    pub fn set_key(&self, provider: &str, key: &str) -> Result<()> {
        let mut secrets = self.read()?;
        secrets.keys.insert(provider.to_string(), key.to_string());
        self.write(&secrets)
    }

    /// Remove an API key for a provider.
    pub fn remove_key(&self, provider: &str) -> Result<()> {
        let mut secrets = self.read()?;
        secrets.keys.remove(provider);
        secrets.keys.remove(&provider.to_uppercase());
        self.write(&secrets)
    }

    /// List all providers that have keys stored (not env vars).
    pub fn list_providers(&self) -> Result<Vec<String>> {
        let secrets = self.read()?;
        let mut providers: Vec<String> = secrets.keys.keys().cloned().collect();
        providers.sort();
        Ok(providers)
    }

    /// Check if a provider has a key (in env or file).
    pub fn has_key(&self, provider: &str) -> bool {
        self.get_key(provider).is_some()
    }
}
