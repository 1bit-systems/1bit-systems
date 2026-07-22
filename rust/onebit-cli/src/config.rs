//! Configuration management for 1bit CLI.
//!
//! Stores settings in `~/.1bit/settings.toml` — mirroring the old TypeScript
//! config structure but as a real data format instead of JSON.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Root data directory (~/.1bit/)
fn data_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("Cannot find home directory — set $HOME")?;
    Ok(PathBuf::from(home).join(".1bit"))
}

/// Agent settings directory
fn agent_dir() -> Result<PathBuf> {
    Ok(data_dir()?.join("agent"))
}

/// Path to the settings file
pub fn settings_path() -> Result<PathBuf> {
    Ok(agent_dir()?.join("settings.toml"))
}

/// All persistent settings for the 1bit agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Theme name
    pub theme: String,
    /// Default provider slug
    pub default_provider: String,
    /// Default model ID (e.g. "deepseek-v4-pro")
    pub default_model: String,
    /// NPU API endpoint
    pub npu_endpoint: String,
    /// Thinking level: off, low, medium, high, max
    pub thinking_level: String,
    /// Registered packages
    pub packages: Vec<String>,
    /// Fallback providers in order (e.g. ["openai", "anthropic"])
    #[serde(default)]
    pub fallback_providers: Vec<String>,
    /// NPU-specific settings
    pub npu: NpuSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NpuSettings {
    /// Path to bitnet_decode binary
    pub bitnet_decode_path: String,
    /// Path to the NPU daemon script
    pub daemon_path: String,
    /// Whether to tune prefill at startup
    pub tune_prefill: bool,
    /// Forced prefill variant (if set)
    pub prefill_variant: Option<u8>,
    /// Whether to pre-decode weights to FP16
    pub fp16_weights: bool,
    /// Port for the API server
    pub api_port: u16,
    /// Port for the Lemond TUI
    pub lemond_port: u16,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: "1bit".into(),
            default_provider: "npu".into(),
            default_model: "qwen3-0.6b-FLM".into(),
            npu_endpoint: "http://127.0.0.1:9090/v1".into(),
            thinking_level: "medium".into(),
            packages: vec![],
            fallback_providers: vec!["openai".into(), "deepseek".into()],
            npu: NpuSettings::default(),
        }
    }
}

impl Default for NpuSettings {
    fn default() -> Self {
        Self {
            bitnet_decode_path: "bitnet_decode".into(),
            daemon_path: String::new(),
            tune_prefill: false,
            prefill_variant: None,
            fp16_weights: false,
            api_port: 9090,
            lemond_port: 13305,
        }
    }
}

impl Settings {
    /// Load settings from disk, or return defaults if the file doesn't exist.
    pub fn load() -> Result<Self> {
        let path = settings_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read settings from {}", path.display()))?;
        toml::from_str(&raw)
            .with_context(|| format!("Failed to parse settings from {}", path.display()))
    }

    /// Save settings to disk.
    pub fn save(&self) -> Result<()> {
        let path = settings_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create ~/.1bit/agent directory")?;
        }
        let raw = toml::to_string_pretty(self)
            .context("Failed to serialize settings")?;
        std::fs::write(&path, &raw)
            .with_context(|| format!("Failed to write settings to {}", path.display()))
    }

    /// Get a single value by dotted key path (e.g. "npu.api_port").
    pub fn get(&self, key: &str) -> Result<String> {
        match key {
            "theme" => Ok(self.theme.clone()),
            "default_provider" | "default-provider" => Ok(self.default_provider.clone()),
            "default_model" | "default-model" => Ok(self.default_model.clone()),
            "npu_endpoint" | "npu-endpoint" => Ok(self.npu_endpoint.clone()),
            "thinking_level" | "thinking-level" => Ok(self.thinking_level.clone()),
            "npu.bitnet_decode_path" | "npu.bitnet-decode-path" => Ok(self.npu.bitnet_decode_path.clone()),
            "npu.daemon_path" | "npu.daemon-path" => Ok(self.npu.daemon_path.clone()),
            "npu.tune_prefill" | "npu.tune-prefill" => Ok(self.npu.tune_prefill.to_string()),
            "npu.prefill_variant" | "npu.prefill-variant" => Ok(format!("{:?}", self.npu.prefill_variant)),
            "npu.fp16_weights" | "npu.fp16-weights" => Ok(self.npu.fp16_weights.to_string()),
            "npu.api_port" | "npu.api-port" => Ok(self.npu.api_port.to_string()),
            "npu.lemond_port" | "npu.lemond-port" => Ok(self.npu.lemond_port.to_string()),
            "fallback_providers" | "fallback-providers" => {
                Ok(self.fallback_providers.join(", "))
            }
            _ => anyhow::bail!("Unknown config key: {key}"),
        }
    }

    /// Set a single value by dotted key path.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "theme" => self.theme = value.into(),
            "default_provider" | "default-provider" => self.default_provider = value.into(),
            "default_model" | "default-model" => self.default_model = value.into(),
            "npu_endpoint" | "npu-endpoint" => self.npu_endpoint = value.into(),
            "thinking_level" | "thinking-level" => self.thinking_level = value.into(),
            "npu.bitnet_decode_path" | "npu.bitnet-decode-path" => {
                self.npu.bitnet_decode_path = value.into();
            }
            "npu.daemon_path" | "npu.daemon-path" => {
                self.npu.daemon_path = value.into();
            }
            "npu.tune_prefill" | "npu.tune-prefill" => {
                self.npu.tune_prefill = value.parse()?;
            }
            "npu.prefill_variant" | "npu.prefill-variant" => {
                self.npu.prefill_variant = Some(value.parse()?);
            }
            "npu.fp16_weights" | "npu.fp16-weights" => {
                self.npu.fp16_weights = value.parse()?;
            }
            "npu.api_port" | "npu.api-port" => {
                self.npu.api_port = value.parse()?;
            }
            "npu.lemond_port" | "npu.lemond-port" => {
                self.npu.lemond_port = value.parse()?;
            }
            "fallback_providers" | "fallback-providers" => {
                self.fallback_providers = value
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            _ => anyhow::bail!("Unknown config key: {key}"),
        }
        self.save()
    }
}
