//! Provider routing — multi-LLM provider support with fallback chains.
//!
//! Routes chat requests to the appropriate provider based on model name or
//! explicit provider selection.  Supports OpenAI-compatible APIs, Anthropic,
//! and custom endpoints — with fallback chains when a provider is unavailable.
//!
//! Architecture matches codewhale: each provider has a base URL, auth mode,
//! and set of models.  Models are resolved by prefix matching (e.g.
//! `deepseek/deepseek-v4-pro` → DeepSeek provider).

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use crate::secrets::SecretsStore;

// ── Provider definitions ────────────────────────────────────────

/// Built-in provider registry.
fn builtin_providers() -> HashMap<&'static str, ProviderDef> {
    let mut m = HashMap::new();

    macro_rules! def {
        ($name:expr, $url:expr, $env:expr) => {
            m.insert($name, ProviderDef {
                id: $name.to_string(),
                name: $name.to_string(),
                base_url: $url.to_string(),
                env_var: Some($env.to_string()),
                auth_mode: "api_key".into(),
            });
        };
    }

    def!("deepseek",       "https://api.deepseek.com/beta",              "DEEPSEEK_API_KEY");
    def!("deepseek-china", "https://api.deepseek.com",                   "DEEPSEEK_API_KEY");
    def!("openai",         "https://api.openai.com/v1",                  "OPENAI_API_KEY");
    def!("anthropic",      "https://api.anthropic.com",                  "ANTHROPIC_API_KEY");
    def!("openrouter",     "https://openrouter.ai/api/v1",               "OPENROUTER_API_KEY");
    def!("nvidia-nim",     "https://integrate.api.nvidia.com/v1",        "NVIDIA_API_KEY");
    def!("together",       "https://api.together.xyz/v1",                "TOGETHER_API_KEY");
    def!("fireworks",      "https://api.fireworks.ai/inference/v1",      "FIREWORKS_API_KEY");
    def!("deepinfra",      "https://api.deepinfra.com/v1/openai",        "DEEPINFRA_API_KEY");
    def!("huggingface",    "https://router.huggingface.co/v1",           "HF_API_KEY");
    def!("sambanova",      "https://api.sambanova.ai/v1",               "SAMBANOVA_API_KEY");
    def!("groq",           "https://api.groq.com/openai/v1",            "GROQ_API_KEY");
    def!("xai",            "https://api.x.ai/v1",                       "XAI_API_KEY");
    def!("mistral",        "https://api.mistral.ai/v1",                 "MISTRAL_API_KEY");
    def!("cohere",         "https://api.cohere.ai/v1",                  "COHERE_API_KEY");
    def!("google",         "https://generativelanguage.googleapis.com/v1", "GOOGLE_API_KEY");
    def!("zai",            "https://api.z.ai/api/coding/paas/v4",       "ZAI_API_KEY");
    def!("siliconflow",    "https://api.siliconflow.com/v1",            "SILICONFLOW_API_KEY");
    def!("ollama",         "http://localhost:11434/v1",                  "OLLAMA_API_KEY");
    def!("stepfun",        "https://api.stepfun.ai/v1",                 "STEPFUN_API_KEY");
    def!("minimax",        "https://api.minimax.io/v1",                 "MINIMAX_API_KEY");
    def!("xiaomi-mimo",    "https://token-plan-sgp.xiaomimimo.com/v1",  "XIAOMI_MIMO_API_KEY");
    def!("moonshot",       "https://api.moonshot.ai/v1",                "MOONSHOT_API_KEY");
    def!("volcengine",     "https://ark.cn-beijing.volces.com/api/coding/v3", "VOLCENGINE_API_KEY");
    def!("atlascloud",     "https://api.atlascloud.ai/v1",              "ATLASCLOUD_API_KEY");
    def!("wanjie",         "https://maas-openapi.wanjiedata.com/api/v1", "WANJIE_API_KEY");
    def!("novita",         "https://api.novita.ai/openai/v1",           "NOVITA_API_KEY");
    def!("sakana",         "https://api.sakana.ai/v1",                  "SAKANA_API_KEY");
    def!("longcat",        "https://api.longcat.chat/openai/v1",        "LONGCAT_API_KEY");
    def!("arcee",          "https://api.arcee.ai/api/v1",               "ARCEE_API_KEY");

    m
}

/// Built-in model → provider mapping.
fn builtin_model_routes() -> Vec<(&'static str, &'static str)> {
    vec![
        // DeepSeek models
        ("deepseek-v4-pro",       "deepseek"),
        ("deepseek-v4-flash",     "deepseek"),
        ("deepseek-reasoner",     "deepseek"),
        ("deepseek-chat",         "deepseek"),
        // OpenAI models
        ("gpt-4",                 "openai"),
        ("gpt-4o",                "openai"),
        ("gpt-4o-mini",          "openai"),
        ("gpt-5",                 "openai"),
        ("o1",                    "openai"),
        ("o3",                    "openai"),
        // Anthropic
        ("claude",                "anthropic"),
        ("claude-3",              "anthropic"),
        ("claude-3.5",            "anthropic"),
        ("claude-4",              "anthropic"),
        // OpenRouter
        ("openrouter/",           "openrouter"),
        // NVIDIA
        ("nvidia/",               "nvidia-nim"),
        ("nemotron",              "nvidia-nim"),
        // Together
        ("together/",             "together"),
        // Fireworks
        ("fireworks/",            "fireworks"),
        // DeepInfra
        ("deepinfra/",            "deepinfra"),
        // HuggingFace
        ("hf/",                   "huggingface"),
        ("huggingface/",          "huggingface"),
        // xAI / Grok
        ("grok",                  "xai"),
        // Mistral
        ("mistral",               "mistral"),
        // Cohere
        ("command",               "cohere"),
        // Ollama (local)
        ("ollama/",               "ollama"),
        // Qwen (local NPU)
        ("qwen3",                 "npu"),
    ]
}

// ── Provider config ─────────────────────────────────────────────

/// A provider definition (from builtins or custom config).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDef {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub env_var: Option<String>,
    #[serde(default = "default_auth_mode")]
    pub auth_mode: String,
}

fn default_auth_mode() -> String {
    "api_key".into()
}

impl ProviderDef {
    /// Get the API key for this provider.
    pub fn get_key(&self, secrets: &SecretsStore) -> Option<String> {
        // Check provider-specific env var first
        if let Some(env) = &self.env_var {
            if let Ok(val) = std::env::var(env) {
                if !val.is_empty() {
                    return Some(val);
                }
            }
        }
        // Check generic env var
        let generic = format!("{}_API_KEY", self.id.to_uppercase().replace('-', "_"));
        if let Ok(val) = std::env::var(&generic) {
            if !val.is_empty() {
                return Some(val);
            }
        }
        // Check secrets store
        secrets.get_key(&self.id)
    }

    /// Build the appropriate chat completion URL for this provider.
    pub fn chat_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if self.id == "anthropic" {
            format!("{base}/v1/messages")
        } else {
            format!("{base}/chat/completions")
        }
    }

    /// Build the models list URL.
    pub fn models_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{base}/models")
    }
}

/// Custom provider from user config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProvider {
    pub id: String,
    pub name: Option<String>,
    pub base_url: String,
    pub auth_mode: Option<String>,
    pub api_key: Option<String>,
    pub models: Option<Vec<String>>,
}

// ── Provider router ─────────────────────────────────────────────

/// Resolves model names to providers and manages the provider chain.
#[derive(Debug, Clone)]
pub struct ProviderRouter {
    /// Built-in providers (id → definition)
    builtins: HashMap<String, ProviderDef>,
    /// Model → provider ID routing table
    model_routes: Vec<(String, String)>,
    /// Custom providers from config
    custom: Vec<CustomProvider>,
    /// API client
    client: Client,
    /// Secrets store
    secrets: SecretsStore,
    /// Default provider ID
    pub default_provider: String,
    /// Fallback chain (provider IDs in order)
    pub fallbacks: Vec<String>,
}

impl ProviderRouter {
    /// Create a new provider router with built-in providers.
    pub fn new(
        default_provider: &str,
        custom: Vec<CustomProvider>,
        fallbacks: Vec<String>,
    ) -> Result<Self> {
        let secrets = SecretsStore::open()?;
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        let builtins = builtin_providers()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        let model_routes: Vec<(String, String)> = builtin_model_routes()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        Ok(Self {
            builtins,
            model_routes,
            custom,
            client,
            secrets,
            default_provider: default_provider.to_string(),
            fallbacks,
        })
    }

    /// Resolve a model name to a provider ID.
    /// Supports `provider/model` syntax for explicit routing.
    pub fn resolve_provider(&self, model: &str) -> String {
        // Check for explicit provider/model syntax
        if let Some((provider, _rest)) = model.split_once('/') {
            if self.builtins.contains_key(provider) || self.custom.iter().any(|c| c.id == provider) {
                return provider.to_string();
            }
        }

        // Check model route table (prefix matching)
        for (prefix, provider_id) in &self.model_routes {
            if model.starts_with(prefix) {
                return provider_id.clone();
            }
        }

        // Fall back to default provider
        self.default_provider.clone()
    }

    /// Get a provider definition by ID.
    pub fn get_provider(&self, id: &str) -> Option<ProviderDef> {
        // Check builtins
        if let Some(p) = self.builtins.get(id) {
            return Some(p.clone());
        }
        // Check custom providers
        if let Some(c) = self.custom.iter().find(|c| c.id == id) {
            return Some(ProviderDef {
                id: c.id.clone(),
                name: c.name.clone().unwrap_or_default(),
                base_url: c.base_url.clone(),
                env_var: None,
                auth_mode: c.auth_mode.clone().unwrap_or_else(default_auth_mode),
            });
        }
        None
    }

    /// Get a list of all available providers (built-in + custom).
    pub fn list_providers(&self) -> Vec<ProviderDef> {
        let mut providers: Vec<ProviderDef> = self.builtins.values().cloned().collect();
        for c in &self.custom {
            providers.push(ProviderDef {
                id: c.id.clone(),
                name: c.name.clone().unwrap_or_default(),
                base_url: c.base_url.clone(),
                env_var: None,
                auth_mode: c.auth_mode.clone().unwrap_or_else(default_auth_mode),
            });
        }
        providers.sort_by(|a, b| a.id.cmp(&b.id));
        providers
    }

    /// Check which providers have keys configured.
    pub fn providers_with_keys(&self) -> Vec<String> {
        let mut available = Vec::new();
        for (id, def) in &self.builtins {
            if def.get_key(&self.secrets).is_some() {
                available.push(id.clone());
            }
        }
        for c in &self.custom {
            // Check if custom provider has a key
            let def = ProviderDef {
                id: c.id.clone(),
                name: String::new(),
                base_url: c.base_url.clone(),
                env_var: None,
                auth_mode: c.auth_mode.clone().unwrap_or_else(default_auth_mode),
            };
            if def.get_key(&self.secrets).is_some() || c.api_key.is_some() {
                available.push(c.id.clone());
            }
        }
        available.sort();
        available
    }

    /// Send a chat completion to a specific provider.
    pub async fn chat(
        &self,
        provider_id: &str,
        model: &str,
        messages: Vec<Value>,
        tools: Option<Vec<Value>>,
    ) -> Result<(String, String)> {
        let provider = self
            .get_provider(provider_id)
            .with_context(|| format!("Unknown provider: {provider_id}"))?;

        let api_key = provider
            .get_key(&self.secrets)
            .with_context(|| format!("No API key for provider '{provider_id}' — set via env or `1bit config auth`"))?;

        let url = provider.chat_url();
        let model_name = if let Some((_prov, rest)) = model.split_once('/') {
            rest
        } else {
            model
        };

        let mut body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "stream": false,
        });

        if let Some(t) = tools {
            if !t.is_empty() {
                body["tools"] = serde_json::Value::Array(t);
            }
        }

        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(120));

        // Set auth header based on provider
        match provider_id {
            "anthropic" => {
                req = req
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", "2023-06-01");
                // Anthropic uses a different request format
                let messages_arr = body["messages"].take();
                body["messages"] = messages_arr;
                body["max_tokens"] = serde_json::json!(4096);
            }
            "huggingface" | "hf" => {
                req = req.header("Authorization", format!("Bearer {api_key}"));
            }
            _ => {
                req = req.header("Authorization", format!("Bearer {api_key}"));
            }
        }

        let resp = req
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Request to {provider_id} failed"))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();

        if !status.is_success() {
            anyhow::bail!("{provider_id} returned HTTP {status}: {text}");
        }

        let parsed: Value = serde_json::from_str(&text)?;
        let content = parsed
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("[no content]")
            .to_string();

        let model_used = parsed
            .pointer("/model")
            .and_then(|v| v.as_str())
            .unwrap_or(model_name);

        Ok((content, model_used.to_string()))
    }

    /// Chat with automatic provider routing and fallback.
    pub async fn chat_with_fallback(
        &self,
        model: &str,
        messages: Vec<Value>,
        tools: Option<Vec<Value>>,
    ) -> Result<(String, String, String)> {
        let primary = self.resolve_provider(model);

        // Build the fallback chain: primary provider first, then configured fallbacks
        let chain: Vec<String> = {
            let mut c = vec![primary.clone()];
            for fb in &self.fallbacks {
                if fb != &primary && !c.contains(fb) {
                    c.push(fb.clone());
                }
            }
            // Add NPU as ultimate fallback if configured
            if !c.contains(&"npu".to_string()) {
                if let Some(npu_def) = self.get_provider("npu") {
                    if npu_def.get_key(&self.secrets).is_some() || true {
                        // NPU is always available (local)
                        c.push("npu".to_string());
                    }
                }
            }
            c
        };

        let mut last_error = String::new();
        for provider_id in &chain {
            // Check if this provider has a key (skip NPU which is local)
            if provider_id != "npu" {
                if let Some(def) = self.get_provider(provider_id) {
                    if def.get_key(&self.secrets).is_none() {
                        tracing::debug!("Skipping {provider_id}: no API key configured");
                        continue;
                    }
                }
            }

            match self.chat(provider_id, model, messages.clone(), tools.clone()).await {
                Ok((content, model_used)) => {
                    return Ok((content, model_used, provider_id.clone()));
                }
                Err(e) => {
                    tracing::warn!("{provider_id} failed, trying fallback: {e}");
                    last_error = e.to_string();
                }
            }
        }

        anyhow::bail!("All providers failed. Last error: {last_error}");
    }

    /// List models available from a specific provider.
    pub async fn list_provider_models(&self, provider_id: &str) -> Result<Vec<String>> {
        let provider = self
            .get_provider(provider_id)
            .with_context(|| format!("Unknown provider: {provider_id}"))?;

        let api_key = match provider.get_key(&self.secrets) {
            Some(k) => k,
            None => return Ok(vec![]),
        };

        let url = provider.models_url();
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .with_context(|| format!("Failed to fetch models from {provider_id}"))?;

        if !resp.status().is_success() {
            return Ok(vec![]);
        }

        let parsed: Value = resp.json().await?;
        let models = parsed
            .pointer("/data")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|id| id.as_str().map(|s| s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}
