//! Backend lifecycle and inference.
//!
//! NPUBackend: spawns `npu_engine_universal` as subprocess,
//!   communicates via stdin/stdout JSON protocol, uses HuggingFace tokenizers.
//! GPUBackend: spawns `lemond` as subprocess, communicates via HTTP.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child as StdChild, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::{error, info, warn};

// ── ChatML prompt building ─────────────────────────────────────────

/// A chat message from an OpenAI-compatible request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
}

/// Build a Qwen-style ChatML prompt from a list of messages.
pub fn build_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut system_set = false;

    for msg in messages {
        let content = match &msg.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => {
                arr.iter()
                    .filter_map(|part| {
                        if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                            part.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            other => other.to_string(),
        };

        match msg.role.as_str() {
            "system" => {
                parts.push(format!("<|im_start|>system\n{content}<|im_end|>"));
                system_set = true;
            }
            "user" => parts.push(format!("<|im_start|>user\n{content}<|im_end|>")),
            "assistant" => parts.push(format!("<|im_start|>assistant\n{content}<|im_end|>")),
            other => parts.push(format!("<|im_start|>{other}\n{content}<|im_end|>")),
        }
    }

    if !system_set {
        parts.insert(0, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>".to_string());
    }
    parts.push("<|im_start|>assistant\n".to_string());

    parts.join("\n")
}

// ── NPU Engine protocol ────────────────────────────────────────────

const EOS_ID: u32 = 151_645; // <|im_end|>

#[derive(Serialize)]
struct NpuRequest {
    tokens: Vec<u32>,
    max_new_tokens: u32,
}

#[derive(Deserialize)]
struct NpuResponse {
    #[serde(default)]
    tokens: Vec<u32>,
}

// ── NPU Backend ─────────────────────────────────────────────────────

pub struct NpuBackend {
    engine_bin: PathBuf,
    _model_path: PathBuf,
    tokenizer_path: PathBuf,
    tokenizer: Option<Tokenizer>,
    inner: Arc<Mutex<Option<NpuInner>>>,
}

struct NpuInner {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pid: u32,
}

impl NpuBackend {
    pub fn new(engine_bin: PathBuf, model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            engine_bin,
            _model_path: model_path,
            tokenizer_path,
            tokenizer: None,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Start the NPU C++ engine subprocess and wait for "Ready." on stderr.
    pub async fn start(&mut self) -> Result<()> {
        if !self.engine_bin.exists() {
            warn!("NPU engine binary not found: {}", self.engine_bin.display());
            return Err(anyhow::anyhow!(
                "NPU engine binary not found: {}. Run: cd ~/engine/npu && make",
                self.engine_bin.display()
            ));
        }
        if !self.tokenizer_path.exists() {
            return Err(anyhow::anyhow!(
                "Tokenizer not found: {}",
                self.tokenizer_path.display()
            ));
        }

        info!("Starting NPU C++ engine...");

        // Load tokenizer (sync is fine — happens once at startup).
        // Tokenizer::from_file returns Result<Tokenizer, Box<dyn StdError>>.
        // Convert to anyhow::Error via map_err.
        let tokenizer = Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        info!("Tokenizer loaded ({})", self.tokenizer_path.display());

        // Spawn engine using tokio::process for async I/O.
        // We'll convert to std::process::Child after taking stdin/stdout.
        let mut child: Child = tokio::process::Command::new(&self.engine_bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn NPU engine: {}", self.engine_bin.display()))?;

        let pid = child.id().unwrap_or(0);

        // Read stderr until "Ready." or timeout
        let stderr = child.stderr.take()
            .context("Failed to capture NPU engine stderr")?;
        let mut stderr_reader = BufReader::new(stderr);
        let mut ready_line = String::new();
        let start = Instant::now();
        let timeout = Duration::from_secs(30);
        let mut ready = false;

        loop {
            ready_line.clear();
            match stderr_reader.read_line(&mut ready_line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = ready_line.trim();
                    if !trimmed.is_empty() {
                        info!("  [npu] {trimmed}");
                    }
                    if trimmed.contains("Ready.") {
                        ready = true;
                        break;
                    }
                }
                Err(e) => {
                    error!("Error reading NPU engine stderr: {e}");
                    break;
                }
            }
            if start.elapsed() > timeout {
                break;
            }
        }

        if !ready {
            let _ = child.kill();
            let _ = child.wait().await;
            return Err(anyhow::anyhow!("NPU engine failed to start within 30s"));
        }

        let stdin = child.stdin.take()
            .context("Failed to capture NPU engine stdin")?;
        let stdout = child.stdout.take()
            .context("Failed to capture NPU engine stdout")?;

        self.tokenizer = Some(tokenizer);
        *self.inner.lock().await = Some(NpuInner {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            pid,
        });

        info!("NPU C++ engine ready (pid={pid})");
        Ok(())
    }

    /// Stop the NPU engine subprocess.
    pub async fn stop(&self) {
        let mut guard = self.inner.lock().await;
        if let Some(mut inner) = guard.take() {
            // Close stdin first to signal graceful shutdown
            let _ = inner.stdin.shutdown().await;
            drop(inner.stdin);
            let _ = inner._child.kill();
            let _ = inner._child.wait().await;
            info!("NPU engine stopped (pid={})", inner.pid);
        }
    }

    /// Check if the NPU backend is running.
    pub async fn is_running(&self) -> bool {
        let guard = self.inner.lock().await;
        guard.is_some()
    }

    /// Run inference via the C++ engine.
    pub async fn chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        _temperature: Option<f32>,
    ) -> Result<serde_json::Value> {
        let tokenizer = self.tokenizer.as_ref()
            .context("NPU engine not initialized (no tokenizer)")?;

        let mut guard = self.inner.lock().await;
        let inner = guard.as_mut()
            .context("NPU engine not running")?;

        // Build prompt and tokenize
        let prompt = build_chatml_prompt(messages);
        let encoded = tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?;
        let tokens: Vec<u32> = encoded.get_ids().to_vec();
        let max_new = max_tokens.unwrap_or(256);

        if tokens.len() > 4096 {
            return Err(anyhow::anyhow!(
                "Prompt too long ({} tokens, max 4096)",
                tokens.len()
            ));
        }

        // Send request as JSON line via stdin
        let request = NpuRequest {
            tokens,
            max_new_tokens: max_new,
        };
        let request_str = serde_json::to_string(&request)?;

        inner.stdin.write_all(request_str.as_bytes()).await?;
        inner.stdin.write_all(b"\n").await?;
        inner.stdin.flush().await?;

        // Read response line from stdout
        let mut line = String::new();
        inner.stdout.read_line(&mut line).await
            .context("Failed to read NPU engine response")?;

        if line.trim().is_empty() {
            return Err(anyhow::anyhow!("NPU engine returned empty response"));
        }

        let resp: NpuResponse = serde_json::from_str(line.trim())
            .with_context(|| format!("Failed to parse NPU response: {line}"))?;

        // Decode tokens, stopping at EOS
        let mut filtered: Vec<u32> = Vec::new();
        for &t in &resp.tokens {
            if t == EOS_ID {
                break;
            }
            filtered.push(t);
        }

        let generated = if filtered.is_empty() {
            String::new()
        } else {
            tokenizer.decode(&filtered, false)
                .map_err(|e| anyhow::anyhow!("Failed to decode output tokens: {e}"))?
        };

        let prompt_tokens = encoded.get_ids().len();
        let completion_tokens = filtered.len();
        let now = chrono::Utc::now().timestamp();

        let response = serde_json::json!({
            "id": format!("chatcmpl-{}", now),
            "object": "chat.completion",
            "created": now,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated,
                },
                "finish_reason": if resp.tokens.last() == Some(&EOS_ID) { "stop" } else { "length" },
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "x-device": "npu",
            "model": model,
        });

        Ok(response)
    }
}

// ── GPU Backend (HTTP to lemond) ────────────────────────────────────

pub struct GpuBackend {
    port: u16,
    process: Arc<Mutex<Option<StdChild>>>,
    client: reqwest::Client,
}

impl GpuBackend {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            process: Arc::new(Mutex::new(None)),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Start the GPU backend (lemond subprocess).
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting GPU backend (lemond) on port {}...", self.port);

        let cache_dir = format!("/tmp/lemonade-gpu-{}", self.port);
        let mut child = Command::new("lemond")
            .arg(&cache_dir)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn lemond. Is it installed?")?;

        let health_url = format!("http://127.0.0.1:{}/api/v1/health", self.port);
        let mut ready = false;

        for _ in 0..30 {
            match self.client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    ready = true;
                    break;
                }
                _ => {}
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Err(anyhow::anyhow!("lemond exited with code {status}"));
                }
                _ => {}
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        if !ready {
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow::anyhow!("lemond failed to start within 30s"));
        }

        info!("GPU backend running (pid={})", child.id());
        *self.process.lock().await = Some(child);
        Ok(())
    }

    /// Stop the GPU backend.
    pub async fn stop(&self) {
        let mut guard = self.process.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
            info!("GPU backend stopped");
        }
    }

    /// Check if the GPU backend is running.
    pub async fn is_running(&self) -> bool {
        let mut guard = self.process.lock().await;
        if let Some(child) = guard.as_mut() {
            match child.try_wait() {
                Ok(None) => true,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Forward a chat completion request to the GPU backend.
    pub async fn chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        extra: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
        });

        if let Some(obj) = body.as_object_mut() {
            for (k, v) in extra {
                if !obj.contains_key(k) {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("GPU backend request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("GPU backend returned {status}: {text}"));
        }

        let value: serde_json::Value = resp.json().await
            .context("Failed to parse GPU backend response")?;

        Ok(value)
    }
}
