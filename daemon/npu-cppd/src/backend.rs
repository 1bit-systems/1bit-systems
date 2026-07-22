//! NPU inference backend — tokenization via `engine/fusion/tokenize` C++ binary
//! and inference via `npu_runner.py` Python subprocess.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// Manages the NPU inference backend.
pub struct NpuBackend {
    repo_root: PathBuf,
    tokenizer_json: PathBuf,
    torch2aie_root: PathBuf,
    torch2aie_venv: PathBuf,
    /// Whether the NPU runner Python module is importable.
    ready: Arc<Mutex<bool>>,
}

impl NpuBackend {
    pub fn new(
        repo_root: PathBuf,
        tokenizer_json: PathBuf,
        torch2aie_root: PathBuf,
        torch2aie_venv: PathBuf,
    ) -> Self {
        Self {
            repo_root,
            tokenizer_json,
            torch2aie_root,
            torch2aie_venv,
            ready: Arc::new(Mutex::new(false)),
        }
    }

    /// Attempt to validate the NPU runner by importing it.
    pub async fn start(&self) -> Result<()> {
        if !self.torch2aie_root.exists() {
            warn!("TORCH2AIE_ROOT not set or not found: {:?}", self.torch2aie_root);
            return Ok(()); // Don't fail — the daemon can still serve health/models
        }

        let python_bin = self.torch2aie_venv.join("bin/python3");
        if !python_bin.exists() {
            warn!("Python binary not found: {:?}", python_bin);
            return Ok(());
        }

        // Test import
        let test_code = format!(
            "import sys, os; sys.path.insert(0, '{}'); from tools.npu_runner import NPURunner; print('OK')",
            self.repo_root.display()
        );

        let result = Command::new(&python_bin)
            .args(["-c", &test_code])
            .env("REPO_ROOT", self.repo_root.to_string_lossy().to_string())
            .env("PYTHONPATH", self.build_pythonpath())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("Failed to launch Python: {}", python_bin.display()))?;

        if result.status.success() {
            let mut ready = self.ready.lock().await;
            *ready = true;
            info!("NPU runner ready");
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            warn!("NPU runner failed to import: {:.200}", stderr);
        }

        Ok(())
    }

    /// Build PYTHONPATH for torch2aie.
    fn build_pythonpath(&self) -> String {
        let mut parts = Vec::new();
        parts.push(self.torch2aie_root.join("toolchain/mlir_aie/python").to_string_lossy().to_string());
        parts.push(self.torch2aie_root.join("examples/qwen3-decode-layer").to_string_lossy().to_string());
        parts.push(self.torch2aie_root.join("examples/qwen3-decode-layer/cases").to_string_lossy().to_string());

        if let Ok(existing) = std::env::var("PYTHONPATH") {
            if !existing.is_empty() {
                parts.push(existing);
            }
        }

        parts.join(":")
    }

    /// Tokenize text via `engine/fusion/tokenize --model <path> --encode <text>`.
    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenize_bin = self.repo_root.join("engine/fusion/tokenize");
        if !tokenize_bin.exists() {
            anyhow::bail!("tokenize binary not found: {}", tokenize_bin.display());
        }

        let output = Command::new(&tokenize_bin)
            .args(["--model", &self.tokenizer_json.to_string_lossy()])
            .arg("--encode")
            .arg(text)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("Failed to run tokenize: {}", tokenize_bin.display()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("tokenize failed: {:.200}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let tokens: Vec<u32> = stdout
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        Ok(tokens)
    }

    /// Detokenize tokens via `engine/fusion/tokenize --model <path> --decode <tokens>`.
    pub async fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let tokenize_bin = self.repo_root.join("engine/fusion/tokenize");
        if !tokenize_bin.exists() {
            anyhow::bail!("tokenize binary not found: {}", tokenize_bin.display());
        }

        let mut args = vec![
            "--model".to_string(),
            self.tokenizer_json.to_string_lossy().to_string(),
            "--decode".to_string(),
        ];
        for t in tokens {
            args.push(t.to_string());
        }

        let output = Command::new(&tokenize_bin)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("Failed to run detokenize: {}", tokenize_bin.display()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("detokenize failed: {:.200}", stderr);
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(text)
    }

    /// Run inference via `npu_runner.py`.
    pub async fn chat(
        &self,
        model: &str,
        prompt_tokens: &[u32],
        max_new: u32,
    ) -> Result<serde_json::Value> {
        let ready = *self.ready.lock().await;
        if !ready {
            anyhow::bail!("NPU engine not ready (TORCH2AIE_ROOT not configured or runner failed)");
        }

        let python_bin = self.torch2aie_venv.join("bin/python3");
        let runner_path = self.repo_root.join("tools/npu_runner.py");

        if !runner_path.exists() {
            anyhow::bail!("NPU runner not found: {}", runner_path.display());
        }

        let request = serde_json::json!({
            "tokens": prompt_tokens,
            "max_new_tokens": max_new,
        });
        let request_str = serde_json::to_string(&request)?;

        // Build environment for the subprocess
        let mut envs: Vec<(String, String)> = std::env::vars().collect();
        envs.push(("REPO_ROOT".to_string(), self.repo_root.to_string_lossy().to_string()));
        envs.push(("PYTHONPATH".to_string(), self.build_pythonpath()));

        let mut child = tokio::process::Command::new(&python_bin)
            .arg(runner_path.to_string_lossy().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .envs(envs)
            .spawn()
            .with_context(|| format!("Failed to spawn NPU runner: {}", runner_path.display()))?;

        // Write request to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(request_str.as_bytes()).await
                .context("Failed to write to NPU runner stdin")?;
            drop(stdin); // close stdin to signal EOF
        }

        let output = child.wait_with_output().await
            .context("Failed to read NPU runner output")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("NPU runner failed (exit={}): {:.200}", output.status, stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let resp: serde_json::Value = serde_json::from_str(stdout.trim())
            .with_context(|| format!("Bad JSON from runner: {:.200}", stdout))?;

        // Build OpenAI-compatible response
        let out_tokens: Vec<u32> = resp.get("tokens")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
            .unwrap_or_default();

        let out_text = self.detokenize(&out_tokens).await.unwrap_or_default();
        let finished = resp.get("finished").and_then(|v| v.as_bool()).unwrap_or(false);

        Ok(serde_json::json!({
            "id": format!("chatcmpl-npu-cppd"),
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": out_text,
                },
                "finish_reason": if finished { "stop" } else { "length" },
            }],
            "usage": {
                "prompt_tokens": prompt_tokens.len(),
                "completion_tokens": out_tokens.len(),
                "total_tokens": prompt_tokens.len() + out_tokens.len(),
            },
        }))
    }

    pub async fn is_ready(&self) -> bool {
        *self.ready.lock().await
    }
}
