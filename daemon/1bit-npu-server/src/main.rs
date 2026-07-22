//! 1bit NPU Chat Server — Zero FLM dependency.
//!
//! Uses `npu_engine_universal --worker` for GEMM ops on NPU.
//! CPU handles: RoPE, attention, KV cache, RMSNorm, SiLU, sampling.
//! Serves Ollama-compatible HTTP API.
//!
//! Usage:
//!   sudo ./1bit-npu-server --port 9090
//!
//! Environment:
//!   NPU_ENGINE_BIN      Path to npu_engine_universal binary
//!   NPU_MODEL_PATH      Path to model.q4nx file
//!   NPU_TOKENIZER_PATH  Path to tokenizer.json
//!   NPU_XCLBIN_DIR      Path to xclbin directory
//!   NPU_MODEL_TAG       Model tag (default: qwen3_0_6b)
//!   NPU_BIND_ADDR       Bind address (default: 127.0.0.1)

mod inference;
mod q4nx;
mod routes;
mod worker;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    name = "1bit-npu-server",
    about = "1bit NPU Chat Server — Zero FLM dependency",
    version = "0.1.0"
)]
struct Args {
    /// HTTP API port
    #[arg(short, long, default_value_t = 9090)]
    port: u16,

    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "1bit_npu_server=info".into()),
        )
        .init();

    let args = Args::parse();

    // ── Resolve paths from env vars ───────────────────────────────
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string());
    let repo_root = PathBuf::from(&home).join("1bit-systems");

    let engine_bin = std::env::var("NPU_ENGINE_BIN").unwrap_or_else(|_| {
        repo_root.join("engine/npu/build/npu_engine_universal").to_string_lossy().to_string()
    });
    let model_path = std::env::var("NPU_MODEL_PATH").unwrap_or_else(|_| {
        format!("{home}/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx")
    });
    let tokenizer_path = std::env::var("NPU_TOKENIZER_PATH").unwrap_or_else(|_| {
        format!("{home}/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")
    });
    let xclbin_dir = std::env::var("NPU_XCLBIN_DIR").unwrap_or_else(|_| {
        repo_root.join("engine/npu/xclbins").to_string_lossy().to_string()
    });
    let model_tag = std::env::var("NPU_MODEL_TAG").unwrap_or_else(|_| "qwen3_0_6b".to_string());
    let bind_addr = std::env::var("NPU_BIND_ADDR").unwrap_or_else(|_| args.bind.clone());

    // ── Load model weights ───────────────────────────────────────
    info!("Loading model from {model_path}");
    let model = q4nx::Q4nxReader::open(&model_path)
        .with_context(|| format!("Failed to load model from {model_path}"))?;
    info!("Model loaded: {}", PathBuf::from(&model_path).file_name().unwrap_or_default().to_string_lossy());

    // ── Start NPU worker ─────────────────────────────────────────
    let mut worker = worker::NpuWorker::new(
        engine_bin,
        model_path,
        model_tag,
        xclbin_dir,
    );
    worker.start()?;

    // ── Create inference engine ───────────────────────────────────
    let cfg = inference::ModelConfig::default();
    let inference = inference::Inference::new(worker, &model, cfg)
        .context("Failed to initialize inference engine")?;
    let inference = Arc::new(Mutex::new(inference));

    // ── Load tokenizer ────────────────────────────────────────────
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
    info!("Tokenizer loaded ({tokenizer_path})");

    // ── Build state and router ────────────────────────────────────
    let state = Arc::new(routes::AppState {
        inference,
        tokenizer,
    });
    let app = routes::build_router(state);

    // ── Start HTTP server ────────────────────────────────────────
    let addr: std::net::SocketAddr = format!("{bind_addr}:{}", args.port)
        .parse()
        .context("Invalid bind address")?;

    if bind_addr != "127.0.0.1" {
        tracing::warn!("⚠️  WARNING: binding to non-localhost. Ensure firewall rules are in place.");
    }

    info!("🚀 1bit NPU chat server: http://{addr}");
    info!("Zero FLM dependency — using npu_engine_universal worker");

    let listener = tokio::net::TcpListener::bind(addr).await
        .context("Failed to bind TCP listener")?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await
        .context("Server error")?;

    Ok(())
}
