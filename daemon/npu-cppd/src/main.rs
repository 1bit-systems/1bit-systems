//! npu-cppd — C++ NPU Engine Daemon
//!
//! OpenAI-compatible HTTP API for the NPU inference engine.
//! Routes chat completions to the NPU runner and handles tokenization.
//!
//! Usage:
//!   sudo ./npu-cppd [--port PORT] [--model MODEL]
//!
//! Environment:
//!   NPU_XCLBIN_DIR       Override xclbin directory (default: auto-detect)
//!   NPU_MODEL_PATH        Override model path (default: auto-detect)
//!   TORCH2AIE_ROOT        Path to torch2aie checkout (required)
//!   TORCH2AIE_VENV        Path to torch2aie virtualenv (default: TORCH2AIE_ROOT/.venv)
//!   TOKENIZER_JSON        Path to tokenizer.json
//!   NPU_CPPD_BIND_ADDR    Bind address (default: 127.0.0.1)

mod backend;
mod routes;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "npu-cppd", about = "C++ NPU Engine Daemon", version = "0.1.0")]
struct Args {
    /// HTTP API port
    #[arg(short, long, default_value_t = 9090)]
    port: u16,

    /// Model name to advertise
    #[arg(long, default_value = "qwen3-0.6b-npu")]
    model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "npu_cppd=info".into()),
        )
        .init();

    let args = Args::parse();

    // ── Resolve paths ───────────────────────────────────────────────
    let repo_root = PathBuf::from(
        std::env::var("REPO_ROOT").unwrap_or_else(|_| {
            format!("{}/1bit-systems",
                std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string()))
        })
    );

    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string());

    // Tokenizer path
    let tokenizer_json = std::env::var("TOKENIZER_JSON").ok()
        .map(PathBuf::from)
        .or_else(|| {
            // Auto-detect
            let candidates = [
                format!("{home}/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"),
                repo_root.join("engine/npu/tokenizer/tokenizer.json").to_string_lossy().to_string(),
            ];
            candidates.iter().find(|p| PathBuf::from(p).exists()).map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            PathBuf::from(&home).join(".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")
        });

    // torch2aie paths
    let torch2aie_root = std::env::var("TORCH2AIE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_default();

    let torch2aie_venv = std::env::var("TORCH2AIE_VENV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if torch2aie_root.exists() {
                torch2aie_root.join(".venv")
            } else {
                PathBuf::new()
            }
        });

    let bind_addr = std::env::var("NPU_CPPD_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());

    // ── Print configuration ─────────────────────────────────────────
    println!("=== NPU C++ Engine Daemon (Rust) ===");
    println!("  Port: {}", args.port);
    println!("  Model: {}", args.model);
    println!("  Tokenizer: {}", tokenizer_json.display());
    println!("  torch2aie: {}",
        if torch2aie_root.exists() {
            torch2aie_root.display().to_string()
        } else {
            "not set (NPU runner disabled)".to_string()
        }
    );

    // ── Start backend ───────────────────────────────────────────────
    let backend = Arc::new(backend::NpuBackend::new(
        repo_root,
        tokenizer_json,
        torch2aie_root,
        torch2aie_venv,
    ));
    backend.start().await?;

    // ── Build router ─────────────────────────────────────────────────
    let state = Arc::new(routes::AppState {
        backend,
        model_name: args.model,
    });
    let app = routes::build_router(state);

    // ── Start HTTP server ───────────────────────────────────────────
    let addr: std::net::SocketAddr = format!("{bind_addr}:{}", args.port)
        .parse()
        .context("Invalid bind address")?;

    if bind_addr == "0.0.0.0" {
        tracing::warn!("⚠️  WARNING: binding to 0.0.0.0 — server is publicly reachable");
    }

    info!("Listening on http://{addr}");
    info!("  Health: http://localhost:{}/health", args.port);
    info!("  API:    http://localhost:{}/v1/chat/completions", args.port);

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
