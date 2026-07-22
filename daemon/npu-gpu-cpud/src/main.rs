//! npu-gpu-cpud — Unified Control Plane Daemon (Rust)
//!
//! Routes inference requests to NPU, GPU, or CPU based on policy.
//! Rust port of the original npu-gpu-cpud.py.
//!
//! Backends:
//!   NPU: Open-source C++ engine (MIT, 97 tok/s Qwen3-0.6B) — spawned as subprocess
//!   GPU: ROCm/WMMA GPU engine via lemond — spawned as subprocess
//!   CPU: not implemented — >8B requests fall back to GPU (see #147)
//!
//! Policy (model_size → device):
//!   < 2B params  → NPU (lowest power)
//!   >= 2B params → GPU (fastest compute)
//!
//! Usage:
//!   sudo ./npu-gpu-cpud [--port PORT] [--npu-port PORT] [--gpu-port PORT]

mod backends;
mod orders;
mod policy;
mod routes;
mod stripe;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    name = "npu-gpu-cpud",
    about = "Unified NPU+GPU Control Plane Daemon",
    version = "0.1.0"
)]
struct Args {
    /// Gateway port (HTTP API)
    #[arg(short, long, default_value_t = 9090)]
    port: u16,

    /// NPU backend port (internal, only used for tracking)
    #[arg(long, default_value_t = 52625)]
    npu_port: u16,

    /// GPU backend port (for lemond)
    #[arg(long, default_value_t = 13305)]
    gpu_port: u16,

    /// Don't auto-start backends
    #[arg(long)]
    no_auto: bool,

    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── Logging ───────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "npu_gpu_cpud=info".into()),
        )
        .init();

    let args = Args::parse();

    // ── Resolve paths ─────────────────────────────────────────────
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string());
    let repo_root = PathBuf::from(&home).join("1bit-systems");

    let engine_bin = std::env::var("NPU_ENGINE_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            repo_root.join("engine/npu/build/npu_engine_universal")
        });

    let model_path = std::env::var("NPU_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(&home).join(".config/flm/models/Qwen3-0.6B-NPU2/model.q4nx")
        });

    let tokenizer_path = std::env::var("NPU_TOKENIZER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(&home).join(".config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json")
        });

    // ── SMTP config ───────────────────────────────────────────────
    let smtp_host = std::env::var("SMTP_HOST").unwrap_or_else(|_| "localhost".to_string());
    let smtp_port = std::env::var("SMTP_PORT")
        .unwrap_or_else(|_| "25".to_string())
        .parse()
        .unwrap_or(25);
    let smtp_user = std::env::var("SMTP_USER").unwrap_or_default();
    let smtp_pass = std::env::var("SMTP_PASS").unwrap_or_default();
    let notify_email = std::env::var("ORDER_NOTIFY_EMAIL")
        .unwrap_or_else(|_| "sales@1bit.systems".to_string());
    let notify_cc = std::env::var("ORDER_NOTIFY_CC").unwrap_or_default();

    // ── Stripe config ─────────────────────────────────────────────
    let stripe_secret_key = std::env::var("STRIPE_SECRET_KEY").unwrap_or_default();
    let stripe_webhook_secret = std::env::var("STRIPE_WEBHOOK_SECRET").unwrap_or_default();

    // ── Banner ────────────────────────────────────────────────────
    println!("{}", "=".repeat(60));
    println!("  NPU + GPU + CPU = Unified Control Plane (Rust)");
    println!("  Gateway: http://{}:{}", args.bind, args.port);
    println!("{}", "=".repeat(60));
    println!();

    // ── Load pending orders ───────────────────────────────────────
    let order_manager = orders::OrderManager::new(
        &repo_root,
        smtp_host,
        smtp_port,
        smtp_user,
        smtp_pass,
        notify_email,
        notify_cc,
    );
    if let Err(e) = order_manager.load_pending().await {
        warn!("Could not load pending orders: {e}");
    }

    // ── Start backends ────────────────────────────────────────────
    let mut npu_backend = backends::NpuBackend::new(
        engine_bin.clone(),
        model_path.clone(),
        tokenizer_path.clone(),
    );
    let mut gpu_backend = backends::GpuBackend::new(args.gpu_port);

    if !args.no_auto {
        info!("Starting backends...");

        // NPU backend
        if let Err(e) = npu_backend.start().await {
            warn!("NPU backend failed to start: {e}");
            warn!("  Check NPU_ENGINE_BIN={}", engine_bin.display());
            warn!("  Check NPU_MODEL_PATH={}", model_path.display());
            warn!("  Check NPU_TOKENIZER_PATH={}", tokenizer_path.display());
        }

        // GPU backend (optional)
        if let Err(e) = gpu_backend.start().await {
            warn!("GPU backend skipped: {e}");
            warn!("  lemond not found or failed — GPU requests will fail");
        }

        println!();
    }

    // ── Load store HTML ───────────────────────────────────────────
    let store_path = repo_root.join("site/store/index.html");
    let store_html = match tokio::fs::read_to_string(&store_path).await {
        Ok(html) => Some(html),
        Err(e) => {
            warn!("Store HTML not found at {}: {e}", store_path.display());
            None
        }
    };

    // ── Build shared state ────────────────────────────────────────
    let state = Arc::new(routes::AppState {
        npu: npu_backend,
        gpu: gpu_backend,
        orders: order_manager,
        store_html,
        start_time: std::time::Instant::now(),
        stripe_secret_key,
        stripe_webhook_secret,
    });

    // ── Build HTTP server ─────────────────────────────────────────
    let app = routes::build_router(state.clone());

    // ── Signal handling for graceful shutdown ─────────────────────
    // When SIGINT or SIGTERM arrives, stop backends and trigger
    // graceful HTTP server shutdown.
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to set up SIGTERM handler");

        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received SIGINT, shutting down...");
            }
            _ = sigterm.recv() => {
                info!("Received SIGTERM, shutting down...");
            }
        }

        info!("Stopping backends...");
        state_clone.npu.stop().await;
        state_clone.gpu.stop().await;
    });

    // ── Zombie reaper ─────────────────────────────────────────────
    // The GPU backend (lemond) often dies on systems without ROCm,
    // leaving defunct children. This task reaps them.
    tokio::spawn(async move {
        loop {
            tokio::task::spawn_blocking(|| {
                #[cfg(unix)]
                unsafe {
                    loop {
                        let pid = libc::waitpid(-1, std::ptr::null_mut(), libc::WNOHANG);
                        if pid <= 0 {
                            break;
                        }
                    }
                }
            })
            .await
            .ok();
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

    // ── Serve ─────────────────────────────────────────────────────
    let addr = format!("{}:{}", args.bind, args.port)
        .parse::<std::net::SocketAddr>()
        .context("Invalid bind address")?;

    if args.bind != "127.0.0.1" {
        warn!("⚠️  WARNING: binding to non-localhost. Ensure firewall rules are in place.");
    }

    info!("Gateway listening on http://{addr}");
    info!("  GET  /v1/health     — Device and backend status");
    info!("  GET  /v1/models     — List available models");
    info!("  POST /v1/chat/completions — Route to NPU/GPU");
    info!("  POST /v1/batch/completions — Batch prefill (NPU multi-token)");

    let listener = tokio::net::TcpListener::bind(addr).await
        .context("Failed to bind TCP listener")?;

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            signal::ctrl_c().await.ok();
        })
        .await
        .context("Server error")?;

    Ok(())
}
