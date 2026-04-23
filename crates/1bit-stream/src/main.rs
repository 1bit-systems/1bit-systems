//! 1bit-stream binary — hosts `.1bl` catalogs over HTTP.
//!
//! Env vars (override CLI defaults):
//!   HALO_STREAM_LISTEN       — bind address (default 127.0.0.1:8150)
//!   HALO_STREAM_CATALOG_DIR  — directory scanned for *.1bl
//!                              (default ~/.local/share/1bit/catalogs)
//!   HALO_STREAM_JWT_SECRET   — HS256 secret for the lossless gate
//!   HALO_STREAM_ADMIN_BEARER — required header for POST /internal/*
//!
//! The reindex happens once on boot; subsequent refreshes come via
//! `POST /internal/reindex`. No filesystem watcher — the publisher
//! controls when catalogs flip live.

use anyhow::Result;
use clap::Parser;
use onebit_stream::{AppState, AuthConfig, build};
use std::net::SocketAddr;
use std::path::PathBuf;
use tower_http::trace::TraceLayer;

#[derive(Parser, Debug)]
#[command(name = "1bit-stream", about, version)]
struct Cli {
    /// Listen address. Env: HALO_STREAM_LISTEN
    #[arg(long, env = "HALO_STREAM_LISTEN", default_value = "127.0.0.1:8150")]
    listen: SocketAddr,

    /// Directory scanned for `*.1bl`. Env: HALO_STREAM_CATALOG_DIR
    #[arg(long, env = "HALO_STREAM_CATALOG_DIR")]
    catalog_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_stream=info,tower_http=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let catalog_dir = cli.catalog_dir.unwrap_or_else(default_catalog_dir);
    std::fs::create_dir_all(&catalog_dir).ok();

    let auth = AuthConfig::from_env();
    let state = AppState::new(catalog_dir.clone(), auth);
    let (count, errs) = state.reindex().await;
    tracing::info!(
        catalog_dir = %catalog_dir.display(),
        loaded = count,
        errors = errs.len(),
        "initial reindex complete",
    );
    for (p, e) in &errs {
        tracing::warn!(path = %p, error = %e, "catalog failed to parse");
    }

    let app = build(state).layer(TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(cli.listen).await?;
    tracing::info!(listen = %cli.listen, "1bit-stream listening");
    axum::serve(listener, app).await?;
    Ok(())
}

fn default_catalog_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("1bit")
        .join("catalogs")
}
