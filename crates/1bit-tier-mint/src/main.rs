//! `1bit-tier-mint` entrypoint.
//!
//! Listens on `HALO_TIER_LISTEN` (default `127.0.0.1:8151`, one port
//! above `1bit-stream` on 8150). Reachable only from localhost; the
//! BTCPay webhook is fronted by Caddy which terminates TLS and
//! proxies in. Patreon's webhook is cloudflared-tunnelled the same
//! way.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use tracing::info;
use tracing_subscriber::EnvFilter;

use onebit_tier_mint::{AppState, Config, build_router};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cfg = Config::from_env()?;
    let state = AppState::new(cfg);
    let app = build_router(state);

    let addr: SocketAddr = std::env::var("HALO_TIER_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:8151".to_string())
        .parse()
        .context("HALO_TIER_LISTEN must be host:port")?;

    info!(%addr, "1bit-tier-mint listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
