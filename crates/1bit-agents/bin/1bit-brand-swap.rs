//! 1bit-brand-swap — rotate the Discord server icon / splash / banner on
//! milestones (Patreon threshold hit, training run shipped, launch day).
//!
//! Discord only accepts image payloads as base64 data-URIs on PATCH
//! /guilds/{guild.id}. This CLI handles the encoding + API call so the
//! operator (or a systemd timer) can flip assets without clicking
//! through the Discord UI.
//!
//! Env:
//!   DISCORD_BOT_TOKEN   required — halo's token, with MANAGE_GUILD perm
//!   GUILD_ID            required — target guild id
//!
//! Usage:
//!   1bit-brand-swap --icon   path/to/icon.gif
//!   1bit-brand-swap --splash path/to/splash.png
//!   1bit-brand-swap --banner path/to/banner.png
//!   1bit-brand-swap --name   "1bit.systems — training run 5"
//!
//! Any combination of the flags above is allowed in a single call; all
//! provided fields are applied in one PATCH. Exits non-zero if Discord
//! rejects the payload (e.g. animated icon without sufficient boost
//! level).

use anyhow::{Context, Result};
use base64::Engine;
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser)]
#[command(
    name = "1bit-brand-swap",
    about = "Swap Discord server icon / splash / banner / name on milestones"
)]
struct Args {
    /// Path to a PNG / JPEG / GIF icon. 512x512 recommended; animated
    /// GIFs require server boost level 1 or higher.
    #[arg(long)]
    icon: Option<PathBuf>,

    /// Path to a PNG invite splash. 960x540 recommended.
    #[arg(long)]
    splash: Option<PathBuf>,

    /// Path to a PNG server banner (top of member list). 960x540.
    #[arg(long)]
    banner: Option<PathBuf>,

    /// New guild name. Leave unset to keep the current one.
    #[arg(long)]
    name: Option<String>,

    /// Audit-log reason so the change is traceable in Discord's
    /// server-settings → audit log pane.
    #[arg(long, default_value = "1bit-brand-swap: milestone asset rotation")]
    reason: String,
}

fn encode_data_uri(path: &PathBuf) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        other => anyhow::bail!("unsupported image extension: {other:?}"),
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let token = std::env::var("DISCORD_BOT_TOKEN")
        .context("DISCORD_BOT_TOKEN must be set (halo's token with MANAGE_GUILD)")?;
    let guild_id: u64 = std::env::var("GUILD_ID")
        .context("GUILD_ID must be set to the target guild id")?
        .parse()
        .context("GUILD_ID must be a u64")?;

    // Build the PATCH payload with whichever fields the operator asked
    // for. Unset flags are simply omitted so Discord leaves them alone.
    let mut payload = serde_json::Map::new();
    if let Some(p) = &args.icon {
        payload.insert("icon".to_string(), json!(encode_data_uri(p)?));
        info!(path = %p.display(), "icon queued");
    }
    if let Some(p) = &args.splash {
        payload.insert("splash".to_string(), json!(encode_data_uri(p)?));
        info!(path = %p.display(), "splash queued");
    }
    if let Some(p) = &args.banner {
        payload.insert("banner".to_string(), json!(encode_data_uri(p)?));
        info!(path = %p.display(), "banner queued");
    }
    if let Some(n) = &args.name {
        payload.insert("name".to_string(), json!(n));
        info!(name = %n, "name queued");
    }
    if payload.is_empty() {
        warn!("nothing to do — pass at least one of --icon --splash --banner --name");
        return Ok(());
    }

    let url = format!("https://discord.com/api/v10/guilds/{guild_id}");
    let client = reqwest::Client::new();
    let resp = client
        .patch(&url)
        .header("Authorization", format!("Bot {token}"))
        .header("X-Audit-Log-Reason", args.reason)
        .json(&serde_json::Value::Object(payload))
        .send()
        .await
        .context("PATCH guild request failed")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("Discord rejected guild edit ({status}): {body}");
    }
    info!(status = %status, "guild edit applied");
    println!("OK — guild {guild_id} edited ({status})");
    Ok(())
}
