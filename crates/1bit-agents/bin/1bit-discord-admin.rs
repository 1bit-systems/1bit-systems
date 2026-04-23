//! 1bit-discord-admin — one-shot admin operations against the
//! 1bit.systems Discord guild. Complements the bot runtime with
//! operator-driven actions that don't belong in the gateway loop:
//!
//!   * `event`   — create a guild scheduled event (training ship,
//!                 launch day, AMA). POST /guilds/{id}/scheduled-events.
//!   * `sticker` — upload a server sticker from a local PNG / APNG /
//!                 Lottie JSON. POST /guilds/{id}/stickers (multipart).
//!   * `stage`   — start a stage instance on an existing stage
//!                 channel. POST /stage-instances.
//!
//! All subcommands share env var auth:
//!   DISCORD_BOT_TOKEN  required — halo's bot token, MANAGE_EVENTS /
//!                      MANAGE_GUILD_EXPRESSIONS / MANAGE_CHANNELS
//!                      depending on subcommand
//!   GUILD_ID           required — target guild id
//!
//! Output is a single line on stdout with the created resource id so
//! the operator can chain (e.g. feed the event id into a notification
//! pipeline). Failures return non-zero with the Discord error body on
//! stderr.

use anyhow::{Context, Result, bail};
use base64::Engine;
use clap::{Parser, Subcommand};
use reqwest::{Client, multipart};
use serde_json::json;
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Parser)]
#[command(
    name = "1bit-discord-admin",
    about = "One-shot admin ops for 1bit.systems Discord"
)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a guild scheduled event (training ship, launch, AMA).
    Event {
        /// Event display name (2-100 chars).
        #[arg(long)]
        name: String,
        /// ISO-8601 start time, e.g. 2026-04-25T18:00:00Z.
        #[arg(long)]
        start: String,
        /// ISO-8601 end time (required for EXTERNAL events).
        #[arg(long)]
        end: Option<String>,
        /// Channel id the event lives in (voice / stage). Required for
        /// VOICE / STAGE_INSTANCE entity types; omit with --location
        /// for EXTERNAL.
        #[arg(long)]
        channel_id: Option<u64>,
        /// External location string (in lieu of a channel id).
        #[arg(long)]
        location: Option<String>,
        /// Optional long description.
        #[arg(long)]
        description: Option<String>,
        /// Optional PNG cover image for the event card.
        #[arg(long)]
        cover: Option<PathBuf>,
    },

    /// Upload a guild sticker (PNG / APNG / Lottie JSON).
    Sticker {
        /// Sticker display name (2-30 chars).
        #[arg(long)]
        name: String,
        /// Short description (2-100 chars).
        #[arg(long)]
        description: String,
        /// Comma-separated emoji tags (e.g. "halo,1bit,training").
        #[arg(long)]
        tags: String,
        /// Path to the sticker file. <= 512 KB for PNG/APNG.
        #[arg(long)]
        file: PathBuf,
    },

    /// Start a stage instance on an existing stage channel (AMA).
    Stage {
        /// Stage channel id.
        #[arg(long)]
        channel_id: u64,
        /// Topic visible in the stage header (1-120 chars).
        #[arg(long)]
        topic: String,
        /// Privacy level: 1 = public (default), 2 = guild-only.
        #[arg(long, default_value_t = 1)]
        privacy: u8,
    },
}

fn encode_data_uri(path: &PathBuf) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path:?}"))?;
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("apng") => "image/apng",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        other => bail!("unsupported image extension: {other:?}"),
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
    let token = std::env::var("DISCORD_BOT_TOKEN").context("DISCORD_BOT_TOKEN must be set")?;
    let guild_id: u64 = std::env::var("GUILD_ID")
        .context("GUILD_ID must be set")?
        .parse()
        .context("GUILD_ID must be a u64")?;

    let client = Client::new();
    let auth = format!("Bot {token}");

    match args.cmd {
        Cmd::Event {
            name,
            start,
            end,
            channel_id,
            location,
            description,
            cover,
        } => {
            create_event(
                &client,
                &auth,
                guild_id,
                &name,
                &start,
                end.as_deref(),
                channel_id,
                location.as_deref(),
                description.as_deref(),
                cover.as_ref(),
            )
            .await
        }
        Cmd::Sticker {
            name,
            description,
            tags,
            file,
        } => create_sticker(&client, &auth, guild_id, &name, &description, &tags, &file).await,
        Cmd::Stage {
            channel_id,
            topic,
            privacy,
        } => create_stage_instance(&client, &auth, channel_id, &topic, privacy).await,
    }
}

async fn create_event(
    client: &Client,
    auth: &str,
    guild_id: u64,
    name: &str,
    start: &str,
    end: Option<&str>,
    channel_id: Option<u64>,
    location: Option<&str>,
    description: Option<&str>,
    cover: Option<&PathBuf>,
) -> Result<()> {
    // Entity type: 1 = STAGE_INSTANCE (channel), 2 = VOICE (channel),
    // 3 = EXTERNAL (location string + end time required). We autodetect
    // from the supplied flags.
    let (entity_type, metadata) = match (channel_id, location) {
        (Some(_), None) => (2u8, None), // VOICE by default; stage variant can be added later
        (None, Some(loc)) => (3u8, Some(json!({"location": loc}))),
        (None, None) => bail!("either --channel-id or --location must be set"),
        (Some(_), Some(_)) => bail!("--channel-id and --location are mutually exclusive"),
    };

    if entity_type == 3 && end.is_none() {
        bail!("EXTERNAL events (--location set) require --end");
    }

    let mut payload = serde_json::Map::new();
    payload.insert("name".to_string(), json!(name));
    payload.insert("scheduled_start_time".to_string(), json!(start));
    payload.insert("privacy_level".to_string(), json!(2)); // GUILD_ONLY
    payload.insert("entity_type".to_string(), json!(entity_type));
    if let Some(e) = end {
        payload.insert("scheduled_end_time".to_string(), json!(e));
    }
    if let Some(cid) = channel_id {
        payload.insert("channel_id".to_string(), json!(cid.to_string()));
    }
    if let Some(meta) = metadata {
        payload.insert("entity_metadata".to_string(), meta);
    }
    if let Some(d) = description {
        payload.insert("description".to_string(), json!(d));
    }
    if let Some(c) = cover {
        payload.insert("image".to_string(), json!(encode_data_uri(c)?));
    }

    let url = format!("https://discord.com/api/v10/guilds/{guild_id}/scheduled-events");
    let resp = client
        .post(&url)
        .header("Authorization", auth)
        .json(&serde_json::Value::Object(payload))
        .send()
        .await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Discord rejected event create ({status}): {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
    let id = parsed.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    info!(event_id = id, name, "scheduled event created");
    println!("OK — event {id} created");
    Ok(())
}

async fn create_sticker(
    client: &Client,
    auth: &str,
    guild_id: u64,
    name: &str,
    description: &str,
    tags: &str,
    file: &PathBuf,
) -> Result<()> {
    // Discord sticker upload uses multipart/form-data: `name`,
    // `description`, `tags` as text parts and `file` as a file part.
    let file_bytes = std::fs::read(file).with_context(|| format!("read {file:?}"))?;
    let filename = file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("sticker")
        .to_string();
    let mime = match file.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("apng") => "image/apng",
        Some("json") => "application/json",
        other => bail!("unsupported sticker format: {other:?} (want png / apng / json)"),
    };

    let form = multipart::Form::new()
        .text("name", name.to_string())
        .text("description", description.to_string())
        .text("tags", tags.to_string())
        .part(
            "file",
            multipart::Part::bytes(file_bytes)
                .file_name(filename)
                .mime_str(mime)?,
        );

    let url = format!("https://discord.com/api/v10/guilds/{guild_id}/stickers");
    let resp = client
        .post(&url)
        .header("Authorization", auth)
        .multipart(form)
        .send()
        .await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Discord rejected sticker upload ({status}): {body}");
    }
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
    let id = parsed.get("id").and_then(|v| v.as_str()).unwrap_or("?");
    info!(sticker_id = id, name, "sticker uploaded");
    println!("OK — sticker {id} uploaded");
    Ok(())
}

async fn create_stage_instance(
    client: &Client,
    auth: &str,
    channel_id: u64,
    topic: &str,
    privacy: u8,
) -> Result<()> {
    let payload = json!({
        "channel_id": channel_id.to_string(),
        "topic": topic,
        "privacy_level": privacy,
    });
    let url = "https://discord.com/api/v10/stage-instances";
    let resp = client
        .post(url)
        .header("Authorization", auth)
        .json(&payload)
        .send()
        .await?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("Discord rejected stage-instance create ({status}): {body}");
    }
    warn!(
        "stage instance created — remember to join the voice channel with a bot/user account to actually speak"
    );
    info!(channel_id, topic, "stage instance started");
    println!("OK — stage instance on channel {channel_id}");
    Ok(())
}
