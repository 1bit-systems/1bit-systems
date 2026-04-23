//! 1bit-helpdesk-setup — one-shot admin tool that shapes a Discord forum
//! channel into the canonical 1bit help-desk layout used by
//! `1bit-watch-discord`.
//!
//! What it does, given a channel that is ALREADY a forum channel
//! (Discord's API cannot convert a text channel to a forum — that step
//! must be done by hand in the Discord settings UI, or the forum must be
//! created fresh):
//!
//!   * sets the channel topic to the standard help-desk guidelines
//!   * installs the canonical tag set — bug, feature, question, pending,
//!     resolved, escalated — each with a matching emoji
//!   * sets default auto-archive to 1 week and a light slowmode
//!
//! Env:
//!   DISCORD_BOT_TOKEN  required — the bot token that has MANAGE_CHANNELS
//!                      on the target guild
//!   CHANNEL_ID         required — the forum channel id to configure
//!
//! Usage:
//!   DISCORD_BOT_TOKEN=… CHANNEL_ID=… 1bit-helpdesk-setup
//!
//! The program is idempotent: running it a second time rewrites the tags
//! and topic to the canonical state without creating duplicates.

use anyhow::{Context, Result, bail};
use serenity::all::{
    AutoArchiveDuration, ChannelId, ChannelType, CreateForumTag, EditChannel, Http,
};
use std::sync::Arc;
use tracing::{error, info};

/// Canonical tag set — lowercased name + a unicode emoji. Three
/// dimensions:
///   * type       — what kind of post (troubleshooting / feature / inquiry)
///   * state      — where the post is in its lifecycle (pending /
///                  resolved / escalated)
///   * severity   — DEFCON-style urgency (1 = worst, 5 = trivial);
///                  halo sets this from keyword heuristics on post
///                  creation and humans can adjust as triage progresses
/// Order in this list is also the order they appear in the forum UI.
const CANONICAL_TAGS: &[(&str, &str)] = &[
    // Type
    ("troubleshooting", "🐛"),
    ("feature",         "✨"),
    ("inquiry",         "❓"),
    // State
    ("pending",         "🕘"),
    ("resolved",        "✅"),
    ("escalated",       "⬆️"),
    // Severity (lower number = worse)
    ("defcon-1",        "🟥"),
    ("defcon-2",        "🟧"),
    ("defcon-3",        "🟨"),
    ("defcon-4",        "🟩"),
    ("defcon-5",        "⬜"),
];

const HELP_DESK_TOPIC: &str = "Help-desk routed from across the 1bit.systems server. \
Every post is auto-created by halo when someone reports a bug, requests a \
feature, or asks a question elsewhere. Tags are set by the bot; a human \
flips `pending` → `resolved` / `escalated` once triaged. Don't post \
directly — ask in your home channel and halo will route here.";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let token = std::env::var("DISCORD_BOT_TOKEN")
        .context("DISCORD_BOT_TOKEN must be set (needs MANAGE_CHANNELS on the guild)")?;
    let channel_id: u64 = std::env::var("CHANNEL_ID")
        .context("CHANNEL_ID must be set to the forum channel id")?
        .parse()
        .context("CHANNEL_ID must be a u64")?;

    let http = Arc::new(Http::new(&token));
    let channel_id = ChannelId::new(channel_id);

    // Confirm the channel is a forum. If it isn't, tell the operator how
    // to fix it instead of silently running against the wrong target.
    let channel = channel_id
        .to_channel(http.as_ref())
        .await
        .context("failed to fetch channel metadata")?;
    let guild_channel = match channel {
        serenity::all::Channel::Guild(gc) => gc,
        other => bail!("channel {channel_id} is not a guild channel: {other:?}"),
    };

    if !matches!(guild_channel.kind, ChannelType::Forum) {
        error!(
            channel_id = %channel_id,
            kind = ?guild_channel.kind,
            "channel is not a Forum — convert it in Discord UI first \
             (channel settings → overview → Forum), or delete + recreate \
             as Forum. this tool only configures existing forum channels."
        );
        bail!("channel {channel_id} is kind {:?}, not Forum", guild_channel.kind);
    }

    info!(
        channel_id = %channel_id,
        name = %guild_channel.name,
        "configuring help-desk forum channel"
    );

    // Build the canonical tag list. Discord dedupes tags by name on edit,
    // so re-running this tool reshapes the set to exactly CANONICAL_TAGS
    // (existing matching tags retain their id; missing ones get created;
    // tags not in the canonical list are removed).
    let tags: Vec<CreateForumTag> = CANONICAL_TAGS
        .iter()
        .map(|(name, emoji)| {
            CreateForumTag::new(*name).emoji(serenity::all::ReactionType::Unicode(
                (*emoji).to_string(),
            ))
        })
        .collect();

    let edit = EditChannel::new()
        .topic(HELP_DESK_TOPIC)
        .rate_limit_per_user(5)
        .default_auto_archive_duration(AutoArchiveDuration::OneWeek)
        .available_tags(tags);

    channel_id
        .edit(http.as_ref(), edit)
        .await
        .context("failed to apply help-desk configuration")?;

    info!(
        channel_id = %channel_id,
        tag_count = CANONICAL_TAGS.len(),
        "help-desk configuration applied"
    );
    println!(
        "OK — configured forum channel {channel_id} with {n} canonical tags",
        n = CANONICAL_TAGS.len()
    );
    Ok(())
}
