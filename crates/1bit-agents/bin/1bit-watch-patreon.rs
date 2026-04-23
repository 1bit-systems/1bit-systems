//! 1bit-watch-patreon — one-shot Patreon poller for 1bit-agents.
//!
//! Walks `GET /api/oauth2/v2/campaigns/{id}/members`, paginating through
//! every page via the `meta.pagination.cursors.next` token, diffs the
//! result against a local snapshot, and for each newly-active patron:
//!
//!   1. Routes a compact JSON payload to [`Quartermaster`] via the shared
//!      [`Registry`] — Quartermaster drafts a short thank-you note via
//!      the LLM-backed specialist path.
//!   2. Posts Quartermaster's reply to Discord `#announcements` via
//!      echo's bot token (if `ECHO_BOT_TOKEN` is set).
//!
//! Rule A: no Python at runtime. Rule D: edition 2024, Rust 1.86.
//!
//! Cadence is systemd-timer driven (see
//! `strixhalo/systemd/1bit-halo-patreon.timer`). The binary is
//! deliberately one-shot: poll, diff, post, write snapshot, exit.
//!
//! Env:
//!   PATREON_ACCESS_TOKEN            required — no token => log info + exit 0
//!   HALO_PATREON_CAMPAIGN_ID        default "15895969" (memory project_patreon_live.md)
//!   HALO_PATREON_STATE_FILE         default ~/.local/state/1bit-halo/patreon-members.json
//!   ECHO_BOT_TOKEN                  optional — enables Discord post
//!   HALO_DISCORD_ANNOUNCE_CHANNEL   channel id for #announcements (numeric)
//!   RUST_LOG                        default onebit_watch_patreon=info
//!
//! First-run: if the state file does not exist, we seed it with the
//! current member list and SKIP the Discord announcement — we don't
//! want to re-announce every pre-existing patron on day one. Subsequent
//! runs diff normally.

use std::time::Duration;

use anyhow::{Context, Result};
use onebit_agents::Registry;
use onebit_agents::watch::patreon::{
    DEFAULT_CAMPAIGN_ID, Member, Snapshot, diff_new_patrons, parse_members_page, read_snapshot,
    state_path_from_env, write_snapshot,
};
use serde_json::json;
use serenity::all::{ChannelId, Http};
use tracing::{info, warn};

const PATREON_API_BASE: &str = "https://www.patreon.com/api/oauth2/v2";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("onebit_watch_patreon=info")
            }),
        )
        .with_target(false)
        .init();

    // Token gate. Missing token => info + exit 0. Same shape as
    // 1bit-watch-github; the systemd timer fires regardless, so we
    // must no-op cleanly when the operator hasn't provisioned a PAT.
    let token = match std::env::var("PATREON_ACCESS_TOKEN") {
        Ok(t) if !t.trim().is_empty() => t,
        _ => {
            info!("no PATREON_ACCESS_TOKEN set, skipping poll");
            return Ok(());
        }
    };

    let campaign_id = std::env::var("HALO_PATREON_CAMPAIGN_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_CAMPAIGN_ID.to_string());
    let state_path = state_path_from_env();

    info!(
        campaign = campaign_id.as_str(),
        state_file = %state_path.display(),
        "1bit-watch-patreon starting one-shot poll"
    );

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .user_agent(concat!("1bit-watch-patreon/", env!("CARGO_PKG_VERSION")))
        .build()?;

    let current_members = fetch_all_members(&http, &token, &campaign_id).await?;
    info!(count = current_members.len(), "fetched current members");

    let prior = read_snapshot(&state_path)
        .with_context(|| format!("reading prior snapshot at {}", state_path.display()))?;
    let diff = diff_new_patrons(prior.as_ref(), &current_members);

    if diff.first_run {
        // Seed-and-skip: write the snapshot but do NOT announce. Every
        // active patron would otherwise get a welcome post on the
        // day we first run this unit, which is the wrong call.
        warn!(
            new_actives = diff.new_patrons.len(),
            "first run detected (no prior snapshot) — seeding state file, skipping announcements"
        );
    } else if diff.new_patrons.is_empty() {
        info!("no new active patrons since last poll");
    } else {
        info!(
            count = diff.new_patrons.len(),
            "new active patrons detected"
        );
        // ECHO_BOT_TOKEN gate: if unset, we log the new-patron list and
        // keep going. The dispatch-to-Quartermaster step is worth
        // running either way — it exercises the LLM pipeline and the
        // reply ends up in the journal even without Discord.
        let discord = build_discord_client();
        let registry = Registry::default_stubs();
        for patron in &diff.new_patrons {
            announce_patron(&registry, discord.as_ref(), patron).await;
        }
    }

    // Always write the current snapshot. Safe against crashes: the
    // write is atomic (tmp + rename), so a poll that blew up mid-
    // announce won't re-announce the same patron on the next run as
    // long as we reached this line.
    let snap = Snapshot::new(current_members);
    write_snapshot(&state_path, &snap)
        .with_context(|| format!("writing snapshot to {}", state_path.display()))?;

    info!("1bit-watch-patreon pass complete");
    Ok(())
}

/// Fetch every page of `campaigns/{id}/members` and flatten into a
/// single `Vec<Member>`. Paginates via the opaque cursor from
/// `meta.pagination.cursors.next`, falling back to `links.next` parsing
/// (see `parse_members_page`) for older response shapes.
///
/// Guards against infinite pagination loops via a hard page cap; 200
/// pages × ~100 rows is 20k patrons, which is well past any realistic
/// campaign.
async fn fetch_all_members(
    http: &reqwest::Client,
    token: &str,
    campaign_id: &str,
) -> Result<Vec<Member>> {
    const PAGE_CAP: usize = 200;
    // Field mask matches the crate-level doc — small enough to keep
    // privacy-sensitive columns out of the snapshot unless we change
    // the mask deliberately.
    const FIELDS: &str =
        "fields%5Bmember%5D=email,full_name,patron_status,currently_entitled_amount_cents";
    let mut out: Vec<Member> = Vec::new();
    let mut cursor: Option<String> = None;

    for page_idx in 0..PAGE_CAP {
        let url = match &cursor {
            Some(c) => format!(
                "{PATREON_API_BASE}/campaigns/{campaign_id}/members?{FIELDS}&page%5Bcursor%5D={}",
                percent_encode(c)
            ),
            None => format!("{PATREON_API_BASE}/campaigns/{campaign_id}/members?{FIELDS}"),
        };

        let resp = http
            .get(&url)
            .bearer_auth(token)
            .send()
            .await
            .context("patreon GET /members")?;
        let status = resp.status();
        let text = resp.text().await.context("reading members body")?;
        if !status.is_success() {
            anyhow::bail!("patreon api returned {status}: {text}");
        }
        let page: serde_json::Value =
            serde_json::from_str(&text).context("parsing members page as JSON")?;
        let (members, next) = parse_members_page(&page)?;

        info!(
            page = page_idx,
            members = members.len(),
            has_next = next.is_some(),
            "fetched members page"
        );
        out.extend(members);

        match next {
            Some(c) => cursor = Some(c),
            None => return Ok(out),
        }
    }

    warn!(
        cap = PAGE_CAP,
        "hit pagination cap while walking /members — returning what we have"
    );
    Ok(out)
}

/// Minimal percent-encoder for cursor tokens. Mirrors the helper in
/// 1bit-mcp-patreon's client — duplicated rather than shared to keep
/// the two crates independent (the MCP shim might move or diverge).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '-' | '_' | '.' | '~' | 'a'..='z' | 'A'..='Z' | '0'..='9' => out.push(c),
            _ => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    out.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    out
}

/// Build a Discord `Http` client if `ECHO_BOT_TOKEN` is set. Returns
/// `None` otherwise — caller treats that as "skip the post, log
/// instead".
fn build_discord_client() -> Option<Http> {
    match std::env::var("ECHO_BOT_TOKEN") {
        Ok(t) if !t.trim().is_empty() => {
            info!("ECHO_BOT_TOKEN configured — will post welcome to Discord");
            Some(Http::new(&format!("Bot {t}")))
        }
        _ => {
            warn!("ECHO_BOT_TOKEN unset — new-patron events will log but not post to Discord");
            None
        }
    }
}

/// Dispatch one patron to Quartermaster and, on success, post the
/// reply to the announcement channel. Every branch logs so operators
/// can see the pipeline end-to-end in the journal even if ECHO_BOT_TOKEN
/// is absent.
async fn announce_patron(registry: &Registry, discord: Option<&Http>, patron: &Member) {
    let name_str = patron
        .full_name
        .clone()
        .unwrap_or_else(|| patron.id.clone());
    info!(
        id = patron.id.as_str(),
        name = name_str.as_str(),
        cents = patron.currently_entitled_amount_cents.unwrap_or(0),
        "dispatching new patron to Quartermaster"
    );

    let payload = json!({
        "event": "patreon.new_patron",
        "patron": {
            "id": patron.id,
            "full_name": patron.full_name,
            "email": patron.email,
            "patron_status": patron.patron_status,
            "currently_entitled_amount_cents": patron.currently_entitled_amount_cents,
        },
    });

    let reply = match registry.dispatch("quartermaster", payload).await {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "Quartermaster dispatch failed — skipping Discord post");
            return;
        }
    };

    let text = reply
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if text.is_empty() {
        warn!("Quartermaster returned empty text — nothing to post");
        return;
    }

    let Some(http) = discord else {
        info!(welcome = %text, "welcome drafted (no ECHO_BOT_TOKEN, skipping post)");
        return;
    };

    let Some(channel_id) = announce_channel_id() else {
        warn!(
            "HALO_DISCORD_ANNOUNCE_CHANNEL unset or invalid — drafted welcome but no channel to post to: {text}"
        );
        return;
    };

    match ChannelId::new(channel_id).say(http, &text).await {
        Ok(_) => info!(
            channel = channel_id,
            "posted welcome to Discord #announcements"
        ),
        Err(e) => warn!(error = %e, "failed to post welcome to Discord"),
    }
}

/// Parse `HALO_DISCORD_ANNOUNCE_CHANNEL` as a numeric Discord channel
/// id. Non-numeric or empty values are ignored (with a warn one level
/// up). The Discord API takes a `u64` snowflake, not a human name.
fn announce_channel_id() -> Option<u64> {
    std::env::var("HALO_DISCORD_ANNOUNCE_CHANNEL")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_encode_preserves_unreserved() {
        assert_eq!(percent_encode("eyJhIjoiYiJ9"), "eyJhIjoiYiJ9");
        assert_eq!(percent_encode("a=b"), "a%3Db");
    }

    #[test]
    fn announce_channel_rejects_non_numeric() {
        // Saved + restored so the test doesn't stomp on operator env.
        let prev = std::env::var("HALO_DISCORD_ANNOUNCE_CHANNEL").ok();
        // SAFETY: single-threaded test mutating one env var we own.
        unsafe {
            std::env::set_var("HALO_DISCORD_ANNOUNCE_CHANNEL", "announcements");
        }
        assert!(announce_channel_id().is_none());
        unsafe {
            std::env::set_var("HALO_DISCORD_ANNOUNCE_CHANNEL", "1234567890");
        }
        assert_eq!(announce_channel_id(), Some(1234567890));
        match prev {
            Some(p) => unsafe { std::env::set_var("HALO_DISCORD_ANNOUNCE_CHANNEL", p) },
            None => unsafe { std::env::remove_var("HALO_DISCORD_ANNOUNCE_CHANNEL") },
        }
    }
}
