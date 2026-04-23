//! 1bit-watch-discord — Discord gateway client that keeps the 17 specialist
//! registry aware of channel activity.
//!
//! Design:
//!   * Lurker by default. We listen on a whitelist of channels
//!     (`HALO_DISCORD_CHANNELS`), classify each message, and dispatch a
//!     compact payload to the relevant specialist via `Registry::dispatch`.
//!   * NO automatic replies. We only post to a channel when directly
//!     `@mentioned` — and only for the `status` subcommand.
//!   * Gracefully degrades if `DISCORD_BOT_TOKEN` is missing: print help
//!     and exit 0. This matters because the systemd unit ships disabled,
//!     so the binary is expected to be invoked on a fresh box with no
//!     token configured.
//!
//! Rule A: no Python at runtime — serenity-rs is pure Rust.
//! Rule D: edition 2024, Rust 1.86.
//!
//! Env:
//!   DISCORD_BOT_TOKEN          required to actually connect
//!   HALO_DISCORD_CHANNELS      comma-separated channel IDs to watch
//!   HALO_SERVER_URL            default http://127.0.0.1:8180
//!   HALO_LANDING_URL           default http://127.0.0.1:8190
//!   RUST_LOG                   default onebit_watch_discord=info

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use onebit_agents::watch::{
    Classification, HELP_TEXT, classify, is_direct_mention, parse_channel_whitelist, strip_mention,
};
use onebit_agents::{Name, Registry};
use serde_json::json;
use serenity::Client;
use serenity::all::{
    AutoArchiveDuration, ChannelType, Context, CreateForumPost, CreateMessage, CreateThread,
    EventHandler, ForumTag, ForumTagId, GatewayIntents, Http, Message, Ready,
};
use serenity::async_trait;
use std::collections::HashMap;
use tokio::sync::OnceCell;
use tracing::{error, info, warn};

const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8180";
const DEFAULT_LANDING_URL: &str = "http://127.0.0.1:8190";

/// Handler shared across every gateway event. `Arc`'d internally by serenity.
///
/// Two Discord tokens live here:
///   * halo's token (passed to the serenity `Client` in `main`) is the
///     gateway listener — lurker only, never posts.
///   * echo's token lives in `echo_http`. Every outbound post
///     (specialist reply, thread creation, `@halo status`) routes
///     through this client so the halo identity stays silent.
///
/// If `ECHO_BOT_TOKEN` is unset, `echo_http` is `None` and outbound
/// posting is skipped with a warning — preserves the lurker-only
/// behaviour as the safe fallback.
struct Handler {
    registry: Arc<Registry>,
    channels: HashSet<u64>,
    http: reqwest::Client,
    echo_http: Option<Arc<Http>>,
    server_url: String,
    landing_url: String,
    /// If set, non-Chat messages in OTHER channels get auto-routed into a
    /// new thread here, and Echo replies only in that thread. Empty-string /
    /// unset disables routing and reverts to reply-in-place behavior.
    help_desk_channel_id: Option<u64>,
    /// Lazy-fetched metadata about the help-desk channel — channel kind
    /// (forum vs text) and the forum tag map keyed by lowercased name.
    /// Populated the first time `route_to_help_desk` runs and reused
    /// across subsequent routes.
    help_desk_meta: OnceCell<HelpDeskMeta>,
}

/// Snapshot of the help-desk channel's shape.
#[derive(Debug, Clone)]
struct HelpDeskMeta {
    is_forum: bool,
    /// Lowercased tag name → tag id. Empty when the channel isn't a forum
    /// or defines no tags.
    tags_by_name: HashMap<String, ForumTagId>,
}

impl HelpDeskMeta {
    fn empty() -> Self {
        Self { is_forum: false, tags_by_name: HashMap::new() }
    }

    fn from_tags(is_forum: bool, tags: &[ForumTag]) -> Self {
        let mut tags_by_name = HashMap::new();
        for t in tags {
            tags_by_name.insert(t.name.to_lowercase(), t.id);
        }
        Self { is_forum, tags_by_name }
    }

    /// Pick a forum-tag id for the given classification plus the implicit
    /// "pending" state. Unknown tags are skipped silently so servers that
    /// haven't configured the exact names still get routed correctly.
    fn tags_for(&self, class: Classification) -> Vec<ForumTagId> {
        let mut out = Vec::new();
        let class_key = match class {
            Classification::BugReport => "bug",
            Classification::FeatureRequest => "feature",
            Classification::Question => "question",
            Classification::Chat => return out,
        };
        if let Some(id) = self.tags_by_name.get(class_key) {
            out.push(*id);
        }
        if let Some(id) = self.tags_by_name.get("pending") {
            out.push(*id);
        }
        out
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!(
            bot = %ready.user.name,
            channels = self.channels.len(),
            "1bit-watch-discord connected"
        );
    }

    async fn message(&self, ctx: Context, msg: Message) {
        // Never react to our own messages or other bots.
        if msg.author.bot {
            return;
        }

        // Whitelist gate.
        //   Empty whitelist  = observe ALL channels the bot is permissioned for.
        //   Non-empty filter = only these channels.
        let channel_id = msg.channel_id.get();
        if !self.channels.is_empty() && !self.channels.contains(&channel_id) {
            return;
        }

        // Lightweight classification + dispatch. We do this even for
        // mentioned messages so the specialist sees the context.
        let class = classify(&msg.content);
        let specialist = class.specialist();

        info!(
            author = %msg.author.name,
            channel = channel_id,
            class = %class,
            specialist = specialist.as_str(),
            "dispatching to specialist"
        );

        let req = json!({
            "source": "discord",
            "channel_id": channel_id,
            "author": msg.author.name,
            "classification": class.as_str(),
            "content": msg.content,
        });
        let dispatch_result = self.registry.dispatch(specialist.as_str(), req).await;

        // Cross-channel help-desk routing:
        //   * If HALO_HELP_DESK_CHANNEL_ID is set AND the message is outside
        //     that channel AND classification is non-Chat, Echo creates a
        //     new thread INSIDE the help-desk and posts the specialist's
        //     reply there. We also drop a short breadcrumb in the original
        //     channel linking to the new thread so the asker can follow.
        //   * If the message IS in the help-desk, current behavior stands.
        //   * If class == Chat, routing is skipped (no noise).
        let should_route = matches!(
            (self.help_desk_channel_id, class),
            (Some(hd), c) if hd != channel_id && !matches!(c, Classification::Chat)
        );

        match dispatch_result {
            Ok(resp) => {
                let text = resp
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if should_route {
                    // Post to help-desk thread even if specialist returned
                    // empty — the user asked for help and we promised a
                    // landing spot. Fallback copy covers the empty case.
                    let body = if text.is_empty() {
                        "Picked this up, a human will follow up here."
                    } else {
                        text
                    };
                    self.route_to_help_desk(&ctx, &msg, class, body).await;
                } else if !text.is_empty() {
                    self.post_response(&ctx, &msg, class, text).await;
                }
            }
            Err(e) => {
                warn!(error = %e, "specialist dispatch failed");
                if should_route {
                    // Even if dispatch failed, route the user so a human
                    // picks it up. Specialist silence must not mean silence.
                    self.route_to_help_desk(
                        &ctx,
                        &msg,
                        class,
                        "Picked this up, a human will follow up here.",
                    )
                    .await;
                }
            }
        }

        // Reply path: ONLY on direct mention. Keeps us off channels
        // otherwise. The bot ID is resolved via the cache — serenity
        // populates this from the READY payload.
        let me = ctx.cache.current_user().id.get();
        if !is_direct_mention(&msg.content, me) {
            return;
        }

        let cmd = strip_mention(&msg.content, me);
        if cmd.starts_with("status") {
            let line = self.status_line().await;
            if let Some(http) = self.echo_http.as_ref() {
                if let Err(e) = msg.channel_id.say(http.as_ref(), line).await {
                    warn!(error = %e, "echo failed to post status reply");
                }
            } else {
                warn!("@halo status requested but ECHO_BOT_TOKEN unset; silent.");
            }
        }
        // Any other `@halo-bot …` command is deliberately ignored for now.
        // We don't want accidental auto-moderation surface.
    }
}

impl Handler {
    /// Post a specialist's text reply via echo's token. BugReport-classified
    /// messages are moved into a newly-created public thread on the original
    /// message so follow-ups stay scoped to one incident. All other
    /// classifications reply in the original channel. If `ECHO_BOT_TOKEN`
    /// is unset, we warn and drop the post — halo never speaks with its
    /// own token.
    async fn post_response(&self, _ctx: &Context, msg: &Message, class: Classification, text: &str) {
        let Some(http) = self.echo_http.as_ref() else {
            warn!(
                "specialist returned text but ECHO_BOT_TOKEN is unset — halo stays silent. \
                 Configure echo's bot token to enable posting."
            );
            return;
        };

        // Specialists are prompted to draft inside ```...``` fences so the LLM
        // produces one clean chunk without editorial padding. Discord posts
        // should read as normal conversation, so strip the wrapping fence
        // before sending. Inner fenced blocks (e.g. a command example inside
        // a longer reply) are preserved — only the outer wrapper goes.
        // sanitize_reply then drops any *empty* inner fences so stubs that
        // emit ```...``` with no body don't leak empty code blocks to chat.
        let unwrapped = unwrap_outer_codeblock(text);
        let cleaned = sanitize_reply(&unwrapped);
        if cleaned.trim().is_empty() {
            warn!(
                "specialist reply had no usable content after sanitize — dropping post"
            );
            return;
        }
        let post_text = cleaned.as_str();

        if matches!(class, Classification::BugReport) {
            let name = format!(
                "troubleshoot: {}",
                truncate_for_thread_name(msg.content.as_str())
            );
            let builder =
                CreateThread::new(name).auto_archive_duration(AutoArchiveDuration::OneDay);
            match msg
                .channel_id
                .create_thread_from_message(http.as_ref(), msg.id, builder)
                .await
            {
                Ok(thread) => {
                    if let Err(e) = thread.id.say(http.as_ref(), post_text).await {
                        warn!(error = %e, thread_id = %thread.id, "echo failed to post into thread");
                    }
                    return;
                }
                Err(e) => {
                    warn!(error = %e, "failed to create troubleshoot thread, falling back to channel reply");
                }
            }
        }

        if let Err(e) = msg.channel_id.say(http.as_ref(), post_text).await {
            warn!(error = %e, "echo failed to post specialist response");
        }
    }

    /// Cross-channel help-desk routing. Two shapes, picked by channel kind:
    ///
    ///   * **Forum channel** — each help request becomes its own forum
    ///     post (the mc-help-desk layout). Classification drives the tag
    ///     (`bug` / `feature` / `question`) and we also stamp `pending`
    ///     when it exists so the triage queue is visible at a glance.
    ///   * **Text channel** — legacy seed+thread flow: a header message
    ///     is posted in the channel, a thread branches off it, and the
    ///     quote + specialist reply go inside.
    ///
    /// The channel kind is fetched once via `OnceCell` on first route and
    /// reused afterwards. All outbound posts go through echo's token; if
    /// echo is unset we log + drop so halo never accidentally speaks.
    async fn route_to_help_desk(&self, ctx: &Context, msg: &Message, class: Classification, reply: &str) {
        let Some(http) = self.echo_http.as_ref() else {
            warn!("help-desk routing requested but ECHO_BOT_TOKEN unset — dropping");
            return;
        };
        let Some(help_desk) = self.help_desk_channel_id else {
            return;
        };

        let help_ch = serenity::all::ChannelId::new(help_desk);
        let thread_name = format!(
            "help-{}-{}",
            sanitize_for_thread(&msg.author.name),
            truncate_for_thread_name(&msg.content)
        );

        // Pre-build the three shared pieces: header, quote, cleaned reply.
        let header = format!(
            "**help from {author}** ({class}) — {jump_url}",
            author = msg.author.name,
            class = class.as_str(),
            jump_url = msg.link(),
        );
        let quote_block = format!("> {}", truncate_for_quote(&msg.content));
        let clean_reply = sanitize_reply(reply);
        let body_reply: String = if clean_reply.trim().is_empty() {
            warn!(
                original_len = reply.len(),
                "specialist reply sanitized to empty — posting placeholder"
            );
            "_Specialist reply came back empty after cleanup — a human will follow up here._"
                .to_string()
        } else {
            clean_reply
        };

        // Resolve channel shape on first use. Keep the cached result even
        // if the fetch fails — empty() means "treat as text channel".
        let meta = self
            .help_desk_meta
            .get_or_init(|| async move {
                match help_ch.to_channel(&ctx.http).await {
                    Ok(serenity::all::Channel::Guild(gc)) => {
                        let is_forum = matches!(gc.kind, ChannelType::Forum);
                        let tags = &gc.available_tags;
                        info!(
                            help_desk = help_desk,
                            is_forum = is_forum,
                            tag_count = tags.len(),
                            "resolved help-desk channel metadata"
                        );
                        HelpDeskMeta::from_tags(is_forum, tags)
                    }
                    Ok(_) => {
                        warn!(help_desk = help_desk, "help-desk channel is not a guild channel");
                        HelpDeskMeta::empty()
                    }
                    Err(e) => {
                        warn!(error = %e, help_desk = help_desk, "failed to resolve help-desk channel metadata — defaulting to text-channel shape");
                        HelpDeskMeta::empty()
                    }
                }
            })
            .await;

        if meta.is_forum {
            // Forum path: one post, one starter message, classification tag.
            let starter = format!("{header}\n\n{quote_block}\n\n{body_reply}");
            let starter = truncate_for_forum_starter(&starter);
            let tags = meta.tags_for(class);
            let tag_count = tags.len();

            let post_builder = CreateForumPost::new(
                thread_name.clone(),
                CreateMessage::new().content(starter),
            )
            .auto_archive_duration(AutoArchiveDuration::OneWeek)
            .set_applied_tags(tags);

            match help_ch.create_forum_post(http.as_ref(), post_builder).await {
                Ok(post) => {
                    info!(
                        post_id = %post.id,
                        origin_channel = msg.channel_id.get(),
                        tag_count,
                        "created help-desk forum post"
                    );
                    let breadcrumb = format!(
                        "Routed to <#{post}> — follow there.",
                        post = post.id.get()
                    );
                    if let Err(e) = msg.channel_id.say(http.as_ref(), breadcrumb).await {
                        warn!(error = %e, "failed to post help-desk breadcrumb");
                    }
                }
                Err(e) => {
                    warn!(error = %e, help_desk = help_desk, "failed to create forum post");
                }
            }
            return;
        }

        // Text-channel path: header message, branched thread, quote + reply inside.
        let seed_msg = match help_ch.say(http.as_ref(), &header).await {
            Ok(m) => m,
            Err(e) => {
                warn!(error = %e, help_desk = help_desk, "failed to seed help-desk post");
                return;
            }
        };

        let thread_builder = CreateThread::new(thread_name)
            .auto_archive_duration(AutoArchiveDuration::OneWeek);
        let thread = match help_ch
            .create_thread_from_message(http.as_ref(), seed_msg.id, thread_builder)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                warn!(error = %e, "failed to spin help-desk thread — seed stays as plain post");
                return;
            }
        };

        info!(
            thread_id = %thread.id,
            origin_channel = msg.channel_id.get(),
            "spun help-desk thread"
        );

        if let Err(e) = thread.id.say(http.as_ref(), quote_block).await {
            warn!(error = %e, "failed to post quote into thread");
        }
        info!(
            thread_id = %thread.id,
            bytes = body_reply.len(),
            "posting specialist reply to help-desk thread"
        );
        if let Err(e) = thread.id.say(http.as_ref(), body_reply).await {
            warn!(error = %e, "failed to post specialist reply into thread");
        }

        let breadcrumb = format!(
            "Routed to <#{thread}> — follow there.",
            thread = thread.id.get()
        );
        if let Err(e) = msg.channel_id.say(http.as_ref(), breadcrumb).await {
            warn!(error = %e, "failed to post help-desk breadcrumb");
        }
    }

    /// Single-line status pulled from 1bit-server /v1/models and
    /// 1bit-landing /metrics. Both probes are best-effort; a missing
    /// endpoint surfaces as "?" in its slot rather than failing the whole
    /// reply.
    async fn status_line(&self) -> String {
        let models_url = format!("{}/v1/models", self.server_url);
        let metrics_url = format!("{}/metrics", self.landing_url);

        let models = self.http.get(&models_url).send().await.ok().and_then(|r| {
            if r.status().is_success() {
                Some(r)
            } else {
                None
            }
        });
        let model_count = match models {
            Some(r) => match r.json::<serde_json::Value>().await {
                Ok(v) => v
                    .get("data")
                    .and_then(|d| d.as_array())
                    .map(|a| a.len().to_string())
                    .unwrap_or_else(|| "?".into()),
                Err(_) => "?".into(),
            },
            None => "?".into(),
        };

        let metrics_ok = self
            .http
            .get(&metrics_url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        let metrics_tag = if metrics_ok { "up" } else { "down" };

        format!(
            "halo: models={} landing={} ({} specialists online)",
            model_count,
            metrics_tag,
            Name::ALL.len()
        )
    }
}

/// If the whole response is wrapped in a fenced code block (```...``` with
/// an optional language tag) return the inner text; otherwise return the
/// input verbatim. We only peel ONE layer — inner code blocks inside a
/// larger reply are kept as-is so real command examples still render as
/// code on Discord.
fn unwrap_outer_codeblock(src: &str) -> String {
    let trimmed = src.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return src.to_string();
    };
    let Some(before_close) = after_open.strip_suffix("```") else {
        return src.to_string();
    };
    // Strip the optional language tag line (e.g. "rust\n", "bash\n").
    // Only drop the first line if it has no spaces and no backticks —
    // that's the shape Discord accepts as a language hint.
    let body = match before_close.split_once('\n') {
        Some((first, rest))
            if !first.is_empty()
                && !first.contains(' ')
                && !first.contains('`')
                && first.len() <= 20 =>
        {
            rest
        }
        _ => before_close,
    };
    body.trim_matches('\n').to_string()
}

/// Strip empty/duplicate code fences and collapse blank-line runs from a
/// specialist reply. Guards against stubs that emit ``` ``` ``` … (Discord
/// renders each as an empty code block, flooding the thread). Keeps a
/// single fence intact when it contains actual content.
fn sanitize_reply(src: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut blank_run = 0;
    let mut in_fence = false;
    let mut fence_has_content = false;
    let mut fence_buf: Vec<&str> = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("```") {
            if in_fence {
                if fence_has_content {
                    out.extend(fence_buf.iter());
                    out.push(line);
                }
                fence_buf.clear();
                in_fence = false;
                fence_has_content = false;
            } else {
                fence_buf.push(line);
                in_fence = true;
                fence_has_content = false;
            }
            continue;
        }
        if in_fence {
            if !t.is_empty() {
                fence_has_content = true;
            }
            fence_buf.push(line);
            continue;
        }
        if t.is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push(line);
            }
        } else {
            blank_run = 0;
            out.push(line);
        }
    }
    // Unterminated fence with content → keep it; without content → drop.
    if in_fence && fence_has_content {
        out.extend(fence_buf.iter());
    }
    out.join("\n").trim().to_string()
}

/// Strip non-alphanumeric from a username so it fits cleanly in a thread
/// name. Discord thread names cap at 100 chars and prefer ASCII.
fn sanitize_for_thread(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let clean = clean.trim_matches('-');
    if clean.len() <= 24 {
        clean.to_string()
    } else {
        clean.chars().take(24).collect()
    }
}

/// Clip a full forum-post starter to Discord's 2000-char message cap.
/// We prefer to keep the header + quote intact and trim the reply tail
/// rather than drop the post entirely. The 4-char ellipsis reserve keeps
/// the limit comfortable even after Discord's own encoding overhead.
fn truncate_for_forum_starter(src: &str) -> String {
    const CAP: usize = 1996;
    if src.chars().count() <= CAP {
        return src.to_string();
    }
    src.chars().take(CAP).chain("…".chars()).collect()
}

/// Truncate a message to ≤400 chars for embedding as a blockquote in the
/// help-desk seed post. Keeps the feed scannable without losing the gist.
fn truncate_for_quote(src: &str) -> String {
    let flat: String = src
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    if flat.chars().count() <= 400 {
        flat
    } else {
        flat.chars().take(397).chain("...".chars()).collect()
    }
}

/// Clip a message body to a thread-title-friendly length (≤80 chars).
/// Strips newlines and collapses whitespace. Used for auto-created
/// troubleshoot threads so titles stay legible.
fn truncate_for_thread_name(src: &str) -> String {
    let flat: String = src
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    let collapsed: String = flat
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() <= 80 {
        collapsed
    } else {
        collapsed
            .chars()
            .take(77)
            .chain("...".chars())
            .collect()
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_watch_discord=info".into()),
        )
        .with_target(false)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    // Token gate. Missing token → print help + exit 0. Do NOT try to
    // connect. Do NOT panic. See watch::HELP_TEXT for the copy.
    let token = match std::env::var("DISCORD_BOT_TOKEN") {
        Ok(t) if !t.trim().is_empty() => t,
        _ => {
            println!("{HELP_TEXT}");
            return Ok(());
        }
    };

    let channels_raw = std::env::var("HALO_DISCORD_CHANNELS").unwrap_or_default();
    let channels: HashSet<u64> = parse_channel_whitelist(&channels_raw).into_iter().collect();
    if channels.is_empty() {
        warn!(
            "HALO_DISCORD_CHANNELS is empty — bot will connect but observe \
             nothing. Set it to a comma-separated list of channel IDs."
        );
    }

    let server_url = std::env::var("HALO_SERVER_URL").unwrap_or_else(|_| DEFAULT_SERVER_URL.into());
    let landing_url =
        std::env::var("HALO_LANDING_URL").unwrap_or_else(|_| DEFAULT_LANDING_URL.into());

    let help_desk_channel_id: Option<u64> = std::env::var("HALO_HELP_DESK_CHANNEL_ID")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    if let Some(hd) = help_desk_channel_id {
        info!(help_desk_channel = hd, "help-desk routing enabled");
    }

    // Echo's token is the posting identity. Halo never posts with its own
    // token — every outbound message, thread creation, status reply flows
    // through this client. Unset => halo stays purely lurker.
    let echo_http = match std::env::var("ECHO_BOT_TOKEN") {
        Ok(t) if !t.trim().is_empty() => {
            info!("ECHO_BOT_TOKEN configured — posting identity active");
            Some(Arc::new(Http::new(&format!("Bot {t}"))))
        }
        _ => {
            warn!(
                "ECHO_BOT_TOKEN unset — halo will classify + dispatch but not post. \
                 Set echo's bot token to enable the reply pipeline."
            );
            None
        }
    };

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .user_agent("1bit-watch-discord/0.1")
        .build()?;

    let registry = Arc::new(Registry::default_stubs());

    let handler = Handler {
        registry,
        channels,
        http,
        echo_http,
        server_url,
        landing_url,
        help_desk_channel_id,
        help_desk_meta: OnceCell::new(),
    };

    // MESSAGE_CONTENT is a privileged intent — operator must enable it on
    // the bot in the Discord developer portal. Without it we still see
    // messages but content is empty and classification reduces to Chat.
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    // Secondary gateway for echo — stateless HTTP wouldn't give echo an
    // "online" presence in the Discord member list. Spin a minimal
    // serenity client with a no-op handler whose only job is to open and
    // hold the websocket, so echo shows green in the guild sidebar. Halo
    // still owns all message processing + outbound posting via
    // `echo_http` (REST). Done in a tokio task so the halo shard loop
    // stays primary; if echo disconnects the handler logs and exits
    // without bringing halo down.
    if let Ok(echo_token) = std::env::var("ECHO_BOT_TOKEN") {
        if !echo_token.trim().is_empty() {
            let echo_token = echo_token.clone();
            tokio::spawn(async move {
                // Echo only needs presence — no intents required for the
                // bare "I'm here" websocket, but GUILDS is the cheapest
                // non-empty intent that doesn't require the MESSAGE
                // privileged flag. Keeps the gateway from refusing the
                // handshake.
                let echo_intents = GatewayIntents::GUILDS;
                let mut echo_client = match Client::builder(&echo_token, echo_intents)
                    .event_handler(EchoPresence)
                    .await
                {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, "echo presence client failed to construct");
                        return;
                    }
                };
                if let Err(e) = echo_client.start().await {
                    warn!(error = %e, "echo presence gateway exited");
                }
            });
        }
    }

    let mut client = Client::builder(&token, intents)
        .event_handler(handler)
        .await?;

    if let Err(e) = client.start().await {
        error!(error = %e, "discord client exited with error");
        return Err(e.into());
    }

    Ok(())
}

/// No-op event handler for the echo presence gateway. Serenity requires
/// an `EventHandler` impl even if we never act on any event — this one
/// just logs that echo came online and then ignores everything. Halo's
/// `Handler` above is the real brain; echo is here purely so the Discord
/// member list shows a green dot next to the echo account while it holds
/// the gateway open.
struct EchoPresence;

#[async_trait]
impl EventHandler for EchoPresence {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!(
            bot = %ready.user.name,
            "echo presence gateway connected — green dot in member list"
        );
    }
}

// The heavy-lifting tests live in `onebit_agents::watch::tests` so they run
// under `cargo test -p 1bit-agents`. The startup-without-token behaviour
// is exercised there via the HELP_TEXT + env-gated main shape; a separate
// integration test below validates that the compiled binary short-circuits
// when the token is unset.
//
// See tests/watch_discord_startup.rs.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwrap_plain_text_passes_through() {
        assert_eq!(unwrap_outer_codeblock("hello world"), "hello world");
    }

    #[test]
    fn unwrap_strips_outer_fence_no_lang() {
        let src = "```\nhello from echo\n```";
        assert_eq!(unwrap_outer_codeblock(src), "hello from echo");
    }

    #[test]
    fn unwrap_strips_outer_fence_with_lang_tag() {
        let src = "```text\nstatus: healthy\nmodels: 1\n```";
        assert_eq!(unwrap_outer_codeblock(src), "status: healthy\nmodels: 1");
    }

    #[test]
    fn unwrap_preserves_inner_code_blocks() {
        // Outer wraps the whole thing; inner fenced block stays as-is so
        // Discord still renders the bash sample as code.
        let src = "```\nTry this command:\n```bash\nls -la\n```\ncheers\n```";
        let out = unwrap_outer_codeblock(src);
        assert!(out.starts_with("Try this command:"));
        assert!(out.contains("```bash\nls -la\n```"));
        assert!(out.trim_end().ends_with("cheers"));
    }

    #[test]
    fn unwrap_unbalanced_fence_passes_through() {
        // Only an opener, no closer — don't claw at the string.
        let src = "```\nforgot to close";
        assert_eq!(unwrap_outer_codeblock(src), src);
    }

    #[test]
    fn unwrap_fence_with_trailing_whitespace() {
        let src = "\n\n```\nanswer\n```\n\n";
        assert_eq!(unwrap_outer_codeblock(src), "answer");
    }

    #[test]
    fn sanitize_drops_multiple_empty_fences() {
        // Specialist stubs that emit ```...``` ```...``` with no body should
        // not leak empty code blocks into Discord (2026-04-23 help-desk bug).
        let src = "```\n\n```\n\n```\n \n```";
        let out = sanitize_reply(src);
        assert!(out.trim().is_empty(), "got: {out:?}");
    }

    #[test]
    fn sanitize_keeps_fence_with_content() {
        let src = "Here is the fix:\n```bash\nsystemctl restart foo\n```\nGood luck!";
        let out = sanitize_reply(src);
        assert!(out.contains("```bash\nsystemctl restart foo\n```"));
        assert!(out.contains("Here is the fix:"));
        assert!(out.contains("Good luck!"));
    }

    #[test]
    fn sanitize_strips_single_empty_fence_between_prose() {
        let src = "Here is the fix:\n```\n```\nGood luck!";
        let out = sanitize_reply(src);
        assert!(!out.contains("```"), "got: {out:?}");
        assert!(out.contains("Here is the fix:"));
        assert!(out.contains("Good luck!"));
    }

    #[test]
    fn unwrap_then_sanitize_kills_nested_empty_fences() {
        // The full post_response path: outer unwrap + sanitize.
        // Mimics a stub that wraps everything AND forgot to fill inner blocks.
        let src = "```\n```\n```\n```";
        let unwrapped = unwrap_outer_codeblock(src);
        let cleaned = sanitize_reply(&unwrapped);
        assert!(cleaned.trim().is_empty(), "got: {cleaned:?}");
    }
}
