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
    AutoArchiveDuration, ButtonStyle, ChannelType, Context, CreateActionRow, CreateButton,
    CreateCommand, CreateCommandOption, CreateForumPost, CreateInteractionResponse,
    CreateInteractionResponseMessage, CreateMessage, CreatePoll, CreatePollAnswer, CreateThread,
    EditThread, EventHandler, ForumTag, ForumTagId, GatewayIntents, GuildId, Http, Interaction,
    Member, Message, Ready, RoleId,
};
use serenity::all::CommandOptionType;
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
    /// Guild the slash commands register under. None → global
    /// registration (Discord caches these for ~1h before propagating).
    guild_id: Option<GuildId>,
    /// Role auto-granted on `guild_member_add`. None disables auto-role.
    member_role_id: Option<RoleId>,
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

    /// Build the replacement tag list when swapping `pending` → a
    /// terminal state (`resolved` or `escalated`). Keeps any existing
    /// classification tags (bug / feature / question) in place — only
    /// the state tag is swapped. Called from slash-command + button
    /// handlers. Existing applied tags come from the post's current
    /// state at the call site.
    fn swap_state_tag(
        &self,
        current_tags: &[ForumTagId],
        target_state: &str,
    ) -> Vec<ForumTagId> {
        let transient: std::collections::HashSet<ForumTagId> = ["pending", "resolved", "escalated"]
            .iter()
            .filter_map(|n| self.tags_by_name.get(*n).copied())
            .collect();
        let mut out: Vec<ForumTagId> = current_tags
            .iter()
            .copied()
            .filter(|t| !transient.contains(t))
            .collect();
        if let Some(id) = self.tags_by_name.get(target_state) {
            out.push(*id);
        }
        out
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        info!(
            bot = %ready.user.name,
            channels = self.channels.len(),
            "1bit-watch-discord connected"
        );

        // Slash-command registration. Guild-scoped is instant; global
        // takes up to an hour to propagate. Prefer HALO_GUILD_ID for
        // iteration.
        let commands = vec![
            CreateCommand::new("status")
                .description("Show halo + landing health + specialist count"),
            CreateCommand::new("ask")
                .description("Dispatch a question to the specialist registry")
                .add_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "question",
                        "What do you want to ask?",
                    )
                    .required(true),
                ),
            CreateCommand::new("resolve")
                .description("Mark the current help-desk post resolved"),
            CreateCommand::new("escalate")
                .description("Mark the current help-desk post escalated for a human"),
            CreateCommand::new("poll")
                .description("Start a Discord native poll in the current channel")
                .add_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "question",
                        "What do you want to ask?",
                    )
                    .required(true),
                )
                .add_option(
                    CreateCommandOption::new(
                        CommandOptionType::String,
                        "options",
                        "Comma-separated answers (2-10), e.g. yes,no,later",
                    )
                    .required(true),
                )
                .add_option(
                    CreateCommandOption::new(
                        CommandOptionType::Integer,
                        "hours",
                        "Poll duration in hours (default 24, max 168)",
                    )
                    .required(false),
                ),
        ];

        let register = if let Some(gid) = self.guild_id {
            gid.set_commands(&ctx.http, commands.clone()).await.map(|_| "guild")
        } else {
            serenity::all::Command::set_global_commands(&ctx.http, commands.clone())
                .await
                .map(|_| "global")
        };
        match register {
            Ok(scope) => info!(
                scope,
                count = commands.len(),
                "registered slash commands"
            ),
            Err(e) => warn!(error = %e, "failed to register slash commands"),
        }
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::Command(cmd) => self.handle_slash_command(&ctx, cmd).await,
            Interaction::Component(btn) => self.handle_button(&ctx, btn).await,
            _ => {}
        }
    }

    async fn guild_member_addition(&self, ctx: Context, new_member: Member) {
        self.handle_member_join(&ctx, new_member).await;
    }

    async fn reaction_add(&self, ctx: Context, reaction: serenity::all::Reaction) {
        self.handle_reaction(&ctx, reaction).await;
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

        // Lightweight classification.
        let class = classify(&msg.content);

        // Chat = silence. We don't dispatch, we don't post, we don't
        // touch the message. Halo is here to triage real asks, not
        // auto-respond to banter in #water-cooler. Bug / feature /
        // question all route to the help-desk forum below.
        if matches!(class, Classification::Chat) {
            return;
        }

        // Skip messages that originated IN the help-desk — that's where
        // follow-ups already live, no point re-routing them to
        // themselves.
        if self.help_desk_channel_id == Some(channel_id) {
            return;
        }

        let specialist = class.specialist();
        info!(
            author = %msg.author.name,
            channel = channel_id,
            class = %class,
            specialist = specialist.as_str(),
            "dispatching to specialist"
        );

        // Visual ack: drop a 👀 reaction on the user's message so they
        // can see halo picked it up, and start a typing indicator in
        // the channel that lasts ~10s (Discord auto-expires) — covers
        // the typical sentinel dispatch window. Both via echo's token;
        // halo stays silent.
        if let Some(echo) = self.echo_http.as_ref() {
            let eyes = serenity::all::ReactionType::Unicode("👀".to_string());
            if let Err(e) = echo.create_reaction(msg.channel_id, msg.id, &eyes).await {
                info!(error = %e, "failed to seed pickup reaction");
            }
            if let Err(e) = echo.broadcast_typing(msg.channel_id).await {
                info!(error = %e, "failed to broadcast typing");
            }
        }

        let req = json!({
            "source": "discord",
            "channel_id": channel_id,
            "author": msg.author.name,
            "classification": class.as_str(),
            "content": msg.content,
        });
        let dispatch_result = self.registry.dispatch(specialist.as_str(), req).await;

        // Every non-chat message routes to help-desk, unconditionally.
        // No in-channel responses anymore — the main channel stays
        // quiet; the forum is where halo speaks.
        match dispatch_result {
            Ok(resp) => {
                let text = resp
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                let body = if text.is_empty() {
                    "Picked this up, a human will follow up here."
                } else {
                    text
                };
                self.route_to_help_desk(&ctx, &msg, class, body).await;
            }
            Err(e) => {
                warn!(error = %e, "specialist dispatch failed");
                self.route_to_help_desk(
                    &ctx,
                    &msg,
                    class,
                    "Picked this up, a human will follow up here.",
                )
                .await;
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
            // Forum path: one post, one starter message, classification tag,
            // plus a button row so the asker / mods can close the loop
            // without typing a command. The specialist prompt demands
            // inline `​`​`​`bash blocks around every command, so the
            // starter renders as prose + scannable command blocks —
            // no outer wrap, otherwise we'd nest and break the inner
            // command fences.
            let starter = format!("{header}\n\n{quote_block}\n\n{body_reply}");
            let starter = truncate_for_forum_starter(&starter);
            let tags = meta.tags_for(class);
            let tag_count = tags.len();

            let action_row = CreateActionRow::Buttons(vec![
                CreateButton::new("hd_resolve")
                    .label("Resolved")
                    .emoji('✅')
                    .style(ButtonStyle::Success),
                CreateButton::new("hd_escalate")
                    .label("Escalate")
                    .emoji('⬆')
                    .style(ButtonStyle::Danger),
                CreateButton::new(format!("hd_reroll:{}", msg.id.get()))
                    .label("Re-ask")
                    .emoji('🔁')
                    .style(ButtonStyle::Secondary),
            ]);

            let post_builder = CreateForumPost::new(
                thread_name.clone(),
                CreateMessage::new()
                    .content(starter)
                    .components(vec![action_row]),
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
                    // Seed 👍/👎 on the starter message so users can rate
                    // the reply quality with one click. For forum posts
                    // the starter message id equals the thread id.
                    seed_feedback_reactions(
                        http.as_ref(),
                        post.id,
                        serenity::all::MessageId::new(post.id.get()),
                    )
                    .await;

                    let breadcrumb = format!(
                        "Routed to <#{post}> — follow there.",
                        post = post.id.get()
                    );
                    if let Err(e) = msg.channel_id.say(http.as_ref(), breadcrumb).await {
                        warn!(error = %e, "failed to post help-desk breadcrumb");
                    }
                    // Delete the asker's original message now that the
                    // conversation has moved to the help-desk post.
                    // Keeps the origin channel clean. Needs
                    // MANAGE_MESSAGES on echo's role.
                    delete_origin_message(http.as_ref(), msg).await;
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
        match thread.id.say(http.as_ref(), body_reply).await {
            Ok(reply_msg) => {
                seed_feedback_reactions(http.as_ref(), thread.id, reply_msg.id).await;
            }
            Err(e) => {
                warn!(error = %e, "failed to post specialist reply into thread");
            }
        }

        let breadcrumb = format!(
            "Routed to <#{thread}> — follow there.",
            thread = thread.id.get()
        );
        if let Err(e) = msg.channel_id.say(http.as_ref(), breadcrumb).await {
            warn!(error = %e, "failed to post help-desk breadcrumb");
        }
        delete_origin_message(http.as_ref(), msg).await;
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

    /// Dispatch slash commands. All responses are ephemeral (only the
    /// invoking user sees them) to keep the channel quiet; public work
    /// like `/resolve` tagging is visible through the tag swap itself.
    async fn handle_slash_command(
        &self,
        ctx: &Context,
        cmd: serenity::all::CommandInteraction,
    ) {
        let name = cmd.data.name.as_str();
        info!(command = name, user = %cmd.user.name, "slash command invoked");

        match name {
            "status" => {
                let line = self.status_line().await;
                ephemeral(ctx, &cmd, &line).await;
            }
            "ask" => {
                let question = cmd
                    .data
                    .options
                    .iter()
                    .find(|o| o.name == "question")
                    .and_then(|o| o.value.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if question.is_empty() {
                    ephemeral(ctx, &cmd, "Give me something to ask — `/ask question:<text>`").await;
                    return;
                }
                let class = classify(&question);
                let specialist = class.specialist();
                let req = serde_json::json!({
                    "source": "discord-slash",
                    "channel_id": cmd.channel_id.get(),
                    "author": cmd.user.name,
                    "classification": class.as_str(),
                    "content": question,
                });
                // Defer so Discord doesn't time out the 3s ack window.
                let defer = cmd
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Defer(
                            CreateInteractionResponseMessage::new().ephemeral(true),
                        ),
                    )
                    .await;
                if let Err(e) = defer {
                    warn!(error = %e, "failed to defer /ask response");
                    return;
                }
                let reply_text = match self.registry.dispatch(specialist.as_str(), req).await {
                    Ok(resp) => resp
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    Err(e) => {
                        warn!(error = %e, "specialist dispatch failed (slash)");
                        "Specialist dispatch failed — check server logs.".to_string()
                    }
                };
                let clean = sanitize_reply(&unwrap_outer_codeblock(&reply_text));
                let final_text = if clean.trim().is_empty() {
                    "Specialist came back empty — nothing to post.".to_string()
                } else {
                    clean
                };
                if let Err(e) = cmd
                    .edit_response(
                        &ctx.http,
                        serenity::all::EditInteractionResponse::new().content(final_text),
                    )
                    .await
                {
                    warn!(error = %e, "failed to post /ask follow-up");
                }
            }
            "resolve" => {
                self.swap_post_state(ctx, &cmd, "resolved").await;
            }
            "escalate" => {
                self.swap_post_state(ctx, &cmd, "escalated").await;
            }
            "poll" => {
                self.handle_poll_command(ctx, &cmd).await;
            }
            _ => {
                ephemeral(ctx, &cmd, &format!("Unknown command: /{name}")).await;
            }
        }
    }

    /// `/poll` — post a Discord native poll in the invoking channel.
    /// Echo owns the message so the poll looks like it came from the
    /// posting identity, matching every other specialist output.
    async fn handle_poll_command(
        &self,
        ctx: &Context,
        cmd: &serenity::all::CommandInteraction,
    ) {
        let Some(http) = self.echo_http.as_ref() else {
            ephemeral(ctx, cmd, "ECHO_BOT_TOKEN unset — can't post a poll").await;
            return;
        };

        let question = cmd
            .data
            .options
            .iter()
            .find(|o| o.name == "question")
            .and_then(|o| o.value.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let options_raw = cmd
            .data
            .options
            .iter()
            .find(|o| o.name == "options")
            .and_then(|o| o.value.as_str())
            .unwrap_or("")
            .to_string();
        let hours = cmd
            .data
            .options
            .iter()
            .find(|o| o.name == "hours")
            .and_then(|o| o.value.as_i64())
            .unwrap_or(24)
            .clamp(1, 168) as u64;

        let answers: Vec<CreatePollAnswer> = options_raw
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| CreatePollAnswer::new().text(s))
            .collect();

        if question.is_empty() || answers.len() < 2 || answers.len() > 10 {
            ephemeral(
                ctx,
                cmd,
                "Need a non-empty question and 2-10 comma-separated options.",
            )
            .await;
            return;
        }

        let poll = CreatePoll::new()
            .question(&question)
            .answers(answers)
            .duration(Duration::from_secs(hours * 3600));

        let msg = CreateMessage::new().poll(poll);
        match cmd.channel_id.send_message(http.as_ref(), msg).await {
            Ok(_) => {
                info!(question = %question, hours, "posted poll");
                ephemeral(ctx, cmd, &format!("Poll posted ({hours}h)")).await;
            }
            Err(e) => {
                warn!(error = %e, "failed to post poll");
                ephemeral(ctx, cmd, &format!("Failed to post poll: {e}")).await;
            }
        }
    }

    /// Handle a button press on a help-desk forum post's action row.
    /// Custom ids:
    ///   * `hd_resolve`                  → swap state tag to resolved
    ///   * `hd_escalate`                 → swap state tag to escalated
    ///   * `hd_reroll:<original_msg_id>` → re-dispatch original message
    async fn handle_button(
        &self,
        ctx: &Context,
        btn: serenity::all::ComponentInteraction,
    ) {
        let id = btn.data.custom_id.clone();
        info!(custom_id = %id, user = %btn.user.name, "button pressed");

        if id == "hd_resolve" {
            self.swap_post_state_component(ctx, &btn, "resolved").await;
        } else if id == "hd_escalate" {
            self.swap_post_state_component(ctx, &btn, "escalated").await;
        } else if let Some(_orig_id) = id.strip_prefix("hd_reroll:") {
            // Reroll kicks the original classification + question back
            // through the specialist registry. We don't have the
            // original Message object at hand here, only the id, so the
            // v1 behaviour is: ack, post a note, and let a human drag
            // the asker back to the origin. A later pass can fetch the
            // original via http::get_message and re-dispatch fully.
            let _ = btn
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content(
                                "Reroll queued. Full re-dispatch from this button is a \
                                 follow-up — for now ping the asker to repost or use \
                                 `/ask <question>` yourself.",
                            )
                            .ephemeral(true),
                    ),
                )
                .await;
        } else {
            let _ = btn
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content(format!("Unknown button: {id}"))
                            .ephemeral(true),
                    ),
                )
                .await;
        }
    }

    /// Flip the help-desk post's state tag (resolved / escalated) via a
    /// slash command. No-ops (with an ephemeral reply) when invoked
    /// outside a forum thread whose parent is the configured help-desk.
    async fn swap_post_state(
        &self,
        ctx: &Context,
        cmd: &serenity::all::CommandInteraction,
        target_state: &str,
    ) {
        let Some(meta) = self.help_desk_meta.get() else {
            ephemeral(ctx, cmd, "Help-desk metadata not resolved yet — route one message first.").await;
            return;
        };
        if !meta.is_forum {
            ephemeral(
                ctx,
                cmd,
                "Help-desk isn't a forum — tag swaps only work in forum channels.",
            )
            .await;
            return;
        }
        match self.swap_tag_on_channel(ctx, cmd.channel_id.get(), meta, target_state).await {
            Ok(()) => {
                ephemeral(ctx, cmd, &format!("✓ marked {target_state}")).await;
            }
            Err(e) => {
                ephemeral(ctx, cmd, &format!("Failed: {e}")).await;
            }
        }
    }

    /// Same as swap_post_state but driven by a button press.
    async fn swap_post_state_component(
        &self,
        ctx: &Context,
        btn: &serenity::all::ComponentInteraction,
        target_state: &str,
    ) {
        let Some(meta) = self.help_desk_meta.get() else {
            let _ = btn
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Help-desk metadata not resolved yet.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        };
        if !meta.is_forum {
            let _ = btn
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Help-desk isn't a forum — tag swaps disabled.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }
        match self.swap_tag_on_channel(ctx, btn.channel_id.get(), meta, target_state).await {
            Ok(()) => {
                let _ = btn
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Message(
                            CreateInteractionResponseMessage::new()
                                .content(format!("✓ marked {target_state}"))
                                .ephemeral(true),
                        ),
                    )
                    .await;
            }
            Err(e) => {
                let _ = btn
                    .create_response(
                        &ctx.http,
                        CreateInteractionResponse::Message(
                            CreateInteractionResponseMessage::new()
                                .content(format!("Failed: {e}"))
                                .ephemeral(true),
                        ),
                    )
                    .await;
            }
        }
    }

    /// Core tag swap: fetch the forum post's current tags, filter out
    /// the transient state tags, push the target state tag if defined,
    /// and apply via EditThread.
    async fn swap_tag_on_channel(
        &self,
        ctx: &Context,
        channel_id: u64,
        meta: &HelpDeskMeta,
        target_state: &str,
    ) -> Result<()> {
        let ch = serenity::all::ChannelId::new(channel_id);
        let current = match ch.to_channel(&ctx.http).await {
            Ok(serenity::all::Channel::Guild(gc)) => gc,
            _ => anyhow::bail!("not a guild channel"),
        };
        let new_tags = meta.swap_state_tag(&current.applied_tags, target_state);
        ch.edit_thread(&ctx.http, EditThread::new().applied_tags(new_tags))
            .await
            .map_err(|e| anyhow::anyhow!("edit_thread failed: {e}"))?;
        Ok(())
    }

    /// Guild-member-add hook: auto-grant the configured member role so
    /// new joiners are "paired" on arrival, then DM them a short
    /// welcome + channel map. Role grant and DM are independent — a
    /// missing role id skips only the role part. Failures log but do
    /// not crash the gateway.
    async fn handle_member_join(&self, ctx: &Context, member: Member) {
        if let Some(role_id) = self.member_role_id {
            info!(
                user = %member.user.name,
                guild = %member.guild_id.get(),
                role = %role_id.get(),
                "auto-granting member role"
            );
            if let Err(e) = ctx
                .http
                .add_member_role(
                    member.guild_id,
                    member.user.id,
                    role_id,
                    Some("auto-pair on member join"),
                )
                .await
            {
                warn!(error = %e, "failed to grant member role");
            }
        }

        // Welcome DM — best effort. Users who block DMs get skipped
        // silently (Discord returns 403 "Cannot send messages to this user").
        let welcome = "**Welcome to 1bit.systems.**\n\n\
            This is a small open-source AI project: native ternary kernels \
            for consumer AMD hardware, open weights, built in public.\n\n\
            • Ask questions anywhere — halo auto-routes bugs / features / \
              questions into the help-desk forum for follow-up.\n\
            • Use `/status` to see model + landing health, `/ask <question>` \
              for a direct specialist reply.\n\
            • Mention `@halo` for a silent status line.\n\n\
            Landing page: https://1bit.systems · GitHub: https://github.com/bong-water-water-bong/1bit-systems";
        match member.user.id.create_dm_channel(&ctx.http).await {
            Ok(dm) => {
                if let Err(e) = dm.id.say(&ctx.http, welcome).await {
                    info!(user = %member.user.name, error = %e, "welcome DM skipped (user likely has DMs closed)");
                } else {
                    info!(user = %member.user.name, "welcome DM sent");
                }
            }
            Err(e) => {
                info!(user = %member.user.name, error = %e, "failed to open DM channel");
            }
        }
    }

    /// Reaction hook: when someone 👍 or 👎 a Sentinel reply we own,
    /// append a single jsonl line to `~/.local/state/1bit-halo/feedback.jsonl`
    /// so the reply-rating signal can later feed a preference-data
    /// pipeline. We only record reactions on messages authored by
    /// *our* bot (echo), so external convos don't leak in. Other
    /// emojis are ignored — keep the signal clean.
    async fn handle_reaction(&self, ctx: &Context, reaction: serenity::all::Reaction) {
        let Some(http) = self.echo_http.as_ref() else {
            return;
        };
        let emoji = match &reaction.emoji {
            serenity::all::ReactionType::Unicode(s) => s.as_str(),
            _ => return, // ignore custom emoji for now
        };
        let verdict = match emoji {
            "👍" => "positive",
            "👎" => "negative",
            _ => return,
        };

        // Load the message that was reacted to. Needs full fetch to see
        // the author + content.
        let msg = match reaction.channel_id.message(&ctx.http, reaction.message_id).await {
            Ok(m) => m,
            Err(e) => {
                info!(error = %e, "failed to fetch reacted message — skipping");
                return;
            }
        };

        // Only record feedback on our echo posts — that's the signal we
        // care about for reply quality. Halo's gateway bot never speaks,
        // so any bot-authored message in a help-desk context is echo.
        let echo_id = {
            // serenity Http doesn't expose get_current_user without a
            // gateway session; rely on the author.bot flag instead and
            // the fact that we only post through echo.
            let _ = http;
            None::<u64>
        };
        if !msg.author.bot {
            return;
        }
        if let Some(expected) = echo_id {
            if msg.author.id.get() != expected {
                return;
            }
        }

        let user_id = reaction.user_id.map(|u| u.get()).unwrap_or(0);
        let entry = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "verdict": verdict,
            "channel_id": reaction.channel_id.get(),
            "message_id": reaction.message_id.get(),
            "user_id": user_id,
            "content": msg.content,
        });

        let path = feedback_log_path();
        if let Err(e) = append_jsonl(&path, &entry) {
            warn!(error = %e, path = %path.display(), "failed to append feedback line");
        } else {
            info!(verdict, message_id = reaction.message_id.get(), "recorded reply feedback");
        }
    }
}

/// Wrap a specialist reply in a single triple-backtick fenced block so
/// it renders as a monospaced container on Discord — visually
/// distinctive from normal chat. Inner triple-backticks in the payload
/// (legit code samples from the model) would otherwise close the outer
/// fence; we neutralise them by inserting a zero-width joiner between
/// each backtick, which renders visually as three backticks but isn't
/// parsed as a fence delimiter.
///
/// If the content is already short enough, the result fits under
/// Discord's 2000-char message cap comfortably (4-byte fence overhead
/// plus ZWJ expansions). For longer payloads the caller should trim
/// first via `truncate_for_forum_starter` or similar.
#[allow(dead_code)]
fn wrap_in_codeblock(text: &str) -> String {
    let escaped = text.replace("```", "`\u{200D}`\u{200D}`");
    format!("```\n{escaped}\n```")
}

/// Delete the asker's original message via echo after their request
/// has been routed into the help-desk. Users hit "post a bug in
/// #water-cooler" and see the routed link in the help-desk; keeping the
/// original message in the origin channel just fragments the thread.
/// Best-effort — a missing MANAGE_MESSAGES perm is logged and skipped
/// so the rest of the pipeline stays intact. Also skips when echo's
/// token is absent (defensive; caller already checked but we don't
/// rely on it).
async fn delete_origin_message(http: &Http, msg: &Message) {
    if let Err(e) = http.delete_message(msg.channel_id, msg.id, Some("routed to help-desk")).await {
        info!(
            error = %e,
            channel_id = %msg.channel_id,
            message_id = %msg.id,
            "failed to delete origin message (MANAGE_MESSAGES missing?)"
        );
    } else {
        info!(
            channel_id = %msg.channel_id,
            message_id = %msg.id,
            "deleted origin message post-route"
        );
    }
}

/// Seed 👍 + 👎 reactions on a message echo just posted, so users can
/// rate specialist replies with one click. Reactions feed
/// `handle_reaction` which appends a row to `feedback.jsonl`. Best
/// effort — failures are logged at debug because permissions or
/// API transients here shouldn't mask the successful post.
async fn seed_feedback_reactions(
    http: &Http,
    channel_id: serenity::all::ChannelId,
    message_id: serenity::all::MessageId,
) {
    for emoji in ["👍", "👎"] {
        let reaction = serenity::all::ReactionType::Unicode(emoji.to_string());
        if let Err(e) = http
            .create_reaction(channel_id, message_id, &reaction)
            .await
        {
            info!(
                emoji,
                channel_id = %channel_id,
                message_id = %message_id,
                error = %e,
                "failed to seed feedback reaction (check ADD_REACTIONS perm)"
            );
        }
    }
}

/// Path where reply-feedback reactions are appended, one json line per
/// reaction. Ensures the parent dir exists on first write. Keeps the
/// path stable across restarts so downstream training pipelines can
/// tail-follow it.
fn feedback_log_path() -> std::path::PathBuf {
    let base = std::env::var("HALO_FEEDBACK_LOG").ok().map(std::path::PathBuf::from);
    base.unwrap_or_else(|| {
        let mut p = dirs::state_dir()
            .or_else(dirs::data_local_dir)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
        p.push("1bit-halo");
        p.push("feedback.jsonl");
        p
    })
}

/// Append a serde_json Value as a single line to a jsonl file, creating
/// parent directories as needed. File is opened in append mode each
/// call — simpler than keeping a long-lived handle, and a reaction
/// once every few minutes doesn't need the throughput.
fn append_jsonl(path: &std::path::Path, value: &serde_json::Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut line = serde_json::to_string(value).unwrap_or_default();
    line.push('\n');
    f.write_all(line.as_bytes())?;
    Ok(())
}

/// Helper: send an ephemeral reply to a slash command. Ephemeral = only
/// the invoking user sees it, no channel noise.
async fn ephemeral(
    ctx: &Context,
    cmd: &serenity::all::CommandInteraction,
    content: &str,
) {
    if let Err(e) = cmd
        .create_response(
            &ctx.http,
            CreateInteractionResponse::Message(
                CreateInteractionResponseMessage::new()
                    .content(content)
                    .ephemeral(true),
            ),
        )
        .await
    {
        warn!(error = %e, "failed to send ephemeral response");
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

    // Guild id for slash-command registration. Without it we fall back to
    // global registration (slower to propagate).
    let guild_id: Option<GuildId> = std::env::var("HALO_GUILD_ID")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(GuildId::new);
    if let Some(gid) = guild_id {
        info!(guild_id = %gid.get(), "slash commands will register guild-scoped");
    } else {
        info!("HALO_GUILD_ID unset — slash commands register globally (slow propagation)");
    }

    // Role auto-granted on guild_member_add. Empty/unset disables
    // auto-role entirely.
    let member_role_id: Option<RoleId> = std::env::var("HALO_MEMBER_ROLE_ID")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(RoleId::new);
    if let Some(rid) = member_role_id {
        info!(role_id = %rid.get(), "auto-role on join enabled");
    }

    let handler = Handler {
        registry,
        channels,
        http,
        echo_http,
        server_url,
        landing_url,
        help_desk_channel_id,
        help_desk_meta: OnceCell::new(),
        guild_id,
        member_role_id,
    };

    // MESSAGE_CONTENT + GUILD_MEMBERS are privileged intents — operator
    // must enable them on the bot in the Discord developer portal.
    // Without MESSAGE_CONTENT classification reduces to Chat; without
    // GUILD_MEMBERS the auto-role-on-join + welcome-DM hooks never fire.
    // GUILD_MESSAGE_REACTIONS is non-privileged; feeds the feedback
    // reaction logger.
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT
        | GatewayIntents::GUILD_MEMBERS
        | GatewayIntents::GUILD_MESSAGE_REACTIONS;

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
