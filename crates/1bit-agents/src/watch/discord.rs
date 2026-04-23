//! Discord watcher plumbing used by the `1bit-watch-discord` binary.
//!
//! Pure logic — no I/O, no serenity types — so the classifier can be
//! exercised in the lib test suite without a live gateway.

use std::fmt;

use crate::Name;

/// Coarse classification of an incoming Discord message.
///
/// Lightweight heuristic: keyword scoring, no ML. The goal is "good enough
/// to route" — the receiving specialist (sentinel / herald / magistrate)
/// does the real triage. False positives are fine; we never auto-reply
/// on the classification alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    /// Something is broken. Stack trace, "error", "panic", "regression".
    BugReport,
    /// A wish or proposal. "feature", "would be nice", "can we add".
    FeatureRequest,
    /// An honest question. Ends with `?` or leads with who/what/where/…
    Question,
    /// Everything else. Lurker-default.
    Chat,
}

impl Classification {
    /// Which specialist consumes this classification.
    ///
    /// * [`Name::Sentinel`] — the watcher. Bug reports are first in line
    ///   because sentinel already tails logs + metrics.
    /// * [`Name::Magistrate`] — the reviewer. Feature requests turn into
    ///   design-review items.
    /// * [`Name::Herald`] — comms. Questions and general chat both route
    ///   here; herald decides whether to escalate or sit on them.
    pub fn specialist(self) -> Name {
        match self {
            Classification::BugReport => Name::Sentinel,
            Classification::FeatureRequest => Name::Magistrate,
            Classification::Question => Name::Herald,
            Classification::Chat => Name::Herald,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Classification::BugReport => "bug_report",
            Classification::FeatureRequest => "feature_request",
            Classification::Question => "question",
            Classification::Chat => "chat",
        }
    }
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classify a raw message body.
///
/// Priority: bug > feature > question > chat. A message that matches
/// multiple categories (e.g. "bug: can we fix the panic?") is labelled
/// as a bug — that's the one we want a human to see first.
/// DEFCON-style severity. 1 = worst (production down / data loss),
/// 5 = trivial (question, nitpick). Keyword-based heuristic — intended
/// to be a starting point the bot auto-stamps on forum posts that
/// humans can bump up or down during triage.
pub fn severity(text: &str) -> u8 {
    let lower = text.to_lowercase();
    const D1: &[&str] = &[
        "production down",
        "data loss",
        "data corruption",
        "security",
        "cve ",
        "unusable",
        "breaks everything",
        "cannot use",
        "can't use at all",
        // Strix-Halo / 1bit-systems specific: these require a power
        // cycle or a full reboot to recover — "really bad" by
        // definition. OPTC hang is the reigning champion.
        "optc",
        "kernel panic",
        "kernel hang",
        "amdgpu hang",
        "machine hung",
        "power cycle",
        "hard lock",
        "wayland crash",
        "wayland frozen",
    ];
    if D1.iter().any(|k| lower.contains(k)) {
        return 1;
    }
    const D2: &[&str] = &[
        "crashed",
        "crash",
        "segfault",
        "sigabrt",
        "sigsegv",
        "sigbus",
        "panic",
        "panicked",
        "blocker",
        "regression",
        "oom",
        "hang",
        "deadlock",
    ];
    if D2.iter().any(|k| lower.contains(k)) {
        return 2;
    }
    const D3: &[&str] = &[
        "broken",
        "fails",
        "failed to",
        "doesn't work",
        "does not work",
        "not working",
        "error:",
        "bug",
    ];
    if D3.iter().any(|k| lower.contains(k)) {
        return 3;
    }
    const D4: &[&str] = &[
        "slow",
        "annoying",
        "minor",
        "quirk",
        "would be nice",
        "nice to have",
        "wish",
        "could we",
    ];
    if D4.iter().any(|k| lower.contains(k)) {
        return 4;
    }
    5
}

pub fn classify(text: &str) -> Classification {
    let lower = text.to_lowercase();

    // Bug signals — hard keywords + common stack-trace shapes.
    const BUG: &[&str] = &[
        "bug",
        "error:",
        "panic",
        "panicked",
        "traceback",
        "stack trace",
        "crash",
        "crashed",
        "segfault",
        "regression",
        "broken",
        "doesn't work",
        "does not work",
        "not working",
        "fails",
        "failed to",
    ];
    if BUG.iter().any(|k| lower.contains(k)) {
        return Classification::BugReport;
    }

    // Feature signals.
    const FEATURE: &[&str] = &[
        "feature request",
        "would be nice",
        "would love",
        "can we add",
        "could we add",
        "please add",
        "proposal:",
        "rfc:",
        "feat:",
        "wishlist",
    ];
    if FEATURE.iter().any(|k| lower.contains(k)) {
        return Classification::FeatureRequest;
    }

    // Question signals. Cheap: trailing `?` or a leading interrogative.
    if lower.trim_end().ends_with('?') {
        return Classification::Question;
    }
    const Q_LEAD: &[&str] = &[
        "how ",
        "how's",
        "what ",
        "what's",
        "why ",
        "why's",
        "when ",
        "where ",
        "who ",
        "which ",
        "can i",
        "can you",
        "could you",
        "should i",
        "is there",
        "are there",
    ];
    let head = lower.trim_start();
    if Q_LEAD.iter().any(|k| head.starts_with(k)) {
        return Classification::Question;
    }

    Classification::Chat
}

/// Parse a comma-separated list of u64 channel IDs.
///
/// Whitespace and empty entries are tolerated. Invalid IDs are logged by
/// the caller and skipped; we return only the valid ones.
pub fn parse_channel_whitelist(raw: &str) -> Vec<u64> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u64>().ok())
        .collect()
}

/// Help text printed when `DISCORD_BOT_TOKEN` is not set. The binary
/// prints this to stdout and exits 0 — a missing token is an expected
/// state on a fresh box, not a bug.
pub const HELP_TEXT: &str = "1bit-watch-discord — Discord presence for 1bit-agents specialists

This bot is a LURKER. By default it observes messages in whitelisted
channels, classifies each one, and routes the gist to the relevant
specialist (sentinel / herald / magistrate). It never replies to the
channel unless directly mentioned with `@halo-bot`.

Required env:
  DISCORD_BOT_TOKEN          Discord bot token (no `Bot ` prefix)
  HALO_DISCORD_CHANNELS      Comma-separated channel IDs to watch

Optional env:
  HALO_SERVER_URL            Default http://127.0.0.1:8180
  HALO_LANDING_URL           Default http://127.0.0.1:8190
  RUST_LOG                   Default onebit_watch_discord=info

Commands (only in whitelisted channels, via @halo-bot mention):
  @halo-bot status           One-line fleet status

To enable on strixhalo:
  systemctl --user edit strix-watch-discord      # drop token.conf in
  systemctl --user enable --now strix-watch-discord

This bot MUST NOT: auto-moderate, spam, DM users, or forward bearer tokens.
";

/// Decide whether a message is a direct mention of this bot.
///
/// Matches `<@BOT_ID>` and `<@!BOT_ID>` (nickname mention form). We
/// deliberately do not match plain-text "@halo-bot" because that's a
/// Discord display-name string — not a routable mention — and we want
/// the bot to stay silent unless a real mention was sent.
pub fn is_direct_mention(content: &str, bot_id: u64) -> bool {
    let plain = format!("<@{bot_id}>");
    let nick = format!("<@!{bot_id}>");
    content.contains(&plain) || content.contains(&nick)
}

/// Strip a leading `<@BOT_ID>` / `<@!BOT_ID>` prefix from `content` and
/// return the remainder, lowercased and trimmed. Used to match the
/// `status` subcommand after an `@halo-bot` mention.
pub fn strip_mention(content: &str, bot_id: u64) -> String {
    let plain = format!("<@{bot_id}>");
    let nick = format!("<@!{bot_id}>");
    let stripped = content.replacen(&plain, "", 1).replacen(&nick, "", 1);
    stripped.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- Classifier --------

    #[test]
    fn classifier_catches_handcrafted_samples() {
        // Bugs
        assert_eq!(
            classify("got a panic in the decode loop"),
            Classification::BugReport
        );
        assert_eq!(
            classify("traceback on startup, crash with SIGSEGV"),
            Classification::BugReport
        );
        assert_eq!(classify("this is broken"), Classification::BugReport);

        // Feature requests
        assert_eq!(
            classify("can we add a streaming API please"),
            Classification::FeatureRequest
        );
        assert_eq!(
            classify("would be nice to have darkmode"),
            Classification::FeatureRequest
        );
        assert_eq!(
            classify("feat: add TTS sharding"),
            Classification::FeatureRequest
        );

        // Questions
        assert_eq!(classify("how do I build this?"), Classification::Question);
        assert_eq!(
            classify("what is the current tok/s"),
            Classification::Question
        );
        assert_eq!(
            classify("are there release notes"),
            Classification::Question
        );

        // Chat fallback
        assert_eq!(classify("gm everyone"), Classification::Chat);
        assert_eq!(classify("shipping a new bench later"), Classification::Chat);
    }

    #[test]
    fn classifier_priority_bug_beats_question() {
        // Mixed: bug keyword wins even with a trailing `?`.
        assert_eq!(
            classify("why is the decoder panicking?"),
            Classification::BugReport
        );
    }

    #[test]
    fn classification_routes_to_expected_specialist() {
        assert_eq!(Classification::BugReport.specialist(), Name::Sentinel);
        assert_eq!(
            Classification::FeatureRequest.specialist(),
            Name::Magistrate
        );
        assert_eq!(Classification::Question.specialist(), Name::Herald);
        assert_eq!(Classification::Chat.specialist(), Name::Herald);
    }

    // -------- Channel parser --------

    #[test]
    fn channel_whitelist_parses_commas_and_whitespace() {
        let ids = parse_channel_whitelist("  100, 200 ,300  ,");
        assert_eq!(ids, vec![100, 200, 300]);
    }

    #[test]
    fn channel_whitelist_skips_invalid() {
        let ids = parse_channel_whitelist("100,not-a-number,200");
        assert_eq!(ids, vec![100, 200]);
    }

    #[test]
    fn channel_whitelist_empty_is_empty() {
        assert!(parse_channel_whitelist("").is_empty());
        assert!(parse_channel_whitelist(" , , ").is_empty());
    }

    // -------- Mention matching --------

    #[test]
    fn direct_mention_matches_both_forms() {
        assert!(is_direct_mention("hey <@12345> status", 12345));
        assert!(is_direct_mention("<@!12345> status", 12345));
        assert!(!is_direct_mention("@halo-bot status", 12345));
        assert!(!is_direct_mention("<@99999> status", 12345));
    }

    #[test]
    fn strip_mention_returns_lowercased_remainder() {
        assert_eq!(strip_mention("<@12345> STATUS", 12345), "status");
        assert_eq!(strip_mention("<@!12345>  Status  ", 12345), "status");
    }

    // -------- Help text + startup gate --------

    #[test]
    fn help_text_mentions_required_envs() {
        assert!(HELP_TEXT.contains("DISCORD_BOT_TOKEN"));
        assert!(HELP_TEXT.contains("HALO_DISCORD_CHANNELS"));
        assert!(HELP_TEXT.contains("lurker") || HELP_TEXT.contains("LURKER"));
    }

    #[test]
    fn severity_defcon_1_for_optc_and_critical() {
        assert_eq!(severity("OPTC hang froze wayland had to power cycle"), 1);
        assert_eq!(severity("production down after the update"), 1);
        assert_eq!(severity("kernel panic on boot"), 1);
        assert_eq!(severity("amdgpu hang, hard lock"), 1);
        assert_eq!(severity("this is unusable"), 1);
    }

    #[test]
    fn severity_defcon_2_for_crashes_and_oom() {
        assert_eq!(severity("decode crashed with SIGABRT"), 2);
        assert_eq!(severity("got a segfault loading the model"), 2);
        assert_eq!(severity("oom killer took it out"), 2);
        assert_eq!(severity("panicked mid-generation"), 2);
    }

    #[test]
    fn severity_defcon_3_for_generic_errors() {
        assert_eq!(severity("error: failed to load gguf"), 3);
        assert_eq!(severity("this feature is broken on my box"), 3);
        assert_eq!(severity("chat completions doesn't work"), 3);
    }

    #[test]
    fn severity_defcon_4_for_minor_complaints() {
        assert_eq!(severity("decode feels slow on long prompts"), 4);
        assert_eq!(severity("minor quirk with the tokenizer"), 4);
        assert_eq!(severity("would be nice to have a progress bar"), 4);
    }

    #[test]
    fn severity_defcon_5_for_plain_questions() {
        assert_eq!(severity("how does the rope convention work?"), 5);
        assert_eq!(severity("what's the tok/s on gfx1151"), 5);
        assert_eq!(severity("morning lads"), 5);
    }

    #[test]
    fn severity_precedence_d1_before_d2() {
        // "OPTC hang" should win over "crashed" — hang is worse.
        assert_eq!(severity("OPTC hang then the service crashed"), 1);
    }
}
