//! watch — shared helpers for the halo-watch-* binaries.
//!
//! Two binaries live next to this module:
//!
//! * `1bit-watch-github` — polls configured repos for new issues + PRs and
//!   routes each event to a specialist via the [`Registry`](crate::Registry).
//! * `1bit-watch-discord` — gateway-resident Discord lurker that classifies
//!   channel messages and routes them to sentinel / herald / magistrate.
//!
//! Shared concerns (classification, env parsing, Registry dispatch) live
//! here so both binaries stay thin. Each binary owns its transport.

pub mod discord;
pub mod github;
pub mod patreon;

// Re-export the Discord helpers at the `watch::` level so the binary can
// `use onebit_agents::watch::{classify, …}` without threading the submodule
// name everywhere. GitHub helpers stay namespaced under `watch::github`
// because they share generic names (`classify`, `Event`) with the Discord
// ones and we want the call site to be explicit.
pub use discord::{
    Classification, HELP_TEXT, classify, is_direct_mention, parse_channel_whitelist, severity,
    strip_mention,
};
