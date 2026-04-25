//! 1bit-helm — native desktop client for 1bit-server.
//!
//! Renamed from `halo-gaia` on 2026-04-20 to avoid clashing with AMD GAIA
//! (amd-gaia.ai) — see `docs/wiki/AMD-GAIA-Integration.md`. Stack choice is
//! egui + eframe (glow backend). Pure Rust, single ELF, no Electron, no
//! web-view.
//!
//! Layout:
//! * transport modules (`client`, `conversation`, `session`, `stream`) —
//!   reused from the scaffold; OpenAI-compat chat client with SSE stream
//!   parser. No UI dependency.
//! * `bearer`     — system keyring + XDG-file fallback for `/v1/*` bearer.
//! * `telemetry`  — long-lived SSE subscription to 1bit-landing.
//! * `models`     — `GET /v1/models` client for the Models pane.
//! * `conv_log`   — JSONL conversation snapshot on close.
//! * `app`        — eframe `App` impl. Four panes behind a top-bar switcher.
//! * `tray`       — KDE Plasma SNI tray logic (gap P1 #7). Pure; the
//!   `1bit-halo-helm-tray` binary wires it to a live dbus host.
//!
//! Brand: **1bit monster**, domain `1bit.systems`. The hero strip + about
//! dialog surface these strings verbatim.

pub mod app;
pub mod bearer;
pub mod client;
pub mod conv_log;
pub mod conversation;
pub mod models;
pub mod session;
pub mod stream;
pub mod telemetry;
// MVP KDE Plasma SNI tray — split out so `cargo test -p onebit-helm`
// can exercise the pure-logic helpers without pulling a dbus
// connection.
pub mod tray;

pub use app::{HelmApp, Pane};
pub use bearer::{Bearer, BearerBackend};
pub use client::HelmClient;
pub use conv_log::{LogEntry, default_root as conv_log_root, read_session, write_session};
pub use conversation::{ChatTurn, Conversation, Role};
pub use models::{ModelCard, fetch_models, parse_models};
pub use session::SessionConfig;
pub use stream::{SseEvent, parse_sse_line};
pub use telemetry::{LiveStats, ServiceDot, TelemetryMsg, parse_stats};

/// Product name surfaced in UI strings (hero strip + about dialog).
pub const BRAND: &str = "1bit monster";
/// Canonical domain for the about dialog.
pub const BRAND_DOMAIN: &str = "1bit.systems";
