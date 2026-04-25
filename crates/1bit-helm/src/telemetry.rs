//! Client for `1bit-landing`'s `/_live/stats` SSE endpoint.
//!
//! One long-lived HTTP connection → a stream of `LiveStats` snapshots at
//! ~1.5 s cadence (matches the server's `SSE_INTERVAL`). On disconnect we
//! reconnect with short backoff. The UI subscribes by pulling
//! [`TelemetryMsg`] off an `mpsc` channel in its egui update loop.
//!
//! Server wire shape (from `crates/1bit-landing/src/telemetry.rs::Stats`):
//!
//! ```json
//! {
//!   "loaded_model": "1bit-monster-2b",
//!   "tok_s_decode": 83.4,
//!   "gpu_temp_c": 54.0,
//!   "gpu_util_pct": 27,
//!   "npu_up": false,
//!   "shadow_burn_exact_pct": 92.1,
//!   "services": [ { "name": "1bit-halo-lemonade", "active": true }, ... ],
//!   "stale": false
//! }
//! ```
//!
//! We tolerate missing fields — a dev server running only some of the
//! probes still parses.

use crate::stream::{SseEvent, parse_sse_line};
use anyhow::Result;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;

/// One-shot snapshot mirrored from `onebit_landing::Stats`. We keep our
/// own struct rather than importing it to avoid a compile-time dep on a
/// server-side crate — helm is the client, not the peer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LiveStats {
    #[serde(default)]
    pub loaded_model: String,
    #[serde(default)]
    pub tok_s_decode: f32,
    #[serde(default)]
    pub gpu_temp_c: f32,
    #[serde(default)]
    pub gpu_util_pct: u8,
    #[serde(default)]
    pub npu_up: bool,
    #[serde(default)]
    pub shadow_burn_exact_pct: f32,
    #[serde(default)]
    pub services: Vec<ServiceDot>,
    #[serde(default)]
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServiceDot {
    pub name: String,
    #[serde(default)]
    pub active: bool,
}

/// Async message emitted by the telemetry worker back to the UI.
#[derive(Debug, Clone)]
pub enum TelemetryMsg {
    /// One parsed snapshot. Newest wins; UI overwrites its cached copy.
    Snapshot(LiveStats),
    /// Worker lost its connection. UI should flip the "reachable" dot
    /// to red until the next Snapshot arrives.
    Disconnected(String),
}

/// Parse the JSON payload of one SSE `data:` line into [`LiveStats`].
/// Pure — no I/O, easy to unit-test.
pub fn parse_stats(data: &str) -> Option<LiveStats> {
    serde_json::from_str::<LiveStats>(data).ok()
}

/// Spawn the background SSE loop. Returns the receiver the UI polls.
///
/// `server_url` is the 1bit-landing base (default `http://127.0.0.1:8190`).
/// The worker is fire-and-forget — when the UI drops the receiver the
/// next send fails, the stream body errors, and the task exits on its own.
pub fn spawn(
    handle: &tokio::runtime::Handle,
    http: reqwest::Client,
    server_url: String,
) -> mpsc::UnboundedReceiver<TelemetryMsg> {
    let (tx, rx) = mpsc::unbounded_channel();
    handle.spawn(async move {
        run_loop(http, server_url, tx).await;
    });
    rx
}

async fn run_loop(
    http: reqwest::Client,
    server_url: String,
    tx: mpsc::UnboundedSender<TelemetryMsg>,
) {
    let url = format!("{}/_live/stats", server_url.trim_end_matches('/'));
    let mut backoff = Duration::from_millis(500);
    loop {
        match connect_once(&http, &url, &tx).await {
            Ok(()) => {
                // Clean close from the server — treat the same as a retry.
                if tx
                    .send(TelemetryMsg::Disconnected("server closed".into()))
                    .is_err()
                {
                    return; // UI gone.
                }
                backoff = Duration::from_millis(500);
            }
            Err(e) => {
                if tx.send(TelemetryMsg::Disconnected(e.to_string())).is_err() {
                    return;
                }
            }
        }
        tokio::time::sleep(backoff).await;
        // Bounded exponential backoff up to 5 s so we don't flood a dead
        // server with reconnects, but still recover quickly when it comes
        // back.
        backoff = (backoff * 2).min(Duration::from_secs(5));
    }
}

async fn connect_once(
    http: &reqwest::Client,
    url: &str,
    tx: &mpsc::UnboundedSender<TelemetryMsg>,
) -> Result<()> {
    let resp = http
        .get(url)
        .header("accept", "text/event-stream")
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("status {}", resp.status());
    }
    let mut bytes = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk?;
        buf.extend_from_slice(&chunk);
        while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
            let line = buf.drain(..=nl).collect::<Vec<u8>>();
            let line = &line[..line.len() - 1];
            let line = std::str::from_utf8(line).unwrap_or("");
            // 1bit-landing emits bare `data: {...}` lines — it does not
            // use `event:`. `parse_sse_line` from our chat client returns
            // `SseEvent::Delta` for `data: {...}` bodies; but that path
            // only yields content when `choices[0].delta.content` is set,
            // which the landing JSON doesn't carry. So we do our own
            // minimal prefix-strip for landing and reuse `parse_sse_line`
            // only for framing concerns (CR trim, blank-line skip).
            match parse_sse_line(line) {
                SseEvent::Done => return Ok(()),
                SseEvent::Ignore => {
                    // `Ignore` covers both keep-alives and any `data:` that
                    // didn't match the OpenAI-chat shape — the landing
                    // payload falls in the latter bucket. We still need to
                    // try to parse it as Stats.
                    if let Some(stats) = extract_landing_payload(line)
                        && let Some(parsed) = parse_stats(&stats)
                        && tx.send(TelemetryMsg::Snapshot(parsed)).is_err()
                    {
                        return Ok(()); // UI gone.
                    }
                }
                SseEvent::Delta(_) => {
                    // Shouldn't happen for /_live/stats, but if some future
                    // server rolls chat-style deltas into the same stream
                    // just skip them.
                }
            }
        }
    }
    Ok(())
}

/// Pull the JSON payload out of a raw SSE line. Returns `None` for
/// non-`data:` lines or blanks.
fn extract_landing_payload(line: &str) -> Option<String> {
    let line = line.trim_end_matches('\r');
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    let payload = line.strip_prefix("data:")?.trim_start();
    if payload == "[DONE]" {
        return None;
    }
    Some(payload.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stats_full_shape() {
        let blob = r#"{
            "loaded_model": "1bit-monster-2b",
            "tok_s_decode": 83.4,
            "gpu_temp_c": 54.0,
            "gpu_util_pct": 27,
            "npu_up": false,
            "shadow_burn_exact_pct": 92.1,
            "services": [
                { "name": "1bit-halo-lemonade", "active": true  },
                { "name": "strix-landing",      "active": true  },
                { "name": "1bit-halo-bitnet",   "active": false }
            ],
            "stale": false
        }"#;
        let s = parse_stats(blob).expect("parses");
        assert_eq!(s.loaded_model, "1bit-monster-2b");
        assert!((s.tok_s_decode - 83.4).abs() < 0.01);
        assert_eq!(s.gpu_util_pct, 27);
        assert!(!s.npu_up);
        assert_eq!(s.services.len(), 3);
        assert!(s.services[0].active);
        assert!(!s.services[2].active);
        assert!(!s.stale);
    }

    #[test]
    fn parse_stats_tolerates_missing_fields() {
        // A dev server with the NPU probe disabled still streams.
        let blob = r#"{ "loaded_model": "m", "tok_s_decode": 1.0 }"#;
        let s = parse_stats(blob).expect("parses");
        assert_eq!(s.loaded_model, "m");
        assert!(!s.npu_up);
        assert!(s.services.is_empty());
    }

    #[test]
    fn parse_stats_rejects_garbage() {
        assert!(parse_stats("not json").is_none());
        assert!(parse_stats("").is_none());
    }

    #[test]
    fn extract_landing_payload_strips_data_prefix() {
        assert_eq!(
            extract_landing_payload("data: {\"a\":1}"),
            Some(r#"{"a":1}"#.to_string())
        );
        // tolerate no-space variant
        assert_eq!(
            extract_landing_payload("data:{\"a\":1}"),
            Some(r#"{"a":1}"#.to_string())
        );
        // trailing CR handled
        assert_eq!(
            extract_landing_payload("data: {\"a\":1}\r"),
            Some(r#"{"a":1}"#.to_string())
        );
    }

    #[test]
    fn extract_landing_payload_drops_non_data_lines() {
        assert_eq!(extract_landing_payload(""), None);
        assert_eq!(extract_landing_payload(": keep-alive"), None);
        assert_eq!(extract_landing_payload("event: message"), None);
        assert_eq!(extract_landing_payload("data: [DONE]"), None);
    }
}
