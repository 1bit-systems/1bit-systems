//! Aggregated live-telemetry source for the landing dashboard.
//!
//! Everything on `/_live/stats` and `/_live/services` ultimately resolves
//! through [`Telemetry::snapshot`]. We keep a ~1 s TTL cache in front of
//! each source so a bursty reload of the page doesn't fork a dozen
//! `rocm-smi` subprocesses.
//!
//! Sources (all 127.0.0.1 only — invariant 1/5 in `Crate-1bit-landing.md`):
//! 1. `http://127.0.0.1:8180/v1/models` — loaded model name.
//! 2. `http://127.0.0.1:8180/metrics` — live tok/s, request counts.
//! 3. `rocm-smi --showtemp --showuse --json` — iGPU temp + utilisation.
//! 4. `xrt-smi examine` exit code (or `/dev/accel/accel0` presence) — NPU up.
//! 5. `~/claude output/shadow-burnin.jsonl` tail — rolling exact-match %.
//! 6. `systemctl --user is-active strix-*` — per-unit state.
//!
//! Every source degrades to a sentinel on failure rather than propagating
//! an error — this endpoint must never 5xx (invariant 5).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// User-scope systemd units we probe for `/_live/services`. Kept small
/// and explicit — random `strix-*` units the user adds are ignored until
/// this list grows.
///
/// Canon (2026-04-25): lemond (the C++ server at /home/bcloud/repos/lemonade/)
/// runs as `1bit-halo-lemonade.service` on :8180. The retired
/// `strix-server` / `halo-server-real` / `1bit-server` units are gone.
/// `strix-lemonade` was the Rust 1bit-lemonade gateway and is also
/// retired. The list below mirrors the canonical short-name set in
/// 1bit-cli/src/status.rs.
pub const TRACKED_SERVICES: &[&str] = &[
    "1bit-halo-lemonade",
    "1bit-halo-bitnet",
    "1bit-halo-sd",
    "1bit-halo-whisper",
    "1bit-halo-kokoro",
    "1bit-halo-agent",
    "strix-landing",
    "strix-echo",
    "strix-burnin",
    "strix-cloudflared",
];

/// Cached live snapshot emitted on `/_live/stats`.
///
/// `stale` flips true when the most recent `collect()` failed to reach
/// lemond and we're serving the previous `loaded_model`.
#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub loaded_model: String,
    pub tok_s_decode: f32,
    pub gpu_temp_c: f32,
    pub gpu_util_pct: u8,
    pub npu_up: bool,
    pub shadow_burn_exact_pct: f32,
    pub services: Vec<ServiceState>,
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ServiceState {
    pub name: String,
    pub active: bool,
}

impl Stats {
    pub fn empty() -> Self {
        Self {
            loaded_model: String::new(),
            tok_s_decode: 0.0,
            gpu_temp_c: 0.0,
            gpu_util_pct: 0,
            npu_up: false,
            shadow_burn_exact_pct: 0.0,
            services: Vec::new(),
            stale: true,
        }
    }
}

/// Path surfaces we sniff to produce a [`Stats`]. Overridable from tests
/// so we don't need rocm-smi / lemond running in CI.
#[derive(Clone)]
pub struct Sources {
    pub http: reqwest::Client,
    pub rocm_smi_bin: PathBuf,
    pub xrt_smi_bin: PathBuf,
    pub accel_dev: PathBuf,
    pub shadow_burnin_jsonl: PathBuf,
    pub systemctl_bin: PathBuf,
    pub services: &'static [&'static str],
    /// Base URL for lemond probes. Tests point this at a mock.
    pub onebit_server_base: String,
}

impl Default for Sources {
    fn default() -> Self {
        Self {
            http: reqwest::Client::builder()
                .user_agent("1bit-landing/telemetry")
                .build()
                .expect("reqwest client"),
            rocm_smi_bin: PathBuf::from("rocm-smi"),
            xrt_smi_bin: PathBuf::from("xrt-smi"),
            accel_dev: PathBuf::from("/dev/accel/accel0"),
            shadow_burnin_jsonl: default_shadow_burnin_path(),
            systemctl_bin: PathBuf::from("systemctl"),
            services: TRACKED_SERVICES,
            onebit_server_base: "http://127.0.0.1:8180".to_string(),
        }
    }
}

fn default_shadow_burnin_path() -> PathBuf {
    // `~/claude output/shadow-burnin.jsonl` — the folder name contains a
    // space on purpose (per project convention).
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/bcloud".to_string());
    PathBuf::from(home).join("claude output/shadow-burnin.jsonl")
}

/// ~1 s TTL cache around the actual source fan-out.
pub struct Telemetry {
    sources: Sources,
    cached: Mutex<Cached>,
    ttl: Duration,
}

struct Cached {
    stats: Stats,
    fetched_at: Option<Instant>,
}

impl Telemetry {
    pub fn new(sources: Sources) -> Self {
        Self {
            sources,
            cached: Mutex::new(Cached {
                stats: Stats::empty(),
                fetched_at: None,
            }),
            ttl: Duration::from_millis(1000),
        }
    }

    /// Wall-clock TTL getter — exposed for tests.
    #[cfg(test)]
    pub fn with_ttl(sources: Sources, ttl: Duration) -> Self {
        Self {
            sources,
            cached: Mutex::new(Cached {
                stats: Stats::empty(),
                fetched_at: None,
            }),
            ttl,
        }
    }

    /// Return the cached snapshot if fresh, else collect a new one.
    pub async fn snapshot(&self) -> Stats {
        {
            let c = self.cached.lock().unwrap();
            if let Some(at) = c.fetched_at
                && at.elapsed() < self.ttl
            {
                return c.stats.clone();
            }
        }

        let fresh = self.collect().await;

        let mut c = self.cached.lock().unwrap();
        c.stats = fresh.clone();
        c.fetched_at = Some(Instant::now());
        fresh
    }

    /// Bypass the cache — for tests + the SSE poll path that already
    /// throttles to 1.5 s.
    pub async fn collect(&self) -> Stats {
        let prev_model = {
            let c = self.cached.lock().unwrap();
            c.stats.loaded_model.clone()
        };

        let (model, stale) = match probe_model(&self.sources).await {
            Some(m) => (m, false),
            None => (prev_model, true),
        };
        let tok_s_decode = probe_tokps(&self.sources).await.unwrap_or(0.0);
        let (gpu_temp_c, gpu_util_pct) = probe_rocm_smi(&self.sources).await;
        let npu_up = probe_npu(&self.sources).await;
        let shadow_burn_exact_pct = probe_shadow_burn(&self.sources.shadow_burnin_jsonl).await;
        let services = probe_services(&self.sources).await;

        Stats {
            loaded_model: model,
            tok_s_decode,
            gpu_temp_c,
            gpu_util_pct,
            npu_up,
            shadow_burn_exact_pct,
            services,
            stale,
        }
    }
}

async fn probe_model(s: &Sources) -> Option<String> {
    let url = format!("{}/v1/models", s.onebit_server_base);
    let r = s
        .http
        .get(&url)
        .timeout(Duration::from_millis(1500))
        .send()
        .await
        .ok()?;
    if !r.status().is_success() {
        return None;
    }
    let v: serde_json::Value = r.json().await.ok()?;
    v.get("data")
        .and_then(|d| d.get(0))
        .and_then(|m| m.get("id"))
        .and_then(|i| i.as_str())
        .map(|s| s.to_string())
}

async fn probe_tokps(s: &Sources) -> Option<f32> {
    let url = format!("{}/metrics", s.onebit_server_base);
    let r = s
        .http
        .get(&url)
        .timeout(Duration::from_millis(1500))
        .send()
        .await
        .ok()?;
    if !r.status().is_success() {
        return None;
    }
    let v: serde_json::Value = r.json().await.ok()?;
    v.get("tokps_recent")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
}

/// Run `rocm-smi --showtemp --showuse --json` and fish out edge temp +
/// GPU-use-%. On any failure (binary missing, non-zero exit, JSON shape
/// change) we return `(0.0, 0)` — the frontend renders that as "—".
///
/// rocm-smi 6.x JSON shape (per-card keyed):
/// ```json
/// { "card0": { "Temperature (Sensor edge) (C)": "48.0",
///              "GPU use (%)": "3" } }
/// ```
async fn probe_rocm_smi(s: &Sources) -> (f32, u8) {
    let out = Command::new(&s.rocm_smi_bin)
        .args(["--showtemp", "--showuse", "--json"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .await;
    let Ok(out) = out else {
        return (0.0, 0);
    };
    if !out.status.success() {
        return (0.0, 0);
    }
    let Ok(v): Result<serde_json::Value, _> = serde_json::from_slice(&out.stdout) else {
        return (0.0, 0);
    };
    parse_rocm_smi_json(&v)
}

pub(crate) fn parse_rocm_smi_json(v: &serde_json::Value) -> (f32, u8) {
    let obj = match v.as_object() {
        Some(o) => o,
        None => return (0.0, 0),
    };
    // Pick the first card* key. Strix Halo has a single iGPU.
    let card = obj
        .iter()
        .find(|(k, _)| k.starts_with("card"))
        .map(|(_, v)| v);
    let Some(card) = card.and_then(|c| c.as_object()) else {
        return (0.0, 0);
    };
    let temp = card
        .iter()
        .find(|(k, _)| k.contains("Temperature") && k.contains("edge"))
        .and_then(|(_, v)| v.as_str())
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    let util = card
        .iter()
        .find(|(k, _)| k.contains("GPU use") || k.eq_ignore_ascii_case("GPU use (%)"))
        .and_then(|(_, v)| v.as_str())
        .and_then(|s| s.trim().trim_end_matches('%').parse::<u8>().ok())
        .unwrap_or(0);
    (temp, util)
}

/// NPU liveness: first try `xrt-smi examine` (fast, returns 0 when a
/// device is enumerated); if the binary is missing fall back to checking
/// that `/dev/accel/accel0` exists.
async fn probe_npu(s: &Sources) -> bool {
    if let Ok(out) = Command::new(&s.xrt_smi_bin)
        .arg("examine")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
    {
        return out.success();
    }
    Path::new(&s.accel_dev).exists()
}

/// Tail the shadow-burnin jsonl and compute the `full_match` ratio over
/// the last N entries. N=200 matches the "~90% byte-exact" rolling-window
/// definition in CLAUDE.md.
async fn probe_shadow_burn(path: &Path) -> f32 {
    const WINDOW: usize = 200;
    let Ok(file) = tokio::fs::File::open(path).await else {
        return 0.0;
    };
    // Cheap tail: read the whole file line-by-line, keep a ring of the
    // last WINDOW lines. shadow-burnin.jsonl is append-only with ~80 B
    // per line, so even at 100k lines this is ~8 MiB — fine.
    let mut ring: Vec<String> = Vec::with_capacity(WINDOW);
    let mut reader = BufReader::new(file).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        if ring.len() == WINDOW {
            ring.remove(0);
        }
        ring.push(line);
    }
    if ring.is_empty() {
        return 0.0;
    }
    let mut matches = 0u32;
    let mut total = 0u32;
    for l in &ring {
        let Ok(v): Result<serde_json::Value, _> = serde_json::from_str(l) else {
            continue;
        };
        total += 1;
        if v.get("full_match")
            .and_then(|b| b.as_bool())
            .unwrap_or(false)
        {
            matches += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        (matches as f32) * 100.0 / (total as f32)
    }
}

async fn probe_services(s: &Sources) -> Vec<ServiceState> {
    let mut out = Vec::with_capacity(s.services.len());
    for name in s.services {
        let active = match Command::new(&s.systemctl_bin)
            .args(["--user", "is-active", name])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
        {
            Ok(st) => st.success(),
            Err(_) => false,
        };
        out.push(ServiceState {
            name: (*name).to_string(),
            active,
        });
    }
    out
}

/// Return `Some(delta)` containing the service rows whose `active` flipped
/// between `prev` and `next`, or `None` if nothing changed. Pure helper
/// so the SSE `/_live/services` handler is trivial to test.
pub fn service_delta(prev: &[ServiceState], next: &[ServiceState]) -> Option<Vec<ServiceState>> {
    let mut changes = Vec::new();
    for n in next {
        let was = prev.iter().find(|p| p.name == n.name).map(|p| p.active);
        if was != Some(n.active) {
            changes.push(n.clone());
        }
    }
    if changes.is_empty() {
        None
    } else {
        Some(changes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rocm_smi_shape() {
        let v = serde_json::json!({
            "card0": {
                "Temperature (Sensor edge) (C)": "52.0",
                "GPU use (%)": "27"
            }
        });
        let (t, u) = parse_rocm_smi_json(&v);
        assert!((t - 52.0).abs() < 0.001);
        assert_eq!(u, 27);
    }

    #[test]
    fn parse_rocm_smi_degrades_on_garbage() {
        let v = serde_json::json!({ "card0": { "bogus": 1 } });
        assert_eq!(parse_rocm_smi_json(&v), (0.0, 0));
        assert_eq!(parse_rocm_smi_json(&serde_json::Value::Null), (0.0, 0));
    }

    #[test]
    fn service_delta_detects_flips() {
        let prev = vec![
            ServiceState {
                name: "a".into(),
                active: true,
            },
            ServiceState {
                name: "b".into(),
                active: false,
            },
        ];
        let next = vec![
            ServiceState {
                name: "a".into(),
                active: true,
            },
            ServiceState {
                name: "b".into(),
                active: true,
            },
        ];
        let d = service_delta(&prev, &next).expect("change detected");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].name, "b");
        assert!(d[0].active);
    }

    #[test]
    fn service_delta_is_none_when_steady() {
        let s = vec![
            ServiceState {
                name: "a".into(),
                active: true,
            },
            ServiceState {
                name: "b".into(),
                active: true,
            },
        ];
        assert!(service_delta(&s, &s).is_none());
    }

    #[tokio::test]
    async fn shadow_burn_counts_full_match_ratio() {
        let dir = tempdir_fallback();
        let p = dir.join("shadow-burnin.jsonl");
        let body = [
            r#"{"ts":"t","prompt_idx":0,"full_match":true}"#,
            r#"{"ts":"t","prompt_idx":1,"full_match":true}"#,
            r#"{"ts":"t","prompt_idx":2,"full_match":false}"#,
            r#"{"ts":"t","prompt_idx":3,"full_match":true}"#,
        ]
        .join("\n");
        tokio::fs::write(&p, body).await.unwrap();
        let pct = probe_shadow_burn(&p).await;
        assert!((pct - 75.0).abs() < 0.01, "expected 75.0 got {pct}");
        let _ = tokio::fs::remove_file(&p).await;
    }

    #[tokio::test]
    async fn shadow_burn_missing_file_returns_zero() {
        let p = PathBuf::from("/tmp/1bit-landing-nonexistent-file-xyz.jsonl");
        let _ = tokio::fs::remove_file(&p).await;
        assert_eq!(probe_shadow_burn(&p).await, 0.0);
    }

    #[tokio::test]
    async fn collect_returns_stats_with_missing_sources() {
        // Point every external at a broken/nonexistent path — collect()
        // must not panic and must yield a Stats with sentinels.
        let sources = Sources {
            http: reqwest::Client::new(),
            rocm_smi_bin: PathBuf::from("/nonexistent/rocm-smi"),
            xrt_smi_bin: PathBuf::from("/nonexistent/xrt-smi"),
            accel_dev: PathBuf::from("/nonexistent/accel0"),
            shadow_burnin_jsonl: PathBuf::from("/nonexistent/shadow.jsonl"),
            systemctl_bin: PathBuf::from("/nonexistent/systemctl"),
            services: TRACKED_SERVICES,
            // Unroutable TEST-NET-1 address so probe_model fails fast
            // without actually hitting lemond on the dev box.
            onebit_server_base: "http://192.0.2.1:1".to_string(),
        };
        let t = Telemetry::new(sources);
        let s = t.collect().await;
        // loaded_model blank, stale flagged (lemond unreachable).
        assert!(s.loaded_model.is_empty());
        assert!(s.stale);
        assert_eq!(s.tok_s_decode, 0.0);
        assert_eq!(s.gpu_temp_c, 0.0);
        assert_eq!(s.gpu_util_pct, 0);
        assert!(!s.npu_up);
        assert_eq!(s.shadow_burn_exact_pct, 0.0);
        // Services must include every tracked entry, all false.
        assert_eq!(s.services.len(), TRACKED_SERVICES.len());
        assert!(s.services.iter().all(|svc| !svc.active));
    }

    #[tokio::test]
    async fn snapshot_respects_cache_ttl() {
        let sources = Sources {
            http: reqwest::Client::new(),
            rocm_smi_bin: PathBuf::from("/nonexistent/rocm-smi"),
            xrt_smi_bin: PathBuf::from("/nonexistent/xrt-smi"),
            accel_dev: PathBuf::from("/nonexistent/accel0"),
            shadow_burnin_jsonl: PathBuf::from("/nonexistent/shadow.jsonl"),
            systemctl_bin: PathBuf::from("/nonexistent/systemctl"),
            services: TRACKED_SERVICES,
            onebit_server_base: "http://192.0.2.1:1".to_string(),
        };
        let t = Telemetry::with_ttl(sources, Duration::from_secs(60));
        let a = t.snapshot().await;
        let b = t.snapshot().await;
        // Same stale+empty data, re-used from cache.
        assert_eq!(a.loaded_model, b.loaded_model);
        assert_eq!(a.stale, b.stale);
    }

    fn tempdir_fallback() -> PathBuf {
        let p = PathBuf::from(format!("/tmp/1bit-landing-test-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
