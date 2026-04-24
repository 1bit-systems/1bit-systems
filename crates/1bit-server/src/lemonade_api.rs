//! Lemonade desktop-compatibility routes (`/api/v1/*`).
//!
//! Lemonade-SDK's desktop client (Tauri-based, lemonade-sdk/lemonade) speaks
//! a richer surface than vanilla OpenAI: a flat `/api/v1/{health,models,
//! load,unload,pull,delete,install,uninstall,system-info,pull/variants}`.
//! These handlers expose the same wire shapes so a stock Lemonade desktop
//! pointed at `http://127.0.0.1:8180` connects without a custom adapter.
//!
//! ## Implementation policy
//!
//! * `health`, `models`, `models/{name}`, `system-info` — fully implemented.
//!   They read from the existing [`AppState`] (registry, backend metadata,
//!   uname/proc) — same data we already serve on `/v1/models`, just wrapped
//!   in Lemonade's richer envelope (recipe, downloaded, owned_by, etc.).
//! * `load`, `unload`, `pull`, `delete`, `install`, `uninstall`,
//!   `pull/variants` — return HTTP 200 with `{status:"not_supported",reason:
//!   "1bit-systems manages models via packages.toml; this endpoint is
//!   informational"}`. Lemonade desktop renders unknown 4xx/5xx as fatal
//!   errors, so we 200 with a clear status field instead.
//!
//! ## Branding
//!
//! `owned_by` stays `"1bit systems"` everywhere (matches `/v1/models`).
//! `recipe` is `"1bit-ternary"` for our `.h1b` weights — a string Lemonade
//! desktop already tolerates as opaque metadata.
//!
//! ## Schema source
//!
//! All response shapes were extracted from `~/repos/lemonade/src/cpp/server/
//! server.cpp` handlers (Lemonade's own C++ implementation, post-v10.2.0).
//! See `~/claude output/lemonade-api-contract.md` for the full reference.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use axum::Json;
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::routes::AppState;

/// Process-start time — captured the first time any Lemonade route is hit.
/// Used by `/api/v1/health` to report `uptime_seconds`. Lazy so the library
/// constructor (`build_router`) doesn't need a side effect.
static START_UNIX: AtomicU64 = AtomicU64::new(0);

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn uptime_secs() -> u64 {
    let start = START_UNIX.load(Ordering::Relaxed);
    if start == 0 {
        let now = now_unix();
        // CAS so concurrent requests agree on the same start instant.
        let _ = START_UNIX.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
        return 0;
    }
    now_unix().saturating_sub(start)
}

/// Build the `/api/v1/*` sub-router. Mount with `.nest("/api/v1", lemonade_router())`.
pub fn lemonade_router() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/models/:name", get(get_model))
        .route("/system-info", get(system_info))
        .route("/load", post(load))
        .route("/unload", post(unload))
        .route("/pull", post(pull))
        .route("/pull/variants", get(pull_variants))
        .route("/delete", post(delete))
        .route("/install", post(install))
        .route("/uninstall", post(uninstall))
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// "Informational" 200 response used for the model-management endpoints
/// 1bit-systems doesn't actually implement (we own model lifecycle via
/// packages.toml, not via HTTP). Lemonade desktop treats 4xx/5xx as fatal,
/// so we return 200 with an explicit `status: "not_supported"` field that
/// the desktop UI can render as a banner instead of crashing.
fn not_supported(endpoint: &str) -> Json<Value> {
    Json(json!({
        "status": "not_supported",
        "endpoint": endpoint,
        "reason": "1bit-systems manages models via packages.toml; this endpoint is informational",
        "message": "1bit-server runs in Lemonade-desktop-compat mode — use `1bit install <model>` from packages.toml to add weights",
    }))
}

/// Render one of the 1bit-server registry entries as a Lemonade-shaped
/// model card (matches `Server::model_info_to_json` in lemonade
/// server.cpp:1264).
fn model_card(id: &str, loaded: bool) -> Value {
    json!({
        "id": id,
        "object": "model",
        "created": 1234567890,                // Lemonade hard-codes this constant too.
        "owned_by": "1bit systems",
        "checkpoint": id,
        "checkpoints": [id],
        "recipe": "1bit-ternary",
        "downloaded": true,
        "suggested": false,
        "labels": ["ternary", "1.58bpw"],
        "composite_models": [],
        "recipe_options": {},
        "size": 0.0,
        // Extras Lemonade clients tolerate as opaque metadata — match the
        // task spec's "richer envelope" requirement.
        "size_bytes": 0_u64,
        "quant_format": "ternary-1.58bpw",
        "context_length": 2048,
        "loaded": loaded,
    })
}

// ─── Handlers ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    model_loaded: Option<String>,
    all_models_loaded: Vec<Value>,
    max_models: Value,
    uptime_seconds: u64,
    runtime: &'static str,
}

async fn health(State(s): State<AppState>) -> Json<HealthResponse> {
    let ids = s.models.ids();
    // We serve a single resident backend today; first registry id is the
    // "loaded" one. The full registry comes back in `all_models_loaded`
    // for parity with Lemonade's multi-model surface.
    let model_loaded = ids.first().cloned();
    let all_models_loaded: Vec<Value> = ids
        .iter()
        .map(|id| {
            json!({
                "model_name": id,
                "type": "llm",
                "device": "gpu",
                "recipe": "1bit-ternary",
                "checkpoint": id,
            })
        })
        .collect();

    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        model_loaded,
        all_models_loaded,
        // Lemonade's CLI reads `max_models.llm` (lemonade_client.cpp:228);
        // 1bit-server today serves one resident backend, so cap = 1.
        max_models: json!({"llm": 1, "embedding": 0, "reranking": 0, "audio": 0, "image": 0, "tts": 0}),
        uptime_seconds: uptime_secs(),
        runtime: "1bit-systems",
    })
}

#[derive(Debug, Deserialize)]
struct ModelsQuery {
    /// Lemonade desktop sends both `show_all=true` (full registry) and
    /// `show_all=false` (downloaded only). For 1bit-systems every
    /// registered model IS downloaded (we own the lifecycle, not the
    /// HTTP), so the response is identical either way — we accept the
    /// param for shape parity but don't branch on it.
    #[serde(default, rename = "show_all")]
    _show_all: Option<bool>,
}

async fn list_models(
    State(s): State<AppState>,
    Query(_q): Query<ModelsQuery>,
) -> Json<Value> {
    let ids = s.models.ids();
    let loaded = ids.first().cloned();
    let data: Vec<Value> = ids
        .iter()
        .map(|id| model_card(id, Some(id) == loaded.as_ref()))
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn get_model(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if !s.models.contains(&name) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("Model '{name}' not found"),
                    "type": "model_not_found",
                    "code": "model_not_found",
                }
            })),
        ));
    }
    let ids = s.models.ids();
    let loaded = ids.first();
    let is_loaded = loaded == Some(&name);
    Ok(Json(model_card(&name, is_loaded)))
}

async fn system_info() -> Json<Value> {
    // Schema mirrors `Server::handle_system_info` (server.cpp:3135) +
    // `SystemInfo::get_system_info_dict` (Linux variant, system_info.cpp:2569).
    // We populate the load-bearing fields Lemonade desktop renders + leave
    // the rest as empty objects (Lemonade tolerates missing keys; only
    // `recipes` is fatal-if-malformed and we emit a non-empty placeholder).
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(0);
    let os_version = read_os_version();
    let memory_gb = read_memory_gb();
    let processor = read_processor();

    Json(json!({
        "OS Version": os_version,
        "Processor": processor,
        "Physical Memory": format!("{} GB", memory_gb),
        "os": "linux",
        "cpu": processor,
        "gpu": "gfx1151",
        "vram_gb": 0,
        "memory_gb": memory_gb,
        "runtime": "1bit-systems",
        "version": env!("CARGO_PKG_VERSION"),
        "devices": {
            "cpu": {
                "name": processor,
                "cores": cpu_threads,
                "threads": cpu_threads,
                "available": true,
                "family": "x86_64",
            },
            "amd_gpu": [{
                "name": "AMD Radeon Graphics (Strix Halo)",
                "available": true,
                "vram_gb": 0,
                "family": "gfx1151",
            }],
        },
        "recipes": {
            "1bit-ternary": {
                "backends": {
                    "rocm": {
                        "state": "installed",
                        "version": env!("CARGO_PKG_VERSION"),
                        "message": "1bit-systems native HIP backend",
                    }
                }
            }
        },
        "no_fetch_executables": true,
    }))
}

fn read_os_version() -> String {
    // Mirrors `LinuxSystemInfo::get_os_version` (system_info.cpp:2576) —
    // assemble Linux-<kernel> (<distro> <ver>) from /proc + /etc/os-release.
    let kernel = std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|s| {
            s.split("version ")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next().map(String::from))
        })
        .unwrap_or_else(|| "unknown".into());
    let mut out = format!("Linux-{kernel}");
    if let Ok(rel) = std::fs::read_to_string("/etc/os-release") {
        let mut name = String::new();
        let mut ver = String::new();
        for line in rel.lines() {
            if let Some(v) = line.strip_prefix("NAME=") {
                name = v.trim_matches('"').to_string();
            } else if let Some(v) = line.strip_prefix("VERSION_ID=") {
                ver = v.trim_matches('"').to_string();
            }
        }
        if !name.is_empty() {
            if ver.is_empty() {
                out.push_str(&format!(" ({name})"));
            } else {
                out.push_str(&format!(" ({name} {ver})"));
            }
        }
    }
    out
}

fn read_memory_gb() -> i64 {
    // /proc/meminfo MemTotal is in kB; round to the nearest GB the way
    // lemonade's `LinuxSystemInfo::get_physical_memory` does.
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|kb| ((kb as f64 / 1024.0 / 1024.0) + 0.5) as i64)
        .unwrap_or(0)
}

fn read_processor() -> String {
    // `lscpu` in Lemonade — we read /proc/cpuinfo directly to avoid a
    // popen + Rule A "no Python at runtime" doesn't apply here, but
    // shelling out for a single string is overkill.
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some((_, name)) = rest.split_once(':') {
                    return name.trim().to_string();
                }
            }
        }
    }
    "Unknown".to_string()
}

// ─── Informational / not-supported handlers ─────────────────────────────

async fn load() -> impl IntoResponse {
    not_supported("/api/v1/load")
}

async fn unload() -> impl IntoResponse {
    not_supported("/api/v1/unload")
}

async fn pull() -> impl IntoResponse {
    not_supported("/api/v1/pull")
}

#[derive(Debug, Deserialize)]
struct PullVariantsQuery {
    #[serde(default)]
    checkpoint: Option<String>,
}

async fn pull_variants(Query(q): Query<PullVariantsQuery>) -> Json<Value> {
    // Match Lemonade's `fetch_pull_variants` shape (hf_variants.cpp:190)
    // with an empty variants list, since we don't pull from HF.
    let ckpt = q.checkpoint.unwrap_or_default();
    Json(json!({
        "status": "not_supported",
        "endpoint": "/api/v1/pull/variants",
        "reason": "1bit-systems manages models via packages.toml; this endpoint is informational",
        "checkpoint": ckpt,
        "recipe": "1bit-ternary",
        "repo_kind": "ternary",
        "suggested_name": "",
        "suggested_labels": [],
        "mmproj_files": [],
        "variants": [],
    }))
}

async fn delete() -> impl IntoResponse {
    not_supported("/api/v1/delete")
}

async fn install() -> impl IntoResponse {
    not_supported("/api/v1/install")
}

async fn uninstall() -> impl IntoResponse {
    not_supported("/api/v1/uninstall")
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_os_version_is_nonempty_on_linux() {
        // Sanity check — on the Strix Halo box this returns
        // "Linux-<kernel> (CachyOS Linux ...)". On CI it's still
        // non-empty. On macOS/Windows we'd skip this test.
        let v = read_os_version();
        assert!(!v.is_empty());
        assert!(v.starts_with("Linux"));
    }

    #[test]
    fn read_memory_gb_is_nonzero_on_linux() {
        // Anything > 0 is fine — CI runners have at least 1 GB.
        assert!(read_memory_gb() > 0);
    }

    #[test]
    fn model_card_branding_is_1bit_systems() {
        let c = model_card("halo-1bit-2b", true);
        assert_eq!(c["owned_by"], "1bit systems");
        assert_eq!(c["object"], "model");
        assert_eq!(c["recipe"], "1bit-ternary");
        assert_eq!(c["quant_format"], "ternary-1.58bpw");
        assert_eq!(c["context_length"], 2048);
        assert_eq!(c["loaded"], true);
    }

    #[test]
    fn not_supported_envelope_has_status_field() {
        let Json(v) = not_supported("/api/v1/foo");
        assert_eq!(v["status"], "not_supported");
        assert_eq!(v["endpoint"], "/api/v1/foo");
        assert!(v["reason"].as_str().unwrap().contains("packages.toml"));
    }
}
