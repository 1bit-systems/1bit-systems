//! 1bit-helm — egui/eframe desktop client against lemond (the canonical
//! OpenAI / Ollama / Anthropic-compat HTTP server at /home/bcloud/repos/lemonade/).
//!
//! Renamed from halo-gaia on 2026-04-20 to avoid a naming collision with
//! AMD GAIA (amd-gaia.ai). See `docs/wiki/AMD-GAIA-Integration.md`.
//!
//! Config is via env (same keys the TUI scaffold used — callers already rely
//! on these):
//!   HALO_HELM_URL      — /v1/* gateway URL (default http://127.0.0.1:8200)
//!   HALO_HELM_LANDING  — /_live/* base URL  (default http://127.0.0.1:8190)
//!   HALO_HELM_MODEL    — model id           (default 1bit-monster-2b)
//!   HALO_HELM_TOKEN    — optional bearer (usually comes from keyring)
//!
//! Legacy HALO_GAIA_* env vars are honored as fallbacks so existing shell
//! profiles keep working without a scripted rewrite.
//!
//! Runtime layout: we spin up a multi-thread tokio runtime alongside
//! eframe's event loop (eframe itself is *not* async). `attach_runtime`
//! hands the app a clone of its `Handle` + a shared `reqwest::Client` so
//! every pane can spawn work onto a single runtime — no per-pane runtime
//! creep.

use anyhow::Result;
use onebit_helm::{HelmApp, SessionConfig};

fn load_config() -> SessionConfig {
    // Default /v1/* gateway is lemonade on :8200. The old scaffold
    // defaulted to :8180 (1bit-server); lemonade is the correct target
    // for chat/completions + models today (project_lemonade_10_2_pivot).
    let url = env_any(&["HALO_HELM_URL", "HALO_GAIA_URL"])
        .unwrap_or_else(|| "http://127.0.0.1:8200".to_string());
    let model = env_any(&["HALO_HELM_MODEL", "HALO_GAIA_MODEL"])
        .unwrap_or_else(|| "1bit-monster-2b".to_string());
    let mut cfg = SessionConfig::new(url, model);
    cfg.bearer = env_any(&["HALO_HELM_TOKEN", "HALO_GAIA_TOKEN"]);
    cfg
}

fn landing_url() -> String {
    env_any(&["HALO_HELM_LANDING", "HALO_GAIA_LANDING"])
        .unwrap_or_else(|| "http://127.0.0.1:8190".to_string())
}

fn env_any(keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Ok(v) = std::env::var(k)
            && !v.is_empty()
        {
            return Some(v);
        }
    }
    None
}

fn main() -> Result<()> {
    // Background runtime for network I/O (SSE stream + REST fetches).
    // eframe owns the main thread for the window event loop; the runtime
    // lives on a side-thread and we hand out `Handle`s to workers.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .thread_name("helm-tokio")
        .build()?;
    let rt_handle = rt.handle().clone();
    // Leak the runtime: it lives for the process lifetime, and dropping
    // a multi-thread runtime from inside a Handle-rooted worker would
    // panic. Simpler to let the OS reap on exit.
    std::mem::forget(rt);

    let cfg = load_config();
    let landing = landing_url();

    let http = reqwest::Client::builder()
        .user_agent(concat!("1bit-helm/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| anyhow::anyhow!("reqwest build: {e}"))?;

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("1bit monster — helm")
            .with_inner_size([1024.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "1bit-helm",
        native_options,
        Box::new(move |cc| {
            let mut app = HelmApp::from_cc(cc, cfg);
            app.landing_url = landing;
            app.attach_runtime(rt_handle, http);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))?;
    Ok(())
}
