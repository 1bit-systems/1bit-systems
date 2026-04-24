//! `1bit-server` binary — OpenAI-compatible HTTP daemon.
//!
//! Replaces `bitnet_decode --server` (C++) and `lemonade-server` (Python).
//! Ships with an [`EchoBackend`] by default; pass `--features real-backend`
//! at build time plus `--model <path>` at launch time to load a real
//! `.h1b` model into `onebit_router::Router` and swap it in.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use onebit_server::middleware::RateLimit;
use onebit_server::registry::ModelRegistry;
use onebit_server::routes::{AppState, build_router_with_state, default_http_client};
use onebit_server::{EchoBackend, InferenceBackend, Metrics, shutdown_signal};

/// 1bit-server — OpenAI-compatible HTTP front door.
#[derive(Parser, Debug)]
#[command(name = "1bit-server", version, about)]
struct Args {
    /// Bind address (host:port). Overrides $HALO_SERVER_BIND.
    #[arg(long, env = "HALO_SERVER_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Path to a .h1b OR .gguf file to load via 1bit-router.
    ///
    /// Ignored when the binary is built without `--features real-backend`.
    /// Format is sniffed from magic bytes + extension; `.h1b` reads its
    /// tokenizer from a sibling `.htok` (matches gen-1's on-disk layout),
    /// `.gguf` reads its tokenizer from the GGUF metadata block.
    #[arg(long, env = "HALO_SERVER_MODEL")]
    model: Option<PathBuf>,

    /// Base URL of the Stable Diffusion sidecar (sd-server). The Layer-A
    /// image proxy at `/v{1,2}/images/generations` forwards to
    /// `{sd_url}/v1/images/generations` with a 900 s timeout.
    #[arg(long, env = "HALO_SD_URL", default_value = "http://127.0.0.1:8081")]
    sd_url: String,

    /// Per-IP requests-per-minute cap on `/v{1,2}/chat/completions`.
    /// Token-bucket: capacity = this value, refill = rpm/60 tokens/sec.
    /// Set to `0` to disable the limiter entirely (useful for the
    /// strixhalo box behind Caddy where tailnet auth already gates).
    #[arg(long, env = "HALO_SERVER_RATE_LIMIT_RPM", default_value_t = 30)]
    rate_limit_rpm: u32,

    /// Directory to scan for `.h1b` models at startup. Discovered files
    /// populate `/v1/models` and gate the `model` field on chat
    /// completions. Leave unset to derive from `--model`'s parent
    /// directory (backwards-compatible) or fall back to
    /// `/home/bcloud/1bit-halo-models/models`.
    ///
    /// The server still only loads **one** backend at a time; this is
    /// discovery + validation, not concurrent multi-model serving.
    #[arg(long, env = "HALO_SERVER_MODELS_DIR")]
    models_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,onebit_server=debug,tower_http=info")),
        )
        .init();

    let args = Args::parse();

    let backend: Arc<dyn InferenceBackend> = build_backend(&args)?;
    info!(sd_url = %args.sd_url, "image proxy upstream configured");
    if args.rate_limit_rpm == 0 {
        info!("rate limiter disabled (--rate-limit-rpm=0)");
    } else {
        info!(
            rpm = args.rate_limit_rpm,
            "per-IP rate limiter enabled on /v{{1,2}}/chat/completions"
        );
    }

    // Resolve the scan directory. Precedence:
    //   1. --models-dir / $HALO_SERVER_MODELS_DIR
    //   2. parent dir of --model (so `--model .../models/x.h1b` keeps
    //      sibling `.h1b` files discoverable without an extra flag)
    //   3. the canonical /home/bcloud/1bit-halo-models/models layout
    let models_dir = args
        .models_dir
        .clone()
        .or_else(|| {
            args.model
                .as_deref()
                .and_then(|m| m.parent())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from("/home/bcloud/1bit-halo-models/models"));
    info!(dir = %models_dir.display(), "scanning for .h1b models");
    let mut registry = ModelRegistry::from_dir(&models_dir);
    // Make sure the backend's advertised model id is reachable even if it
    // lives outside the scan dir — keeps /v1/models honest when someone
    // runs with `--model /tmp/foo.h1b` and no `--models-dir`.
    for card in backend.list_models() {
        registry.ensure_id(card.id);
    }
    info!(count = registry.entries().len(), ids = ?registry.ids(), "model registry ready");

    // HALO_CHAT_TEMPLATE picks the server-wide default prompt template
    // (`llama3` | `short` | `raw`). The `short` / `raw` variants skip
    // part or all of the Llama-3 framing on the prefill path — see
    // `chat_template` module + `docs/wiki/Chat-Template-Options.md`.
    // Clients may override per-request via `X-Halo-Chat-Template`.
    let default_chat_template = onebit_server::ChatTemplate::from_env();
    if default_chat_template != onebit_server::ChatTemplate::default() {
        info!(
            template = ?default_chat_template,
            "HALO_CHAT_TEMPLATE override active"
        );
    }

    let state = AppState {
        backend,
        metrics: Arc::new(Metrics::new()),
        sd_base_url: Arc::new(args.sd_url.clone()),
        http_client: default_http_client(),
        rate_limit: Arc::new(RateLimit::new(args.rate_limit_rpm)),
        models: Arc::new(registry),
        default_chat_template,
    };
    let app = build_router_with_state(state);

    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    info!(addr = %args.bind, "1bit-server listening");

    // `into_make_service_with_connect_info::<SocketAddr>()` is what makes
    // the client IP visible to the rate-limiter middleware via
    // `ConnectInfo<SocketAddr>`. Without it the extractor misses and the
    // limiter falls through to allow — see `middleware::rate_limit`.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .context("axum serve failed")?;

    info!("1bit-server stopped cleanly");
    Ok(())
}

#[cfg(feature = "real-backend")]
fn build_backend(args: &Args) -> Result<Arc<dyn InferenceBackend>> {
    use onebit_server::backend::RealBackend;
    match &args.model {
        Some(path) => {
            info!(model = %path.display(), "real-backend enabled — loading 1bit-router");
            let real = RealBackend::new(path)
                .with_context(|| format!("1bit-router load {}", path.display()))?;
            Ok(Arc::new(real))
        }
        None => {
            // real-backend compiled in but no model path provided — keep the
            // stub so the binary still starts (useful for health checks /
            // reading /v1/models before anything is loaded).
            info!("real-backend compiled but no --model given; using EchoBackend");
            Ok(Arc::new(EchoBackend::new()))
        }
    }
}

#[cfg(not(feature = "real-backend"))]
fn build_backend(args: &Args) -> Result<Arc<dyn InferenceBackend>> {
    if args.model.is_some() {
        tracing::warn!(
            "`--model` was passed but this binary was built without `--features real-backend`; \
             ignoring and using EchoBackend"
        );
    }
    Ok(Arc::new(EchoBackend::new()))
}
