//! 1bit engine — OpenAI-compatible HTTP frontend for AMD Strix Halo.
//!
//! A pure-Rust axum proxy that spawns the C++/HIP `bitnet_decode --server`
//! binary as a subprocess and reverse-proxies OpenAI-compatible requests
//! to it. The Rust layer does no inference itself; all kernel work lives
//! in librocm_cpp.so, reached via the child process.
//!
//! Architecture:
//!   axum HTTP server (:13305)
//!     → spawns bitnet_decode --server as subprocess
//!     → health-checks until ready
//!     → proxies /v1/* requests with streaming passthrough

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use reqwest::Client;
use std::net::SocketAddr;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::sleep;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "onebit", about = "1-bit inference engine for Strix Halo")]
struct Args {
    /// Path to .h1b model file
    #[arg(short, long, default_value = "./model.h1b")]
    model: String,

    /// Port for the OpenAI-compatible API
    #[arg(short, long, default_value_t = 13305)]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Path to bitnet_decode binary
    #[arg(long, default_value = "bitnet_decode")]
    bitnet_decode: String,

    /// Internal port for bitnet_decode (random if not set)
    #[arg(long, default_value_t = 0)]
    backend_port: u16,

    /// Extra args to pass to bitnet_decode
    #[arg(long, default_value = "")]
    bitnet_args: String,

    /// Run prefill kernel auto-tuning at startup (picks fastest variant)
    #[arg(long)]
    tune_prefill: bool,

    /// Force a specific prefill variant (0=4i, 1=4h, 2=4k, 3=4f, 4=FP16-B, 5=4c, 6=4g)
    #[arg(long)]
    prefill_variant: Option<u8>,

    /// Pre-decode weights to FP16 at load time for maximum prefill throughput
    #[arg(long)]
    fp16_weights: bool,
}

#[derive(Clone)]
struct AppState {
    client: Client,
    backend_url: String,
    _child: Arc<std::sync::Mutex<Option<Child>>>,
}

impl Drop for AppState {
    fn drop(&mut self) {
        // Best-effort kill on destructor (panics/signals).
        if let Ok(mut guard) = self._child.lock()
            && let Some(ref mut c) = *guard
        {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit=info".into()),
        )
        .init();

    let args = Args::parse();

    // Find a free port for the backend if not specified
    let backend_port = if args.backend_port == 0 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    } else {
        args.backend_port
    };

    // Build command
    // bitnet_decode uses positional CLI: <model> --server <port> [flags].
    // The model path goes first (argv[1]), then --server <port> to enable
    // HTTP server mode, then remaining flags.
    let mut cmd = Command::new(&args.bitnet_decode);
    cmd.arg(&args.model)
        .arg("--server")
        .arg(backend_port.to_string())
        .arg("--bind")
        .arg("127.0.0.1");

    // Kernel tuning flags
    if args.tune_prefill {
        cmd.arg("--tune-prefill");
        info!("Kernel auto-tuning enabled — will benchmark prefill variants at startup");
    }
    if let Some(v) = args.prefill_variant {
        cmd.arg("--prefill-variant").arg(v.to_string());
        info!("Forcing prefill variant: {v}");
    }
    if args.fp16_weights {
        cmd.arg("--fp16-weights");
        info!("FP16 weight pre-decode enabled — maximum prefill throughput");
    }

    if !args.bitnet_args.is_empty() {
        for arg in args.bitnet_args.split_whitespace() {
            cmd.arg(arg);
        }
    }

    // Set ROCm environment for the child
    cmd.env("HSA_OVERRIDE_GFX_VERSION", "11.5.1");
    cmd.env("HSA_ENABLE_SDMA", "0");

    info!("Starting bitnet_decode on port {backend_port}...");
    let mut child = cmd
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("Failed to start bitnet_decode. Is rocm-cpp installed?");

    let backend_url = format!("http://127.0.0.1:{backend_port}");
    let client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .unwrap();

    // Wait for backend to be ready
    info!("Waiting for bitnet_decode to be ready...");
    for i in 0..120 {
        match client.get(format!("{backend_url}/health")).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!("bitnet_decode ready after {i}s");
                break;
            }
            _ => {
                if i > 0 && i % 10 == 0 {
                    info!("Still waiting... ({i}s)");
                }
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            error!("bitnet_decode exited early with status: {status:?}");
            std::process::exit(1);
        }
        sleep(Duration::from_secs(1)).await;
    }

    if let Ok(Some(status)) = child.try_wait() {
        error!("bitnet_decode exited with status: {status:?}");
        std::process::exit(1);
    }

    let state = AppState {
        client,
        backend_url: backend_url.clone(),
        _child: Arc::new(std::sync::Mutex::new(Some(child))),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(proxy_get))
        .route("/v1/models/{name}", get(proxy_get))
        .route("/v1/chat/completions", post(proxy_post))
        .route("/v1/completions", post(proxy_post))
        .route("/v1/embeddings", post(proxy_post))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse().unwrap();
    info!("1bit engine listening on http://{addr}");
    info!("Backend: {backend_url}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn proxy_get(State(state): State<AppState>, uri: axum::http::Uri) -> Response {
    let path = uri.path().to_string();
    let url = format!("{}{}", state.backend_url, path);

    match state.client.get(&url).send().await {
        Ok(resp) => {
            let status = resp.status();
            let headers = resp.headers().clone();
            let body = resp.bytes().await.unwrap_or_default();
            let mut response = Response::new(Body::from(body));
            *response.status_mut() = status;
            for (k, v) in headers.iter() {
                if k != "transfer-encoding" && k != "content-encoding" {
                    response.headers_mut().insert(k.clone(), v.clone());
                }
            }
            response
        }
        Err(e) => {
            error!("GET proxy error: {e}");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

async fn proxy_post(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let path = uri.path().to_string();
    let url = format!("{}{}", state.backend_url, path);

    let mut request = state
        .client
        .post(&url)
        .header("Content-Type", "application/json");

    if let Some(auth) = headers.get("authorization") {
        request = request.header("authorization", auth);
    }

    match request.body(body).send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let is_stream = resp_headers
                .get("content-type")
                .map(|v| v.to_str().unwrap_or("").contains("text/event-stream"))
                .unwrap_or(false);

            if is_stream {
                let stream = resp.bytes_stream();
                let body = Body::from_stream(stream);
                let mut response = Response::new(body);
                *response.status_mut() = status;
                for (k, v) in resp_headers.iter() {
                    if k != "transfer-encoding" {
                        response.headers_mut().insert(k.clone(), v.clone());
                    }
                }
                response
            } else {
                let body = resp.bytes().await.unwrap_or_default();
                let mut response = Response::new(Body::from(body));
                *response.status_mut() = status;
                for (k, v) in resp_headers.iter() {
                    if k != "transfer-encoding" && k != "content-encoding" {
                        response.headers_mut().insert(k.clone(), v.clone());
                    }
                }
                response
            }
        }
        Err(e) => {
            error!("POST proxy error: {e}");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_app() -> Router {
        let state = AppState {
            client: reqwest::Client::new(),
            backend_url: "http://127.0.0.1:1".into(),
            _child: Arc::new(std::sync::Mutex::new(None)),
        };
        Router::new()
            .route("/health", get(health))
            .with_state(state)
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn cli_args_parse() {
        use clap::Parser;
        let args =
            Args::try_parse_from(["onebit", "--model", "test.h1b", "--port", "9999"]).unwrap();
        assert_eq!(args.model, "test.h1b");
        assert_eq!(args.port, 9999);
        assert_eq!(args.host, "127.0.0.1");
    }

    #[test]
    fn cli_args_defaults() {
        use clap::Parser;
        let args = Args::try_parse_from(["onebit"]).unwrap();
        assert_eq!(args.model, "./model.h1b");
        assert_eq!(args.port, 13305);
        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.bitnet_decode, "bitnet_decode");
    }

    #[tokio::test]
    async fn get_proxy_returns_bad_gateway_when_backend_down() {
        let state = AppState {
            client: reqwest::Client::new(),
            backend_url: "http://127.0.0.1:65535".into(),
            _child: Arc::new(std::sync::Mutex::new(None)),
        };
        let app = Router::new()
            .route("/v1/models", get(proxy_get))
            .with_state(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn post_proxy_returns_bad_gateway_when_backend_down() {
        let state = AppState {
            client: reqwest::Client::new(),
            backend_url: "http://127.0.0.1:65535".into(),
            _child: Arc::new(std::sync::Mutex::new(None)),
        };
        let app = Router::new()
            .route("/v1/chat/completions", post(proxy_post))
            .with_state(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn cli_args_tune_prefill() {
        use clap::Parser;
        let args = Args::try_parse_from(["onebit", "--tune-prefill"]).unwrap();
        assert!(args.tune_prefill);
        assert!(!args.fp16_weights);
    }

    #[test]
    fn cli_args_fp16_weights() {
        use clap::Parser;
        let args =
            Args::try_parse_from(["onebit", "--fp16-weights", "--prefill-variant", "4"]).unwrap();
        assert!(args.fp16_weights);
        assert_eq!(args.prefill_variant, Some(4));
    }
}
