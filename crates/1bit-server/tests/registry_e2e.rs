//! End-to-end integration tests for the model registry.
//!
//! Covers the two halves of the registry contract:
//!
//! 1. `/v1/models` enumerates every `.h1b` file discovered in the scan dir.
//! 2. `/v1/chat/completions` rejects requests whose `model` field is not in
//!    the registry with a 400 + OpenAI-shaped error envelope, while known
//!    ids proceed to the backend normally.
//!
//! We build the router directly with a populated `AppState` (bypassing
//! `build_router` so we can inject a registry that doesn't match the
//! EchoBackend's default `list_models()` result) and drive it over a real
//! loopback socket.

use std::fs;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use onebit_server::backend::EchoBackend;
use onebit_server::metrics::Metrics;
use onebit_server::middleware::RateLimit;
use onebit_server::registry::ModelRegistry;
use onebit_server::routes::{AppState, build_router_with_state, default_http_client};
use tokio::net::TcpListener;

/// Tempdir scoped to a test run — auto-cleans on drop.
struct TmpDir(PathBuf);
impl TmpDir {
    fn new(tag: &str) -> Self {
        let root = std::env::var_os("CARGO_TARGET_TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let pid = std::process::id();
        let nonce: u64 = fastrand::u64(..);
        let path = root.join(format!("onebit-server-reg-e2e-{tag}-{pid}-{nonce:016x}"));
        fs::create_dir_all(&path).expect("create tempdir");
        Self(path)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn touch(p: &Path) {
    let mut f = fs::File::create(p).expect("create file");
    f.write_all(b"0").expect("write byte");
}

/// Build a registry-backed AppState, bind to an ephemeral port, spawn the
/// server, and return the base URL.
async fn spawn_with_registry(registry: ModelRegistry) -> String {
    let state = AppState {
        backend: Arc::new(EchoBackend::new()),
        metrics: Arc::new(Metrics::new()),
        sd_base_url: Arc::new("http://127.0.0.1:8081".to_string()),
        http_client: default_http_client(),
        rate_limit: Arc::new(RateLimit::new(0)),
        models: Arc::new(registry),
        default_chat_template: onebit_server::ChatTemplate::default(),
    };
    let app = build_router_with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr: SocketAddr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });
    format!("http://{addr}")
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest::Client build")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_endpoint_returns_all_discovered() {
    let dir = TmpDir::new("models-endpoint");
    touch(&dir.path().join("alpha.h1b"));
    touch(&dir.path().join("beta.h1b"));
    touch(&dir.path().join("gamma.h1b"));
    // Sidecar on beta just to prove it doesn't break the wire shape.
    fs::write(
        dir.path().join("beta.json"),
        r#"{"description": "second model"}"#,
    )
    .expect("write sidecar");

    let registry = ModelRegistry::from_dir(dir.path());
    let base = spawn_with_registry(registry).await;
    let client = http_client();

    let resp = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .expect("send /v1/models");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let v: serde_json::Value = resp.json().await.expect("parse /v1/models");
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().expect("data is array");
    let ids: Vec<String> = data
        .iter()
        .map(|c| c["id"].as_str().unwrap_or("").to_string())
        .collect();
    assert_eq!(ids, vec!["alpha", "beta", "gamma"], "got {ids:?}");
    // Existing ModelCard schema preserved (no new fields leaking).
    for card in data {
        assert_eq!(card["object"], "model");
        assert_eq!(card["owned_by"], "1bit systems");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_completion_bad_model_returns_400_with_hint() {
    let dir = TmpDir::new("bad-model");
    touch(&dir.path().join("halo-1bit-2b.h1b"));
    touch(&dir.path().join("halo-bitnet-2b-tq2.h1b"));
    let registry = ModelRegistry::from_dir(dir.path());
    let base = spawn_with_registry(registry).await;
    let client = http_client();

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "does-not-exist",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("send bad-model chat");
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    let v: serde_json::Value = resp.json().await.expect("parse error envelope");
    // OpenAI-shaped error: { error: { message, type, code } }
    let err = v.get("error").expect("error envelope present");
    assert_eq!(err["type"], "invalid_request_error");
    assert_eq!(err["code"], "bad_request");
    let msg = err["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("does-not-exist"),
        "message missing requested id: {msg:?}"
    );
    assert!(
        msg.contains("halo-1bit-2b") && msg.contains("halo-bitnet-2b-tq2"),
        "message missing available ids hint: {msg:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_completion_good_model_proceeds() {
    let dir = TmpDir::new("good-model");
    touch(&dir.path().join("halo-1bit-2b.h1b"));
    let registry = ModelRegistry::from_dir(dir.path());
    let base = spawn_with_registry(registry).await;
    let client = http_client();

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "halo-1bit-2b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        }))
        .send()
        .await
        .expect("send good-model chat");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let v: serde_json::Value = resp.json().await.expect("parse chat response");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "stub");
    assert_eq!(v["model"], "halo-1bit-2b");
}

/// The v2 alias (Caddy's canonical gen-2 path) goes through the same
/// handler → same validation logic. Regression guard so future refactors
/// don't skip validation on one of the two routes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_chat_completion_bad_model_also_rejected() {
    let dir = TmpDir::new("v2-bad-model");
    touch(&dir.path().join("halo-1bit-2b.h1b"));
    let registry = ModelRegistry::from_dir(dir.path());
    let base = spawn_with_registry(registry).await;
    let client = http_client();

    let resp = client
        .post(format!("{base}/v2/chat/completions"))
        .json(&serde_json::json!({
            "model": "unknown",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("send v2 bad-model");
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

/// When the registry is empty (no `--models-dir` set, EchoBackend stub),
/// the server stays permissive so the default smoke path keeps working.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_registry_accepts_any_model() {
    let base = spawn_with_registry(ModelRegistry::empty()).await;
    let client = http_client();

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "anything-goes",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        }))
        .send()
        .await
        .expect("send chat");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    // And /v1/models returns a valid-but-empty envelope.
    let list: serde_json::Value = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .expect("send models")
        .json()
        .await
        .expect("parse models");
    assert_eq!(list["object"], "list");
    assert!(list["data"].as_array().unwrap().is_empty());
}
