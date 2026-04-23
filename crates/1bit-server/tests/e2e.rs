//! End-to-end HTTP integration tests for 1bit-server.
//!
//! Unlike the unit tests in `src/routes.rs` which drive the axum `Router`
//! via `tower::ServiceExt::oneshot` (in-process, no sockets), these tests
//! bind the router to a real `TcpListener` on `127.0.0.1:0`, spawn the
//! axum server on a `tokio::spawn`, and hit it with a real `reqwest::Client`
//! over loopback. This exercises the full stack: kernel TCP, axum's
//! `serve`, hyper's connection handling, SSE framing over the wire, and
//! reqwest's JSON + stream decoders.
//!
//! All tests use `EchoBackend` (the default, non-`real-backend` feature),
//! so they run under plain `cargo test -p onebit-server --test e2e` with
//! no GPU / model weights needed.
//!
//! The `e2e_real_backend_chat_smoke` test is feature-gated and `#[ignore]`
//! so it only runs when the operator explicitly opts in with
//! `--features real-backend -- --ignored`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use onebit_server::backend::EchoBackend;
use onebit_server::routes::build_router;
use tokio::net::TcpListener;

/// Bind the EchoBackend-backed router to an ephemeral port on localhost,
/// spawn it on the current tokio runtime, and return the base URL
/// (`http://127.0.0.1:<port>`) to hit.
///
/// The task handle is dropped on purpose — tokio will cancel the server
/// task when the test function returns, which is fine for short-lived
/// integration tests.
async fn spawn_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr: SocketAddr = listener.local_addr().expect("local_addr");
    let app = build_router(Arc::new(EchoBackend::new()));
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });
    format!("http://{addr}")
}

/// `reqwest::Client` with `.no_proxy()` so that host-wide `http_proxy` /
/// `https_proxy` env vars (common on dev boxes behind corp proxies or tailnet
/// sidecars) don't intercept the loopback call and break the test.
fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest::Client build")
}

// ─── 1. Non-streaming chat completion ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_chat_completions_non_streaming() {
    let base = spawn_server().await;
    let client = http_client();

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        }))
        .send()
        .await
        .expect("send chat completion");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let v: serde_json::Value = resp.json().await.expect("parse json");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "stub");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(
        v["id"]
            .as_str()
            .unwrap_or_default()
            .starts_with("chatcmpl-"),
        "expected chatcmpl- id, got {v}"
    );
    assert!(
        v["usage"]["total_tokens"].as_u64().unwrap_or(0) >= 1,
        "expected non-zero total_tokens, got {v}"
    );
}

// ─── 2. Streaming chat completion via SSE ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_chat_completions_streaming_sse() {
    let base = spawn_server().await;
    let client = http_client();

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("send streaming chat completion");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );

    // Drain the stream into a single buffer — SSE is short here (EchoBackend
    // streams "stub" char-by-char, so we expect a handful of frames).
    let mut bytes = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk");
        bytes.extend_from_slice(&chunk);
    }
    let text = String::from_utf8(bytes).expect("utf8 sse body");

    // Role opener arrives first.
    assert!(
        text.contains("\"role\":\"assistant\""),
        "missing role opener; body =\n{text}"
    );
    // EchoBackend yields "stub" char-by-char → at least 4 content deltas.
    let content_frames = text.matches("\"content\":\"").count();
    assert!(
        content_frames >= 4,
        "expected >=4 content frames, got {content_frames}; body =\n{text}"
    );
    // Final frame carries finish_reason.
    assert!(
        text.contains("\"finish_reason\":\"stop\""),
        "missing finish_reason=stop; body =\n{text}"
    );
    // OpenAI [DONE] sentinel terminates the stream.
    assert!(
        text.contains("data: [DONE]"),
        "missing [DONE] sentinel; body =\n{text}"
    );
}

// ─── 3. /v1/models returns a non-empty list ──────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_models_list_nonempty() {
    let base = spawn_server().await;
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
    assert!(!data.is_empty(), "expected non-empty model list, got {v}");
    assert_eq!(data[0]["object"], "model");
    assert!(
        data[0]["id"].as_str().is_some(),
        "first model missing id: {v}"
    );
    assert_eq!(data[0]["owned_by"], "1bit systems");
}

// ─── 4. /healthz works without a model loaded ────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_healthz_ok_without_model() {
    // EchoBackend intentionally has no model — /healthz must still answer
    // 200 so systemd's `ExecStartPost=curl --fail /healthz` succeeds during
    // the stub window before a real backend is wired up.
    let base = spawn_server().await;
    let client = http_client();

    let resp = client
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("send /healthz");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body = resp.text().await.expect("healthz body");
    assert!(body.trim() == "ok", "expected body 'ok', got {body:?}");

    // `/health` alias (C++ compat) should behave identically.
    let resp2 = client
        .get(format!("{base}/health"))
        .send()
        .await
        .expect("send /health");
    assert_eq!(resp2.status(), reqwest::StatusCode::OK);
}

// ─── 5. Malformed JSON → OpenAI-shaped JSON error envelope ──────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_malformed_request_returns_json_error() {
    let base = spawn_server().await;
    let client = http_client();

    // Empty messages[] → our handler returns ServerError::BadRequest, which
    // IntoResponse serializes as OpenAI's `{error:{message,type,...}}`
    // envelope with status 400.
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::json!({
                "model": "bitnet-b1.58-2b-4t",
                "messages": [],
            })
            .to_string(),
        )
        .send()
        .await
        .expect("send malformed chat completion");
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    let v: serde_json::Value = resp.json().await.expect("parse error json");
    assert!(
        v.get("error").is_some(),
        "error envelope missing 'error' key: {v}"
    );
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .to_lowercase()
            .contains("messages"),
        "expected 'messages' in error message, got {v}"
    );
}

// ─── 6. Real backend smoke — feature-gated + ignored by default ─────────

/// Smoke test against a live `RealBackend` (ROCm + model weights on disk).
///
/// Gated on the `real-backend` cargo feature so default CI builds don't
/// try to link librocm_cpp, and `#[ignore]` so even a local
/// `cargo test --features real-backend` skips it unless the operator
/// passes `-- --ignored` explicitly.
///
/// Run with:
/// ```sh
/// cargo test -p onebit-server --test e2e \
///     --features real-backend -- --ignored e2e_real_backend_chat_smoke
/// ```
#[cfg(feature = "real-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn e2e_real_backend_chat_smoke() {
    use onebit_server::backend::RealBackend;
    use std::path::PathBuf;

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let h1b = std::env::var("HALO_MODEL_H1B")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("{home}/1bit systems/models/halo-1bit-2b.h1b")));
    if !h1b.exists() {
        eprintln!(
            "skipping e2e_real_backend_chat_smoke: {} not found",
            h1b.display()
        );
        return;
    }

    let backend =
        Arc::new(RealBackend::new(&h1b).expect("RealBackend init (GPU + model weights required)"));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local_addr");
    let app = build_router(backend);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum serve");
    });
    let base = format!("http://{addr}");
    let client = http_client();

    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [{"role": "user", "content": "Say the single word 'halo' and stop."}],
            "max_tokens": 8,
            "stream": false,
        }))
        .send()
        .await
        .expect("send real-backend chat completion");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let v: serde_json::Value = resp.json().await.expect("parse real-backend json");
    assert_eq!(v["object"], "chat.completion");
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        !content.is_empty(),
        "expected non-empty real-backend content, got {v}"
    );
    assert!(
        v["usage"]["completion_tokens"].as_u64().unwrap_or(0) >= 1,
        "expected >=1 completion_tokens, got {v}"
    );
}
