//! HTTP-level tests for the request-id + rate-limit middleware pair.
//!
//! The unit tests in `src/middleware.rs` cover the bucket math; this file
//! drives the middleware through a real [`axum::Router`] + `tower::oneshot`
//! so we catch wiring regressions (wrong layer order, missing extension,
//! lost `Retry-After`) that pure unit tests can't see.
//!
//! Isolation notes:
//! * We build our own `AppState` per test rather than reusing the library
//!   default, so the `rate_limit` field is always the one we want.
//! * The rate-limit middleware only fires when a `ConnectInfo<SocketAddr>`
//!   extension exists on the request. We inject one via
//!   [`axum::extract::connect_info::MockConnectInfo`] because `oneshot`
//!   doesn't go through the TCP listener.
//! * Tracing-subscriber state is NOT touched here — see
//!   `tests/middleware_tracing.rs` for the one case that needs it (an
//!   isolated binary keeps the global subscriber from bleeding across
//!   tests).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::connect_info::MockConnectInfo;
use axum::http::{Request, StatusCode};
use onebit_server::backend::EchoBackend;
use onebit_server::metrics::Metrics;
use onebit_server::middleware::{REQUEST_ID_HEADER, RateLimit};
use onebit_server::routes::{AppState, build_router_with_state, default_http_client};
use tower::ServiceExt;

fn sock(ip: [u8; 4], port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3])), port)
}

fn app_with_limiter(rpm: u32) -> Router {
    let state = AppState {
        backend: Arc::new(EchoBackend::new()),
        metrics: Arc::new(Metrics::new()),
        sd_base_url: Arc::new("http://127.0.0.1:8081".to_string()),
        http_client: default_http_client(),
        rate_limit: Arc::new(RateLimit::new(rpm)),
        models: Arc::new(onebit_server::registry::ModelRegistry::empty()),
        default_chat_template: onebit_server::ChatTemplate::default(),
    };
    // `layer(MockConnectInfo)` stamps the extension on every inbound
    // request so the rate-limiter sees a client IP. In production the
    // `into_make_service_with_connect_info::<SocketAddr>()` call in
    // `main.rs` does the equivalent.
    build_router_with_state(state).layer(MockConnectInfo(sock([127, 0, 0, 1], 55555)))
}

fn chat_request(uri: &str) -> Request<Body> {
    let body = serde_json::json!({
        "model": "bitnet-b1.58-2b-4t",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
    })
    .to_string();
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn request_id_minted_when_absent() {
    let app = app_with_limiter(0);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let rid = resp
        .headers()
        .get(REQUEST_ID_HEADER)
        .expect("x-request-id must be present on response")
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        rid.len() >= 16,
        "request-id should be a uuid-ish opaque string, got {rid:?}"
    );
}

#[tokio::test]
async fn request_id_preserved_when_supplied() {
    let app = app_with_limiter(0);
    let supplied = "caddy-trace-abc-123";
    let req = Request::builder()
        .uri("/healthz")
        .header(REQUEST_ID_HEADER, supplied)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.headers()
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok()),
        Some(supplied),
        "inbound request-id must be echoed back unchanged"
    );
}

#[tokio::test]
async fn rate_limit_disabled_allows_burst() {
    let app = app_with_limiter(0);
    for _ in 0..20 {
        let resp = app
            .clone()
            .oneshot(chat_request("/v1/chat/completions"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn rate_limit_fires_after_capacity() {
    // Capacity=3 ⇒ three allowed, fourth rejected within the same second.
    let app = app_with_limiter(3);
    for i in 0..3 {
        let resp = app
            .clone()
            .oneshot(chat_request("/v1/chat/completions"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "request {i} should pass");
    }
    let resp = app
        .clone()
        .oneshot(chat_request("/v1/chat/completions"))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "4th request must 429"
    );
    // Retry-After header present and >=1.
    let ra = resp
        .headers()
        .get("retry-after")
        .expect("Retry-After header")
        .to_str()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    assert!(ra >= 1, "Retry-After seconds should be >= 1, got {ra}");
    // OpenAI-shaped error body.
    let bytes = to_bytes(resp.into_body(), 4 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");
    assert_eq!(v["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn rate_limit_applies_to_v2_chat_too() {
    // Doubles as a "v2/chat/completions route exists and returns the same
    // shape as v1" check — we drain to OK twice, then confirm the same
    // limiter instance fires at the third call.
    let app = app_with_limiter(2);
    for _ in 0..2 {
        let resp = app
            .clone()
            .oneshot(chat_request("/v2/chat/completions"))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v2/chat/completions must route to the same handler as /v1"
        );
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
    }
    let resp = app
        .clone()
        .oneshot(chat_request("/v2/chat/completions"))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "rate limit must apply to /v2/chat/completions, not only /v1"
    );
}

#[tokio::test]
async fn rate_limit_does_not_apply_to_models_or_health() {
    // Capacity=1 so any fan-out would 429 if the limiter applied here.
    let app = app_with_limiter(1);
    // Drain the bucket on the chat route.
    let _ = app
        .clone()
        .oneshot(chat_request("/v1/chat/completions"))
        .await
        .unwrap();
    // Now hammer the non-chat routes — all should stay 200.
    for _ in 0..10 {
        let r = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK, "/healthz must not rate-limit");
    }
    for _ in 0..5 {
        let r = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK, "/v1/models must not rate-limit");
    }
}

#[tokio::test]
async fn rate_limit_429_still_carries_request_id() {
    // The limiter short-circuits next.run(req) — verify the outer
    // request_id layer still sees the response and stamps the header.
    let app = app_with_limiter(1);
    let _ = app
        .clone()
        .oneshot(chat_request("/v1/chat/completions"))
        .await
        .unwrap();
    let resp = app
        .clone()
        .oneshot(chat_request("/v1/chat/completions"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        resp.headers().get(REQUEST_ID_HEADER).is_some(),
        "request-id must be stamped even on a rate-limit 429"
    );
}
