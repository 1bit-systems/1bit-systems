//! Isolated integration test: request-id middleware + tracing.
//!
//! `tracing_subscriber::fmt().try_init()` installs a **process-global**
//! dispatcher. If we initialize it inside a test case in the same binary
//! as other tests, the subscriber state leaks: subsequent tests either
//! see double-logged lines, or their own `try_init()` silently fails.
//!
//! By living in its own integration-test file, cargo compiles this as a
//! separate binary — the subscriber globals are fresh for this test and
//! cannot interfere with `tests/middleware.rs`. Worth the extra file.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::connect_info::MockConnectInfo;
use axum::http::{Request, StatusCode};
use onebit_server::backend::EchoBackend;
use onebit_server::metrics::Metrics;
use onebit_server::middleware::{REQUEST_ID_HEADER, RateLimit};
use onebit_server::routes::{AppState, build_router_with_state, default_http_client};
use tower::ServiceExt;
use tracing_subscriber::EnvFilter;

fn app() -> Router {
    let state = AppState {
        backend: Arc::new(EchoBackend::new()),
        metrics: Arc::new(Metrics::new()),
        sd_base_url: Arc::new("http://127.0.0.1:8081".to_string()),
        http_client: default_http_client(),
        rate_limit: Arc::new(RateLimit::new(0)),
        models: Arc::new(onebit_server::registry::ModelRegistry::empty()),
        default_chat_template: onebit_server::ChatTemplate::default(),
    };
    build_router_with_state(state).layer(MockConnectInfo(SocketAddr::new(
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        55555,
    )))
}

#[tokio::test]
async fn request_id_flows_with_tracing_subscriber_installed() {
    // Best-effort init — if something else has already installed a
    // subscriber we don't care, we just want to confirm that turning
    // tracing on doesn't break the middleware wiring.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("onebit_server=trace"))
        .with_test_writer()
        .try_init();

    let resp = app()
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
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .expect("x-request-id on response");
    assert!(!rid.is_empty());
}
