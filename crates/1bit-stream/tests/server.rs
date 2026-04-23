//! Integration tests — spin up the real axum app on a random port and
//! hit it over HTTP.

use onebit_stream::{AppState, AuthConfig, build};
use std::net::SocketAddr;
use tempfile::tempdir;
use tokio::net::TcpListener;

async fn spawn(state: AppState) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = build(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

#[tokio::test]
async fn empty_catalog_dir_lists_empty_array() {
    let dir = tempdir().unwrap();
    let state = AppState::new(dir.path().to_path_buf(), AuthConfig::new(None, None));
    let (count, errs) = state.reindex().await;
    assert_eq!(count, 0);
    assert!(errs.is_empty(), "empty dir should not produce errors, got {errs:?}");

    let addr = spawn(state).await;

    let client = reqwest::Client::new();
    let res = client
        .get(format!("http://{addr}/v1/catalogs"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let v: serde_json::Value = res.json().await.unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn health_endpoint() {
    let dir = tempdir().unwrap();
    let state = AppState::new(dir.path().to_path_buf(), AuthConfig::new(None, None));
    let addr = spawn(state).await;
    let body = reqwest::get(format!("http://{addr}/v1/health"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(body, "ok");
}

#[tokio::test]
async fn missing_catalog_returns_404() {
    let dir = tempdir().unwrap();
    let state = AppState::new(dir.path().to_path_buf(), AuthConfig::new(None, None));
    let addr = spawn(state).await;
    let res = reqwest::get(format!("http://{addr}/v1/catalogs/does-not-exist"))
        .await
        .unwrap();
    assert_eq!(res.status(), 404);
}

#[tokio::test]
async fn reindex_endpoint_open_without_admin_bearer() {
    // With admin_bearer = None, /internal/reindex is effectively open —
    // the intended deployment pins the server to 127.0.0.1 so only the
    // operator can hit it. Setting HALO_STREAM_ADMIN_BEARER closes this.
    let dir = tempdir().unwrap();
    let state = AppState::new(dir.path().to_path_buf(), AuthConfig::new(None, None));
    let addr = spawn(state).await;
    let client = reqwest::Client::new();
    let res = client
        .post(format!("http://{addr}/internal/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let v: serde_json::Value = res.json().await.unwrap();
    assert_eq!(v["loaded"], 0);
}
