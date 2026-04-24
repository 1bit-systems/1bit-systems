//! Integration tests for the Lemonade-desktop-compat `/api/v1/*` routes.
//!
//! Drives the axum router via `tower::ServiceExt::oneshot` (in-process,
//! no socket) so they run under plain `cargo test -p onebit-server` with
//! no GPU / Lemonade desktop instance needed. The schema asserts here pin
//! the exact response shape Lemonade desktop expects — see
//! `crates/1bit-server/src/lemonade_api.rs` for the policy + source refs.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use onebit_server::backend::EchoBackend;
use onebit_server::routes::build_router;
use serde_json::Value;
use tower::ServiceExt;

fn app() -> Router {
    build_router(Arc::new(EchoBackend::new()))
}

async fn get_json(uri: &str) -> (StatusCode, Value) {
    let resp = app()
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request build"),
        )
        .await
        .expect("router oneshot");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 64 * 1024)
        .await
        .expect("body bytes");
    let v: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "GET {uri} returned non-JSON body: {e}\nbody = {:?}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, v)
}

async fn post_json(uri: &str, body: Value) -> (StatusCode, Value) {
    let resp = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request build"),
        )
        .await
        .expect("router oneshot");
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 64 * 1024)
        .await
        .expect("body bytes");
    let v: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "POST {uri} returned non-JSON body: {e}\nbody = {:?}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, v)
}

// ─── 1. /api/v1/health ───────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok_envelope() {
    let (status, v) = get_json("/api/v1/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "ok");
    assert_eq!(v["runtime"], "1bit-systems");
    assert!(v["version"].is_string(), "version field must be a string");
    // `model_loaded` is null with EchoBackend (no .h1b on disk).
    assert!(v["model_loaded"].is_null() || v["model_loaded"].is_string());
    assert!(v["all_models_loaded"].is_array());
    assert_eq!(v["max_models"]["llm"], 1);
    assert!(v["uptime_seconds"].as_u64().is_some());
}

// ─── 2. /api/v1/models ───────────────────────────────────────────────────

#[tokio::test]
async fn models_returns_lemonade_envelope() {
    let (status, v) = get_json("/api/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().expect("data must be array");
    // EchoBackend advertises one model id ("bitnet-b1.58-2b-4t") which
    // build_router seeds into the registry at construction.
    assert!(!data.is_empty(), "EchoBackend should surface at least 1 model");
    let first = &data[0];
    assert_eq!(first["object"], "model");
    assert_eq!(first["owned_by"], "1bit systems");
    assert_eq!(first["recipe"], "1bit-ternary");
    assert_eq!(first["quant_format"], "ternary-1.58bpw");
    assert_eq!(first["context_length"], 2048);
    assert!(first["id"].is_string());
}

#[tokio::test]
async fn models_show_all_query_does_not_break() {
    // Lemonade desktop sends `?show_all=false` and `?show_all=true`; both
    // must 200 with the same envelope shape.
    let (status, _) = get_json("/api/v1/models?show_all=true").await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get_json("/api/v1/models?show_all=false").await;
    assert_eq!(status, StatusCode::OK);
}

// ─── 3. /api/v1/models/{name} ────────────────────────────────────────────

#[tokio::test]
async fn model_by_id_returns_card_when_present() {
    // EchoBackend's id is "bitnet-b1.58-2b-4t" (see backend::EchoBackend).
    let (status, v) = get_json("/api/v1/models/bitnet-b1.58-2b-4t").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["id"], "bitnet-b1.58-2b-4t");
    assert_eq!(v["owned_by"], "1bit systems");
    assert_eq!(v["object"], "model");
}

#[tokio::test]
async fn model_by_id_returns_404_when_missing() {
    let (status, v) = get_json("/api/v1/models/nope-not-real").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(v["error"]["message"].as_str().unwrap().contains("not found"));
    assert_eq!(v["error"]["code"], "model_not_found");
}

// ─── 4. /api/v1/system-info ──────────────────────────────────────────────

#[tokio::test]
async fn system_info_returns_lemonade_shape() {
    let (status, v) = get_json("/api/v1/system-info").await;
    assert_eq!(status, StatusCode::OK);
    // Lemonade-shape (legacy keys + our extra flat ones).
    assert!(v["OS Version"].is_string());
    assert!(v["Processor"].is_string());
    assert!(v["Physical Memory"].is_string());
    assert_eq!(v["os"], "linux");
    assert_eq!(v["gpu"], "gfx1151");
    assert_eq!(v["runtime"], "1bit-systems");
    assert!(v["devices"]["cpu"].is_object());
    assert!(v["devices"]["amd_gpu"].is_array());
    // Must contain at least one recipe entry — Lemonade CLI's `lemonade
    // recipes` walks this map and crashes on empty.
    let recipes = v["recipes"].as_object().expect("recipes must be object");
    assert!(!recipes.is_empty());
}

// ─── 5-11. POST stubs return 200 with status:"not_supported" ─────────────

#[tokio::test]
async fn load_returns_not_supported() {
    let (status, v) = post_json(
        "/api/v1/load",
        serde_json::json!({"model_name": "halo-1bit-2b"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
    assert!(v["reason"].as_str().unwrap().contains("packages.toml"));
}

#[tokio::test]
async fn unload_returns_not_supported() {
    let (status, v) = post_json("/api/v1/unload", serde_json::json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
}

#[tokio::test]
async fn pull_returns_not_supported() {
    let (status, v) = post_json(
        "/api/v1/pull",
        serde_json::json!({"model_name": "user.foo", "checkpoint": "x/y"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
}

#[tokio::test]
async fn pull_variants_returns_not_supported_with_lemonade_shape() {
    let (status, v) = get_json("/api/v1/pull/variants?checkpoint=meta-llama/Llama-3.2-1B").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
    // Lemonade desktop indexes into these fields even on the empty case;
    // include them so a destructure doesn't blow up.
    assert!(v["variants"].is_array());
    assert_eq!(v["checkpoint"], "meta-llama/Llama-3.2-1B");
}

#[tokio::test]
async fn delete_returns_not_supported() {
    let (status, v) = post_json(
        "/api/v1/delete",
        serde_json::json!({"model_name": "halo-1bit-2b"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
}

#[tokio::test]
async fn install_returns_not_supported() {
    let (status, v) = post_json(
        "/api/v1/install",
        serde_json::json!({"recipe": "1bit-ternary", "backend": "rocm"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
}

#[tokio::test]
async fn uninstall_returns_not_supported() {
    let (status, v) = post_json(
        "/api/v1/uninstall",
        serde_json::json!({"recipe": "1bit-ternary", "backend": "rocm"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "not_supported");
}

// ─── Smoke: existing /v1/* routes still work ─────────────────────────────

#[tokio::test]
async fn existing_v1_models_route_still_works() {
    // Sanity check that nesting `/api/v1` didn't accidentally clobber
    // the OpenAI `/v1/models` surface. Both must coexist.
    let (status, v) = get_json("/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["object"], "list");
}

#[tokio::test]
async fn existing_healthz_route_still_works() {
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
}
