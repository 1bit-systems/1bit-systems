//! Integration test: mocked BTCPay webhook → JWT body round-trips.

use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use jsonwebtoken::{DecodingKey, Validation, decode};
use serde_json::Value;
use tower::ServiceExt;

use onebit_tier_mint::jwt::Claims;
use onebit_tier_mint::{AppState, Config, build_router};
use onebit_tier_mint::btcpay::sign_for_test as btcpay_sign;

fn test_state() -> AppState {
    AppState::for_tests(Config {
        jwt_secret: b"test-jwt-hmac-secret-at-least-32b-long".to_vec(),
        btcpay_webhook_secret: b"btcpay-secret".to_vec(),
        patreon_webhook_secret: b"patreon-secret".to_vec(),
        issuer: "1bit.systems".to_string(),
        jwt_ttl: Duration::from_secs(3600),
    })
}

#[tokio::test]
async fn btcpay_invoice_settled_mints_jwt() {
    let state = test_state();
    let app = build_router(state.clone());

    let body = serde_json::json!({
        "type": "InvoiceSettled",
        "invoiceId": "inv-integration-1",
        "metadata": {}
    })
    .to_string();
    let sig = btcpay_sign(&state.cfg.btcpay_webhook_secret, body.as_bytes());

    let req = Request::builder()
        .method("POST")
        .uri("/btcpay/webhook")
        .header("content-type", "application/json")
        .header("BTCPay-Sig", sig)
        .body(Body::from(body))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["ok"], Value::Bool(true));
    let token = json["jwt"].as_str().expect("jwt in response");

    // Verify the JWT with the same secret.
    let mut v = Validation::new(jsonwebtoken::Algorithm::HS256);
    v.set_issuer(&["1bit.systems"]);
    let decoded = decode::<Claims>(
        token,
        &DecodingKey::from_secret(&state.cfg.jwt_secret),
        &v,
    )
    .unwrap();
    assert_eq!(decoded.claims.tier, "premium");
    assert_eq!(decoded.claims.btcpay_invoice.as_deref(), Some("inv-integration-1"));

    // Poll endpoint returns the same JWT on first hit then 202 on second.
    let poll1 = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/tier/poll/inv-integration-1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(poll1.status(), StatusCode::OK);

    let poll2 = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/tier/poll/inv-integration-1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(poll2.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn btcpay_bad_signature_rejected() {
    let state = test_state();
    let app = build_router(state);

    let body = br#"{"type":"InvoiceSettled","invoiceId":"x"}"#.to_vec();

    let req = Request::builder()
        .method("POST")
        .uri("/btcpay/webhook")
        .header("BTCPay-Sig", "sha256=deadbeef")
        .body(Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn health_ok() {
    let state = test_state();
    let app = build_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
