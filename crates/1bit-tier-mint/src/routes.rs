//! HTTP handlers. Kept thin — validation + signing + cache writes only.

use std::time::SystemTime;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::state::{AppState, PollEntry, unix_now};
use crate::{btcpay, jwt};

/// 10-minute TTL on poll cache entries. The storefront is expected to
/// poll every 1-2 s for roughly the Lightning settle window (~a few
/// seconds) so this is a generous upper bound.
const POLL_CACHE_TTL_SECS: u64 = 600;

pub async fn health() -> &'static str {
    "ok"
}

// -----------------------------------------------------------------------------
// BTCPay webhook
// -----------------------------------------------------------------------------

#[derive(Serialize)]
pub struct BtcpayWebhookResponse {
    pub ok: bool,
    /// Only populated on `InvoiceSettled` — lets BTCPay echo the JWT
    /// back to the storefront if it's been configured to forward the
    /// response. Not relied on; the poll endpoint is canonical.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwt: Option<String>,
}

pub async fn btcpay_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // BTCPay sends the sig as `BTCPay-Sig` per docs. The exact casing /
    // prefix is not fully nailed down — see the open question in the
    // wiki. We check multiple common spellings defensively.
    let sig = headers
        .get("btcpay-sig")
        .or_else(|| headers.get("BTCPay-Sig"))
        .or_else(|| headers.get("BTCPAY-SIG"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if !btcpay::verify_signature(&state.cfg.btcpay_webhook_secret, &body, sig) {
        warn!("btcpay webhook: signature mismatch");
        return (StatusCode::UNAUTHORIZED, "bad signature").into_response();
    }

    let event: btcpay::BtcPayEvent = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            warn!(?e, "btcpay webhook: malformed body");
            return (StatusCode::BAD_REQUEST, "malformed body").into_response();
        }
    };

    // We only care about settlement events. Anything else gets a 200
    // so BTCPay stops retrying.
    if event.event_type != "InvoiceSettled" {
        return (StatusCode::OK, Json(BtcpayWebhookResponse { ok: true, jwt: None })).into_response();
    }

    // Defensive: if this invoice was already revoked (admin backoffice
    // reasons) refuse to mint.
    if state.revoked.contains(&event.invoice_id) {
        warn!(invoice = %event.invoice_id, "btcpay webhook: invoice is revoked");
        return (StatusCode::CONFLICT, "invoice revoked").into_response();
    }

    let token = match jwt::mint_btcpay(
        &state.cfg.jwt_secret,
        &event.invoice_id,
        state.cfg.jwt_ttl,
        &state.cfg.issuer,
    ) {
        Ok(t) => t,
        Err(e) => {
            warn!(?e, "btcpay webhook: mint failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "mint failed").into_response();
        }
    };

    state.poll_cache.insert(
        event.invoice_id.clone(),
        PollEntry {
            jwt: token.clone(),
            minted_at: SystemTime::now(),
        },
    );
    info!(invoice = %event.invoice_id, "minted premium JWT (btcpay)");

    (
        StatusCode::OK,
        Json(BtcpayWebhookResponse { ok: true, jwt: Some(token) }),
    )
        .into_response()
}

// -----------------------------------------------------------------------------
// Poll + revoke
// -----------------------------------------------------------------------------

#[derive(Serialize)]
pub struct PollResponse {
    pub jwt: String,
}

pub async fn tier_poll(
    State(state): State<AppState>,
    Path(invoice_id): Path<String>,
) -> Response {
    let cache = &state.poll_cache;
    // Expire stale entries lazily.
    if let Some(entry) = cache.get(&invoice_id) {
        let age = SystemTime::now()
            .duration_since(entry.minted_at)
            .unwrap_or_default()
            .as_secs();
        if age > POLL_CACHE_TTL_SECS {
            drop(entry);
            cache.remove(&invoice_id);
            return (StatusCode::GONE, "expired").into_response();
        }
    }

    // Atomic take — first poll wins.
    if let Some((_, entry)) = cache.remove(&invoice_id) {
        return (StatusCode::OK, Json(PollResponse { jwt: entry.jwt })).into_response();
    }

    // Unknown or not-yet-settled: 202 tells the storefront to keep polling.
    (StatusCode::ACCEPTED, "pending").into_response()
}

#[derive(Deserialize)]
pub struct RevokeRequest {
    pub id: String,
    /// Admin bearer — checked against `HALO_TIER_HMAC_SECRET` as a
    /// simple shared-secret for now. A proper admin auth story is
    /// follow-up work.
    pub admin_token: String,
}

pub async fn tier_revoke(
    State(state): State<AppState>,
    Json(req): Json<RevokeRequest>,
) -> Response {
    // Cheap admin gate: compare the admin token (unencoded) against
    // the JWT signing secret. Not perfect but keeps the surface area
    // small; real admin needs its own key.
    if req.admin_token.as_bytes() != state.cfg.jwt_secret.as_slice() {
        return (StatusCode::UNAUTHORIZED, "bad admin token").into_response();
    }
    state.revoked.insert(req.id.clone());
    info!(id = %req.id, at = unix_now(), "admin revoke");
    (StatusCode::OK, "revoked").into_response()
}
