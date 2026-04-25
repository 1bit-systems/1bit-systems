//! `1bit-tier-mint` — mints Premium tier JWTs from paid BTCPay events.
//!
//! # Rails
//!
//! * **BTCPay Lightning**: customer pays an invoice; BTCPay POSTs
//!   `InvoiceSettled` to `/btcpay/webhook`. We verify the HMAC-SHA256
//!   signature (`BTCPay-Sig` header) against `HALO_BTCPAY_WEBHOOK_SECRET`,
//!   mint a JWT keyed to the invoice id, and stash it in an in-memory
//!   poll cache so the browser that's polling `GET /tier/poll/:invoice_id`
//!   can collect it.
//!
//! Callers of `/v1/catalogs/:slug/lossless` on 1bit-stream just verify
//! the JWT — see `docs/wiki/tier-jwt-flow.md` for the full write-up.
//!
//! # What this service does NOT do
//!
//! - Does not persist minted JWTs (state is a mint-and-forget event log).
//! - Does not verify the JWT on behalf of 1bit-stream — that's the stream
//!   server's job, it just needs the same HMAC secret.
//! - Does not accept raw card / fiat flow — that's whatever the storefront
//!   wires up to BTCPay.
//!
//! Rule A: pure Rust, no Python, no runtime interpreter.

pub mod btcpay;
pub mod jwt;
pub mod routes;
pub mod state;

pub use state::{AppState, Config};

use axum::Router;
use axum::routing::{get, post};

/// Build the axum router. Extracted so integration tests can mount it
/// without spawning the full binary.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/health", get(routes::health))
        .route("/btcpay/webhook", post(routes::btcpay_webhook))
        .route("/tier/poll/:invoice_id", get(routes::tier_poll))
        .route("/tier/revoke", post(routes::tier_revoke))
        .with_state(state)
}
