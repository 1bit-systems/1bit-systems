//! Shared service state: secrets, JWT signer, revocation list, poll cache.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use dashmap::{DashMap, DashSet};

/// Immutable service config, cloned from env on boot.
#[derive(Clone)]
pub struct Config {
    /// HS256 secret for signing minted JWTs. Shared with 1bit-stream so
    /// it can verify. Rotated quarterly; during a rotation window both
    /// the new and old keys are accepted on the verify side — see
    /// `docs/wiki/tier-jwt-flow.md`.
    pub jwt_secret: Vec<u8>,
    /// BTCPay-side shared secret; HMAC-SHA256 of the request body must
    /// match the `BTCPay-Sig` header.
    pub btcpay_webhook_secret: Vec<u8>,
    /// Patreon-side shared secret; MD5-HMAC of the request body must
    /// match the `X-Patreon-Signature` header.
    pub patreon_webhook_secret: Vec<u8>,
    /// JWT issuer claim (`iss`).
    pub issuer: String,
    /// JWT lifetime from mint. Short, because `/tier/refresh` can
    /// re-mint against the same invoice as long as it's not revoked.
    pub jwt_ttl: Duration,
}

impl Config {
    /// Load from env. Returns a usable config or bails.
    pub fn from_env() -> Result<Self> {
        let jwt_secret = std::env::var("HALO_TIER_HMAC_SECRET")
            .context("HALO_TIER_HMAC_SECRET must be set (see strixhalo/systemd/user/1bit-tier-mint.service.env)")?
            .into_bytes();
        if jwt_secret.len() < 32 {
            anyhow::bail!("HALO_TIER_HMAC_SECRET too short; need ≥ 32 bytes of entropy");
        }
        let btcpay_webhook_secret = std::env::var("HALO_BTCPAY_WEBHOOK_SECRET")
            .context("HALO_BTCPAY_WEBHOOK_SECRET must be set")?
            .into_bytes();
        let patreon_webhook_secret = std::env::var("HALO_PATREON_WEBHOOK_SECRET")
            .context("HALO_PATREON_WEBHOOK_SECRET must be set")?
            .into_bytes();
        Ok(Self {
            jwt_secret,
            btcpay_webhook_secret,
            patreon_webhook_secret,
            issuer: "1bit.systems".to_string(),
            jwt_ttl: Duration::from_secs(60 * 60 * 24 * 30), // 30 days
        })
    }
}

/// Revoke list entries are keyed by the originating invoice / member id.
/// The stream server reads `jti` = that id out of the JWT and consults
/// this set before serving the lossless endpoint.
///
/// In production this would be durable (sqlite on disk). For the MVP
/// we keep it in memory and persist on write — a TODO is noted in
/// the wiki doc. Rebooting the service loses pending revocations; fine
/// for MVP but not for real money.
#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    /// `invoice_id` / `patreon_member_id` → minted JWT. Populated by the
    /// webhook handler; drained by `/tier/poll/:id`. Entries expire on
    /// first read (the customer only needs to collect once) or after
    /// 10 minutes, whichever comes first.
    pub poll_cache: Arc<DashMap<String, PollEntry>>,
    /// Revoked invoice / member ids. Stream server consults via the
    /// shared sqlite file — this in-memory set is the write-through
    /// layer; a background task would flush to disk. Out of scope for
    /// the MVP skeleton.
    pub revoked: Arc<DashSet<String>>,
}

pub struct PollEntry {
    pub jwt: String,
    pub minted_at: SystemTime,
}

impl AppState {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg: Arc::new(cfg),
            poll_cache: Arc::new(DashMap::new()),
            revoked: Arc::new(DashSet::new()),
        }
    }

    /// For test-only use — injects a fixed config without env.
    #[doc(hidden)]
    pub fn for_tests(cfg: Config) -> Self {
        Self::new(cfg)
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
