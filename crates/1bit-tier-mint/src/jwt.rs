//! JWT minting. HS256, short-ish expiry, claims shape locked by
//! `docs/wiki/tier-jwt-flow.md`.
//!
//! The stream server on port 8150 verifies with the same HMAC secret
//! (`HALO_TIER_HMAC_SECRET`). That's the only coupling point — neither
//! side shares a database.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use jsonwebtoken::{EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Claims {
    /// Subject — stable id for the paying customer. We don't have an
    /// account system so `sub` = invoice id (BTCPay) or Patreon member id.
    pub sub: String,
    /// Tier gate — currently only `"premium"` is minted. Leaving the
    /// field open in case we add further tiers.
    pub tier: String,
    /// Issuer. Always `"1bit.systems"`.
    pub iss: String,
    /// Unix expiry.
    pub exp: u64,
    /// Unix issue time.
    pub iat: u64,
    /// JWT id; same as `sub` so the revoke list only needs one key.
    pub jti: String,
    /// Provenance. Exactly one of these is set; the other is `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub btcpay_invoice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patreon_member_id: Option<String>,
}

/// Mint a Premium JWT for a BTCPay invoice.
pub fn mint_btcpay(secret: &[u8], invoice_id: &str, ttl: Duration, issuer: &str) -> Result<String> {
    let claims = base_claims(invoice_id, ttl, issuer);
    let claims = Claims {
        btcpay_invoice: Some(invoice_id.to_string()),
        ..claims
    };
    sign(secret, &claims)
}

/// Mint a Premium JWT for a Patreon active patron.
pub fn mint_patreon(
    secret: &[u8],
    member_id: &str,
    ttl: Duration,
    issuer: &str,
) -> Result<String> {
    let claims = base_claims(member_id, ttl, issuer);
    let claims = Claims {
        patreon_member_id: Some(member_id.to_string()),
        ..claims
    };
    sign(secret, &claims)
}

fn base_claims(id: &str, ttl: Duration, issuer: &str) -> Claims {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Claims {
        sub: id.to_string(),
        tier: "premium".to_string(),
        iss: issuer.to_string(),
        exp: now + ttl.as_secs(),
        iat: now,
        jti: id.to_string(),
        btcpay_invoice: None,
        patreon_member_id: None,
    }
}

fn sign(secret: &[u8], claims: &Claims) -> Result<String> {
    let header = Header::new(jsonwebtoken::Algorithm::HS256);
    let key = EncodingKey::from_secret(secret);
    Ok(encode(&header, claims, &key)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{DecodingKey, Validation, decode};

    #[test]
    fn mint_and_verify_btcpay() {
        let secret = b"test-secret-at-least-32-bytes-long-x";
        let token = mint_btcpay(secret, "inv-abc", Duration::from_secs(3600), "1bit.systems").unwrap();
        let mut v = Validation::new(jsonwebtoken::Algorithm::HS256);
        v.set_issuer(&["1bit.systems"]);
        let decoded = decode::<Claims>(&token, &DecodingKey::from_secret(secret), &v).unwrap();
        assert_eq!(decoded.claims.tier, "premium");
        assert_eq!(decoded.claims.btcpay_invoice.as_deref(), Some("inv-abc"));
        assert!(decoded.claims.patreon_member_id.is_none());
        assert_eq!(decoded.claims.jti, "inv-abc");
    }

    #[test]
    fn mint_patreon_sets_member_id_only() {
        let secret = b"test-secret-at-least-32-bytes-long-x";
        let token = mint_patreon(secret, "m-42", Duration::from_secs(60), "1bit.systems").unwrap();
        let mut v = Validation::new(jsonwebtoken::Algorithm::HS256);
        v.set_issuer(&["1bit.systems"]);
        let decoded = decode::<Claims>(&token, &DecodingKey::from_secret(secret), &v).unwrap();
        assert_eq!(decoded.claims.patreon_member_id.as_deref(), Some("m-42"));
        assert!(decoded.claims.btcpay_invoice.is_none());
    }

    #[test]
    fn bad_secret_fails_verify() {
        let token = mint_btcpay(b"secret-one-at-least-32-bytes-long-aa", "x", Duration::from_secs(60), "1bit.systems").unwrap();
        let v = Validation::new(jsonwebtoken::Algorithm::HS256);
        let err = decode::<Claims>(&token, &DecodingKey::from_secret(b"different-secret-xxxxxxxxxxxxxx"), &v);
        assert!(err.is_err());
    }
}
