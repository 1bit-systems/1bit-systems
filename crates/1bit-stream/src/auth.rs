//! HS256 JWT verification for the `/v1/catalogs/:slug/lossless` gate.
//!
//! Premium tier = any valid HS256 token whose `tier` claim equals
//! `"premium"` and whose `exp` (if present) is in the future. The BTCPay
//! integration lives in a sibling crate — by the time the request hits
//! here, we only care that the token is signed with the server's shared
//! secret.
//!
//! Admin endpoints (`POST /internal/reindex`) use a separate shared
//! bearer token. That's also validated through this module for
//! consistency; it's not JWT, just a literal Bearer compare.

use axum::http::{HeaderMap, StatusCode, header};
use jsonwebtoken::{DecodingKey, Validation};
use serde::{Deserialize, Serialize};

/// Server-side auth configuration. Built once at startup and held inside
/// `AppState`.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// HS256 secret used for JWT verification. `None` disables the
    /// lossless gate — useful in dev; in prod the env var is required.
    pub jwt_secret: Option<Vec<u8>>,
    /// Shared bearer for `/internal/*` admin routes. `None` means the
    /// admin surface is open to loopback only (dev default).
    pub admin_bearer: Option<String>,
}

impl AuthConfig {
    pub fn from_env() -> Self {
        Self {
            jwt_secret: std::env::var("HALO_STREAM_JWT_SECRET")
                .ok()
                .map(|s| s.into_bytes()),
            admin_bearer: std::env::var("HALO_STREAM_ADMIN_BEARER").ok(),
        }
    }

    /// Test / CLI helper — build an auth config explicitly.
    pub fn new(jwt_secret: Option<Vec<u8>>, admin_bearer: Option<String>) -> Self {
        Self {
            jwt_secret,
            admin_bearer,
        }
    }
}

/// Claim shape we expect on premium tokens. Extra claims are allowed and
/// ignored.
#[derive(Debug, Serialize, Deserialize)]
pub struct PremiumClaims {
    #[serde(default)]
    pub sub: String,
    #[serde(default)]
    pub tier: String,
    #[serde(default)]
    pub exp: Option<i64>,
}

/// Outcome of a premium-gate check. `Allow` passes through to the
/// handler; any other variant turns into an HTTP status.
#[derive(Debug)]
pub enum GateOutcome {
    Allow,
    MissingHeader,
    BadScheme,
    BadToken,
    WrongTier,
    ServerMisconfigured,
}

impl GateOutcome {
    pub fn as_status(&self) -> StatusCode {
        match self {
            GateOutcome::Allow => StatusCode::OK,
            GateOutcome::MissingHeader | GateOutcome::BadScheme | GateOutcome::BadToken => {
                StatusCode::UNAUTHORIZED
            }
            GateOutcome::WrongTier => StatusCode::FORBIDDEN,
            GateOutcome::ServerMisconfigured => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

/// Verify that `headers` carries a valid Bearer JWT with `tier=premium`.
pub fn check_premium(cfg: &AuthConfig, headers: &HeaderMap) -> GateOutcome {
    let Some(secret) = cfg.jwt_secret.as_ref() else {
        // Lossless gate deliberately disabled. Be loud about it — a
        // publisher who forgot to set the env var should see the status,
        // not a silent pass.
        return GateOutcome::ServerMisconfigured;
    };

    let Some(raw) = headers.get(header::AUTHORIZATION) else {
        return GateOutcome::MissingHeader;
    };
    let Ok(val) = raw.to_str() else {
        return GateOutcome::BadScheme;
    };
    let Some(token) = val.strip_prefix("Bearer ").or_else(|| val.strip_prefix("bearer ")) else {
        return GateOutcome::BadScheme;
    };

    let key = DecodingKey::from_secret(secret);
    let mut v = Validation::new(jsonwebtoken::Algorithm::HS256);
    // `exp` is optional in our scheme; set required_spec_claims empty so
    // jsonwebtoken doesn't reject token-without-exp. If exp IS present,
    // it's validated because leeway + now are still compared.
    v.required_spec_claims.clear();
    v.validate_exp = true;

    let decoded = match jsonwebtoken::decode::<PremiumClaims>(token, &key, &v) {
        Ok(d) => d,
        Err(_) => return GateOutcome::BadToken,
    };

    if decoded.claims.tier != "premium" {
        return GateOutcome::WrongTier;
    }
    GateOutcome::Allow
}

/// Constant-time-ish compare against the admin bearer. `None` + loopback
/// is allowed in dev; callers enforce the loopback part by binding to
/// 127.0.0.1 at the listener level.
pub fn check_admin(cfg: &AuthConfig, headers: &HeaderMap) -> GateOutcome {
    let Some(expected) = cfg.admin_bearer.as_ref() else {
        return GateOutcome::Allow;
    };
    let Some(raw) = headers.get(header::AUTHORIZATION) else {
        return GateOutcome::MissingHeader;
    };
    let Ok(val) = raw.to_str() else {
        return GateOutcome::BadScheme;
    };
    let Some(token) = val.strip_prefix("Bearer ").or_else(|| val.strip_prefix("bearer ")) else {
        return GateOutcome::BadScheme;
    };
    if constant_time_eq(token.as_bytes(), expected.as_bytes()) {
        GateOutcome::Allow
    } else {
        GateOutcome::BadToken
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use jsonwebtoken::{EncodingKey, Header};

    fn hdrs(bearer: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {bearer}")).unwrap(),
        );
        h
    }

    fn sign(secret: &[u8], claims: &PremiumClaims) -> String {
        jsonwebtoken::encode(
            &Header::new(jsonwebtoken::Algorithm::HS256),
            claims,
            &EncodingKey::from_secret(secret),
        )
        .unwrap()
    }

    #[test]
    fn allows_valid_premium_token() {
        let secret = b"test-secret".to_vec();
        let cfg = AuthConfig::new(Some(secret.clone()), None);
        let claims = PremiumClaims {
            sub: "user-1".into(),
            tier: "premium".into(),
            exp: None,
        };
        let token = sign(&secret, &claims);
        let out = check_premium(&cfg, &hdrs(&token));
        assert!(matches!(out, GateOutcome::Allow));
    }

    #[test]
    fn rejects_wrong_tier() {
        let secret = b"test-secret".to_vec();
        let cfg = AuthConfig::new(Some(secret.clone()), None);
        let token = sign(
            &secret,
            &PremiumClaims {
                sub: "user-1".into(),
                tier: "lossy".into(),
                exp: None,
            },
        );
        let out = check_premium(&cfg, &hdrs(&token));
        assert!(matches!(out, GateOutcome::WrongTier));
    }

    #[test]
    fn rejects_bad_signature() {
        let secret_a = b"a".to_vec();
        let secret_b = b"b".to_vec();
        let cfg = AuthConfig::new(Some(secret_a), None);
        let token = sign(
            &secret_b,
            &PremiumClaims {
                sub: "u".into(),
                tier: "premium".into(),
                exp: None,
            },
        );
        assert!(matches!(
            check_premium(&cfg, &hdrs(&token)),
            GateOutcome::BadToken
        ));
    }

    #[test]
    fn missing_header() {
        let cfg = AuthConfig::new(Some(b"s".to_vec()), None);
        let out = check_premium(&cfg, &HeaderMap::new());
        assert!(matches!(out, GateOutcome::MissingHeader));
    }

    #[test]
    fn server_misconfigured_when_no_secret() {
        let cfg = AuthConfig::new(None, None);
        let out = check_premium(&cfg, &hdrs("whatever"));
        assert!(matches!(out, GateOutcome::ServerMisconfigured));
    }

    #[test]
    fn admin_bearer_matches() {
        let cfg = AuthConfig::new(None, Some("abc".into()));
        assert!(matches!(check_admin(&cfg, &hdrs("abc")), GateOutcome::Allow));
        assert!(matches!(
            check_admin(&cfg, &hdrs("nope")),
            GateOutcome::BadToken
        ));
    }

    #[test]
    fn admin_open_when_not_configured() {
        let cfg = AuthConfig::new(None, None);
        assert!(matches!(
            check_admin(&cfg, &HeaderMap::new()),
            GateOutcome::Allow
        ));
    }
}
