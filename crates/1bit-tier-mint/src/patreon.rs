//! Patreon webhook verification.
//!
//! Patreon signs with MD5-HMAC (yes, MD5 — it's the documented scheme)
//! keyed on the webhook's shared secret. Signature is in the
//! `X-Patreon-Signature` header, as hex.
//!
//! We only use Patreon for pledge lifecycle events (`members:create`,
//! `members:update`). For billing-fine-grained events you'd also watch
//! `pledges:*` but those are deprecated in v2 — member status supersedes.

use hmac::{Hmac, Mac};
use md5::Md5;
use serde::Deserialize;

type HmacMd5 = Hmac<Md5>;

/// Patreon webhooks deliver JSON:API v2 "document" envelopes. We only
/// look at a narrow slice — the included `member` resource's attrs.
#[derive(Debug, Deserialize)]
pub struct PatreonEvent {
    pub data: PatreonMemberData,
    #[serde(default)]
    pub included: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct PatreonMemberData {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub attributes: PatreonMemberAttrs,
}

#[derive(Debug, Deserialize)]
pub struct PatreonMemberAttrs {
    #[serde(default)]
    pub patron_status: Option<String>, // active_patron | declined_patron | former_patron
    #[serde(default)]
    pub full_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
}

pub fn verify_signature(secret: &[u8], body: &[u8], header: &str) -> bool {
    let Ok(expected) = hex::decode(header.trim()) else {
        return false;
    };
    let Ok(mut mac) = HmacMd5::new_from_slice(secret) else {
        return false;
    };
    mac.update(body);
    mac.verify_slice(&expected).is_ok()
}

pub fn sign_for_test(secret: &[u8], body: &[u8]) -> String {
    let mut mac = HmacMd5::new_from_slice(secret).expect("hmac key");
    mac.update(body);
    hex::encode(mac.finalize().into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_md5_hmac() {
        let secret = b"patreon-secret";
        let body = br#"{"data":{"id":"m-1","type":"member","attributes":{"patron_status":"active_patron"}}}"#;
        let sig = sign_for_test(secret, body);
        assert!(verify_signature(secret, body, &sig));
    }

    #[test]
    fn wrong_secret_fails() {
        let body = b"payload";
        let sig = sign_for_test(b"a", body);
        assert!(!verify_signature(b"b", body, &sig));
    }
}
