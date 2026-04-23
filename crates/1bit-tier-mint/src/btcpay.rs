//! BTCPay Server webhook verification.
//!
//! BTCPay signs request bodies with HMAC-SHA256 keyed on the shared
//! secret configured in the store's webhook settings. The signature
//! is sent as hex, prefixed `sha256=`, in the `BTCPay-Sig` header.
//! (See open question in `docs/wiki/tier-jwt-flow.md` §"BTCPay sig
//! header" — we lock the exact header string to whatever the first
//! real `InvoiceSettled` delivers.)

use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Subset of BTCPay webhook payload we care about. BTCPay sends a
/// fair bit more; we deserialize defensively.
#[derive(Debug, Deserialize)]
pub struct BtcPayEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(rename = "invoiceId")]
    pub invoice_id: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Constant-time compare of expected vs observed MAC.
pub fn verify_signature(secret: &[u8], body: &[u8], header: &str) -> bool {
    // Header format per BTCPay docs: `sha256=<hex digest>`. Tolerate
    // bare hex too, since I've seen both in the wild across versions.
    let hex = header.trim().strip_prefix("sha256=").unwrap_or(header.trim());
    let Ok(expected) = hex::decode(hex) else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret) else {
        return false;
    };
    mac.update(body);
    mac.verify_slice(&expected).is_ok()
}

/// Compute the hex HMAC used by BTCPay clients / our integration tests.
pub fn sign_for_test(secret: &[u8], body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).expect("hmac key");
    mac.update(body);
    format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_signature() {
        let secret = b"webhook-secret";
        let body = br#"{"type":"InvoiceSettled","invoiceId":"abc"}"#;
        let header = sign_for_test(secret, body);
        assert!(verify_signature(secret, body, &header));
    }

    #[test]
    fn tampered_body_fails() {
        let secret = b"webhook-secret";
        let body = br#"{"type":"InvoiceSettled","invoiceId":"abc"}"#;
        let header = sign_for_test(secret, body);
        let evil = br#"{"type":"InvoiceSettled","invoiceId":"xxx"}"#;
        assert!(!verify_signature(secret, evil, &header));
    }

    #[test]
    fn accepts_bare_hex_header() {
        let secret = b"s";
        let body = b"hello";
        let header = sign_for_test(secret, body);
        let bare = header.trim_start_matches("sha256=").to_string();
        assert!(verify_signature(secret, body, &bare));
    }
}
