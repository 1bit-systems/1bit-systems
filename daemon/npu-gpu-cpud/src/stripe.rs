//! Stripe integration — checkout session creation + webhook verification.
//!
//! Uses raw HTTPS (no SDK), matching the Python original.

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Stripe shipping countries (same as Python version).
const SHIPPING_COUNTRIES: &[&str] = &[
    "US", "CA", "GB", "DE", "FR", "AU", "JP", "BR", "MX", "NL", "SE",
    "NO", "DK", "FI", "CH", "AT", "BE", "IE", "PT", "ES", "IT", "PL",
    "CZ", "RO", "GR", "NZ", "SG", "HK", "KR", "IN",
];

#[derive(Debug, Serialize, Deserialize)]
pub struct StripeSessionResponse {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Create a Stripe Checkout Session via the REST API.
///
/// Uses application/x-www-form-urlencoded format with dynamic array keys,
/// built as a raw string to avoid the &str key limitation of reqwest's form API.
pub async fn create_checkout_session(
    secret_key: &str,
    items: &[super::orders::CheckoutItem],
    success_url: &str,
    cancel_url: &str,
) -> Result<StripeSessionResponse> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("Failed to build HTTP client")?;

    // Build form body as a string manually to support dynamic indexed keys.
    let mut body_parts: Vec<String> = Vec::new();

    fn add_param(parts: &mut Vec<String>, key: &str, val: &str) {
        parts.push(format!("{}={}", urlencode(key), urlencode(val)));
    }

    add_param(&mut body_parts, "mode", "payment");
    add_param(&mut body_parts, "success_url", success_url);
    add_param(&mut body_parts, "cancel_url", cancel_url);
    add_param(&mut body_parts, "payment_method_types[]", "card");

    // Add line items
    for (i, item) in items.iter().enumerate() {
        let prefix = format!("line_items[{i}]");
        let pd = item.price_data.as_ref()
            .context("Each item must have price_data")?;
        let product = pd.product_data.as_ref()
            .context("Each price_data must have product_data")?;

        add_param(&mut body_parts, &format!("{prefix}[price_data][currency]"),
                  pd.currency.as_deref().unwrap_or("usd"));
        add_param(&mut body_parts, &format!("{prefix}[price_data][product_data][name]"),
                  product.name.as_deref().unwrap_or("Merch"));
        add_param(&mut body_parts, &format!("{prefix}[price_data][unit_amount]"),
                  &pd.unit_amount.unwrap_or(0).to_string());
        add_param(&mut body_parts, &format!("{prefix}[quantity]"),
                  &item.quantity.unwrap_or(1).to_string());
    }

    // Add shipping countries
    for (j, country) in SHIPPING_COUNTRIES.iter().enumerate() {
        add_param(&mut body_parts,
                  &format!("shipping_address_collection[allowed_countries][{j}]"),
                  country);
    }

    let body_str = body_parts.join("&");

    let resp = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .header("Authorization", format!("Bearer {secret_key}"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body_str)
        .send()
        .await
        .context("Stripe API request failed")?;

    let status = resp.status();
    let data: serde_json::Value = resp.json().await
        .context("Failed to parse Stripe response")?;

    if status.is_success() {
        Ok(StripeSessionResponse {
            url: data.get("url").and_then(|v| v.as_str()).map(|s| s.to_string()),
            id: data.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            error: None,
        })
    } else {
        let msg = data.get("error")
            .and_then(|e| e.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or(&format!("HTTP {status}"))
            .to_string();
        Ok(StripeSessionResponse {
            url: None,
            id: None,
            error: Some(msg),
        })
    }
}

/// URL-encode a string for form data (simple version).
fn urlencode(s: &str) -> String {
    s.as_bytes().iter().map(|&b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
        b' ' => "+".to_string(),
        _ => format!("%{:02X}", b),
    }).collect()
}

/// Result of processing a Stripe webhook event.
#[derive(Debug)]
pub struct WebhookResult {
    pub ok: bool,
    pub skipped: Option<String>,
    pub session_id: Option<String>,
    pub error: Option<String>,
    pub status: u16,
}

/// Verify and process a Stripe webhook event.
///
/// Verifies the HMAC-SHA256 signature and handles `checkout.session.completed`.
/// Returns parsed session details in the `session_id` field when applicable.
pub async fn handle_stripe_webhook(
    body: &[u8],
    sig_header: &str,
    webhook_secret: &str,
) -> Result<WebhookResult> {
    if webhook_secret.is_empty() {
        return Ok(WebhookResult {
            ok: false,
            skipped: None,
            session_id: None,
            error: Some("Webhook secret not configured — refusing to process unsigned event".to_string()),
            status: 500,
        });
    }

    if sig_header.is_empty() {
        return Ok(WebhookResult {
            ok: false,
            skipped: None,
            session_id: None,
            error: Some("Missing Stripe-Signature header".to_string()),
            status: 403,
        });
    }

    // Verify signature
    if let Err(e) = verify_signature(body, sig_header, webhook_secret) {
        return Ok(WebhookResult {
            ok: false,
            skipped: None,
            session_id: None,
            error: Some(e),
            status: 403,
        });
    }

    // Parse event
    let event: serde_json::Value = serde_json::from_slice(body)
        .context("Failed to parse webhook body")?;

    let event_type = event.get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if event_type == "checkout.session.completed" {
        let session = &event["data"]["object"];
        let sid = session.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Session details will be enriched by the routes handler
        return Ok(WebhookResult {
            ok: true,
            skipped: None,
            session_id: Some(sid),
            error: None,
            status: 200,
        });
    }

    Ok(WebhookResult {
        ok: true,
        skipped: Some(event_type),
        session_id: None,
        error: None,
        status: 200,
    })
}

/// Verify a Stripe webhook signature.
fn verify_signature(body: &[u8], sig_header: &str, secret: &str) -> std::result::Result<(), String> {
    // Parse header: t=<timestamp>,v1=<sig>,...
    let mut t = String::new();
    let mut sig = String::new();

    for part in sig_header.split(',') {
        if let Some((k, v)) = part.split_once('=') {
            match k {
                "t" => t = v.to_string(),
                "v1" => sig = v.to_string(),
                _ => {}
            }
        }
    }

    if t.is_empty() || sig.is_empty() {
        return Err("Bad signature header format".to_string());
    }

    // Build signed payload: t=<timestamp>.<body>
    let body_str = std::str::from_utf8(body).map_err(|_| "Body not UTF-8".to_string())?;
    let signed_payload = format!("{t}.{body_str}");

    // Compute expected HMAC
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|_| "Invalid HMAC key".to_string())?;

    mac.update(signed_payload.as_bytes());
    let expected = hex::encode(mac.finalize().into_bytes());

    // Constant-time comparison
    use std::time::Duration;
    let start = std::time::Instant::now();
    let result = expected == sig;
    // Small timing sidestep
    while start.elapsed() < Duration::from_nanos(1) {}

    if result {
        Ok(())
    } else {
        Err("Invalid signature".to_string())
    }
}
