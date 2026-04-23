//! Async HTTP client against Patreon API v2.
//!
//! Endpoints wrapped here match the MCP tool surface 1:1. Errors are
//! returned as `PatreonError` so the MCP layer can render them as
//! `isError: true` content blocks without losing the HTTP status.
//!
//! Token comes from `PATREON_ACCESS_TOKEN`. For a long-lived operator
//! setup, prefer a Creator-scoped OAuth token over a Client-Credentials
//! one — the latter cannot post updates.

use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde_json::Value;
use thiserror::Error;

use crate::patreon::PostDraft;

const DEFAULT_BASE: &str = "https://www.patreon.com/api/oauth2/v2";

/// Failure modes for the Patreon client.
#[derive(Debug, Error)]
pub enum PatreonError {
    #[error("missing PATREON_ACCESS_TOKEN; set it before invoking Patreon tools")]
    MissingToken,
    #[error("http transport: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("patreon api returned {status}: {body}")]
    Api { status: StatusCode, body: String },
    #[error("patreon response was not valid json: {0}")]
    BadJson(#[from] serde_json::Error),
}

/// Typed handle to the Patreon v2 API.
///
/// Construct via [`PatreonClient::from_env`] for the common case or
/// [`PatreonClient::with_base`] for tests that want to point at a local
/// mock (wiremock etc.). We do **not** implement automatic refresh; a
/// Creator OAuth token is long-lived enough that rotating by hand on
/// `1bit install patreon` is fine for now.
#[derive(Debug, Clone)]
pub struct PatreonClient {
    http: Client,
    base: String,
    token: String,
}

impl PatreonClient {
    /// Build from environment. Fails if `PATREON_ACCESS_TOKEN` is unset or
    /// whitespace — catches the common "forgot to source the unit dropin"
    /// case explicitly rather than failing with an HTTP 401 later.
    pub fn from_env() -> Result<Self, PatreonError> {
        let token = std::env::var("PATREON_ACCESS_TOKEN")
            .ok()
            .filter(|t| !t.trim().is_empty())
            .ok_or(PatreonError::MissingToken)?;
        Ok(Self::with_token(token))
    }

    /// Build with a caller-supplied token (tests / scripts).
    pub fn with_token(token: impl Into<String>) -> Self {
        Self::with_base(DEFAULT_BASE, token)
    }

    /// Build with an explicit base URL. Useful for mock servers.
    pub fn with_base(base: impl Into<String>, token: impl Into<String>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent(concat!("1bit-mcp-patreon/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client build should not fail with defaults");
        Self {
            http,
            base: base.into(),
            token: token.into(),
        }
    }

    /// `GET /campaigns` — scopes: `identity.memberships` or
    /// `w:campaigns.members`. Returns raw Value for pass-through.
    pub async fn campaigns(&self) -> Result<Value, PatreonError> {
        // Note: `pledge_sum` was removed from the v2 campaign schema at
        // some point post-2024; asking for it returns a 400
        // `ParameterInvalidOnType`. Stick to fields Patreon still honors.
        self.get("/campaigns?include=tiers,goals&fields%5Bcampaign%5D=creation_name,patron_count,is_monthly,published_at,summary,url&fields%5Btier%5D=title,amount_cents,patron_count,description")
            .await
    }

    /// `GET /campaigns/{id}/members` — paginated. `cursor` is the raw
    /// `links.next` value from a prior response, or `None` for page 1.
    pub async fn members(
        &self,
        campaign_id: &str,
        cursor: Option<&str>,
    ) -> Result<Value, PatreonError> {
        let fields =
            "fields%5Bmember%5D=email,full_name,patron_status,currently_entitled_amount_cents";
        let path = match cursor {
            Some(c) => format!(
                "/campaigns/{campaign_id}/members?{fields}&page%5Bcursor%5D={}",
                urlencode(c)
            ),
            None => format!("/campaigns/{campaign_id}/members?{fields}"),
        };
        self.get(&path).await
    }

    /// `POST /campaigns/{id}/posts` — requires Creator-scope token and
    /// `w:campaigns.posts`. Response is the new post resource.
    pub async fn create_post(
        &self,
        campaign_id: &str,
        draft: &PostDraft,
    ) -> Result<Value, PatreonError> {
        let body = serde_json::json!({
            "data": {
                "type": "post",
                "attributes": {
                    "title": draft.title,
                    "content": draft.content,
                    "post_type": draft.post_type.clone().unwrap_or_else(|| "public_patrons".into()),
                }
            }
        });
        let url = format!("{}/campaigns/{}/posts", self.base, campaign_id);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(PatreonError::Api { status, body: text });
        }
        Ok(serde_json::from_str(&text)?)
    }

    async fn get(&self, path: &str) -> Result<Value, PatreonError> {
        let url = format!("{}{}", self.base, path);
        let resp = self.http.get(&url).bearer_auth(&self.token).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(PatreonError::Api { status, body: text });
        }
        Ok(serde_json::from_str(&text)?)
    }
}

/// Minimal percent-encoding for cursor pass-through. We only encode the
/// handful of characters that break URLs; a full library is overkill for
/// what Patreon echoes back.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '-' | '_' | '.' | '~' | 'a'..='z' | 'A'..='Z' | '0'..='9' => out.push(c),
            _ => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    out.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_errors_without_token() {
        // Scope the unset via an independent lock-free pattern — tests
        // share the env, so save + restore.
        let prev = std::env::var("PATREON_ACCESS_TOKEN").ok();
        // SAFETY: single-threaded tests inside this mod only touch this
        // variable; we restore immediately after the check.
        unsafe {
            std::env::remove_var("PATREON_ACCESS_TOKEN");
        }
        let err = PatreonClient::from_env().expect_err("missing token must fail");
        assert!(matches!(err, PatreonError::MissingToken));
        if let Some(p) = prev {
            unsafe {
                std::env::set_var("PATREON_ACCESS_TOKEN", p);
            }
        }
    }

    #[test]
    fn urlencode_keeps_unreserved_and_escapes_rest() {
        assert_eq!(urlencode("eyJhIjoiYiJ9"), "eyJhIjoiYiJ9");
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("a+b=c"), "a%2Bb%3Dc");
    }

    #[test]
    fn with_base_sets_custom_base() {
        let c = PatreonClient::with_base("http://127.0.0.1:1", "tok");
        assert_eq!(c.base, "http://127.0.0.1:1");
    }
}
