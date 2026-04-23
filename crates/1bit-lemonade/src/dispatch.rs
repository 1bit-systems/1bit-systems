//! Dispatch layer.
//!
//! Two concepts live here:
//!
//! 1. [`Dispatch`] — a static data decision recorded in the model registry
//!    (run this model locally on a 1bit-router backend, or proxy to some
//!    upstream URL). Unchanged from v0.
//!
//! 2. [`Upstream`] — the runtime **trait** through which HTTP handlers in
//!    [`crate::routes`] talk to whatever inference service backs the
//!    gateway today. [`HaloServer`] is the concrete impl pointing at
//!    `127.0.0.1:8180`; a future `Lemond` or `Ort` (ONNX Runtime) impl
//!    slots in without touching the axum routes.
//!
//! This satisfies spec invariant 3 (upstream-agnostic dispatch). The route
//! layer holds an `Arc<dyn Upstream>` and never reaches for reqwest or a
//! hardcoded URL.

use async_trait::async_trait;
use axum::body::Body;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Static per-model dispatch decision — survived from v0 so the registry
/// serde contract doesn't change. Not to be confused with the [`Upstream`]
/// trait, which is the runtime plumbing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Dispatch {
    Local { router: String },
    Upstream { url: String },
}

impl Dispatch {
    pub fn is_local(&self) -> bool {
        matches!(self, Dispatch::Local { .. })
    }
    pub fn is_upstream(&self) -> bool {
        matches!(self, Dispatch::Upstream { .. })
    }
}

/// Request envelope we hand an [`Upstream`]. We pre-buffer the body in the
/// route layer so the trait doesn't need to care about axum-specific body
/// types, which keeps the mock impl in tests trivial.
#[derive(Debug, Clone)]
pub struct UpstreamRequest {
    pub content_type: Option<String>,
    pub body: Bytes,
}

/// Response produced by an [`Upstream`]. `body` may be a buffered or a
/// streaming body; the route layer forwards it unchanged so SSE chunks
/// flush to the client with their original framing.
pub struct UpstreamResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Body,
}

/// All HTTP errors an upstream can raise — connect failure, non-2xx status
/// mapping, etc. We keep this as [`anyhow::Error`] on purpose: 1bit-lemonade
/// is pure plumbing, there's no structured error handling to preserve.
pub type UpstreamError = anyhow::Error;

/// The runtime dispatch interface.
///
/// Object-safe by design: the routes layer holds an `Arc<dyn Upstream>`
/// and swaps impls at boot (or, eventually, per-request based on a model
/// registry lookup).
#[async_trait]
pub trait Upstream: Send + Sync + std::fmt::Debug {
    /// Human-readable name for metrics labels + log lines (`"1bit-server"`,
    /// `"lemond"`, `"ort"`, `"mock"`).
    fn name(&self) -> &str;

    /// `POST /v1/chat/completions` forwarded upstream.
    async fn chat_completions(
        &self,
        req: UpstreamRequest,
    ) -> Result<UpstreamResponse, UpstreamError>;

    /// `POST /v1/completions` forwarded upstream.
    async fn completions(&self, req: UpstreamRequest) -> Result<UpstreamResponse, UpstreamError>;

    /// `GET /v1/models` passthrough. No request body.
    async fn models(&self) -> Result<UpstreamResponse, UpstreamError>;

    /// Cheap liveness probe — a `GET /_health` or equivalent returning true
    /// if the upstream's TCP + health responds within a short timeout.
    /// Used by the `onebit_lemonade_upstream_up` gauge.
    async fn health(&self) -> bool;
}

/// The concrete upstream we ship today: 1bit-server on 127.0.0.1:8180.
#[derive(Debug, Clone)]
pub struct HaloServer {
    client: reqwest::Client,
    base: url::Url,
}

impl HaloServer {
    /// Build a new HaloServer upstream at `base`. Fails if `base` is not a
    /// parseable URL. The shared reqwest client carries our user-agent and
    /// reuses connections across requests.
    pub fn new(base: impl AsRef<str>) -> anyhow::Result<Self> {
        let mut s = base.as_ref().to_string();
        // Normalise: strip trailing slash so join("/v1/...") works cleanly.
        while s.ends_with('/') {
            s.pop();
        }
        // reqwest doesn't require url::Url parsing, but we hold one so
        // downstream code can inspect host/port for logs + the upstream_up
        // probe.
        let base = url::Url::parse(&format!("{s}/"))
            .map_err(|e| anyhow::anyhow!("invalid upstream url {s:?}: {e}"))?;
        let client = reqwest::Client::builder()
            .user_agent("1bit-lemonade/0.1")
            .build()?;
        Ok(Self { client, base })
    }

    /// The parsed upstream URL (with trailing `/`).
    pub fn base(&self) -> &url::Url {
        &self.base
    }

    /// Construct with a caller-supplied reqwest client. Useful for tests
    /// that want a very short timeout.
    pub fn with_client(base: impl AsRef<str>, client: reqwest::Client) -> anyhow::Result<Self> {
        let mut s = base.as_ref().to_string();
        while s.ends_with('/') {
            s.pop();
        }
        let base = url::Url::parse(&format!("{s}/"))
            .map_err(|e| anyhow::anyhow!("invalid upstream url {s:?}: {e}"))?;
        Ok(Self { client, base })
    }

    async fn post_forward(
        &self,
        path: &str,
        req: UpstreamRequest,
    ) -> Result<UpstreamResponse, UpstreamError> {
        // Strip leading slash so url::Url::join doesn't drop the base path.
        let rel = path.trim_start_matches('/');
        let url = self
            .base
            .join(rel)
            .map_err(|e| anyhow::anyhow!("join {path}: {e}"))?;

        let mut forward = reqwest::header::HeaderMap::new();
        if let Some(ct) = req.content_type.as_deref()
            && let Ok(v) = reqwest::header::HeaderValue::from_str(ct)
        {
            forward.insert(reqwest::header::CONTENT_TYPE, v);
        }
        let resp = self
            .client
            .post(url.clone())
            .headers(forward)
            .body(req.body.to_vec())
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(%url, "upstream post failed: {e}");
                anyhow::anyhow!("upstream: {e}")
            })?;

        let status = resp.status().as_u16();
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        // Stream the body through so SSE chunks keep their framing.
        let stream = resp.bytes_stream();
        Ok(UpstreamResponse {
            status,
            content_type: ct,
            body: Body::from_stream(stream),
        })
    }
}

#[async_trait]
impl Upstream for HaloServer {
    fn name(&self) -> &str {
        "1bit-server"
    }

    async fn chat_completions(
        &self,
        req: UpstreamRequest,
    ) -> Result<UpstreamResponse, UpstreamError> {
        self.post_forward("/v1/chat/completions", req).await
    }

    async fn completions(&self, req: UpstreamRequest) -> Result<UpstreamResponse, UpstreamError> {
        self.post_forward("/v1/completions", req).await
    }

    async fn models(&self) -> Result<UpstreamResponse, UpstreamError> {
        let url = self
            .base
            .join("v1/models")
            .map_err(|e| anyhow::anyhow!("join /v1/models: {e}"))?;
        let resp = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("upstream: {e}"))?;
        let status = resp.status().as_u16();
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let stream = resp.bytes_stream();
        Ok(UpstreamResponse {
            status,
            content_type: ct,
            body: Body::from_stream(stream),
        })
    }

    async fn health(&self) -> bool {
        // 1bit-server exposes /_health on the same port. We deliberately
        // use a short timeout — this probe runs on the /metrics path so
        // slow 1bit-server response can't block a Prometheus scrape.
        let url = match self.base.join("_health") {
            Ok(u) => u,
            Err(_) => return false,
        };
        match self
            .client
            .get(url)
            .timeout(std::time::Duration::from_millis(500))
            .send()
            .await
        {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn dispatch_variants_match() {
        let l = Dispatch::Local {
            router: "onebit_router::bitnet".into(),
        };
        let u = Dispatch::Upstream {
            url: "https://api.openai.com".into(),
        };
        assert!(l.is_local() && !l.is_upstream());
        assert!(u.is_upstream() && !u.is_local());
        match &l {
            Dispatch::Local { router } => assert_eq!(router, "onebit_router::bitnet"),
            Dispatch::Upstream { .. } => unreachable!(),
        }
        match &u {
            Dispatch::Upstream { url } => assert!(url.starts_with("https://")),
            Dispatch::Local { .. } => unreachable!(),
        }
    }

    #[test]
    fn onebit_server_normalises_trailing_slashes() {
        let u = HaloServer::new("http://127.0.0.1:8180///").unwrap();
        assert_eq!(u.base().as_str(), "http://127.0.0.1:8180/");
        let u2 = HaloServer::new("http://127.0.0.1:8180").unwrap();
        assert_eq!(u2.base().as_str(), "http://127.0.0.1:8180/");
    }

    #[test]
    fn onebit_server_rejects_garbage_url() {
        assert!(HaloServer::new("not-a-url").is_err());
    }

    /// Compile-time-enforced proof of object safety — the trait is used
    /// through `Arc<dyn Upstream>` in [`crate::routes::AppState`], so any
    /// future method addition that breaks object safety fails the build
    /// here instead of in a harder-to-diagnose state-construction site.
    #[test]
    fn upstream_trait_is_object_safe() {
        fn _accepts(_: Arc<dyn Upstream>) {}
        let u: Arc<dyn Upstream> = Arc::new(HaloServer::new("http://127.0.0.1:65535").unwrap());
        _accepts(u);
    }

    /// Mock upstream that always returns a canned JSON body with the given
    /// status. Lives here so the /metrics + route tests can reuse it.
    #[derive(Debug)]
    struct MockUpstream {
        pub status: u16,
        pub body: String,
        pub healthy: bool,
    }

    #[async_trait]
    impl Upstream for MockUpstream {
        fn name(&self) -> &str {
            "mock"
        }
        async fn chat_completions(
            &self,
            _req: UpstreamRequest,
        ) -> Result<UpstreamResponse, UpstreamError> {
            Ok(UpstreamResponse {
                status: self.status,
                content_type: "application/json".into(),
                body: Body::from(self.body.clone()),
            })
        }
        async fn completions(
            &self,
            _req: UpstreamRequest,
        ) -> Result<UpstreamResponse, UpstreamError> {
            Ok(UpstreamResponse {
                status: self.status,
                content_type: "application/json".into(),
                body: Body::from(self.body.clone()),
            })
        }
        async fn models(&self) -> Result<UpstreamResponse, UpstreamError> {
            Ok(UpstreamResponse {
                status: 200,
                content_type: "application/json".into(),
                body: Body::from(r#"{"object":"list","data":[]}"#),
            })
        }
        async fn health(&self) -> bool {
            self.healthy
        }
    }

    #[tokio::test]
    async fn mock_upstream_returns_expected_shape() {
        let m = MockUpstream {
            status: 200,
            body: r#"{"id":"cmpl-1","object":"chat.completion"}"#.into(),
            healthy: true,
        };
        let req = UpstreamRequest {
            content_type: Some("application/json".into()),
            body: Bytes::from_static(b"{}"),
        };
        let resp = m.chat_completions(req).await.unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.content_type, "application/json");
        let bytes = axum::body::to_bytes(resp.body, 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
    }
}
