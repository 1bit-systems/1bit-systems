//! Inference backend abstraction.
//!
//! `1bit-server` is HTTP plumbing only — actual token generation happens
//! behind the [`InferenceBackend`] trait. Today the sole implementation is
//! [`EchoBackend`], which returns the string `"stub"` token-by-token. Once
//! `1bit-router` grows a real dispatcher it will implement this same trait
//! and be swapped in at `main.rs` wiring time; no handler code needs to
//! change.
//!
//! The streaming contract is a `Stream<Item = Result<String, ServerError>>`
//! where each item is a single delta chunk (typically one token, but we do
//! not enforce granularity — the backend may coalesce). The non-streaming
//! path is expressed as `generate()` returning the fully assembled text
//! plus a [`GenerationUsage`] token count.
//!
//! We rely on native `async fn` in traits (stable since Rust 1.75, pinned
//! here by the workspace `rust-version = 1.85`) plus `Send` bounds via the
//! `return_type_notation`-free form: we return a `Pin<Box<dyn Future>>` on
//! object-safe call sites where needed, but the primary trait uses plain
//! AFIT — callers of the `dyn` form go through the
//! `InferenceBackendDyn` convenience wrapper below.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::stream::{self, Stream};

use crate::api::{ChatMessage, ModelCard, Usage};
use crate::chat_template::ChatTemplate;
use crate::error::ServerError;

/// Opaque boxed stream of deltas produced during generation.
///
/// `Send` is required because axum handlers run on a multi-threaded tokio
/// runtime and the stream outlives the handler (it is consumed by the SSE
/// writer task).
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String, ServerError>> + Send + 'static>>;

/// Parameters forwarded from the OpenAI request to the backend.
///
/// We deliberately keep this minimal and bag-of-options — the concrete
/// backends know which knobs they honor. Anything the backend ignores is
/// silently dropped.
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    /// Prompt-template rendering to apply to `messages` before dispatch.
    /// Resolved by the HTTP layer from (in order): `X-Halo-Chat-Template`
    /// header → `HALO_CHAT_TEMPLATE` env var → [`ChatTemplate::Llama3`].
    /// See `crate::chat_template` for the exact bytes emitted by each.
    pub chat_template: ChatTemplate,
}

/// Token-count accounting reported back to the OpenAI client.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerationUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

impl GenerationUsage {
    pub fn into_api(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.prompt_tokens + self.completion_tokens,
        }
    }
}

/// Boxed send-future alias — the hand-written object-safe form of AFIT.
type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The only thing the HTTP layer asks of a generation engine.
///
/// Written in the object-safe "return `Pin<Box<Future>>`" style rather than
/// native AFIT so we can hold it as `Arc<dyn InferenceBackend>` in axum's
/// shared state. (AFIT + `dyn` is still nightly-only at the time of writing;
/// this is the same trick `tower::Service` uses.)
///
/// Implementations are almost always easier to write via the blanket
/// wrapper in the [`impl_backend!`](crate::impl_backend) macro, or by hand
/// using `Box::pin(async move { ... })`.
pub trait InferenceBackend: Send + Sync + 'static {
    /// Non-streaming completion.
    fn generate<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        params: &'a GenerationParams,
    ) -> BoxFut<'a, Result<(String, GenerationUsage), ServerError>>;

    /// Streaming completion. Returns a stream of delta strings; the handler
    /// wraps each one in an OpenAI-shaped SSE `data:` event.
    fn generate_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        params: &'a GenerationParams,
    ) -> BoxFut<'a, Result<TokenStream, ServerError>>;

    /// Model cards the backend *can* serve. Historically this was the
    /// only source for `GET /v1/models`; as of the `ModelRegistry`
    /// landing, the on-disk registry is the source of truth for the HTTP
    /// surface. This method is still consulted once at startup so a
    /// backend that loads weights from outside the `--models-dir` scan
    /// path (e.g. `--model /tmp/foo.h1b`) still shows up in
    /// `/v1/models` via `ModelRegistry::ensure_id`. Keep overrides here
    /// narrow — the registry does the real work.
    fn list_models(&self) -> Vec<ModelCard> {
        vec![ModelCard::halo("bitnet-b1.58-2b-4t")]
    }

    /// Optional perplexity scoring path.
    ///
    /// Backends that can expose per-token log-probabilities override this;
    /// the default returns `Backend("ppl not supported on this backend")`
    /// so stub / echo builds fail the `/ppl` endpoint gracefully (501-ish
    /// surfaced as a 500 via our error mapping, with a clear message).
    fn perplexity<'a>(
        &'a self,
        _text: String,
        _stride: u32,
        _max_tokens: u32,
    ) -> BoxFut<'a, Result<PerplexityOutput, ServerError>> {
        Box::pin(async move {
            Err(ServerError::Backend(
                "ppl not supported on this backend (rebuild 1bit-server with \
                 --features real-backend and load a model via --model)"
                    .into(),
            ))
        })
    }
}

/// Backend-facing PPL result, mirroring
/// [`onebit_router::PerplexityResult`] but without pulling the router into
/// the default build.
#[derive(Debug, Clone, Copy)]
pub struct PerplexityOutput {
    pub mean_nll: f64,
    pub perplexity: f64,
    pub tokens: u32,
    pub elapsed_ms: f64,
}

// ─── EchoBackend ─────────────────────────────────────────────────────────

/// Stub backend — returns the literal string `"stub"` regardless of input.
///
/// This is the seam until `1bit-router` grows a real dispatcher. Kept in
/// this crate (not behind a feature flag) so integration tests can exercise
/// the full HTTP surface without a model on disk.
#[derive(Debug, Default, Clone)]
pub struct EchoBackend {
    /// If set, the reply becomes `{reply_prefix}{first_user_message}` —
    /// handy for tests that want to assert round-tripping without relying
    /// on the literal `"stub"`.
    pub reply_prefix: Option<String>,
}

impl EchoBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            reply_prefix: Some(prefix.into()),
        }
    }

    fn render(&self, messages: &[ChatMessage]) -> String {
        match &self.reply_prefix {
            Some(p) => {
                let last_user = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "user")
                    .map(|m| m.content.as_str())
                    .unwrap_or("");
                format!("{p}{last_user}")
            }
            None => "stub".to_string(),
        }
    }
}

impl InferenceBackend for EchoBackend {
    fn generate<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        _params: &'a GenerationParams,
    ) -> BoxFut<'a, Result<(String, GenerationUsage), ServerError>> {
        Box::pin(async move {
            let text = self.render(messages);
            // Rough prompt token count: whitespace-split across all messages.
            let prompt_tokens = messages
                .iter()
                .map(|m| m.content.split_whitespace().count() as u32)
                .sum();
            let completion_tokens = text.split_whitespace().count() as u32;
            Ok((
                text,
                GenerationUsage {
                    prompt_tokens,
                    completion_tokens,
                },
            ))
        })
    }

    fn generate_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        _params: &'a GenerationParams,
    ) -> BoxFut<'a, Result<TokenStream, ServerError>> {
        Box::pin(async move {
            // Stream character-by-character so tests can verify that the
            // SSE path is really chunked and not silently buffering the
            // whole body. Real backends emit one token per chunk.
            let text = self.render(messages);
            let chunks: Vec<Result<String, ServerError>> =
                text.chars().map(|c| Ok(c.to_string())).collect();
            let s: TokenStream = Box::pin(stream::iter(chunks));
            Ok(s)
        })
    }
}

/// Shared-state handle stored in axum's router extension.
///
/// We use `Arc<dyn InferenceBackend>` (not a generic) so the router type is
/// concrete — generics over the whole `Router` percolate into every
/// handler signature and make tests painful.
pub type SharedBackend = Arc<dyn InferenceBackend>;

// ─── RealBackend ─────────────────────────────────────────────────────────
//
// Adapter that turns a `onebit_router::Router` into an `InferenceBackend`.
// Only compiled under the `real-backend` feature so the default build
// stays free of ROCm link-time requirements.
#[cfg(feature = "real-backend")]
pub mod real {
    //! Adapter from [`onebit_router::Router`] to the server's
    //! [`super::InferenceBackend`] trait.

    use super::*;
    use futures::StreamExt;
    use onebit_core::sampler::SamplerConfig;
    use onebit_router::{Router, RouterRequest};

    /// Thin wrapper — the router does all the work; we just translate
    /// OpenAI-shaped types into `RouterRequest` and back.
    pub struct RealBackend {
        router: Arc<Router>,
    }

    impl RealBackend {
        /// Load the model + tokenizer and spin up the router. Blocks
        /// the current thread while weights upload to the GPU
        /// (~3 s for halo-1bit-2b.h1b on gfx1151). Call this once at
        /// server startup, not per request.
        pub fn new(h1b_path: &std::path::Path) -> Result<Self, crate::ServerError> {
            let router = Router::new(h1b_path)
                .map_err(|e| crate::ServerError::Backend(format!("router init: {e}")))?;
            Ok(Self {
                router: Arc::new(router),
            })
        }

        /// Access the underlying router (mostly useful for tests + warm-up).
        pub fn router(&self) -> &Router {
            &self.router
        }

        fn build_request(messages: &[ChatMessage], params: &GenerationParams) -> RouterRequest {
            // Prompt rendering is delegated to `crate::chat_template`. The
            // default variant ([`ChatTemplate::Llama3`]) reproduces the
            // BitNet-B1.58-2B-4T framing gen-1 bitnet_decode emits:
            //
            //   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            //
            // per turn, followed by a trailing
            // `<|start_header_id|>assistant<|end_header_id|>\n\n` that
            // kicks the model into assistant-reply mode. That framing is
            // the reason a naive "Role: content<|eot_id|>Assistant: "
            // form mismatched the trained template and made the model
            // emit a single `!` then EOT (smoke-tested 2026-04-22 against
            // :8180 while gen-1 at :8080 returned full replies). /v1 and
            // /v2 both route through here, so they tokenize identically
            // to the C++ server when the caller leaves the template
            // alone.
            //
            // The `Short` and `Raw` variants skip part or all of the
            // framing to close the HTTP-vs-kernel tok/s gap on short
            // single-turn requests — see the `chat_template` module docs
            // for exact bytes + trade-offs.
            //
            // User-supplied content is sanitized inside each renderer
            // (any `<|...|>` sequence becomes `«scrubbed»`) so a crafted
            // message cannot smuggle control tokens past the tokenizer
            // and synthesize a system turn. BOS is prepended by the
            // tokenizer inside the router.
            let prompt = params.chat_template.render(messages);

            let mut cfg = SamplerConfig::default();
            if let Some(t) = params.temperature {
                cfg.temperature = t;
            }
            if let Some(p) = params.top_p {
                cfg.top_p = p;
            }

            RouterRequest {
                prompt,
                max_new_tokens: params.max_tokens.max(1),
                sampler: cfg,
                stop: Vec::new(),
            }
        }
    }

    impl super::InferenceBackend for RealBackend {
        fn generate<'a>(
            &'a self,
            messages: &'a [ChatMessage],
            params: &'a GenerationParams,
        ) -> BoxFut<'a, Result<(String, GenerationUsage), ServerError>> {
            let router = self.router.clone();
            let req = Self::build_request(messages, params);
            Box::pin(async move {
                let resp = router
                    .generate(req)
                    .await
                    .map_err(|e| ServerError::Backend(format!("{e}")))?;
                Ok((
                    resp.text,
                    GenerationUsage {
                        prompt_tokens: resp.prompt_tokens,
                        completion_tokens: resp.completion_tokens,
                    },
                ))
            })
        }

        fn generate_stream<'a>(
            &'a self,
            messages: &'a [ChatMessage],
            params: &'a GenerationParams,
        ) -> BoxFut<'a, Result<TokenStream, ServerError>> {
            let router = self.router.clone();
            let req = Self::build_request(messages, params);
            Box::pin(async move {
                let inner = router
                    .generate_stream(req)
                    .await
                    .map_err(|e| ServerError::Backend(format!("{e}")))?;
                let mapped = inner.map(|res| res.map_err(|e| ServerError::Backend(format!("{e}"))));
                let stream: TokenStream = Box::pin(mapped);
                Ok(stream)
            })
        }

        fn list_models(&self) -> Vec<ModelCard> {
            vec![ModelCard::halo(self.router.model_id().to_string())]
        }

        fn perplexity<'a>(
            &'a self,
            text: String,
            stride: u32,
            max_tokens: u32,
        ) -> BoxFut<'a, Result<super::PerplexityOutput, ServerError>> {
            let router = self.router.clone();
            Box::pin(async move {
                let res = router
                    .perplexity(text, stride as usize, max_tokens as usize)
                    .await
                    .map_err(|e| ServerError::Backend(format!("ppl: {e}")))?;
                Ok(super::PerplexityOutput {
                    mean_nll: res.mean_nll,
                    perplexity: res.perplexity,
                    tokens: res.scored_tokens as u32,
                    elapsed_ms: res.elapsed_ms,
                })
            })
        }
    }
}

#[cfg(feature = "real-backend")]
pub use real::RealBackend;
