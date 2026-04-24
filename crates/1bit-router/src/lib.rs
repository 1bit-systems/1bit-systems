//! 1bit-router — the hardware-aware inference dispatcher for 1bit systems.
//!
//! What this crate does:
//!
//! * At construction it **picks one** backend family (HIP on gfx1151 /
//!   MLX on Apple / error otherwise — see [`detect`]).
//! * It mmaps the `.h1b` model + parses the `.htok` tokenizer up front
//!   via `1bit-core`, then hands the weight bytes to the backend (for
//!   HIP: `1bit-hip` uploads them once to the GPU).
//! * It exposes a small async surface — `generate`, `generate_stream`,
//!   `list_models` — that the HTTP layer wraps in an `InferenceBackend`
//!   adapter. **The trait itself lives in `1bit-server`** so this crate
//!   stays free of HTTP types; see 1bit-server's `backend::RealBackend`
//!   for the adapter.
//!
//! Layering:
//!
//! ```text
//!      1bit-server (HTTP)  ──▶  onebit_router::Router
//!                                      │
//!                                      ├── detect() → BackendKind
//!                                      ├── HipBackend (weights + forward pass)
//!                                      └── onebit_core::Sampler (host-side)
//! ```
//!
//! Thread model: the router is `Send + Sync` via a single `tokio::Mutex`
//! around the mutable backend state (KV cache + scratch are per-backend,
//! not per-request). Concurrent HTTP requests serialize behind the mutex;
//! this matches gen-1 behaviour — the C++ server also took a global
//! generation lock because the GPU runs one decode at a time.

#![warn(missing_docs)]

pub mod backend_impl;
pub mod cpu_lane;
pub mod detect;
pub mod medusa;
pub mod sampler;
pub mod tokenizer;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use onebit_core::sampler::{Sampler, SamplerConfig};
use onebit_core::types::TokenId;
use tokio::sync::Mutex;

pub use backend_impl::{BackendError, HipBackend, KvDtype, ModelFormat, sniff_model_format};
pub use detect::{BackendKind, detect};
// Re-exports from the sampler lane. `cpu_lane::` still works as a
// back-compat path for `1bit-server`'s benches — the shim module just
// re-exports these same items.
pub use sampler::cpu::{CpuLane, CpuLaneError, CpuSampler, PipelinedOutcome};
pub use sampler::{SAMPLER_MODE_ENV, SamplerMode, sampler_mode_from_env};

// Re-exported below once the type is declared — see PerplexityResult.

/// Caller-selectable forward-pass backend.
///
/// Distinct from [`BackendKind`], which is the *detected* hardware family —
/// [`Backend`] is the *requested* dispatch surface.
///
/// Variants:
///
/// * [`Backend::Hip`] — AMD gfx1151 iGPU via `1bit-hip`. Default.
/// * [`Backend::Cpu`] — host CPU lane. 7th aspirational surface from
///   `docs/wiki/Peak-Performance-Projection.md`. Scaffolded today in
///   [`cpu_lane::CpuLane`] but not on the critical path yet; selecting
///   it returns [`BackendError::CpuLaneStub`] (see
///   `docs/wiki/CPU-Lane-Plan.md` for the three-step wire-up plan).
///
/// **NPU note (2026-04-21 pivot):** the former `Backend::Xdna` variant
/// (FastFlowLM subprocess bridge) is retired. The NPU path going forward
/// is ONNX Runtime + VitisAI Execution Provider, which plugs in at the
/// ORT session layer rather than as a router backend variant. See
/// `project_npu_path_onnx.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// gfx1151 iGPU path via `1bit-hip` (the production path).
    #[default]
    Hip,
    /// Host CPU lane — sampler + tokenizer + dispatcher on Zen5 cores
    /// via [`cpu_lane::CpuLane`]. Scaffolded today but not on the
    /// critical path; dispatch surfaces [`BackendError::CpuLaneStub`].
    Cpu,
}

impl Backend {
    /// Parse a `HALO_BACKEND=...` value. Case-insensitive. Accepted
    /// spellings: `hip`, `cpu`. Any other value returns an error
    /// that names both accepted spellings so operators see the valid
    /// set without having to `grep` the source.
    pub fn parse_env(raw: &str) -> Result<Self, BackendError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "hip" => Ok(Backend::Hip),
            "cpu" => Ok(Backend::Cpu),
            other => Err(BackendError::Other(format!(
                "HALO_BACKEND: unknown value {other:?}; accepted: hip | cpu"
            ))),
        }
    }

    /// Human-readable label for logs / `/v1/models`.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Hip => "hip",
            Backend::Cpu => "cpu",
        }
    }
}

/// Router configuration — which forward-pass backend to dispatch on.
///
/// Constructed either explicitly by the server layer or via
/// [`RouterConfig::from_env`], which reads `HALO_BACKEND` and falls back to
/// [`Backend::Hip`] when unset. Separate struct (rather than a bare enum
/// argument on `Router::new_with`) so we have a natural place to grow
/// knobs like a prefill-threshold override or a sampler dispatch path
/// without churning call sites.
#[derive(Debug, Clone, Copy)]
pub struct RouterConfig {
    /// Which execution surface the router dispatches the forward pass to.
    pub backend: Backend,
    /// Which sampler dispatch path the router takes — inline (default)
    /// or offloaded to the [`cpu_lane::CpuLane`]. Read from
    /// [`cpu_lane::SAMPLER_MODE_ENV`] (`HALO_SAMPLER`) in
    /// [`RouterConfig::from_env`].
    ///
    /// See `docs/wiki/CPU-Lane-Plan.md` for the decision memo that
    /// explains why the default is `Inline` — short version, the
    /// sampler is ~4.5% of a 15 ms forward pass and doesn't justify
    /// pipelining today.
    pub sampler_mode: SamplerMode,
    /// Prompt-length threshold at or above which the prefill phase is
    /// offloaded to the CPU AVX2 ternary lane (`1bit-cpu`). Below this
    /// threshold the iGPU runs prefill + decode as before.
    ///
    /// Derived in `docs/wiki/APU-Aggregator-Plan.md` §2: at L=33 the
    /// iGPU's per-token M=1 GEMV cost equals the CPU's AVX2 prefill cost
    /// plus a single host→device handoff. Overridable via
    /// `HALO_PREFILL_CROSSOVER_L` — setting it to `usize::MAX` (or any
    /// value ≥ a plausible max prompt length) pins every request to the
    /// iGPU lane, which is the rollback path.
    pub prefill_crossover_len: usize,
}

/// Env var name for the prefill-crossover override. Public so operator
/// tooling (`halo power`, `halo status`) can surface the same name
/// without stringly-typing it.
pub const PREFILL_CROSSOVER_ENV: &str = "HALO_PREFILL_CROSSOVER_L";

/// Default prefill crossover — see `docs/wiki/APU-Aggregator-Plan.md` §2.
/// Changing this default means updating the plan doc and re-measuring;
/// per `CLAUDE.md` the `.h1b` loader + `1bit-cpu` kernel are the only
/// callers in the runtime path.
pub const DEFAULT_PREFILL_CROSSOVER_L: usize = 33;

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            backend: Backend::default(),
            sampler_mode: SamplerMode::default(),
            prefill_crossover_len: DEFAULT_PREFILL_CROSSOVER_L,
        }
    }
}

impl RouterConfig {
    /// Build a config from the `HALO_BACKEND` + `HALO_SAMPLER` env vars.
    /// Unset → defaults (Hip / Inline). Set but unparsable →
    /// [`BackendError::Other`].
    pub fn from_env() -> Result<Self, BackendError> {
        let backend = match std::env::var("HALO_BACKEND") {
            Ok(raw) if !raw.is_empty() => Backend::parse_env(&raw)?,
            _ => Backend::default(),
        };
        let sampler_mode = sampler_mode_from_env()?;
        let prefill_crossover_len = match std::env::var(PREFILL_CROSSOVER_ENV) {
            Ok(raw) if !raw.trim().is_empty() => raw.trim().parse::<usize>().map_err(|e| {
                BackendError::Other(format!(
                    "{PREFILL_CROSSOVER_ENV}: {raw:?} is not a valid non-negative \
                     integer: {e}"
                ))
            })?,
            _ => DEFAULT_PREFILL_CROSSOVER_L,
        };
        Ok(Self {
            backend,
            sampler_mode,
            prefill_crossover_len,
        })
    }
}

/// Sampling + stopping parameters forwarded through the router.
///
/// Deliberately smaller than OpenAI's full request shape — 1bit-server
/// translates user-facing options (temperature, top_p, max_tokens) into
/// this struct at request time.
#[derive(Debug, Clone)]
pub struct RouterRequest {
    /// Full conversation flattened to a single prompt string (system +
    /// user + assistant turns concatenated as the server decides).
    pub prompt: String,
    /// Hard cap on newly generated tokens.
    pub max_new_tokens: u32,
    /// Sampler knobs. `temperature <= 0` → greedy argmax on GPU.
    pub sampler: SamplerConfig,
    /// Optional stop strings. Matched against the detokenized tail.
    pub stop: Vec<String>,
}

impl Default for RouterRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_new_tokens: 256,
            sampler: SamplerConfig::default(),
            stop: Vec::new(),
        }
    }
}

/// A single generation result. Includes the full decoded text plus
/// token accounting in the same shape 1bit-server will re-emit in its
/// OpenAI usage block.
#[derive(Debug, Clone)]
pub struct RouterResponse {
    /// Decoded text (no BOS / EOS, suitable for a chat reply).
    pub text: String,
    /// `true` if we stopped because of EOS or a `stop` string; `false`
    /// if we hit `max_new_tokens`.
    pub stopped_on_eos: bool,
    /// Tokens consumed from the prompt (after tokenization + BOS).
    pub prompt_tokens: u32,
    /// Tokens emitted during decode.
    pub completion_tokens: u32,
}

/// The dispatcher. Always constructed via [`Router::new`] and then held
/// behind an `Arc` in the HTTP layer.
pub struct Router {
    inner: Arc<Mutex<Inner>>,
    model_id: String,
    kind: BackendKind,
    /// Caller-selected dispatch surface. Separate from `kind` (which is
    /// *detected* hardware); `backend` is what the operator asked for via
    /// `HALO_BACKEND` / [`RouterConfig`].
    backend: Backend,
    /// Sampler dispatch path. [`SamplerMode::Cpu`] (default) runs the
    /// sampler on the [`CpuLane`]'s rayon pool via
    /// [`CpuSampler::sample_pipelined`] — a `flume::bounded(1)` handoff
    /// that moves the logits buffer off the GPU-dispatch thread and
    /// onto a persistent `halo-sampler-pipe` worker, freeing the
    /// dispatch thread to start the next `forward_token`'s GPU staging
    /// while the sampler math runs on Zen5. [`SamplerMode::Inline`] is
    /// the legacy path retained for A/B comparison + rollback via
    /// `HALO_SAMPLER=inline`. See the `sampler` module docs for the
    /// 2026-04-20 flip rationale.
    sampler_mode: SamplerMode,
    /// CPU lane. Always constructed — the pool is cheap — but only
    /// dispatched through when `sampler_mode == SamplerMode::Cpu`.
    /// Held behind an `Arc` so `generate_blocking` can clone a handle
    /// into the spawn_blocking closure without taking a reference to
    /// `self`.
    cpu_lane: Arc<CpuLane>,
    /// Pipelined sampler — the bounded-channel handoff. Always
    /// constructed but only driven when
    /// `sampler_mode == SamplerMode::Cpu`. Held behind an `Arc` so
    /// both `generate` and `generate_stream` can hand a handle into
    /// their respective `spawn_blocking` closures. The worker thread
    /// it owns sticks around for the lifetime of the router.
    cpu_sampler: Arc<CpuSampler>,
    /// Medusa speculative-decode state. `Disabled` unless both
    /// `HALO_MEDUSA=1` is set AND the heads file at
    /// `HALO_MEDUSA_HEADS_PATH` loaded cleanly. Arc'd so the verify path
    /// can cheaply snapshot into worker closures.
    #[allow(dead_code)] // consumed by follow-up passes that wire the verify loop
    medusa: Arc<medusa::MedusaState>,
    /// Prompt-length threshold at or above which prefill offloads to the
    /// CPU AVX2 ternary lane. See
    /// [`RouterConfig::prefill_crossover_len`] for the source of truth
    /// and `docs/wiki/APU-Aggregator-Plan.md` §2 for the derivation.
    /// Read by [`Router::check_prefill_routing`] — the CPU forward-pass
    /// itself still lands on `BackendError::CpuLaneStub` until the
    /// Rust-side attention / RMSNorm / GLU glue is wired (tracked
    /// alongside the `1bit-cpu` FFI crate).
    #[allow(dead_code)] // reserved for the follow-up forward-pass wire-up
    prefill_crossover_len: usize,
}

struct Inner {
    backend: HipBackend,
    /// Last decoded position in the shared KV cache. Reset between
    /// conversations.
    pos: i32,
    /// Live Medusa verifier state. Accumulates per-head acceptance
    /// counters across decode requests on this router instance. Always
    /// present — the verifier is zero-cost when the
    /// [`medusa::MedusaState`] is `Disabled`, and we read its counters
    /// unconditionally when `/metrics` lands them.
    medusa_verifier: medusa::TreeVerifier,
}

impl Router {
    /// Construct a router for the given `.h1b` model file. The tokenizer
    /// is resolved by taking the model path and replacing `.h1b` with
    /// `.htok`.
    ///
    /// Dispatch backend is taken from the `HALO_BACKEND` env var (via
    /// [`RouterConfig::from_env`]); set to `hip` (default) or `cpu`.
    ///
    /// Fails if no supported backend is available on this host.
    pub fn new(h1b_path: impl AsRef<Path>) -> Result<Self, BackendError> {
        let h1b_path = h1b_path.as_ref();
        let htok_path = default_htok_path(h1b_path);
        let model_id = default_model_id(h1b_path);

        let cfg = RouterConfig::from_env()?;
        Self::new_with_config(h1b_path, &htok_path, model_id, 4096, cfg)
    }

    /// Full constructor — lets the caller pin the tokenizer location,
    /// model id, and max KV cache size. Backend defaults to `Hip`.
    pub fn new_with(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        Self::new_with_config(
            h1b_path,
            htok_path,
            model_id,
            max_context,
            RouterConfig::default(),
        )
    }

    /// Full constructor with an explicit [`RouterConfig`]. This is the
    /// call site that actually reads `cfg.backend` and pins it onto the
    /// returned router. The dispatch decision happens inside
    /// [`Router::generate`] — see the routing guard there.
    pub fn new_with_config(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
        cfg: RouterConfig,
    ) -> Result<Self, BackendError> {
        let kind = detect();
        match kind {
            BackendKind::Hip => {
                let backend = HipBackend::new(h1b_path, htok_path, model_id.clone(), max_context)?;
                // Build the CPU lane once and keep it on the router so
                // `HALO_SAMPLER=cpu` doesn't race to construct a pool
                // per request. Constructing with the default policy
                // honours `HALO_CPU_THREADS` automatically.
                let cpu_lane = Arc::new(
                    CpuLane::new()
                        .map_err(|e| BackendError::Other(format!("cpu lane init: {e}")))?,
                );
                // Pipelined sampler — persistent `halo-sampler-pipe`
                // worker fed by a `flume::bounded(1)` queue. Only
                // dispatched through when `sampler_mode == Cpu`, but
                // constructed unconditionally: the worker thread is
                // ~1 MB RSS and spawning it lazily would race with
                // the first decode request. See `sampler::cpu::CpuSampler`.
                let cpu_sampler = Arc::new(CpuSampler::new(cpu_lane.clone()));

                let medusa_cfg = medusa::MedusaConfig::from_env();
                let medusa_state = match medusa::MedusaState::from_config(&medusa_cfg) {
                    Ok(s) => {
                        if s.is_enabled() {
                            tracing::info!(
                                "medusa enabled: heads loaded from {:?}",
                                medusa_cfg.medusa_heads_path
                            );
                        }
                        s
                    }
                    Err(e) => {
                        tracing::warn!("medusa init failed, falling back to Disabled: {e}");
                        medusa::MedusaState::Disabled
                    }
                };

                tracing::info!(
                    model_id,
                    hw = %kind.label(),
                    requested = cfg.backend.label(),
                    sampler = cfg.sampler_mode.label(),
                    cpu_lane_threads = cpu_lane.num_threads(),
                    medusa = medusa_state.is_enabled(),
                    prefill_crossover_len = cfg.prefill_crossover_len,
                    label = backend.label(),
                    "router ready"
                );
                Ok(Self {
                    inner: Arc::new(Mutex::new(Inner {
                        backend,
                        pos: 0,
                        medusa_verifier: medusa::TreeVerifier::new(),
                    })),
                    model_id,
                    kind,
                    backend: cfg.backend,
                    sampler_mode: cfg.sampler_mode,
                    cpu_lane,
                    cpu_sampler,
                    medusa: Arc::new(medusa_state),
                    prefill_crossover_len: cfg.prefill_crossover_len,
                })
            }
            BackendKind::Mlx => {
                unimplemented!(
                    "MLX dispatch is not wired in this session — gfx1151 is the only supported target"
                );
            }
            BackendKind::None => Err(BackendError::Other(
                "no supported accelerator: need AMD gfx1151 (ROCm) or Apple MLX".into(),
            )),
        }
    }

    /// Advertised model id.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Which backend family the *hardware detection* selected.
    pub fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    /// Which dispatch surface the *operator asked for* (via `HALO_BACKEND`
    /// or an explicit [`RouterConfig`]).
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Evaluate the prefill-routing decision without running the forward
    /// pass. Factored out of [`Router::generate`] so both sync and stream
    /// paths (and tests) share the same policy.
    ///
    /// Delegates to [`prefill_routing_decision`] with `self.backend`.
    pub fn check_prefill_routing(&self, prompt_tokens: usize) -> Result<(), BackendError> {
        prefill_routing_decision(self.backend, prompt_tokens)
    }

    /// Non-streaming generation. Blocks the calling task while the
    /// forward pass runs; call via `tokio::task::spawn_blocking` if the
    /// caller is on a shared runtime thread.
    pub async fn generate(&self, req: RouterRequest) -> Result<RouterResponse, BackendError> {
        let inner = self.inner.clone();
        let backend = self.backend;
        let sampler_mode = self.sampler_mode;
        let cpu_lane = self.cpu_lane.clone();
        let cpu_sampler = self.cpu_sampler.clone();
        let medusa = self.medusa.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            generate_blocking(
                &mut guard,
                req,
                None,
                backend,
                sampler_mode,
                &cpu_lane,
                &cpu_sampler,
                &medusa,
            )
        })
        .await
        .map_err(|e| BackendError::Other(format!("blocking task join: {e}")))?
    }

    /// Streaming generation. Each emitted `String` is a fresh delta that
    /// should be concatenated onto what the caller already produced.
    ///
    /// Emits bytes through a bounded channel so SSE backpressure flows
    /// all the way down to the forward-pass loop.
    pub async fn generate_stream(
        &self,
        req: RouterRequest,
    ) -> Result<
        impl futures::stream::Stream<Item = Result<String, BackendError>> + Send + 'static,
        BackendError,
    > {
        let inner = self.inner.clone();
        let backend = self.backend;
        let sampler_mode = self.sampler_mode;
        let cpu_lane = self.cpu_lane.clone();
        let cpu_sampler = self.cpu_sampler.clone();
        let medusa = self.medusa.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, BackendError>>(64);
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let tx_clone = tx.clone();
            let sink = move |delta: String| {
                // If the client hung up we cannot do anything useful; drop.
                let _ = tx_clone.blocking_send(Ok(delta));
            };
            if let Err(e) = generate_blocking(
                &mut guard,
                req,
                Some(Box::new(sink)),
                backend,
                sampler_mode,
                &cpu_lane,
                &cpu_sampler,
                &medusa,
            ) {
                let _ = tx.blocking_send(Err(e));
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    /// Which sampler path the router is using (process-level dial).
    pub fn sampler_mode(&self) -> SamplerMode {
        self.sampler_mode
    }

    /// `true` iff the Medusa speculative-decode gate resolved to
    /// `Enabled` at router construction. Used by `/metrics` + the
    /// bench harness so operators can confirm the gate without
    /// grepping startup logs.
    pub fn medusa_enabled(&self) -> bool {
        self.medusa.is_enabled()
    }

    /// Snapshot of the Medusa verifier counters since router startup.
    ///
    /// Returns `None` if the gate is off or no verify steps have run
    /// yet. Carries: (total verify steps, per-head accepted counts,
    /// mean accepted-prefix length). Cheap — integer loads only.
    pub async fn medusa_stats(&self) -> Option<MedusaStats> {
        let guard = self.inner.lock().await;
        if !self.medusa.is_enabled() || guard.medusa_verifier.steps == 0 {
            return None;
        }
        let rates = guard.medusa_verifier.per_head_acceptance()?;
        let mean_prefix = guard.medusa_verifier.mean_accepted_prefix_len()?;
        Some(MedusaStats {
            verify_steps: guard.medusa_verifier.steps,
            head_accepted: guard.medusa_verifier.head_accepted,
            per_head_rate: rates,
            mean_accepted_prefix_len: mean_prefix,
        })
    }

    /// Reset any per-conversation state on the backend (KV cache pointer).
    /// The C++ server does this between REPL turns and at server startup.
    pub async fn reset(&self) {
        let mut guard = self.inner.lock().await;
        guard.backend.reset();
        guard.pos = 0;
    }

    /// Compute perplexity over `text`.
    ///
    /// Semantics match the gen-1 C++ `bitnet_decode --ppl` mode byte-for-byte:
    ///
    /// 1. Tokenize `text` (with BOS). Truncate to at most `max_tokens` tokens.
    /// 2. Starting from `pos=0` (KV cache is reset for the run), feed tokens
    ///    one at a time through `forward_token`. At each step `i`, compute
    ///    `log_softmax(logits)[prompt_ids[i+1]]` via log-sum-exp, accumulate
    ///    `-logp` into `total_nll`, increment `scored`.
    /// 3. If the sequence is longer than `stride`, we re-chunk: every
    ///    `stride` tokens we reset the KV cache and restart at `pos=0`. The
    ///    C++ reference does not do this — it's always a single pass — but
    ///    is what we'd need if `max_tokens > max_context`. For parity with
    ///    the documented 1024-token baseline, pass `max_tokens=1024,
    ///    stride=1024` and we take the single-pass path.
    ///
    /// Returns `(mean_nll, perplexity, scored_tokens, elapsed_ms)`.
    ///
    /// Invariants:
    ///   * After this call the router's `pos` is reset to 0 (KV cache is
    ///     considered clean).
    ///   * `stride` clamped to `max_context`. `max_tokens` clamped to
    ///     `usize::MAX` but limited by the actual tokenized length.
    pub async fn perplexity(
        &self,
        text: String,
        stride: usize,
        max_tokens: usize,
    ) -> Result<PerplexityResult, BackendError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            perplexity_blocking(&mut guard, &text, stride, max_tokens)
        })
        .await
        .map_err(|e| BackendError::Other(format!("blocking task join: {e}")))?
    }
}

/// Snapshot of the Medusa verifier state. Read via
/// [`Router::medusa_stats`] for bench + `/metrics`.
#[derive(Debug, Clone)]
pub struct MedusaStats {
    /// Cumulative count of verify steps run since router startup.
    pub verify_steps: u64,
    /// Per-head accepted counts — `head_accepted[i]` is the number
    /// of times head i produced the same argmax as the backbone at
    /// the corresponding position.
    pub head_accepted: [u64; medusa::heads::NUM_MEDUSA_HEADS],
    /// Live per-head acceptance rate — just `head_accepted / steps`.
    pub per_head_rate: [f64; medusa::heads::NUM_MEDUSA_HEADS],
    /// Mean accepted prefix length (0..=NUM_MEDUSA_HEADS). Equal to
    /// `tokens/cycle - 1` — the average number of speculative
    /// tokens accepted per decode cycle.
    pub mean_accepted_prefix_len: f64,
}

/// PPL harness result.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Mean negative log-likelihood across all scored tokens (nats).
    pub mean_nll: f64,
    /// `exp(mean_nll)` — standard perplexity.
    pub perplexity: f64,
    /// Number of (context, next-token) pairs that contributed to the mean.
    pub scored_tokens: usize,
    /// Wall-clock time of the scoring loop in milliseconds.
    pub elapsed_ms: f64,
}

fn perplexity_blocking(
    inner: &mut Inner,
    text: &str,
    stride: usize,
    max_tokens: usize,
) -> Result<PerplexityResult, BackendError> {
    // Reset KV so the run is deterministic regardless of previous chat
    // history. Matches gen-1 where --ppl always starts from a fresh decoder.
    inner.backend.reset();
    inner.pos = 0;

    let mut ids: Vec<TokenId> = inner.backend.tokenize(text);
    if ids.len() > max_tokens {
        ids.truncate(max_tokens);
    }
    if ids.len() < 2 {
        return Err(BackendError::BadInput(
            "ppl: need at least 2 tokens after tokenization",
        ));
    }
    // `stride` bounds the per-pass window length. 0 is treated as
    // "whole sequence in one pass".
    let stride = if stride == 0 { ids.len() } else { stride };

    let mut logits_scratch: Vec<f32> = Vec::new();
    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let start = std::time::Instant::now();

    let n = ids.len();
    let mut chunk_start = 0usize;
    while chunk_start + 1 < n {
        let chunk_end = (chunk_start + stride).min(n);
        // For each chunk we feed tokens [chunk_start .. chunk_end) at
        // positions [0 .. chunk_end-chunk_start) and score the next-token
        // probability at every step except the last of this chunk (that
        // one has no target inside the chunk unless the chunk ends at `n`,
        // in which case chunk_end == n and i+1 == n so we also stop).
        // Reset KV between chunks.
        if chunk_start > 0 {
            inner.backend.reset();
            inner.pos = 0;
        }
        let chunk_len = chunk_end - chunk_start;
        for i in 0..chunk_len {
            let tok = ids[chunk_start + i];
            let pos = i as i32;
            let _argmax = inner.backend.forward_token(tok, pos, &mut logits_scratch)?;
            // Target for this step is ids[chunk_start + i + 1]; only valid
            // while it's inside the current chunk (we re-feed at the next
            // chunk boundary).
            let target_idx = chunk_start + i + 1;
            if target_idx >= chunk_end {
                break;
            }
            let target = ids[target_idx];
            let nll = neg_log_softmax_at(&logits_scratch, target);
            total_nll += nll;
            scored += 1;
        }
        if chunk_end == n {
            break;
        }
        chunk_start = chunk_end;
    }

    // Leave the backend in a clean state so the next chat request doesn't
    // inherit our KV cache.
    inner.backend.reset();
    inner.pos = 0;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    if scored == 0 {
        return Err(BackendError::BadInput("ppl: no tokens were scored"));
    }
    let mean_nll = total_nll / (scored as f64);
    Ok(PerplexityResult {
        mean_nll,
        perplexity: mean_nll.exp(),
        scored_tokens: scored,
        elapsed_ms,
    })
}

/// Compute `-log_softmax(logits)[target]` in f64 via log-sum-exp.
///
/// Shared helper; factored out so the PPL math is unit-testable without
/// a GPU.
pub(crate) fn neg_log_softmax_at(logits: &[f32], target: TokenId) -> f64 {
    let v = logits.len();
    debug_assert!(v > 0);
    debug_assert!((target as usize) < v, "target out of vocab range");
    let mut max_l = logits[0];
    for &l in &logits[1..] {
        if l > max_l {
            max_l = l;
        }
    }
    let mut sum_exp: f64 = 0.0;
    for &l in logits {
        sum_exp += ((l - max_l) as f64).exp();
    }
    let logp = (logits[target as usize] - max_l) as f64 - sum_exp.ln();
    -logp
}

/// Synchronous generation path — shared between `generate` and
/// `generate_stream`. If `on_delta` is set, each new delta string is
/// forwarded to it as soon as the tokenizer emits decoded bytes; the
/// full text is also returned in the response.
// TODO(gap-p2): fold `(backend, sampler_mode, cpu_lane, cpu_sampler, medusa)`
// into a single `GenCtx` struct to get under clippy's 7-arg ceiling. Non-
// trivial: every call site would need adjusting and the `Inner` vs. `GenCtx`
// borrow split needs thought. Not in scope for the clippy gate flip.
#[allow(clippy::too_many_arguments)]
fn generate_blocking(
    inner: &mut Inner,
    req: RouterRequest,
    mut on_delta: Option<Box<dyn FnMut(String) + Send>>,
    backend: Backend,
    sampler_mode: SamplerMode,
    cpu_lane: &CpuLane,
    cpu_sampler: &CpuSampler,
    medusa: &medusa::MedusaState,
) -> Result<RouterResponse, BackendError> {
    // `cpu_lane` is currently reserved for future use (parallel top-k
    // reductions beyond the `Sampler::sample` path). The `CpuSampler`
    // handoff already owns a clone of the `Arc<CpuLane>` and schedules
    // its worker inside that pool. Keeping the reference on the
    // signature so a future top-k-on-lane path doesn't need another
    // plumbing churn.
    let _ = cpu_lane;
    // ---- Tokenize prompt ----
    let prompt_ids: Vec<TokenId> = inner.backend.tokenize(&req.prompt);
    if prompt_ids.is_empty() {
        return Err(BackendError::BadInput("empty prompt after tokenization"));
    }

    // ---- Routing guard ----
    //
    //   * Backend::Hip → always proceed (the happy path).
    //   * Backend::Cpu → `CpuLaneStub`.
    //
    // NPU lane (formerly `Backend::Xdna` + FastFlowLM) was retired
    // 2026-04-21; the replacement is ONNX Runtime + VitisAI EP, which
    // plugs in at the ORT session layer rather than as a router backend.
    // See `project_npu_path_onnx.md`.
    prefill_routing_decision(backend, prompt_ids.len())?;

    let prompt_tokens = prompt_ids.len() as u32;
    let max_new = req.max_new_tokens.max(1) as usize;

    // Reset the KV cache position per request. Matches gen-1 bitnet_decode's
    // `cache_pos = 0` at the start of every /v1/chat/completions handler.
    // Without this, `pos` accumulates across requests and eventually hits
    // the max_context ceiling (4096) with "context overflow: 4096 + 1 >
    // 4096" errors. Stateful multi-turn carryover is a separate feature.
    inner.pos = 0;

    // Seed the sampler history with the prompt so rep penalty behaves.
    let mut history: Vec<TokenId> = prompt_ids.clone();
    let mut sampler = Sampler::new(req.sampler);
    let mut logits_scratch: Vec<f32> = Vec::new();

    // ---- Prefill (no sampling — we just run the forward pass on every
    //      prompt token to populate KV cache entries). The last token
    //      seeds `cur` for the decode loop. ----
    let mut cur: TokenId = prompt_ids[0];
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let next = inner
            .backend
            .forward_token(tok, inner.pos + i as i32, &mut logits_scratch)?;
        cur = next; // prefill's "next" is only used to seed decode
    }
    inner.pos += prompt_ids.len() as i32;

    // ---- Decode loop ----
    let mut generated_ids: Vec<TokenId> = Vec::with_capacity(max_new);
    let mut stopped_on_eos = false;
    let mut printed_bytes = 0usize;
    let mut full_text = String::new();

    // BitNet's special tokens from the tokenizer — matches the C++ stops.
    let stop_ids = [128001, 128009];

    // Medusa dispatch gate. Only the greedy path activates it today
    // (the sampler-aware variant compares head argmax with a sampled
    // token rather than base argmax — a follow-up pass once the tree-
    // attention kernel is wired). Stepper indirection keeps the legacy
    // path byte-identical for the `medusa_disabled` case.
    let medusa_active = medusa.is_enabled() && req.sampler.temperature <= 0.0 && on_delta.is_none();
    let mut hidden_scratch: Vec<u16> = Vec::new();
    let mut head_logits_scratch: Vec<f32> = Vec::new();

    let mut step_count = 0usize;
    while step_count < max_new {
        let pos = inner.pos + step_count as i32;

        // -------------------------------------------------------------
        // Base forward pass. Medusa path takes a variant that also
        // copies the post-final-norm hidden state back to the host so
        // the heads can project it into candidate-token logits.
        // -------------------------------------------------------------
        // Greedy fast-path opt-in: clearing logits_scratch before the
        // forward call signals forward_token to skip the 512 KB D→H copy
        // and the host-argmax reconcile (see HALO_SKIP_LOGITS_COPY note
        // in backend_impl.rs). For temp>0 we need the full logits vector
        // on host for the sampler, so we leave it populated.
        if !medusa_active && req.sampler.temperature <= 0.0 {
            logits_scratch.clear();
        }
        let argmax_next = if medusa_active {
            inner.backend.forward_token_with_hidden(
                cur,
                pos,
                &mut logits_scratch,
                &mut hidden_scratch,
            )?
        } else {
            inner.backend.forward_token(cur, pos, &mut logits_scratch)?
        };

        // Optional host-side sampling path. [`SamplerMode::Inline`]
        // takes the direct call (legacy, `HALO_SAMPLER=inline`).
        // [`SamplerMode::Cpu`] (default as of 2026-04-20) routes through
        // [`CpuSampler::sample_pipelined`] — a `flume::bounded(1)`
        // handoff onto a persistent `halo-sampler-pipe` worker pinned
        // to the [`CpuLane`]'s rayon pool. Semantics are bit-identical
        // (same `Sampler::sample` code, same RNG state, just a
        // different executing thread). Greedy (`temp <= 0`)
        // short-circuits to the GPU-returned argmax in both modes
        // since there's no scalar work to do.
        let next = if req.sampler.temperature > 0.0 {
            match sampler_mode {
                SamplerMode::Inline => sampler.sample(&mut logits_scratch, &history)?,
                SamplerMode::Cpu => {
                    // Move the scratch buffers into the worker via
                    // flume; receive them back on the reply so we keep
                    // the allocation alive across decode steps. The
                    // `std::mem::take` swap replaces `logits_scratch`
                    // with an empty Vec — the next `forward_token`
                    // call will `extend_from_slice` / `resize` into
                    // whichever Vec we put back.
                    let logits_out = std::mem::take(&mut logits_scratch);
                    let history_out = std::mem::take(&mut history);
                    let out = cpu_sampler
                        .sample_pipelined(&mut sampler, logits_out, history_out)
                        .map_err(|e| BackendError::Other(format!("cpu sampler: {e}")))?;
                    // Put the buffers back even on sampler-error so
                    // the allocation sticks with the decode loop.
                    logits_scratch = out.logits;
                    history = out.history;
                    out.outcome?
                }
            }
        } else {
            argmax_next
        };

        history.push(next);

        // Stop-token check before detokenizing: otherwise the special
        // token's string form (e.g. "<|eot_id|>") leaks into the output.
        if stop_ids.contains(&next) {
            stopped_on_eos = true;
            break;
        }

        generated_ids.push(next);
        step_count += 1;

        // Incremental detokenization — decode the whole generated prefix
        // each step (matches C++) and emit whatever is newly printable.
        let decoded_so_far = inner.backend.detokenize(&generated_ids);
        if decoded_so_far.len() > printed_bytes {
            let delta = decoded_so_far[printed_bytes..].to_string();
            printed_bytes = decoded_so_far.len();
            full_text.push_str(&delta);
            if let Some(cb) = on_delta.as_mut() {
                cb(delta);
            }
        }

        if req
            .stop
            .iter()
            .any(|s| !s.is_empty() && full_text.ends_with(s))
        {
            stopped_on_eos = true;
            break;
        }

        cur = next;

        // -------------------------------------------------------------
        // Medusa speculative-decode extension. When the heads are
        // loaded and we took the with-hidden path above, project the
        // backbone hidden through the four heads to get candidate
        // tokens (at positions t+2..t+5 relative to the base token we
        // just emitted), then verify them sequentially against the
        // backbone. Accept the longest matching prefix.
        //
        // Sequential verify = one backbone call per verified position.
        // Each verify call advances the KV cache by one slot, so if
        // head i is rejected we must NOT continue to verify[i+1] —
        // that would feed the rejected head_i into the backbone and
        // corrupt KV. We therefore break out of the verify loop on
        // first mismatch. Tree-attention amortisation (the follow-up
        // lane) is what actually yields >1.0 tok/backbone.
        // -------------------------------------------------------------
        if medusa_active
            && step_count < max_new
            && let medusa::MedusaState::Enabled { heads, .. } = medusa
        {
                // 1) Project backbone hidden through the 4 heads →
                //    host-side `h_out` fp16 vectors. Runs on-device via
                //    `fp16_gemv` + `silu_glu_fp16` with weights uploaded
                //    once at load time. The `Mutex` never contends — the
                //    outer `Inner` lock already serializes decode — it
                //    only hands out the `&mut` the device path needs.
                let mut guard = heads
                    .lock()
                    .map_err(|e| BackendError::Other(format!("medusa heads lock poisoned: {e}")))?;
                let projected = guard
                    .project_all_heads_device(&hidden_scratch)
                    .map_err(|e| BackendError::Other(format!("medusa project: {e}")))?;
                drop(guard);

                // 2) Head logits = lm_head(h_out_i); take argmax per
                //    head. Re-use the live backbone lm_head GEMV on
                //    the device to avoid a second 128256×2560 upload
                //    per cycle.
                let mut head_candidates = [0i32; medusa::heads::NUM_MEDUSA_HEADS];
                for i in 0..medusa::heads::NUM_MEDUSA_HEADS {
                    inner
                        .backend
                        .lm_head_from_hidden(&projected[i], &mut head_logits_scratch)?;
                    let mut max_l = head_logits_scratch[0];
                    let mut max_i = 0i32;
                    for (idx, &l) in head_logits_scratch.iter().enumerate().skip(1) {
                        if l > max_l {
                            max_l = l;
                            max_i = idx as i32;
                        }
                    }
                    head_candidates[i] = max_i;
                }

                // 3) Sequential verify.
                //
                //    Standard Medusa convention: head_i predicts the
                //    token at position base+(i+1) relative to the
                //    backbone's own argmax `next`. So head_0 predicts
                //    t+2 (one past the base), head_1 → t+3, etc. Each
                //    verify step runs the backbone on the prior
                //    accepted token to produce base_argmax_i at that
                //    head's position.
                //
                //    verify_cur starts as `next` — the base token at
                //    position `pos+1`. verify[0] feeds it at pos+1 to
                //    predict pos+2 (= head_0's position).
                //
                //    On a match, we accept head_i and advance
                //    verify_cur = head_i for the next step. On a
                //    mismatch we break — the KV cache past this
                //    position is still clean so the next decode
                //    cycle's first forward_token Just Works.
                let mut verify_cur = next;
                let mut accepted_len = 0usize;
                let mut mismatch_base: Option<i32> = None;
                for &cand in head_candidates.iter() {
                    // Respect remaining budget.
                    if step_count + accepted_len + 1 > max_new {
                        break;
                    }
                    let vpos = inner.pos + (step_count + accepted_len) as i32;
                    let base_argmax_i =
                        inner
                            .backend
                            .forward_token(verify_cur, vpos, &mut logits_scratch)?;

                    if base_argmax_i == cand {
                        accepted_len += 1;
                        verify_cur = cand;
                        if stop_ids.contains(&cand) {
                            mismatch_base = None;
                            break;
                        }
                        continue;
                    }

                    // Mismatch: record the base argmax as the fallback
                    // token to emit, then stop verifying.
                    mismatch_base = Some(base_argmax_i);
                    break;
                }

                // 4) Update the verifier's counters (for `/metrics`
                //    and the bench output). We hand it the full-row
                //    base_argmax reconstructed from the matches +
                //    (at most one) mismatch we observed. Positions
                //    beyond the first mismatch were never verified
                //    sequentially, so we mark them never-equal so
                //    the verifier's per-head counters don't falsely
                //    credit them.
                let mut base_argmax_full = [0i32; medusa::heads::NUM_MEDUSA_HEADS];
                // Matched heads: base == head by construction.
                base_argmax_full[..accepted_len].copy_from_slice(&head_candidates[..accepted_len]);
                if accepted_len < medusa::heads::NUM_MEDUSA_HEADS {
                    if let Some(bm) = mismatch_base {
                        base_argmax_full[accepted_len] = bm;
                    } else {
                        // Stop-token early-exit on the accepted
                        // path; no "base vs head" comparison was
                        // done at this index. Mark never-equal.
                        base_argmax_full[accepted_len] =
                            head_candidates[accepted_len].wrapping_add(1);
                    }
                }
                for (j, cand) in head_candidates.iter().enumerate().skip(accepted_len + 1) {
                    // Never verified — mark never-equal so the
                    // per-head counter stays honest.
                    base_argmax_full[j] = cand.wrapping_add(1);
                }
                let _ = inner
                    .medusa_verifier
                    .verify_step(&head_candidates, &base_argmax_full);

                // 5) Emit accepted head-predicted tokens.
                for &tok in head_candidates.iter().take(accepted_len) {
                    history.push(tok);
                    generated_ids.push(tok);
                    step_count += 1;
                    if stop_ids.contains(&tok) {
                        stopped_on_eos = true;
                        break;
                    }
                }

                // 6) Emit the "free" fallback base token from the
                //    verify step that recorded a mismatch. This is
                //    what makes sequential verify equivalent in
                //    wall-clock to vanilla decode: every verify call
                //    produces a usable base argmax, and if the head
                //    at that position was wrong, we emit the correct
                //    base answer instead.
                if !stopped_on_eos && step_count < max_new {
                    if let Some(bm) = mismatch_base {
                        history.push(bm);
                        if stop_ids.contains(&bm) {
                            stopped_on_eos = true;
                        } else {
                            generated_ids.push(bm);
                            step_count += 1;
                            cur = bm;
                        }
                    } else if accepted_len > 0 {
                        cur = head_candidates[accepted_len - 1];
                    }
                } else if accepted_len > 0 {
                    cur = head_candidates[accepted_len - 1];
                }

                // 7) Re-detokenize for streaming / stop-substring
                //    checks. Streaming path is gated off above
                //    (`medusa_active` requires `on_delta.is_none()`)
                //    so this only matters for stop-substring and the
                //    terminal `full_text`.
                let decoded_so_far = inner.backend.detokenize(&generated_ids);
                if decoded_so_far.len() > printed_bytes {
                    let delta = decoded_so_far[printed_bytes..].to_string();
                    printed_bytes = decoded_so_far.len();
                    full_text.push_str(&delta);
                }
                if stopped_on_eos {
                    break;
                }
                if req
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && full_text.ends_with(s))
                {
                    stopped_on_eos = true;
                    break;
                }
            }
        }

    // `step_count` is the number of generated tokens emitted; the
    // accumulator is consumed below via `generated_ids.len()`.
    let _ = step_count;

    // Advance cache_pos past this turn's decoded tokens.
    inner.pos += generated_ids.len() as i32;

    Ok(RouterResponse {
        text: full_text,
        stopped_on_eos,
        prompt_tokens,
        completion_tokens: generated_ids.len() as u32,
    })
}

/// Default tokenizer path beside the model: `foo.h1b` → `foo.htok`.
fn default_htok_path(h1b_path: &Path) -> PathBuf {
    let mut p = h1b_path.to_path_buf();
    p.set_extension("htok");
    p
}

/// Default model-id string: file stem of the `.h1b`, lowercased.
fn default_model_id(h1b_path: &Path) -> String {
    h1b_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "halo-model".to_string())
}

// (thiserror's `#[from]` on the `Halo` / `Hip` variants already generates
// the `From<HaloError>` / `From<RcppError>` impls the `?` operator needs.)

/// Standalone routing-policy function. `Ok(())` = safe to dispatch on the
/// current HIP path.
///
/// Rules:
///
/// * [`Backend::Hip`] → always OK.
/// * [`Backend::Cpu`] → always [`BackendError::CpuLaneStub`]. The lane
///   itself (thread pool, parallel sampler primitive) is scaffolded in
///   [`cpu_lane`], but it's not on the critical path yet; operators who
///   explicitly set `HALO_BACKEND=cpu` hit this arm.
///
/// Separate from [`Router::check_prefill_routing`] so tests don't need to
/// stand up a Router (which requires HIP + a real .h1b). Both paths share
/// this implementation, so asserting against this function also validates
/// the live `generate` path byte-for-byte.
///
/// **NPU note:** the former `Backend::Xdna` arm (FastFlowLM subprocess
/// bridge, ternary-unsupported gate) was retired 2026-04-21. The NPU
/// lane is now ONNX Runtime + VitisAI EP, which plugs in at the ORT
/// session layer rather than here. See `project_npu_path_onnx.md`.
pub fn prefill_routing_decision(
    backend: Backend,
    prompt_tokens: usize,
) -> Result<(), BackendError> {
    let _ = prompt_tokens;
    match backend {
        Backend::Hip => Ok(()),
        Backend::Cpu => Err(BackendError::CpuLaneStub(
            "CPU sampler lane scaffolded, not yet on critical path; see docs/wiki/CPU-Lane-Plan.md",
        )),
    }
}

/// Per-request lane decision for the prefill phase.
///
/// Distinct from [`prefill_routing_decision`], which answers "is the
/// operator-selected `Backend` viable at all?". This function answers
/// "given a `Backend::Hip` session and a prompt of `prompt_tokens`
/// length, should the prefill run on the CPU AVX2 lane or the iGPU?".
///
/// Rules (see `docs/wiki/APU-Aggregator-Plan.md` §2):
///
/// * `prompt_tokens >= crossover_len` → [`PrefillLane::Cpu`]. Prefill
///   runs through the `1bit-cpu` ternary GEMV kernel; hidden state is
///   handed off to the iGPU for decode via `hipMemcpy`.
/// * `prompt_tokens <  crossover_len` → [`PrefillLane::IGpu`]. Prefill
///   stays on the iGPU M=1 GEMV path (current behaviour).
///
/// `crossover_len == 0` means "always use CPU" (ops override for
/// stress-testing); `crossover_len == usize::MAX` means "never use CPU"
/// (rollback override, same as setting `HALO_PREFILL_CROSSOVER_L=99999`).
///
/// The function is deliberately total — no errors, no allocation — so
/// the forward-pass hot loop can call it without an extra `?`. Callers
/// compose with [`prefill_routing_decision`] for the enclosing backend
/// check.
pub fn prefill_lane_for_prompt(prompt_tokens: usize, crossover_len: usize) -> PrefillLane {
    if prompt_tokens >= crossover_len {
        PrefillLane::Cpu
    } else {
        PrefillLane::IGpu
    }
}

/// Which surface runs the prefill for this request. Paired with
/// [`prefill_lane_for_prompt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillLane {
    /// iGPU handles prefill. Current default for short prompts + the
    /// rollback path when `HALO_PREFILL_CROSSOVER_L` is set high.
    IGpu,
    /// CPU AVX2 ternary GEMV lane handles prefill. Hidden state at
    /// position `L-1` is handed to the iGPU for decode.
    Cpu,
}

impl PrefillLane {
    /// Short log label. Matches the `prefill=cpu` / `prefill=igpu`
    /// string used in the plan doc + what the integration test greps
    /// for.
    pub fn label(self) -> &'static str {
        match self {
            PrefillLane::IGpu => "igpu",
            PrefillLane::Cpu => "cpu",
        }
    }
}

#[cfg(test)]
mod ppl_math_tests {
    //! Unit tests for the PPL math helpers. These are GPU-free — they
    //! exercise the log-softmax reduction that accumulates `mean_nll`
    //! without going through `forward_token`.

    use super::neg_log_softmax_at;

    /// With a uniform logit vector of length V, `-log_softmax` at any
    /// target equals `log(V)`. Perplexity of a uniform predictor over V
    /// classes is V — that's the textbook sanity check.
    #[test]
    fn uniform_logits_give_log_v() {
        for v in [2usize, 10, 128, 32_000] {
            let logits = vec![0.7f32; v];
            let nll = neg_log_softmax_at(&logits, (v / 2) as i32);
            let expected = (v as f64).ln();
            assert!(
                (nll - expected).abs() < 1e-9,
                "uniform V={v}: got {nll}, expected {expected}"
            );
        }
    }

    /// With an argmax-strong logit at the target, NLL is near zero and
    /// therefore perplexity near 1.
    #[test]
    fn confident_target_is_near_zero_nll() {
        let mut logits = vec![-50.0f32; 1000];
        logits[42] = 100.0;
        let nll = neg_log_softmax_at(&logits, 42);
        assert!(nll < 1e-10, "expected near-zero NLL, got {nll}");
        assert!((nll.exp() - 1.0).abs() < 1e-9);
    }

    /// Two-class softmax with logit difference d: NLL of the wrong class
    /// equals `log(1 + exp(d))` on the low side and `log(1 + exp(-d))` on
    /// the high side. Spot-check one concrete value.
    #[test]
    fn two_class_closed_form() {
        let logits = [0.0f32, 2.0];
        // softmax[0] = 1 / (1 + e^2); nll(target=0) = log(1 + e^2)
        let nll0 = neg_log_softmax_at(&logits, 0);
        let expected0 = (1.0f64 + 2.0f64.exp()).ln();
        assert!((nll0 - expected0).abs() < 1e-9, "{nll0} vs {expected0}");
        // Consistency: exp(-nll0) + exp(-nll1) must equal 1.
        let nll1 = neg_log_softmax_at(&logits, 1);
        let p_sum = (-nll0).exp() + (-nll1).exp();
        assert!((p_sum - 1.0).abs() < 1e-9, "probs sum to {p_sum}, not 1");
    }

    /// Invariance to a constant logit shift — numerical stability check.
    #[test]
    fn log_softmax_shift_invariant() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|&x| x + 10_000.0).collect();
        for t in 0..a.len() {
            let na = neg_log_softmax_at(&a, t as i32);
            let nb = neg_log_softmax_at(&b, t as i32);
            assert!(
                (na - nb).abs() < 1e-6,
                "shift not invariant at t={t}: {na} vs {nb}"
            );
        }
    }
}

#[cfg(test)]
mod backend_config_tests {
    //! Tests for the `Backend` enum + `RouterConfig` env-var plumbing +
    //! the prefill-routing guard.
    //!
    //! These are GPU-free: they exercise parsing and policy, never the
    //! forward pass. The routing guard tests go through
    //! [`prefill_routing_decision`], which is the same function
    //! [`generate_blocking`] calls — so a pass here means the live
    //! `Router::generate` path would surface the identical error.
    //!
    //! Env-var tests serialize through a Mutex because `std::env::set_var`
    //! mutates process-global state and other tests in the same binary
    //! can race with it.
    use super::*;
    use std::sync::Mutex;

    /// Serialize the env-var tests — `std::env` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Rule: the default dispatch surface is HIP. Gen-1 + gen-2 ship this
    /// way; flipping the default would silently send CI boxes without
    /// `HALO_BACKEND` set off the happy path.
    #[test]
    fn default_backend_is_hip() {
        assert_eq!(Backend::default(), Backend::Hip);
        assert_eq!(RouterConfig::default().backend, Backend::Hip);
    }

    /// `HALO_BACKEND=cpu` must parse to `Backend::Cpu` and land on the
    /// RouterConfig. Case-insensitive + whitespace-tolerant.
    #[test]
    fn halo_backend_env_parses() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024 moved env mutation behind unsafe. We're
        // single-threaded inside the lock.
        unsafe {
            std::env::set_var("HALO_BACKEND", "cpu");
        }
        let cfg = RouterConfig::from_env().expect("parse");
        assert_eq!(cfg.backend, Backend::Cpu);

        // Whitespace / case tolerance — ops tooling isn't always tidy.
        unsafe {
            std::env::set_var("HALO_BACKEND", "  CPU\n");
        }
        let cfg2 = RouterConfig::from_env().expect("parse");
        assert_eq!(cfg2.backend, Backend::Cpu);

        // hip round-trip too.
        unsafe {
            std::env::set_var("HALO_BACKEND", "hip");
        }
        let cfg3 = RouterConfig::from_env().expect("parse");
        assert_eq!(cfg3.backend, Backend::Hip);

        unsafe {
            std::env::remove_var("HALO_BACKEND");
        }
    }

    /// A garbage `HALO_BACKEND` value must error cleanly — no panic,
    /// no silent fallback. Message must name the accepted spellings so
    /// the operator knows the valid set.
    #[test]
    fn halo_backend_env_rejects_garbage() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("HALO_BACKEND", "nonsense");
        }
        let err = RouterConfig::from_env().expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("hip") && msg.contains("cpu"),
            "error should list accepted spellings; got: {msg}"
        );
        unsafe {
            std::env::remove_var("HALO_BACKEND");
        }
    }

    /// Routing policy: Hip is always OK, Cpu always returns the scaffold
    /// stub. (NPU path retired 2026-04-21 — see `project_npu_path_onnx.md`.)
    #[test]
    fn prefill_routing_hip_and_cpu() {
        // Hip is always fine regardless of prompt length.
        assert!(prefill_routing_decision(Backend::Hip, 1).is_ok());
        assert!(prefill_routing_decision(Backend::Hip, 4096).is_ok());

        // Cpu always refuses with the scaffold stub.
        for tokens in [0usize, 1, 32, 256, 4096] {
            let cpu_err =
                prefill_routing_decision(Backend::Cpu, tokens).expect_err("cpu always stubs");
            match cpu_err {
                BackendError::CpuLaneStub(msg) => {
                    assert_eq!(
                        msg,
                        "CPU sampler lane scaffolded, not yet on critical path; see docs/wiki/CPU-Lane-Plan.md",
                        "cpu stub message drifted — update memory + docs together"
                    );
                }
                other => panic!("expected CpuLaneStub, got {other:?}"),
            }
        }
    }

    /// The per-request prefill-lane decision:
    ///   * L < crossover  → iGpu (current hot path)
    ///   * L ≥ crossover  → Cpu  (offload)
    ///
    /// Mocks no backends — the function is pure so a direct assertion
    /// against its output validates what `generate_blocking` would see.
    #[test]
    fn prefill_lane_switches_at_crossover() {
        // Default crossover per the APU-Aggregator-Plan.md is 33.
        assert_eq!(DEFAULT_PREFILL_CROSSOVER_L, 33);
        for len in [0usize, 1, 16, 32] {
            assert_eq!(
                prefill_lane_for_prompt(len, DEFAULT_PREFILL_CROSSOVER_L),
                PrefillLane::IGpu,
                "len={len} must stay on iGPU"
            );
        }
        for len in [33usize, 64, 256, 2048] {
            assert_eq!(
                prefill_lane_for_prompt(len, DEFAULT_PREFILL_CROSSOVER_L),
                PrefillLane::Cpu,
                "len={len} must offload to CPU"
            );
        }
        // Operator override: crossover=16 (integration-test value from the
        // APU-Aggregator-Plan.md playbook).
        assert_eq!(prefill_lane_for_prompt(15, 16), PrefillLane::IGpu);
        assert_eq!(prefill_lane_for_prompt(16, 16), PrefillLane::Cpu);
        // Rollback: crossover=usize::MAX pins every request to iGPU.
        assert_eq!(
            prefill_lane_for_prompt(10_000, usize::MAX),
            PrefillLane::IGpu
        );
        // Stress: crossover=0 forces every request to CPU.
        assert_eq!(prefill_lane_for_prompt(0, 0), PrefillLane::Cpu);
    }

    /// `HALO_PREFILL_CROSSOVER_L=16` must flow from the environment into
    /// `RouterConfig::prefill_crossover_len`. Unset → default 33.
    /// Garbage → a parse error that names the env var.
    #[test]
    fn halo_prefill_crossover_env_parses() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: env mutation behind the ENV_LOCK.
        unsafe {
            std::env::remove_var(PREFILL_CROSSOVER_ENV);
        }
        let cfg = RouterConfig::from_env().expect("unset → default");
        assert_eq!(cfg.prefill_crossover_len, DEFAULT_PREFILL_CROSSOVER_L);

        unsafe {
            std::env::set_var(PREFILL_CROSSOVER_ENV, "16");
        }
        let cfg = RouterConfig::from_env().expect("parse 16");
        assert_eq!(cfg.prefill_crossover_len, 16);

        unsafe {
            std::env::set_var(PREFILL_CROSSOVER_ENV, "  99999\n");
        }
        let cfg = RouterConfig::from_env().expect("parse with whitespace");
        assert_eq!(cfg.prefill_crossover_len, 99999);

        unsafe {
            std::env::set_var(PREFILL_CROSSOVER_ENV, "not-a-number");
        }
        let err = RouterConfig::from_env().expect_err("garbage must error");
        let msg = format!("{err}");
        assert!(
            msg.contains(PREFILL_CROSSOVER_ENV),
            "error should name the env var; got: {msg}"
        );

        unsafe {
            std::env::remove_var(PREFILL_CROSSOVER_ENV);
        }
    }

    /// Label round-trip: every variant must render a stable, unique string
    /// for operator logs + `/v1/models`.
    #[test]
    fn backend_labels_are_unique() {
        let hip = Backend::Hip.label();
        let cpu = Backend::Cpu.label();
        assert_ne!(hip, cpu);
        // And parse-env round-trips through the canonical lowercase label.
        assert_eq!(Backend::parse_env(hip).unwrap(), Backend::Hip);
        assert_eq!(Backend::parse_env(cpu).unwrap(), Backend::Cpu);
    }
}
