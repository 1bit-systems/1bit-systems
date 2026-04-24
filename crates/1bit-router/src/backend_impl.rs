// Future — C++ mirror: see `rocm-cpp/router-cpp/src/backend_hip.cpp`. The
// Phase 2 port kills the sampler-pipe IPC + async overhead (50-80% of
// kernel tok/s) by running the forward pass synchronously from C++ with
// no tokio / flume hop between logits and sampler. This Rust source stays
// as the reference implementation for one release cycle.
//! HIP backend — forward pass + weight upload.
//!
//! This is the Rust port of the `forward_token` lambda at
//! `rocm-cpp/tools/bitnet_decode.cpp:408-487`. The math is identical and
//! uses exactly the same rocm-cpp kernel names; only the host-side glue
//! (allocation, lifecycle, streaming) is rewritten in safe Rust on top of
//! `1bit-hip`'s safe wrappers.
//!
//! Design:
//!   * [`HipBackend`] owns every device allocation for the lifetime of the
//!     process. Weights are uploaded once at `new()` and never re-touched.
//!   * Scratch / activation buffers are preallocated once, reused per token.
//!   * The KV cache is sized for `max_context` (default 4096) and lives for
//!     the life of the backend — sessions reset `cache_pos` to 0 rather
//!     than re-allocating.
//!
//! Everything here is synchronous and single-threaded: the HTTP layer above
//! us runs this on a tokio blocking thread via `spawn_blocking`. Making
//! forward-token itself async would just hand control back to tokio between
//! GPU kernel launches, which on gfx1151 take a handful of microseconds —
//! scheduling overhead would dominate.

use std::fs::File;
use std::io::Read as _;
use std::path::Path;

use half::f16;
use onebit_core::gguf::{GGUF_MAGIC, GgufFile};
use onebit_core::h1b::{H1B_MAGIC, H1bConfig, H1bFile, H1bLayerOffsets, H1bWeightFormat, Span};
use onebit_core::htok::HtokFile;
use onebit_hip as hip;
use onebit_hip::{DeviceBuffer, DeviceMutPtr, DevicePtr, HipStream, RcppError};

use crate::tokenizer::ByteLevelBpe;

/// Which on-disk model container the router detected at `--model <path>`.
///
/// Resolved by [`sniff_model_format`] from a cheap 4-byte peek at the file
/// head plus the filename extension. The router holds the enum for logs /
/// observability; actual loading is handled per-variant inside
/// [`HipBackend::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Native halo `.h1b` — packed ternary weights already in HaloV2 layout.
    H1b,
    /// Public GGUF v3 BitNet export (e.g. `microsoft/bitnet-b1.58-2B-4t-gguf`).
    Gguf,
}

/// Peek at `path` to decide which loader we should dispatch to.
///
/// The extension is the hint and the first 4 bytes are the authority:
///
/// * `.h1b` + magic starting `H1B`  → [`ModelFormat::H1b`].
/// * `.gguf` + magic `GGUF`        → [`ModelFormat::Gguf`].
/// * mismatch (wrong magic for the claimed extension, or an unknown
///   extension altogether) → [`BackendError::BadInput`] naming both
///   accepted formats so the operator knows what the router understands.
///
/// Note: halo's `.h1b` parser checks only the first three magic bytes,
/// so `H1B\0` / `H1Bx` are both accepted here — same relaxed rule.
pub fn sniff_model_format(path: &Path) -> Result<ModelFormat, BackendError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Read the first 4 bytes — enough for both magics.
    let mut head = [0u8; 4];
    {
        let mut f = File::open(path)
            .map_err(|e| BackendError::Other(format!("open {}: {}", path.display(), e)))?;
        f.read_exact(&mut head).map_err(|e| {
            BackendError::Other(format!(
                "read magic from {} (file too small?): {}",
                path.display(),
                e
            ))
        })?;
    }

    let is_h1b_magic = head[..3] == H1B_MAGIC[..3];
    let is_gguf_magic = head == GGUF_MAGIC;

    match ext.as_str() {
        "h1b" => {
            if is_h1b_magic {
                Ok(ModelFormat::H1b)
            } else {
                Err(BackendError::Other(format!(
                    "{} has .h1b extension but magic is {:?}; expected H1B\\0 or GGUF",
                    path.display(),
                    head
                )))
            }
        }
        "gguf" => {
            if is_gguf_magic {
                Ok(ModelFormat::Gguf)
            } else {
                Err(BackendError::Other(format!(
                    "{} has .gguf extension but magic is {:?}; expected GGUF or H1B\\0",
                    path.display(),
                    head
                )))
            }
        }
        // Unknown extension: trust the magic if it's one we recognise,
        // otherwise bail with the full list of accepted formats.
        _ => {
            if is_h1b_magic {
                Ok(ModelFormat::H1b)
            } else if is_gguf_magic {
                Ok(ModelFormat::Gguf)
            } else {
                Err(BackendError::Other(format!(
                    "unrecognised model file {}: extension {:?}, magic {:?}; \
                     expected a .h1b or .gguf file",
                    path.display(),
                    ext,
                    head
                )))
            }
        }
    }
}

/// Errors surfaced from the router backend. Kept small so the server layer
/// can format them into OpenAI-shaped JSON without lossy stringification.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Model loader / format error (bad .h1b, wrong version, ...).
    #[error(transparent)]
    Halo(#[from] onebit_core::HaloError),
    /// HIP runtime or rocm-cpp kernel error.
    #[error(transparent)]
    Hip(#[from] RcppError),
    /// The request asked for more tokens than the KV cache can hold.
    #[error("context overflow: {used} + {new_tokens} > {limit}")]
    Context {
        /// Tokens already resident in the KV cache.
        used: usize,
        /// Tokens the request wants to add on top.
        new_tokens: usize,
        /// Hard cap set at router construction (default 4096).
        limit: usize,
    },
    /// This .h1b uses a packing version 1bit-router's forward pass doesn't
    /// know how to dispatch yet (currently only HaloV2 is wired).
    #[error("unsupported weight format: {0:?}")]
    UnsupportedFormat(H1bWeightFormat),
    /// Caught bad inputs before kicking off GEMVs.
    #[error("bad input: {0}")]
    BadInput(&'static str),
    /// A peer backend has been selected but its real dispatch path is not
    /// compiled / loaded in this build.
    ///
    /// Historical note: this variant surfaced the XDNA 2 + FastFlowLM
    /// bridge when that path was stubbed. That bridge was retired
    /// 2026-04-21 (the NPU lane is now ONNX Runtime + VitisAI EP, which
    /// plugs in outside the router). The variant is retained because
    /// future backend additions may reach for the same "built, but not
    /// loaded" shape.
    #[error("{0}")]
    NotYetWired(&'static str),
    /// The CPU lane is scaffolded (see [`crate::cpu_lane`]) but not yet
    /// on the critical path. Dispatching through `HALO_BACKEND=cpu`
    /// today hits this arm rather than panicking; the lane's real
    /// purpose is to run the sampler + tokenizer in parallel while the
    /// iGPU grinds the next token, and that wire-up is tracked in
    /// `docs/wiki/CPU-Lane-Plan.md`.
    ///
    /// Distinct from [`BackendError::NotYetWired`] because the CPU lane
    /// *does* have working code — it's just not on the SSE hot path
    /// yet. Keeping the two error kinds separate lets the HTTP layer
    /// log + count them differently without regex on the message.
    #[error("{0}")]
    CpuLaneStub(&'static str),
    /// Anything else.
    #[error("router: {0}")]
    Other(String),
}

/// Per-layer device-resident weights. Every field is a sized device buffer
/// we uploaded once at `new()` and never mutate.
struct LayerWeights {
    input_norm: DeviceBuffer<u16>,     // fp16 [hs]
    post_attn_norm: DeviceBuffer<u16>, // fp16 [hs]
    attn_sub_norm: DeviceBuffer<u16>,  // fp16 [hs]
    ffn_sub_norm: DeviceBuffer<u16>,   // fp16 [is]

    q_packed: DeviceBuffer<u8>,
    q_scales: DeviceBuffer<f32>,
    k_packed: DeviceBuffer<u8>,
    k_scales: DeviceBuffer<f32>,
    v_packed: DeviceBuffer<u8>,
    v_scales: DeviceBuffer<f32>,
    o_packed: DeviceBuffer<u8>,
    o_scales: DeviceBuffer<f32>,
    gate_packed: DeviceBuffer<u8>,
    gate_scales: DeviceBuffer<f32>,
    up_packed: DeviceBuffer<u8>,
    up_scales: DeviceBuffer<f32>,
    down_packed: DeviceBuffer<u8>,
    down_scales: DeviceBuffer<f32>,
}

/// Model-level device weights.
struct ModelWeights {
    embedding: DeviceBuffer<u16>,  // fp16 [vocab * hs]
    final_norm: DeviceBuffer<u16>, // fp16 [hs]
    layers: Vec<LayerWeights>,
}

/// Scratch buffers reused per-token. Allocated once at `new()`.
struct Scratch {
    // Residual + attention activations
    x_fp32: DeviceBuffer<f32>,      // [hs]
    x_fp16: DeviceBuffer<u16>,      // [hs] — embedding read target
    normed: DeviceBuffer<u16>,      // [hs]
    x_i8: DeviceBuffer<i8>,         // [hs_k]
    x_scale_dev: DeviceBuffer<f32>, // [1]

    q_fp16: DeviceBuffer<u16>, // [nh * hd]
    k_fp16: DeviceBuffer<u16>, // [nkv * hd]
    v_fp16: DeviceBuffer<u16>, // [nkv * hd]
    o_fp16: DeviceBuffer<u16>, // [hs]

    gate_fp16: DeviceBuffer<u16>,      // [is]
    up_fp16: DeviceBuffer<u16>,        // [is]
    down_fp16: DeviceBuffer<u16>,      // [hs]
    silu_out: DeviceBuffer<u16>,       // [is]
    silu_i8: DeviceBuffer<i8>,         // [is_k]
    silu_scale_dev: DeviceBuffer<f32>, // [1]

    logits: DeviceBuffer<f32>,       // [vocab]
    next_tok_dev: DeviceBuffer<i32>, // [1]
}

/// KV-cache dtype selector.
///
/// `I8` is the default (2× bandwidth savings for the KV reads at long
/// context — weights still dominate at short L, but at L=2048 on 2B-4T
/// they're the same order of magnitude: ~400 MB weights vs ~458 MB KV
/// at fp16 → 229 MB with i8). `F16` is kept as a debug fallback selectable
/// through `HALO_KV_DTYPE=f16` for PPL A/B against the i8 path.
///
/// Geramy flagged the wire-up gap on 2026-04-23 ("K/V is the biggest RAM
/// BW spender"); the kernels + Rust wrappers already exist in `1bit-hip`
/// (`kv_cache_attn_decode_i8`, `quantize_fp16_to_i8_rowscale`), only the
/// router's decode hot path was still calling the fp16 variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvDtype {
    /// FP16 K/V cache — the pre-2026-04-23 default. Kept for A/B PPL
    /// debugging; selectable via `HALO_KV_DTYPE=f16`.
    F16,
    /// INT8 K/V cache with per-`(pos, kv_head)` FP16 scales; dequant
    /// fused inside `rcpp_kv_cache_attn_decode_i8`. Default.
    #[default]
    I8,
}

impl KvDtype {
    /// Env var the router checks at backend construction. Value is
    /// case-insensitive; unset → [`KvDtype::default`] (I8).
    pub const ENV: &'static str = "HALO_KV_DTYPE";

    /// Parse `HALO_KV_DTYPE=...`. Accepts `i8`/`int8`, `f16`/`fp16`/`half`.
    /// Unknown values surface as [`BackendError::Other`] naming the valid
    /// set so operators don't have to grep the source.
    pub fn parse_env(raw: &str) -> Result<Self, BackendError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "i8" | "int8" => Ok(KvDtype::I8),
            "f16" | "fp16" | "half" => Ok(KvDtype::F16),
            other => Err(BackendError::Other(format!(
                "HALO_KV_DTYPE: unknown value {other:?}; accepted: i8 | f16"
            ))),
        }
    }

    /// Resolve from the process env. Absent or empty → default (I8).
    pub fn from_env() -> Result<Self, BackendError> {
        match std::env::var(Self::ENV) {
            Ok(raw) if !raw.trim().is_empty() => Self::parse_env(&raw),
            _ => Ok(Self::default()),
        }
    }

    /// Human-readable label for logs / metrics.
    pub fn label(self) -> &'static str {
        match self {
            KvDtype::F16 => "f16",
            KvDtype::I8 => "i8",
        }
    }
}

/// Per-layer KV cache storage. Active variant is picked once at `new()`
/// from [`KvDtype::from_env`]; the forward pass reads the discriminant
/// on `self.kv_dtype` (one load, branch-predicted) and dispatches.
///
/// INT8 layout matches what `rcpp_kv_cache_attn_decode_i8` expects:
/// - `k_i8` / `v_i8`: `[max_seq_len, num_kv_heads, head_dim]` i8
/// - `k_scales_f16` / `v_scales_f16`: `[max_seq_len, num_kv_heads]` fp16
///
/// Scales are FP16 because the kernel loads them as `__half` and the
/// `quantize_fp16_to_i8_rowscale` launch writes FP16 scales directly —
/// no host-side cast required.
enum KvCache {
    /// FP16 K/V (pre-2026-04-23 behaviour). `max_context * nkv * hd` each.
    F16 {
        k: DeviceBuffer<u16>,
        v: DeviceBuffer<u16>,
    },
    /// INT8 K/V + per-`(pos, head)` FP16 scales.
    I8 {
        k_i8: DeviceBuffer<i8>,
        v_i8: DeviceBuffer<i8>,
        k_scales: DeviceBuffer<u16>, // fp16 bits
        v_scales: DeviceBuffer<u16>, // fp16 bits
    },
}

/// The actual HIP backend. Holds weights, scratch, KV cache, tokenizer,
/// and a running decode position.
pub struct HipBackend {
    cfg: H1bConfig,
    weights: ModelWeights,
    scratch: Scratch,
    kv: Vec<KvCache>,
    tokenizer: ByteLevelBpe,
    max_context: usize,
    model_id: String,
    hip_label: String,
    /// BitNet v2 online-Hadamard dispatch state. `true` iff the crate was
    /// built with the `bitnet-v2` feature **and** the loaded `.h1b` has
    /// `H1B_FLAG_HADAMARD_ROTATED` set on its `reserved` cfg slot. Only
    /// when both are true does [`HipBackend::forward_token`] dispatch
    /// `hadamard_rotate_fp16_device` at each pre-GEMV quant site.
    ///
    /// Plumbed through `new()` once so the hot path is a single
    /// branch-prediction-friendly field load rather than re-reading the
    /// cfg every layer. See
    /// `/home/bcloud/repos/1bit-halo-workspace/docs/wiki/BitNet-v2-Hadamard-Plan.md`
    /// for the feature-gate matrix.
    hadamard_active: bool,
    /// Debug counter: how many Hadamard rotations the forward pass has
    /// dispatched since `new()`. Only incremented in test builds; the
    /// release hot path doesn't pay for the atomic.
    #[cfg(any(test, feature = "bitnet-v2"))]
    hadamard_dispatch_count: std::sync::atomic::AtomicU64,
    /// KV cache dtype — resolved once at `new()` from [`KvDtype::from_env`]
    /// and never changed for the life of the backend. Lives here (rather
    /// than on the variant) so the hot path can dispatch with a single
    /// enum discriminant load instead of walking the KvCache match on
    /// every layer.
    kv_dtype: KvDtype,
}

impl HipBackend {
    /// Load a model file + `.htok` from disk and upload every weight to the
    /// current HIP device. Allocates all scratch + KV cache up front.
    ///
    /// `model_path` may be either `.h1b` (native halo packing) or `.gguf`
    /// (public BitNet GGUF). The format is sniffed via [`sniff_model_format`]
    /// and dispatched to the matching private loader. `htok_path` is only
    /// consumed on the `.h1b` path — on the GGUF path the tokenizer comes
    /// out of the file's own metadata KVs and `htok_path` is ignored.
    ///
    /// `model_id` is the name advertised through `/v1/models`.
    /// `max_context` bounds the KV cache.
    pub fn new(
        model_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        match sniff_model_format(model_path)? {
            ModelFormat::H1b => {
                tracing::info!(path = %model_path.display(), "dispatch: .h1b loader");
                Self::load_h1b_into_hip(model_path, htok_path, model_id, max_context)
            }
            ModelFormat::Gguf => {
                tracing::info!(path = %model_path.display(), "dispatch: .gguf loader");
                load_gguf_into_hip(model_path, model_id, max_context)
            }
        }
    }

    /// Loader for native `.h1b` files — the path this crate has shipped
    /// since gen-2 day one. Split out of [`Self::new`] in 2026-04 so the
    /// format dispatch can sit above it.
    fn load_h1b_into_hip(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        let h1b = H1bFile::open(h1b_path)?;
        let tok = HtokFile::open(htok_path)?;
        let tokenizer = ByteLevelBpe::from_htok(tok);

        let cfg = *h1b.config();
        // Forward pass here implements HaloV2 packing only. Sherry / TQ1
        // dispatch is a follow-up — the kernel wrappers already exist in
        // 1bit-hip so adding them is a one-match switch.
        if cfg.weight_format()? != H1bWeightFormat::HaloV2 {
            return Err(BackendError::UnsupportedFormat(cfg.weight_format()?));
        }

        tracing::info!(
            hidden_size = cfg.hidden_size,
            intermediate_size = cfg.intermediate_size,
            num_layers = cfg.num_layers,
            num_heads = cfg.num_heads,
            num_kv_heads = cfg.num_kv_heads,
            vocab_size = cfg.vocab_size,
            version = cfg.version,
            "uploading .h1b weights to GPU"
        );

        // Pick the default device. Most hosts only have one; on multi-GPU
        // rigs we'd surface a selector, but Strix Halo is single-iGPU.
        hip::set_device(0)?;
        let hip_label = format!(
            "hip device 0 of {} available",
            hip::device_count().unwrap_or(0)
        );

        let hs = cfg.hidden_size as usize;
        let is_ = cfg.intermediate_size as usize;
        let nh = cfg.num_heads as usize;
        let nkv = cfg.num_kv_heads as usize;
        let hd = cfg.head_dim()? as usize;
        let vocab = cfg.vocab_size as usize;
        let n_layers = cfg.num_layers as usize;
        let hs_k = hs; // HaloV2 kernels take K = hs directly.
        let is_k = is_;

        // ---------- Weights ----------
        let model_off = *h1b.model_offsets();
        let embedding = upload_fp32_as_fp16(&h1b, model_off.embedding, vocab * hs)?;
        let final_norm = upload_fp32_as_fp16(&h1b, model_off.final_norm, hs)?;

        let mut layers = Vec::with_capacity(n_layers);
        for (idx, lo) in h1b.layer_offsets().iter().enumerate() {
            layers.push(upload_layer(&h1b, lo, hs, is_, idx)?);
        }

        let weights = ModelWeights {
            embedding,
            final_norm,
            layers,
        };

        // ---------- Scratch ----------
        let scratch = Scratch {
            x_fp32: DeviceBuffer::alloc(hs)?,
            x_fp16: DeviceBuffer::alloc(hs)?,
            normed: DeviceBuffer::alloc(hs)?,
            x_i8: DeviceBuffer::alloc_zeroed(hs_k)?,
            x_scale_dev: DeviceBuffer::alloc(1)?,

            q_fp16: DeviceBuffer::alloc(nh * hd)?,
            k_fp16: DeviceBuffer::alloc(nkv * hd)?,
            v_fp16: DeviceBuffer::alloc(nkv * hd)?,
            o_fp16: DeviceBuffer::alloc(hs)?,

            gate_fp16: DeviceBuffer::alloc(is_)?,
            up_fp16: DeviceBuffer::alloc(is_)?,
            down_fp16: DeviceBuffer::alloc(hs)?,
            silu_out: DeviceBuffer::alloc(is_)?,
            silu_i8: DeviceBuffer::alloc_zeroed(is_k)?,
            silu_scale_dev: DeviceBuffer::alloc(1)?,

            logits: DeviceBuffer::alloc(vocab)?,
            next_tok_dev: DeviceBuffer::alloc(1)?,
        };

        // ---------- KV cache ----------
        // Resolve the dtype once; log it so serve-path ops can confirm
        // which cache is live on a given boot without grep-on-pid-env.
        let kv_dtype = KvDtype::from_env()?;
        tracing::info!(
            dtype = kv_dtype.label(),
            max_context,
            nkv,
            hd,
            "allocating KV cache"
        );

        let mut kv = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            kv.push(match kv_dtype {
                KvDtype::F16 => KvCache::F16 {
                    k: DeviceBuffer::alloc(max_context * nkv * hd)?,
                    v: DeviceBuffer::alloc(max_context * nkv * hd)?,
                },
                KvDtype::I8 => KvCache::I8 {
                    k_i8: DeviceBuffer::alloc(max_context * nkv * hd)?,
                    v_i8: DeviceBuffer::alloc(max_context * nkv * hd)?,
                    k_scales: DeviceBuffer::alloc(max_context * nkv)?,
                    v_scales: DeviceBuffer::alloc(max_context * nkv)?,
                },
            });
        }

        hip::device_synchronize()?;

        tracing::info!(
            layers = n_layers,
            max_context,
            "1bit-router HIP backend ready"
        );

        // --- BitNet v2 H-BitLinear feature gate resolution ---------------------
        // The live dispatch matrix (see docs/wiki/BitNet-v2-Hadamard-Plan.md §6):
        //   feature OFF              → never rotate (status quo).
        //   feature ON  + flag ON    → rotate at every pre-GEMV quant site.
        //   feature ON  + flag OFF   → pass-through, warn-once. Prevents a
        //       dev-build accidentally double-quantizing a stock model whose
        //       weights were not trained under Hadamard (correctness bug:
        //       rotating activations on W-unrotated weights produces garbage
        //       logits).
        let hadamard_active = Self::resolve_hadamard_gate(&cfg);

        Ok(Self {
            cfg,
            weights,
            scratch,
            kv,
            tokenizer,
            max_context,
            model_id,
            hip_label,
            hadamard_active,
            #[cfg(any(test, feature = "bitnet-v2"))]
            hadamard_dispatch_count: std::sync::atomic::AtomicU64::new(0),
            kv_dtype,
        })
    }

    /// Centralise the BitNet-v2 gate decision so tests can exercise the
    /// matrix without standing up a full `HipBackend`.
    ///
    /// The gate is the logical AND of:
    ///   * **`HALO_BITNET_V2=1` in the process environment** — runtime
    ///     switch, not a compile feature. We need V1 to stay available for
    ///     shadow-burnin parity, so flipping V2 must never require a
    ///     rebuild. Any value other than `"1"` (including unset, `"0"`, or
    ///     anything else) keeps the gate off.
    ///   * **Model flag `H1B_FLAG_HADAMARD_ROTATED`** — set by the offline
    ///     requantizer on checkpoints trained under W' = W · H^T. Enabling
    ///     V2 against an unrotated checkpoint would produce garbage logits,
    ///     so we refuse at load time even if the env says yes.
    ///
    /// The historical `bitnet-v2` cargo feature is retained as a compile-
    /// time override (treated as if `HALO_BITNET_V2=1` were set) so the
    /// feature-gated integration tests still function. The env var alone
    /// is sufficient in production.
    fn resolve_hadamard_gate(cfg: &H1bConfig) -> bool {
        let env_on = std::env::var("HALO_BITNET_V2")
            .map(|v| v == "1")
            .unwrap_or(false);
        let feature_on = cfg!(feature = "bitnet-v2");
        let gate_requested = env_on || feature_on;

        if !gate_requested {
            // Silence the field-read warning under `cfg(not(feature = ...))`.
            let _ = cfg.is_hadamard_rotated();
            return false;
        }

        if cfg.is_hadamard_rotated() {
            tracing::info!(
                env = env_on,
                feature = feature_on,
                "bitnet-v2: online Hadamard rotation ENABLED \
                 (gate requested + model H1B_FLAG_HADAMARD_ROTATED set)"
            );
            true
        } else {
            tracing::warn!(
                env = env_on,
                feature = feature_on,
                "bitnet-v2 requested (env or feature), but loaded model does \
                 NOT have H1B_FLAG_HADAMARD_ROTATED set — passing through \
                 unrotated. Rotate the checkpoint offline (see \
                 docs/wiki/BitNet-v2-Hadamard-Plan.md §4) to enable."
            );
            false
        }
    }

    /// Test-only: how many Hadamard rotations the forward pass has
    /// dispatched since this backend was constructed. Exposed for the
    /// `hook_fires_exactly_once_per_site_per_layer` test below; production
    /// code should not rely on it.
    #[cfg(any(test, feature = "bitnet-v2"))]
    pub fn hadamard_dispatch_count(&self) -> u64 {
        self.hadamard_dispatch_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Human-readable backend label for `/v1/models` and logs.
    pub fn label(&self) -> &str {
        &self.hip_label
    }

    /// Model id advertised on `/v1/models`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Encode a prompt string into BitNet token ids (with BOS prepended).
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        self.tokenizer.encode(text, /* add_bos */ true)
    }

    /// Decode a sequence of ids into raw bytes.
    pub fn detokenize(&self, ids: &[i32]) -> String {
        self.tokenizer.decode(ids)
    }

    /// BitNet v2 H-BitLinear online rotation hook. Called immediately
    /// before each of the four per-layer activation-quant sites; no-op
    /// when `self.hadamard_active` is false.
    ///
    /// Rotates `self.scratch.normed` **in place** (`x_in == y_out` is
    /// explicitly supported by the device kernel) across the first `k`
    /// elements. `k` is always `hs` at Q/K/V/O and `is_` at the FFN-down
    /// site; both are multiples of 128 on BitNet-2B-4T (2560 = 20·128,
    /// 6912 = 54·128) so the kernel's divisibility precondition is
    /// satisfied without a runtime guard.
    ///
    /// The rotation is cheap (7 butterfly stages per block, no LDS on
    /// stages 0..4) but we keep the call site behind the `hadamard_active`
    /// gate so a `bitnet-v2` build on a non-rotated model has zero kernel
    /// overhead, not just "correct output".
    ///
    /// Free-function form (rather than `&mut self`) so call sites inside
    /// the layer loop don't collide with the `ly = &self.weights.layers[l]`
    /// immutable borrow. Takes the two fields it actually needs by
    /// reference — mutable only for the scratch buffer.
    #[inline]
    fn maybe_hadamard_rotate_normed(
        scratch: &mut Scratch,
        hadamard_active: bool,
        #[cfg(any(test, feature = "bitnet-v2"))] dispatch_count: &std::sync::atomic::AtomicU64,
        k: i32,
        stream: HipStream,
    ) -> Result<(), BackendError> {
        if !hadamard_active {
            return Ok(());
        }
        // In-place rotation: src == dst. Device kernel completes each
        // lane's load before issuing the store, so aliasing is safe.
        let ptr = scratch.normed.as_device_mut_ptr();
        let src = DevicePtr(ptr.0 as *const u16);
        ok(hip::hadamard_rotate_fp16_device(src, ptr, k, stream))?;
        #[cfg(any(test, feature = "bitnet-v2"))]
        dispatch_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Variant of [`Self::maybe_hadamard_rotate_normed`] targeting the
    /// `silu_out` scratch (the FFN-down pre-quant site). Same contract.
    #[inline]
    fn maybe_hadamard_rotate_silu(
        scratch: &mut Scratch,
        hadamard_active: bool,
        #[cfg(any(test, feature = "bitnet-v2"))] dispatch_count: &std::sync::atomic::AtomicU64,
        k: i32,
        stream: HipStream,
    ) -> Result<(), BackendError> {
        if !hadamard_active {
            return Ok(());
        }
        let ptr = scratch.silu_out.as_device_mut_ptr();
        let src = DevicePtr(ptr.0 as *const u16);
        ok(hip::hadamard_rotate_fp16_device(src, ptr, k, stream))?;
        #[cfg(any(test, feature = "bitnet-v2"))]
        dispatch_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Run one forward token and return (next_token_id, full logits copy).
    ///
    /// `logits_out` lets the caller reuse a host-side buffer across calls
    /// (sized to `vocab_size`). It's always overwritten on success.
    ///
    /// This is the sampled-decode hot path. Greedy-only callers should
    /// prefer [`Self::forward_token_greedy`] — same forward pass, no
    /// 512 KB D→H copy, no host-argmax reconcile. It takes `&mut self`
    /// because every scratch buffer and the KV cache are borrowed mutably.
    pub fn forward_token(
        &mut self,
        token_id: i32,
        pos: i32,
        logits_out: &mut Vec<f32>,
    ) -> Result<i32, BackendError> {
        let dev_next = self.forward_pass_to_device_argmax(token_id, pos)?;
        let vocab = self.cfg.vocab_size;

        // 0.1.0 fast path: when the caller doesn't need logits (callsite
        // left `logits_out` empty or opted in via HALO_SKIP_LOGITS_COPY=1
        // for shadow-burnin parity toggling), skip the 512 KB D→H copy
        // and the O(vocab) host-argmax reconcile. Kept for backwards
        // compatibility with callers that opportunistically clear
        // `logits_out`; new greedy-only callers should use the dedicated
        // [`Self::forward_token_greedy`] entry point instead — same
        // fast-path behaviour, no env gate, no host Vec<f32> needed.
        let skip_copy = !std::env::var("HALO_SKIP_LOGITS_COPY")
            .map(|v| matches!(v.as_str(), "0" | "false" | "no" | ""))
            .unwrap_or(false)
            && logits_out.is_empty();
        if skip_copy {
            return Ok(dev_next);
        }

        // Copy logits back to host for non-greedy sampling paths.
        if logits_out.len() != vocab as usize {
            logits_out.resize(vocab as usize, 0.0);
        }
        self.scratch
            .logits
            .copy_to_slice(logits_out.as_mut_slice())?;

        // Deterministic host argmax: strict `>` preserves the first
        // (lowest) index on ties. If the device agreed, we keep its
        // answer (zero-cost); if not, the device hit a tie and we
        // override with the lower index.
        let next = host_argmax_lowest_index(logits_out, dev_next);

        Ok(next)
    }

    /// Dedicated greedy-only fast path. Runs the same forward pass as
    /// [`Self::forward_token`] but **skips the logits D→H memcpy
    /// entirely** and returns the on-device argmax index verbatim.
    ///
    /// Why a dedicated entry point (rather than relying on
    /// [`Self::forward_token`] + empty `logits_out` +
    /// `HALO_SKIP_LOGITS_COPY`)?
    ///
    ///   * **No env dance.** Callers that know they want greedy (temp=0
    ///     or top-k=1 with no logprob-return) get the fast path by
    ///     construction, regardless of how `HALO_SKIP_LOGITS_COPY` is
    ///     set.
    ///   * **No host-argmax reconcile.** The reconcile existed only for
    ///     bit-exact tie determinism during the C++↔Rust shadow-burnin
    ///     cutover. Greedy callers that accept device-argmax semantics
    ///     don't pay that O(vocab) scan.
    ///   * **No 512 KB allocation pressure.** The `logits_out` `Vec<f32>`
    ///     doesn't need to exist on the call path at all.
    ///
    /// Savings on halo-1bit-2b (vocab=128256, fp32 logits = 512 KB):
    /// ~20-40 µs skipped memcpy + ~50-80 µs skipped host-scan per
    /// token. At 64-token decode context that's roughly 2-5 tok/s back;
    /// the gain grows as decode tok/s grows because the per-token fixed
    /// overhead becomes a larger fraction of each step. See
    /// `benchmarks/greedy-fast-path.sh` for the harness.
    pub fn forward_token_greedy(
        &mut self,
        token_id: i32,
        pos: i32,
    ) -> Result<i32, BackendError> {
        self.forward_pass_to_device_argmax(token_id, pos)
    }

    /// Shared forward-pass core used by both [`Self::forward_token`]
    /// and [`Self::forward_token_greedy`]. Runs the full 30-layer stack,
    /// dispatches the on-device argmax, synchronises, and returns the
    /// device-argmax token id. Leaves `self.scratch.logits` populated
    /// on device so callers may optionally memcpy it back.
    fn forward_pass_to_device_argmax(
        &mut self,
        token_id: i32,
        pos: i32,
    ) -> Result<i32, BackendError> {
        if (pos as usize) >= self.max_context {
            return Err(BackendError::Context {
                used: pos as usize,
                new_tokens: 1,
                limit: self.max_context,
            });
        }

        let hs = self.cfg.hidden_size;
        let is_ = self.cfg.intermediate_size;
        let nh = self.cfg.num_heads;
        let nkv = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim()?;
        let vocab = self.cfg.vocab_size;
        let rope_theta = self.cfg.rope_theta;
        let rms_eps = self.cfg.rms_norm_eps;
        let scale = 1.0f32 / (hd as f32).sqrt();

        let stream = HipStream::DEFAULT;

        // Seed the FP32 residual stream from the FP16 embedding.
        ok(hip::embedding_lookup_fp16(
            self.weights.embedding.as_device_ptr(),
            token_id,
            self.scratch.x_fp16.as_device_mut_ptr(),
            hs,
            stream,
        ))?;
        // Zero x_fp32, then accumulate the FP16 embedding into it.
        // SAFETY: x_fp32 is a valid device buffer we own; we memset exactly
        // its allocation size.
        unsafe {
            let rc = hip::ffi::hipMemsetAsync(
                self.scratch.x_fp32.as_device_mut_ptr().0 as *mut core::ffi::c_void,
                0,
                self.scratch.x_fp32.byte_len(),
                core::ptr::null_mut(),
            );
            if rc != hip::ffi::HIP_SUCCESS {
                return Err(BackendError::Hip(RcppError::HipError));
            }
        }
        ok(hip::residual_add_fp32_from_fp16(
            self.scratch.x_fp32.as_device_mut_ptr(),
            self.scratch.x_fp16.as_device_ptr(),
            hs,
            stream,
        ))?;

        for l in 0..self.cfg.num_layers as usize {
            // ---------- Attention block ----------
            let ly = &self.weights.layers[l];

            ok(hip::rmsnorm_fp32_in_fp16_out(
                self.scratch.x_fp32.as_device_ptr(),
                ly.input_norm.as_device_ptr(),
                self.scratch.normed.as_device_mut_ptr(),
                rms_eps,
                hs,
                stream,
            ))?;
            // Site #1: Hadamard rotation of the residual stream before
            // Q/K/V pre-quantization. See plan doc §3 — this is the
            // highest-outlier site (RMSNorm output feeds three projections).
            Self::maybe_hadamard_rotate_normed(
                &mut self.scratch,
                self.hadamard_active,
                #[cfg(any(test, feature = "bitnet-v2"))]
                &self.hadamard_dispatch_count,
                hs,
                stream,
            )?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.normed.as_device_ptr(),
                self.scratch.x_i8.as_device_mut_ptr(),
                self.scratch.x_scale_dev.as_device_mut_ptr(),
                hs,
                stream,
            ))?;
            let x_scale = self.scratch.x_scale_dev.copy_to_host_scalar()?;

            // Q/K/V projections (HaloV2 kernel → fp16 direct).
            ok(hip::ternary_gemv_halo_f16(
                ly.q_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.q_scales.as_device_ptr(),
                self.scratch.q_fp16.as_device_mut_ptr(),
                nh * hd,
                hs,
                stream,
            ))?;
            ok(hip::ternary_gemv_halo_f16(
                ly.k_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.k_scales.as_device_ptr(),
                self.scratch.k_fp16.as_device_mut_ptr(),
                nkv * hd,
                hs,
                stream,
            ))?;
            ok(hip::ternary_gemv_halo_f16(
                ly.v_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.v_scales.as_device_ptr(),
                self.scratch.v_fp16.as_device_mut_ptr(),
                nkv * hd,
                hs,
                stream,
            ))?;

            // RoPE on Q and K.
            ok(hip::rope_fp16(
                self.scratch.q_fp16.as_device_mut_ptr(),
                pos,
                rope_theta,
                nh,
                hd,
                stream,
            ))?;
            ok(hip::rope_fp16(
                self.scratch.k_fp16.as_device_mut_ptr(),
                pos,
                rope_theta,
                nkv,
                hd,
                stream,
            ))?;

            // Append K/V to cache at slot `pos`, then run attention.
            //
            // Two dispatch paths:
            //   * F16 — plain D→D memcpy of the just-computed fp16 K/V into
            //     the cache, then split-KV flash-decode with fp16 reads.
            //   * I8  — per-row quantize (one launch for K, one for V;
            //     num_rows = nkv, row_len = hd) directly into the i8 +
            //     scales slots, then flash-decode with int8 reads and
            //     fused per-(pos, head) dequant.
            //
            // Layout matches the kernel's expected strides verbatim:
            //   K_i8 [max_seq_len, nkv, hd]       → slot = pos * nkv * hd
            //   K_scales [max_seq_len, nkv] fp16  → slot = pos * nkv
            // (see rocm-cpp/src/kv_cache_attn_i8.hip lines 10-18).
            let kv_slot = (pos as usize) * (nkv as usize) * (hd as usize);
            let scale_slot = (pos as usize) * (nkv as usize);

            match &mut self.kv[l] {
                KvCache::F16 { k, v } => {
                    let kv_bytes = (nkv as usize) * (hd as usize) * core::mem::size_of::<u16>();
                    // SAFETY: kv_slot <= max_context*nkv*hd by the bounds-check
                    // at the top of forward_token; we write exactly
                    // `nkv*hd*2` bytes which fits in the allocation.
                    unsafe {
                        let rc = hip::ffi::hipMemcpyAsync(
                            k.offset_mut(kv_slot).0 as *mut core::ffi::c_void,
                            self.scratch.k_fp16.as_device_ptr().0 as *const core::ffi::c_void,
                            kv_bytes,
                            hip::ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                            core::ptr::null_mut(),
                        );
                        if rc != hip::ffi::HIP_SUCCESS {
                            return Err(BackendError::Hip(RcppError::HipError));
                        }
                        let rc = hip::ffi::hipMemcpyAsync(
                            v.offset_mut(kv_slot).0 as *mut core::ffi::c_void,
                            self.scratch.v_fp16.as_device_ptr().0 as *const core::ffi::c_void,
                            kv_bytes,
                            hip::ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                            core::ptr::null_mut(),
                        );
                        if rc != hip::ffi::HIP_SUCCESS {
                            return Err(BackendError::Hip(RcppError::HipError));
                        }
                    }

                    ok(hip::kv_cache_attn_decode_fd(
                        self.scratch.q_fp16.as_device_ptr(),
                        k.as_device_ptr(),
                        v.as_device_ptr(),
                        self.scratch.o_fp16.as_device_mut_ptr(),
                        nh,
                        nkv,
                        hd,
                        pos + 1,
                        scale,
                        stream,
                    ))?;
                }
                KvCache::I8 {
                    k_i8,
                    v_i8,
                    k_scales,
                    v_scales,
                } => {
                    // SAFETY: kv_slot + nkv*hd ≤ max_context*nkv*hd and
                    // scale_slot + nkv ≤ max_context*nkv — both bounded by
                    // the `pos < max_context` check at forward_token's top.
                    let (k_i8_slot, v_i8_slot, k_sc_slot, v_sc_slot) = unsafe {
                        (
                            k_i8.offset_mut(kv_slot),
                            v_i8.offset_mut(kv_slot),
                            k_scales.offset_mut(scale_slot),
                            v_scales.offset_mut(scale_slot),
                        )
                    };

                    // Quantize the freshly-projected K/V into the cache at
                    // slot `pos`. num_rows = nkv (one scale per kv-head),
                    // row_len = hd. Kernel writes fp16 scale directly.
                    ok(hip::quantize_fp16_to_i8_rowscale(
                        self.scratch.k_fp16.as_device_ptr(),
                        k_i8_slot,
                        k_sc_slot,
                        nkv,
                        hd,
                        stream,
                    ))?;
                    ok(hip::quantize_fp16_to_i8_rowscale(
                        self.scratch.v_fp16.as_device_ptr(),
                        v_i8_slot,
                        v_sc_slot,
                        nkv,
                        hd,
                        stream,
                    ))?;

                    ok(hip::kv_cache_attn_decode_i8(
                        self.scratch.q_fp16.as_device_ptr(),
                        k_i8.as_device_ptr(),
                        v_i8.as_device_ptr(),
                        k_scales.as_device_ptr(),
                        v_scales.as_device_ptr(),
                        self.scratch.o_fp16.as_device_mut_ptr(),
                        nh,
                        nkv,
                        hd,
                        pos + 1,
                        scale,
                        stream,
                    ))?;
                }
            }

            // BitNet b1.58: attn_sub_norm between attention and O projection.
            ok(hip::rmsnorm_fp16(
                self.scratch.o_fp16.as_device_ptr(),
                ly.attn_sub_norm.as_device_ptr(),
                self.scratch.normed.as_device_mut_ptr(),
                rms_eps,
                hs,
                stream,
            ))?;
            // Site #2: Hadamard rotation before O projection's pre-quant.
            Self::maybe_hadamard_rotate_normed(
                &mut self.scratch,
                self.hadamard_active,
                #[cfg(any(test, feature = "bitnet-v2"))]
                &self.hadamard_dispatch_count,
                hs,
                stream,
            )?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.normed.as_device_ptr(),
                self.scratch.x_i8.as_device_mut_ptr(),
                self.scratch.x_scale_dev.as_device_mut_ptr(),
                hs,
                stream,
            ))?;
            let x_scale = self.scratch.x_scale_dev.copy_to_host_scalar()?;
            ok(hip::ternary_gemv_halo_f16(
                ly.o_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.o_scales.as_device_ptr(),
                self.scratch.o_fp16.as_device_mut_ptr(),
                hs,
                nh * hd,
                stream,
            ))?;
            ok(hip::residual_add_fp32_from_fp16(
                self.scratch.x_fp32.as_device_mut_ptr(),
                self.scratch.o_fp16.as_device_ptr(),
                hs,
                stream,
            ))?;

            // ---------- FFN block ----------
            ok(hip::rmsnorm_fp32_in_fp16_out(
                self.scratch.x_fp32.as_device_ptr(),
                ly.post_attn_norm.as_device_ptr(),
                self.scratch.normed.as_device_mut_ptr(),
                rms_eps,
                hs,
                stream,
            ))?;
            // Site #3: Hadamard rotation before gate+up pre-quant.
            Self::maybe_hadamard_rotate_normed(
                &mut self.scratch,
                self.hadamard_active,
                #[cfg(any(test, feature = "bitnet-v2"))]
                &self.hadamard_dispatch_count,
                hs,
                stream,
            )?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.normed.as_device_ptr(),
                self.scratch.x_i8.as_device_mut_ptr(),
                self.scratch.x_scale_dev.as_device_mut_ptr(),
                hs,
                stream,
            ))?;
            let x_scale = self.scratch.x_scale_dev.copy_to_host_scalar()?;

            ok(hip::ternary_gemv_halo_f16(
                ly.gate_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.gate_scales.as_device_ptr(),
                self.scratch.gate_fp16.as_device_mut_ptr(),
                is_,
                hs,
                stream,
            ))?;
            ok(hip::ternary_gemv_halo_f16(
                ly.up_packed.as_device_ptr(),
                self.scratch.x_i8.as_device_ptr(),
                x_scale,
                ly.up_scales.as_device_ptr(),
                self.scratch.up_fp16.as_device_mut_ptr(),
                is_,
                hs,
                stream,
            ))?;

            // Fused ReLU² GLU + ffn_sub_norm (FP32 interior → FP16 out).
            ok(hip::relu2_glu_rmsnorm_fp16(
                self.scratch.gate_fp16.as_device_ptr(),
                self.scratch.up_fp16.as_device_ptr(),
                ly.ffn_sub_norm.as_device_ptr(),
                self.scratch.silu_out.as_device_mut_ptr(),
                rms_eps,
                is_,
                stream,
            ))?;
            // Site #4: Hadamard rotation before FFN-down pre-quant.
            // Targets silu_out (size `is_ = intermediate_size`, 6912 = 54·128
            // on 2B-4T so divisibility by 128 holds).
            Self::maybe_hadamard_rotate_silu(
                &mut self.scratch,
                self.hadamard_active,
                #[cfg(any(test, feature = "bitnet-v2"))]
                &self.hadamard_dispatch_count,
                is_,
                stream,
            )?;
            ok(hip::quantize_fp16_to_i8(
                self.scratch.silu_out.as_device_ptr(),
                self.scratch.silu_i8.as_device_mut_ptr(),
                self.scratch.silu_scale_dev.as_device_mut_ptr(),
                is_,
                stream,
            ))?;
            let silu_scale = self.scratch.silu_scale_dev.copy_to_host_scalar()?;

            ok(hip::ternary_gemv_halo_f16(
                ly.down_packed.as_device_ptr(),
                self.scratch.silu_i8.as_device_ptr(),
                silu_scale,
                ly.down_scales.as_device_ptr(),
                self.scratch.down_fp16.as_device_mut_ptr(),
                hs,
                is_,
                stream,
            ))?;
            ok(hip::residual_add_fp32_from_fp16(
                self.scratch.x_fp32.as_device_mut_ptr(),
                self.scratch.down_fp16.as_device_ptr(),
                hs,
                stream,
            ))?;
        }

        // ---------- Final norm + tied LM head ----------
        ok(hip::rmsnorm_fp32_in_fp16_out(
            self.scratch.x_fp32.as_device_ptr(),
            self.weights.final_norm.as_device_ptr(),
            self.scratch.normed.as_device_mut_ptr(),
            rms_eps,
            hs,
            stream,
        ))?;
        ok(hip::fp16_gemv(
            self.weights.embedding.as_device_ptr(),
            self.scratch.normed.as_device_ptr(),
            self.scratch.logits.as_device_mut_ptr(),
            vocab,
            hs,
            stream,
        ))?;

        // Greedy argmax on device. Callers of [`Self::forward_token`]
        // may optionally reconcile against a deterministic host-side
        // scan for bit-exact tie determinism;
        // [`Self::forward_token_greedy`] trusts the device answer
        // verbatim (no D→H logits copy, no host reconcile).
        //
        // Background on the reconcile: on a tie (two tokens with the
        // exact same fp32 logit), the HIP warp reduction can land on
        // either tied lane depending on the per-thread stride-scan.
        // Host reconcile — when invoked — forces
        // lowest-token-id-wins-on-tie, the total order matching a
        // plain linear scan on both gen-1 and gen-2.
        ok(hip::argmax_fp32(
            self.scratch.logits.as_device_ptr(),
            self.scratch.next_tok_dev.as_device_mut_ptr(),
            vocab,
            stream,
        ))?;
        hip::device_synchronize()?;
        let dev_next = self.scratch.next_tok_dev.copy_to_host_scalar()?;

        Ok(dev_next)
    }

    /// Reset decode state at the start of a new conversation.
    pub fn reset(&mut self) {
        // Nothing to do on the GPU — reused slots are overwritten on next
        // forward_token. Host-side position is tracked by the caller.
    }

    /// Parsed `.h1b` config header (dimensions, rope params, ...).
    pub fn config(&self) -> &H1bConfig {
        &self.cfg
    }

    /// KV cache dtype resolved at construction. Surfaces through
    /// `/v1/models` / `/metrics` so operators can confirm which cache
    /// variant the live process is running on without walking proc env.
    pub fn kv_dtype(&self) -> KvDtype {
        self.kv_dtype
    }

    /// Variant of [`Self::forward_token`] that also exposes the
    /// post-final-norm, pre-`lm_head` hidden state back to the host.
    ///
    /// Runs the full forward pass exactly like [`Self::forward_token`] —
    /// argmax, logits copy, return value. After the base-model logits are
    /// computed the `scratch.normed` device buffer still holds the
    /// post-final-RMSNorm hidden state (the `lm_head` GEMV only reads
    /// from it; it never overwrites), so we issue one extra D→H memcpy
    /// and hand `hidden_out` back to the Medusa path.
    ///
    /// `hidden_out` is resized to `hidden_size` fp16 (as `u16`) elements
    /// on success. The caller reuses this buffer across decode cycles to
    /// avoid per-step allocation.
    ///
    /// Cost: one extra `hidden_size` fp16 D→H copy on top of the regular
    /// forward pass (2 × hs = 5120 bytes on BitNet-2B-4T). Measured
    /// overhead on gfx1151 is sub-microsecond — dwarfed by the 30-layer
    /// forward itself.
    pub fn forward_token_with_hidden(
        &mut self,
        token_id: i32,
        pos: i32,
        logits_out: &mut Vec<f32>,
        hidden_out: &mut Vec<u16>,
    ) -> Result<i32, BackendError> {
        let next = self.forward_token(token_id, pos, logits_out)?;
        let hs = self.cfg.hidden_size as usize;
        if hidden_out.len() != hs {
            hidden_out.resize(hs, 0);
        }
        self.scratch
            .normed
            .copy_to_slice(hidden_out.as_mut_slice())?;
        Ok(next)
    }

    /// Run the existing fp16×fp16 `lm_head` GEMV against a
    /// caller-supplied host hidden state and return the resulting logits.
    ///
    /// Used by the Medusa speculative-decode path: each head produces a
    /// projected hidden `h_out` on the host (four scalar fp16 GEMVs +
    /// residual in [`crate::medusa::heads::MedusaHeads::project`]), and
    /// this method re-runs the already-uploaded backbone `lm_head`
    /// against that hidden to produce per-head logits. Reusing the live
    /// device-resident embedding matrix avoids a second upload and keeps
    /// the `tied_embedding` invariant intact.
    ///
    /// The `normed` scratch is clobbered as a side effect — the caller
    /// must not rely on its previous contents. Logits are copied back to
    /// the host in `logits_out` (resized to `vocab_size` on success).
    pub fn lm_head_from_hidden(
        &mut self,
        hidden: &[u16],
        logits_out: &mut Vec<f32>,
    ) -> Result<(), BackendError> {
        let hs = self.cfg.hidden_size as usize;
        let vocab = self.cfg.vocab_size;
        if hidden.len() != hs {
            return Err(BackendError::BadInput(
                "lm_head_from_hidden: hidden length must equal hidden_size",
            ));
        }
        // Upload the host-side hidden into the `normed` scratch so the
        // live `fp16_gemv(embedding, normed, logits)` dispatch works
        // unchanged. `normed` is already sized to `[hs]`.
        self.scratch.normed.copy_from_slice(hidden)?;
        let stream = HipStream::DEFAULT;
        ok(hip::fp16_gemv(
            self.weights.embedding.as_device_ptr(),
            self.scratch.normed.as_device_ptr(),
            self.scratch.logits.as_device_mut_ptr(),
            vocab,
            self.cfg.hidden_size,
            stream,
        ))?;
        hip::device_synchronize()?;
        if logits_out.len() != vocab as usize {
            logits_out.resize(vocab as usize, 0.0);
        }
        self.scratch
            .logits
            .copy_to_slice(logits_out.as_mut_slice())?;
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

#[inline]
fn ok(status: onebit_hip::RcppStatus) -> Result<(), BackendError> {
    status.into_result().map_err(BackendError::from)
}

/// Deterministic argmax with lowest-index-wins-on-tie semantics.
///
/// Fast path: if `dev_next` already points at the argmax (which it does
/// whenever the max logit is unique), return it immediately — a single
/// bounds-check and indexed read.
///
/// Slow path: when the device kernel's lane-reduction landed on a tied
/// value, walk the logit vector once and return the lowest index whose
/// logit equals the max. This forces a stable total order across every
/// decode, independent of warp layout / thread count, and fixes the
/// shadow-burnin prompt-7 divergence where ~equal logits on two digit
/// tokens produced v1="1" vs v2="0" 100% of the time.
#[inline]
fn host_argmax_lowest_index(logits: &[f32], dev_next: i32) -> i32 {
    if logits.is_empty() {
        return 0;
    }
    let dev_idx = dev_next as usize;
    let dev_val = logits.get(dev_idx).copied().unwrap_or(f32::NEG_INFINITY);
    // Strict `>`: first (lowest-index) occurrence of the max wins.
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    // Device and host agree whenever the max is unique. They only diverge
    // on exact-logit ties — in which case host wins (lower index).
    if best_val == dev_val && dev_idx < logits.len() {
        // NaN-safe: NaN != NaN so this branch won't fire on NaN max.
        // Prefer the lower of (dev_idx, best_idx) since both carry the max.
        return dev_idx.min(best_idx) as i32;
    }
    best_idx as i32
}

/// Upload a span of on-disk FP32 as FP16 to the device. This matches the
/// C++ loader's `read_fp32_as_fp16` helper, except we read from the mmap
/// instead of an ifstream.
fn upload_fp32_as_fp16(
    file: &H1bFile,
    span: Span,
    expected_count: usize,
) -> Result<DeviceBuffer<u16>, BackendError> {
    let bytes = file.tensor_bytes(span);
    if bytes.len() != expected_count * core::mem::size_of::<f32>() {
        return Err(BackendError::BadInput("fp32 span size mismatch"));
    }
    // Convert fp32 → fp16 on the host, then upload. Lossy but matches
    // gen-1 numerics (same cast).
    let mut host_fp16: Vec<u16> = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        host_fp16.push(f16::from_f32(v).to_bits());
    }
    let mut buf = DeviceBuffer::alloc(expected_count)?;
    buf.copy_from_slice(&host_fp16)?;
    Ok(buf)
}

/// Upload a raw-bytes span to the device as `u8`. Used for packed ternary
/// weights which are already in the exact on-device layout.
fn upload_u8(file: &H1bFile, span: Span) -> Result<DeviceBuffer<u8>, BackendError> {
    let bytes = file.tensor_bytes(span);
    let mut buf = DeviceBuffer::alloc(bytes.len())?;
    buf.copy_from_slice(bytes)?;
    Ok(buf)
}

/// Upload a span of FP32 scales as-is.
fn upload_fp32(
    file: &H1bFile,
    span: Span,
    expected_count: usize,
) -> Result<DeviceBuffer<f32>, BackendError> {
    let bytes = file.tensor_bytes(span);
    if bytes.len() != expected_count * core::mem::size_of::<f32>() {
        return Err(BackendError::BadInput("fp32 scale span size mismatch"));
    }
    let mut host = Vec::with_capacity(expected_count);
    for chunk in bytes.chunks_exact(4) {
        host.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut buf = DeviceBuffer::alloc(expected_count)?;
    buf.copy_from_slice(&host)?;
    Ok(buf)
}

fn upload_layer(
    file: &H1bFile,
    lo: &H1bLayerOffsets,
    hs: usize,
    is_: usize,
    idx: usize,
) -> Result<LayerWeights, BackendError> {
    let _ = idx; // used only for logging
    let input_norm = upload_fp32_as_fp16(file, lo.input_norm, hs)?;
    let post_attn_norm = upload_fp32_as_fp16(file, lo.post_attn_norm, hs)?;
    let attn_sub_norm = upload_fp32_as_fp16(file, lo.attn_sub_norm, hs)?;
    let ffn_sub_norm = upload_fp32_as_fp16(file, lo.ffn_sub_norm, is_)?;

    let q_packed = upload_u8(file, lo.q_packed)?;
    let q_scales = upload_fp32(file, lo.q_scales, lo.q_scales.len / 4)?;
    let k_packed = upload_u8(file, lo.k_packed)?;
    let k_scales = upload_fp32(file, lo.k_scales, lo.k_scales.len / 4)?;
    let v_packed = upload_u8(file, lo.v_packed)?;
    let v_scales = upload_fp32(file, lo.v_scales, lo.v_scales.len / 4)?;
    let o_packed = upload_u8(file, lo.o_packed)?;
    let o_scales = upload_fp32(file, lo.o_scales, lo.o_scales.len / 4)?;
    let gate_packed = upload_u8(file, lo.gate_packed)?;
    let gate_scales = upload_fp32(file, lo.gate_scales, lo.gate_scales.len / 4)?;
    let up_packed = upload_u8(file, lo.up_packed)?;
    let up_scales = upload_fp32(file, lo.up_scales, lo.up_scales.len / 4)?;
    let down_packed = upload_u8(file, lo.down_packed)?;
    let down_scales = upload_fp32(file, lo.down_scales, lo.down_scales.len / 4)?;

    Ok(LayerWeights {
        input_norm,
        post_attn_norm,
        attn_sub_norm,
        ffn_sub_norm,
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        o_packed,
        o_scales,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        down_packed,
        down_scales,
    })
}

// Explicitly shadow these to keep warnings quiet when used by `lib.rs`.
#[allow(dead_code)]
fn _use_devices(_d: DevicePtr<u8>, _m: DeviceMutPtr<u8>) {}

// ----------------------------------------------------------------------------
// GGUF loader — dispatch target for `sniff_model_format(..) == Gguf`.
//
// Two sprints deep now:
//   1. Plumbing — parse header, extract `BitnetHeader`, walk the standard
//      llama.cpp tensor-name grid, return raw mmap slices. DONE.
//   2. Bit-unpack — IQ2_S → halo v2 (2 bpw) on the host side via
//      `onebit_core::gguf::unpack::iq2_s_to_halo_v2`. DONE (2026-04-19).
//
// What's still open is device-side integration: the unpacked payload +
// fp16 absmean scales need to be `memcpy_h2d`'d into `DeviceBuffer<u8>` /
// `DeviceBuffer<f16>` slots on `HipBackend`, and those pointers stashed
// where the forward-pass code can find them. Today we just drop the host
// copies after the unpack succeeds — the loader will still tap out at the
// outer `unimplemented!` since it can't construct a usable `HipBackend`.
// ----------------------------------------------------------------------------

/// GGUF tensor-name grid (llama.cpp convention). We build the expected
/// name for each layer / global up front and fetch the byte slice via
/// [`GgufFile::tensor`]. Producers have diverged here historically — if we
/// ever need to support alternate spellings (`attn_qkv` fused, etc.) we'll
/// add them as a fallback list, not by globbing the tensor directory.
fn layer_tensor_name(layer_idx: usize, tail: &str) -> String {
    format!("blk.{}.{}", layer_idx, tail)
}

/// Dispatch target for `.gguf` — mirrors the shape of
/// [`HipBackend::load_h1b_into_hip`] but reads from [`GgufFile`] instead.
///
/// Scope today (post-unpack sprint):
///   1. Parse header + tensor directory.
///   2. Extract [`onebit_core::BitnetHeader`] (hidden_size, num_layers,
///      rope_theta, rms_norm_eps, ...).
///   3. For each layer, look up the canonical llama.cpp tensor names and
///      pull the raw bytes out of the mmap.
///   4. Host-side bit-unpack via
///      [`onebit_core::gguf::unpack::iq2_s_to_halo_v2`]. The unpacked
///      payload is dropped immediately — device-buffer integration is
///      deliberately split into its own sprint.
///   5. Bail with `unimplemented!` after the directory walk succeeds —
///      we've proven the unpack works end-to-end on a real file, but
///      the `HipBackend` we'd want to return still has no device
///      allocations.
///
/// The signature deliberately matches [`HipBackend::load_h1b_into_hip`]
/// minus the separate `htok_path` — GGUFs carry the tokenizer inline, so
/// there is nothing to pass.
fn load_gguf_into_hip(
    gguf_path: &Path,
    model_id: String,
    max_context: usize,
) -> Result<HipBackend, BackendError> {
    let _ = model_id;
    let _ = max_context;

    let g = GgufFile::open(gguf_path)?;
    let hdr = g.read_bitnet_metadata()?;

    tracing::info!(
        architecture = %hdr.architecture,
        block_count = hdr.block_count,
        embedding_length = hdr.embedding_length,
        feed_forward_length = hdr.feed_forward_length,
        attention_head_count = hdr.attention_head_count,
        attention_head_count_kv = hdr.attention_head_count_kv,
        rope_freq_base = hdr.rope_freq_base,
        tokens = hdr.tokens.len(),
        "parsed GGUF BitnetHeader"
    );

    // Globals (llama.cpp canonical names).
    let _token_embd = g.tensor("token_embd.weight");
    let _output_norm = g.tensor("output_norm.weight");
    let _output = g.tensor("output.weight");
    let _rope_freqs = g.tensor("rope_freqs.weight"); // often absent

    // Per-layer directory sweep. Grab each expected tensor's dtype + bytes;
    // the moment we find an IQ2_S / TQ2_0 payload we tap out with an
    // explicit TODO so the bit-unpacking is a single-function follow-up.
    for l in 0..hdr.block_count as usize {
        for tail in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
        ] {
            let name = layer_tensor_name(l, tail);
            let info = g.tensor_info(&name).ok_or_else(|| {
                BackendError::Other(format!(
                    "GGUF missing expected tensor {name} (llama.cpp convention)"
                ))
            })?;
            let _bytes = g.tensor(&name).ok_or_else(|| {
                BackendError::Other(format!(
                    "GGUF tensor {name} has unknown dtype / size (got {:?})",
                    info.dtype
                ))
            })?;
            // This is where the requantizer will live. Plumbed as a single
            // call so a future agent can swap it in without touching
            // `sniff_model_format` / `HipBackend::new`.
            gguf_tensor_to_halo_packed(info.dtype, _bytes)?;
        }
    }

    // We got through the directory walk and host-side unpack — but we
    // still don't have device-resident weights. Stop here so a bad
    // config can still fail fast during integration testing. Wiring up
    // `DeviceBuffer<u8>` + `memcpy_h2d` is the next sprint.
    unimplemented!(
        "GGUF → HipBackend device-buffer integration is the next sprint; \
         host-side IQ2_S unpack is landed and exercised above"
    )
}

/// Requantizer shim — bit-unpacks GGUF tensor bytes into halo's packed
/// ternary layout on the host. **This sprint does the CPU-side unpack
/// only**; device-buffer integration (allocating `DeviceBuffer<u8>`s for
/// the packed payload + fp32 scales, `memcpy_h2d`, stashing the device
/// pointers on [`HipBackend`]) is still the next sprint. That split is
/// deliberate: the unpacker is backend-agnostic and lives in
/// [`onebit_core::gguf::unpack`] where it can be tested without HIP.
///
/// Dtype dispatch:
///   * `IQ2_S` — routed through [`onebit_core::gguf::unpack::iq2_s_to_halo_v2`].
///     Produces halo v2 packed bytes + per-super-block fp16 absmean scales.
///   * `F16` / `F32` / `BF16` — norms + embeddings. Still TODO; these
///     need a typed upload, not a bit-unpack.
///   * `TQ2_0` / `TQ1_0` — llama.cpp's newer native BitNet formats. TODO.
fn gguf_tensor_to_halo_packed(
    dtype: onebit_core::GgufTensorType,
    bytes: &[u8],
) -> Result<(), BackendError> {
    use onebit_core::GgufTensorType::*;
    match dtype {
        // FP16 / FP32 norms + embeddings — easy, land these first.
        F16 | F32 | BF16 => {
            // TODO(tier-1): typed upload into a DeviceBuffer<f16 / f32>.
            // No bit-unpack needed, just a bytewise memcpy_h2d.
            Ok(())
        }
        // BitNet's canonical public 2-bit ternary packing in llama.cpp.
        IQ2_S => {
            use onebit_core::gguf::unpack;
            // n_weights is derivable from the payload size: each super-block
            // is 82 bytes for 256 weights. Round down; the caller gave us
            // a whole tensor so this should be exact.
            let n_blocks = bytes.len() / unpack::IQ2_S_BLOCK_BYTES;
            let n_weights = n_blocks * 256;
            let mut packed = vec![0u8; n_weights.div_ceil(4)];
            let _scales = unpack::iq2_s_to_halo_v2(bytes, &mut packed, n_weights).map_err(|e| {
                BackendError::Other(format!("IQ2_S → halo v2 bit-unpack failed: {e}"))
            })?;
            // TODO(tier-1): memcpy_h2d the `packed` and `_scales` vectors
            // into DeviceBuffer<u8> / DeviceBuffer<f16> slots on
            // HipBackend, then drop the host copies. We deliberately don't
            // do this yet — the unpack is tested in isolation in
            // 1bit-core, and the device-buffer plumbing is its own PR.
            Ok(())
        }
        TQ2_0 | TQ1_0 => {
            // TODO(1bit-core/unpack): port llama.cpp's base-3 packing
            // (block_tq2_0 / block_tq1_0). Bit-layout is different from
            // IQ2_S — 5 ternaries per byte for TQ1_0 via the
            // d0+3·d1+9·d2+27·d3+81·d4 polynomial.
            unimplemented!("TQ2_0 / TQ1_0 → halo 2-bit repack is the next sprint");
        }
        other => Err(BackendError::Other(format!(
            "GGUF dtype {other:?} is not a BitNet-supported tensor format"
        ))),
    }
}

// ----------------------------------------------------------------------------
// Tests — format sniffing only. Construction of a real `HipBackend` needs
// a GPU and is covered in `tests/smoke_hip.rs` behind `--features hip` +
// `#[ignore]`.
// ----------------------------------------------------------------------------

#[cfg(test)]
mod format_sniff_tests {
    use super::*;
    use onebit_core::gguf::GGUF_MAGIC;
    use onebit_core::h1b::H1B_MAGIC;
    use std::io::Write;

    /// Write a throwaway temp file in the per-test tempdir and return its
    /// path. We keep the file on disk for the duration of the test — tests
    /// are short and tempdir cleans up on drop.
    fn scratch_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("1bit-router-sniff-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn sniff_h1b_by_extension_and_magic() {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&H1B_MAGIC);
        let p = scratch_file("good.h1b", &bytes);
        assert_eq!(sniff_model_format(&p).unwrap(), ModelFormat::H1b);
    }

    #[test]
    fn sniff_gguf_by_extension_and_magic() {
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&GGUF_MAGIC);
        let p = scratch_file("good.gguf", &bytes);
        assert_eq!(sniff_model_format(&p).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn sniff_rejects_mismatch_with_helpful_error() {
        // `.h1b` extension, but magic says GGUF. Operator probably renamed
        // the file — error message must name both accepted magics.
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&GGUF_MAGIC);
        let p = scratch_file("lying.h1b", &bytes);
        let err = sniff_model_format(&p).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("H1B") || s.contains("GGUF"), "got: {s}");
    }

    #[test]
    fn sniff_falls_back_to_magic_for_unknown_extension() {
        // No extension at all — we still accept the file if the magic is
        // unambiguous. Mirrors llama.cpp's behaviour.
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(&GGUF_MAGIC);
        let p = scratch_file("anonymous", &bytes);
        assert_eq!(sniff_model_format(&p).unwrap(), ModelFormat::Gguf);
    }

    #[test]
    fn sniff_rejects_unknown_magic_outright() {
        let bytes = b"XXXX\0\0\0\0".to_vec();
        let p = scratch_file("random.bin", &bytes);
        assert!(sniff_model_format(&p).is_err());
    }

    /// Live-model dispatch test: if the halo-1bit-2b.h1b is on disk, the
    /// sniffer must route it to the `.h1b` loader.
    ///
    /// `#[ignore]` because it needs the ~700 MB model file present; run
    /// manually with `cargo test -p 1bit-router --release -- --ignored`.
    #[test]
    #[ignore = "needs $HOME/1bit systems/models/halo-1bit-2b.h1b on disk"]
    fn live_h1b_model_routes_to_h1b_path() {
        let home = std::env::var("HOME").unwrap();
        let p = std::path::PathBuf::from(format!("{home}/1bit systems/models/halo-1bit-2b.h1b"));
        if !p.exists() {
            eprintln!("skipping: model not present at {}", p.display());
            return;
        }
        let f = sniff_model_format(&p).expect("sniff should succeed");
        assert_eq!(f, ModelFormat::H1b);
    }

    /// Host argmax + tie-break: unique max → device answer flows through,
    /// tied max → lowest-index wins. Regression guard for the shadow-burnin
    /// prompt-7 divergence ("chemical symbol for gold" → v1="1" vs v2="0"
    /// 100% of rounds, fixed 2026-04-20).
    #[test]
    fn host_argmax_picks_lowest_index_on_tie() {
        // Unique max at index 2: device answer passes through.
        let l = [0.1f32, 0.5, 3.2, 0.5, -1.0];
        assert_eq!(host_argmax_lowest_index(&l, 2), 2);

        // Tied max at indices 3 and 7 — device landed on the higher index
        // (HIP warp-reduce lane 7), host override must pick 3.
        let mut l = vec![0.0f32; 16];
        l[3] = 5.25;
        l[7] = 5.25;
        assert_eq!(host_argmax_lowest_index(&l, 7), 3);
        // Idempotent when device already picked the lower index.
        assert_eq!(host_argmax_lowest_index(&l, 3), 3);

        // Device OOB (defensive) + empty input: no panic.
        let l = [1.0f32, 2.0, 3.0];
        assert_eq!(host_argmax_lowest_index(&l, 99), 2);
        assert_eq!(host_argmax_lowest_index(&[], 0), 0);
    }

    /// Synthetic-GGUF dispatch test: build a minimal in-memory GGUF buffer
    /// (1 F32 tensor, correct header + alignment) and assert the sniffer
    /// routes it to the GGUF loader without needing to construct a
    /// `HipBackend`. Stops short of actually calling
    /// [`load_gguf_into_hip`], which hits `unimplemented!` on the first
    /// tensor — the goal is to prove the dispatch fired.
    #[test]
    #[ignore = "synthetic GGUF would succeed at sniff but panic inside load_gguf_into_hip today"]
    fn synthetic_gguf_routes_to_gguf_path() {
        // Minimal valid GGUF: magic + v3 + 0 tensors + 0 kvs, padded.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        // pad to 32-byte alignment for the (empty) tensor data region.
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }

        let p = scratch_file("synth.gguf", &buf);
        let f = sniff_model_format(&p).expect("sniff should succeed");
        assert_eq!(f, ModelFormat::Gguf, "should route to gguf loader");

        // We deliberately don't call `load_gguf_into_hip` — today it would
        // `unimplemented!` on the first tensor directory walk. The
        // assertion above is the whole point: dispatch fired.
    }
}

#[cfg(test)]
mod kv_dtype_tests {
    //! Unit tests for the `KvDtype` env parser + default. These cover the
    //! only two failure modes the serve path can hit: an unset env (must
    //! default to I8 so the 2× long-context bandwidth win ships by default)
    //! and a misspelled override (must surface a helpful error naming the
    //! valid spellings). Live KV quantization is a GPU-side concern
    //! exercised by the `--ignored` smoke_hip tests.
    use super::*;
    use std::sync::Mutex;

    /// Serializes env mutation across parallel cargo threads, same pattern
    /// as the bitnet_v2_hadamard_tests module below.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn default_is_i8() {
        // The whole point of the wire-up is the 2× bandwidth win by
        // default — regress this and the memory-bound long-context path
        // silently goes back to fp16 reads.
        assert_eq!(KvDtype::default(), KvDtype::I8);
    }

    #[test]
    fn parse_env_accepts_canonical_spellings() {
        assert_eq!(KvDtype::parse_env("i8").unwrap(), KvDtype::I8);
        assert_eq!(KvDtype::parse_env("INT8").unwrap(), KvDtype::I8);
        assert_eq!(KvDtype::parse_env("  i8  ").unwrap(), KvDtype::I8);
        assert_eq!(KvDtype::parse_env("f16").unwrap(), KvDtype::F16);
        assert_eq!(KvDtype::parse_env("FP16").unwrap(), KvDtype::F16);
        assert_eq!(KvDtype::parse_env("half").unwrap(), KvDtype::F16);
    }

    #[test]
    fn parse_env_rejects_unknown_with_helpful_error() {
        let err = KvDtype::parse_env("q4").unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("i8"), "expected mention of i8 in: {s}");
        assert!(s.contains("f16"), "expected mention of f16 in: {s}");
    }

    #[test]
    fn from_env_defaults_when_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: ENV_LOCK serializes every reader/writer in this module.
        unsafe { std::env::remove_var(KvDtype::ENV) };
        assert_eq!(KvDtype::from_env().unwrap(), KvDtype::default());
    }

    #[test]
    fn from_env_honours_override() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: ENV_LOCK serializes every reader/writer in this module.
        unsafe { std::env::set_var(KvDtype::ENV, "f16") };
        let got = KvDtype::from_env().unwrap();
        // Always restore before asserting so a panic doesn't poison the
        // process for later tests.
        unsafe { std::env::remove_var(KvDtype::ENV) };
        assert_eq!(got, KvDtype::F16);
    }
}

#[cfg(test)]
mod bitnet_v2_hadamard_tests {
    //! GPU-free tests for the BitNet v2 online-Hadamard gate.
    //!
    //! The live kernel dispatch requires a GPU + a real checkpoint, so it
    //! ships as an `#[ignore]` integration test (see
    //! `tests/smoke_hip.rs`). The policy matrix — `HALO_BITNET_V2` env ×
    //! `H1B_FLAG_HADAMARD_ROTATED` bit — is covered here because it's
    //! pure Rust and directly guards whether the hot path dispatches the
    //! rotation at all.
    //!
    //! Cases (runtime env, no rebuild required):
    //!   * env unset / "0"            → never rotate, even on flagged models.
    //!   * env "1" + model flag ON    → rotate at every site.
    //!   * env "1" + model flag OFF   → pass-through, warn-once.
    //!
    //! The historical `bitnet-v2` compile feature still acts as an
    //! equivalent-to-env override. These tests access process env, so we
    //! serialize them behind a `Mutex` to avoid parallel cargo threads
    //! stomping on one another's `HALO_BITNET_V2` writes.
    use super::*;
    use onebit_core::H1B_FLAG_HADAMARD_ROTATED;
    use onebit_core::types::{DEFAULT_RMS_NORM_EPS, DEFAULT_ROPE_THETA};
    use std::sync::Mutex;

    /// Serializes every test in this module. `std::env::set_var` is
    /// process-global; cargo runs tests in parallel threads, so without
    /// this lock a concurrent reader/writer race would make the gate
    /// non-deterministic.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard: remembers the pre-test `HALO_BITNET_V2` value and
    /// restores (or clears) it on drop so that later tests and the live
    /// process start from the expected state.
    struct EnvGuard {
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(val: Option<&str>) -> Self {
            let prev = std::env::var("HALO_BITNET_V2").ok();
            match val {
                // SAFETY: we hold `ENV_LOCK` for the full scope of every
                // caller. No other thread in this test module will read or
                // write HALO_BITNET_V2 while the guard is live.
                Some(v) => unsafe { std::env::set_var("HALO_BITNET_V2", v) },
                None => unsafe { std::env::remove_var("HALO_BITNET_V2") },
            }
            Self { prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: same as EnvGuard::set — callers hold ENV_LOCK.
            match &self.prev {
                Some(v) => unsafe { std::env::set_var("HALO_BITNET_V2", v) },
                None => unsafe { std::env::remove_var("HALO_BITNET_V2") },
            }
        }
    }

    fn cfg_with_reserved(reserved: i32) -> H1bConfig {
        H1bConfig {
            version: 2,
            hidden_size: 2560,
            intermediate_size: 6912,
            num_layers: 28,
            num_heads: 20,
            num_kv_heads: 5,
            vocab_size: 128_256,
            max_seq_len: 4096,
            tie_embeddings: 1,
            reserved,
            rope_theta: DEFAULT_ROPE_THETA,
            rms_norm_eps: DEFAULT_RMS_NORM_EPS,
        }
    }

    /// Env unset → gate always off, regardless of the model flag, as long
    /// as the compile-feature override is also off. This is the default
    /// behaviour every CI host and every shipping build observes.
    #[test]
    #[cfg(not(feature = "bitnet-v2"))]
    fn gate_is_off_when_env_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(None);

        let not_rotated = cfg_with_reserved(0);
        let rotated = cfg_with_reserved(H1B_FLAG_HADAMARD_ROTATED);
        assert!(
            !HipBackend::resolve_hadamard_gate(&not_rotated),
            "env unset + flag off → off (expected default)"
        );
        assert!(
            !HipBackend::resolve_hadamard_gate(&rotated),
            "env unset + flag ON → STILL off (env is the outer switch)"
        );
    }

    /// Env explicitly "0" (or anything other than "1") also keeps the gate
    /// off. Guards against operators typing the wrong value.
    #[test]
    #[cfg(not(feature = "bitnet-v2"))]
    fn gate_is_off_for_non_one_env_values() {
        let _g = ENV_LOCK.lock().unwrap();
        let rotated = cfg_with_reserved(H1B_FLAG_HADAMARD_ROTATED);

        for bad in ["", "0", "true", "yes", "on", "2", "01"] {
            let _env = EnvGuard::set(Some(bad));
            assert!(
                !HipBackend::resolve_hadamard_gate(&rotated),
                "HALO_BITNET_V2={bad:?} must not enable the gate"
            );
        }
    }

    /// Env "1" + model flag ON → gate resolves to true. Decode will
    /// dispatch `hadamard_rotate_fp16_device` at every pre-GEMV site.
    /// Runs regardless of compile feature because the env-var path is the
    /// supported production entry point.
    #[test]
    fn gate_fires_when_env_on_and_model_flagged() {
        let _g = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(Some("1"));

        let rotated = cfg_with_reserved(H1B_FLAG_HADAMARD_ROTATED);
        assert!(
            HipBackend::resolve_hadamard_gate(&rotated),
            "HALO_BITNET_V2=1 + flag ON → must enable the rotation hook"
        );
    }

    /// Env "1" + model flag OFF → gate resolves to false (pass-through +
    /// warn-once). Guards against double-quantizing a stock checkpoint
    /// whose weights were not trained under Hadamard.
    #[test]
    fn gate_is_passthrough_when_env_on_but_model_not_flagged() {
        let _g = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(Some("1"));

        let not_rotated = cfg_with_reserved(0);
        assert!(
            !HipBackend::resolve_hadamard_gate(&not_rotated),
            "env on + flag off → must NOT rotate (model wasn't trained for it)"
        );
        // Bit-0 is the only bit the accessor reads — other reserved bits
        // don't accidentally flip the gate either.
        let unrelated = cfg_with_reserved(0x1000_0000);
        assert!(
            !HipBackend::resolve_hadamard_gate(&unrelated),
            "unrelated reserved bits must not enable the hook"
        );
    }

    /// Compile feature on + model flag on → gate fires even when the env
    /// isn't set. Preserves the pre-env feature-gated test path.
    #[test]
    #[cfg(feature = "bitnet-v2")]
    fn gate_fires_when_feature_on_and_model_flagged() {
        let _g = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::set(None);

        let rotated = cfg_with_reserved(H1B_FLAG_HADAMARD_ROTATED);
        assert!(
            HipBackend::resolve_hadamard_gate(&rotated),
            "feature on + flag ON → must enable the rotation hook"
        );
    }
}
