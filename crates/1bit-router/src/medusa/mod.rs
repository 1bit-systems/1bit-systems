//! Medusa speculative-decoding scaffold.
//!
//! Status: **scaffolding only.** Wires the FFI + state-machine plumbing so
//! a later pass can drop in the retrained 4-head weights from
//! `parrishcorcoran/MedusaBitNet-2B-4T`. Not on the server hot path — the
//! router's `generate_blocking` does not call into this module yet.
//!
//! Recommendation from `docs/wiki/Medusa-Integration-Plan.md` is **DEFER**
//! behind Sherry + BitNet v2. The scaffold exists to pin down the call
//! site + type surface while the prerequisite work ships — it does **not**
//! flip the defer decision.
//!
//! # Design
//!
//! The module has three concerns, one per file:
//!
//! * [`heads`] — the four typed speculative heads. Each head projects the
//!   frozen backbone's post-final-norm hidden state to a vocab-sized
//!   logits row via the small-M ternary GEMM in `onebit-hip`. Today the
//!   heads are a `path`-stub: construction either succeeds with a
//!   handle that knows its weight path, or returns
//!   [`MedusaError::WeightsNotFound`] if no file exists there. Live
//!   weight loading lands in the retrained-weights pass.
//! * [`verify`] — the tree-attention verification state machine. Accepts
//!   the longest prefix of head-predicted tokens that matches the base
//!   model's argmax at each position. Sequential today; the plan doc
//!   discusses a tree-attention kernel for a later pass.
//! * This file — the public surface: [`MedusaConfig`], [`MedusaState`],
//!   [`MedusaError`], and the feature gate.
//!
//! # Feature gate
//!
//! The whole path is OFF by default. It only activates when **both**:
//!
//! 1. The environment variable `HALO_MEDUSA=1` is set at router
//!    construction time, and
//! 2. The loader found real head weights at the configured path.
//!
//! Either condition alone yields a disabled [`MedusaState`] — the server
//! dispatches the regular `forward_token` path as if Medusa were not
//! compiled in. This matches the BitNet v2 `H1B_FLAG_HADAMARD_ROTATED`
//! gate at the kernel level.

pub mod heads;
pub mod loader;
pub mod verify;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use heads::{MedusaHead, MedusaHeads, NUM_MEDUSA_HEADS};
pub use verify::{TreeVerifier, VerifyOutcome};

/// Environment-variable gate. Set to `"1"` to opt in at router startup.
///
/// Any other value — including unset, empty, or "0" — leaves the router on
/// the non-Medusa dispatch path.
pub const HALO_MEDUSA_ENV: &str = "HALO_MEDUSA";

/// Router-side Medusa configuration. Populated by the server layer from
/// env / CLI at startup and passed into [`MedusaState::new`].
#[derive(Debug, Clone, Default)]
pub struct MedusaConfig {
    /// Path to the serialized head weights. The retrained-weights pass
    /// will define the on-disk format (leaning toward the existing
    /// `.h1b` single-file convention with a new magic). For now, the
    /// scaffold only checks whether the file exists at that path.
    pub medusa_heads_path: Option<PathBuf>,
}

impl MedusaConfig {
    /// Build a config from environment variables. Reads
    /// `HALO_MEDUSA_HEADS_PATH` (optional). The `HALO_MEDUSA=1` enable
    /// gate is read separately at [`MedusaState::new`] — keeping the
    /// path config independent means ops can pin a path in systemd
    /// without accidentally enabling the speculative path.
    pub fn from_env() -> Self {
        let medusa_heads_path = std::env::var("HALO_MEDUSA_HEADS_PATH")
            .ok()
            .map(PathBuf::from);
        Self { medusa_heads_path }
    }

    /// `true` if the `HALO_MEDUSA=1` gate is set. Does *not* consult the
    /// weights path — see the module docstring for the two-of-two rule.
    pub fn gate_enabled() -> bool {
        matches!(std::env::var(HALO_MEDUSA_ENV).as_deref(), Ok("1"))
    }
}

/// Error surface for the Medusa path. Kept as its own error type (not
/// folded into `BackendError`) so the router scaffold can construct + test
/// it without dragging the full backend error taxonomy into the module.
/// The server layer converts as needed when the wire-up pass lands.
#[derive(Debug, thiserror::Error)]
pub enum MedusaError {
    /// Gate is off — `HALO_MEDUSA=1` was not set at startup. Not an
    /// error per se; returned as `Err(Disabled)` from helpers so
    /// constructors have a uniform `Result` shape without threading an
    /// `Option<MedusaState>` through every call site. The server treats
    /// this as "fall through to the regular forward path".
    #[error("medusa path disabled (HALO_MEDUSA != 1)")]
    Disabled,
    /// Gate is on but the head weights file doesn't exist at the
    /// configured path. Operator needs to download
    /// `parrishcorcoran/MedusaBitNet-2B-4T` (or our retrained version)
    /// and point `HALO_MEDUSA_HEADS_PATH` at it.
    #[error("medusa heads weights not found at {path}")]
    WeightsNotFound {
        /// Path that was searched.
        path: PathBuf,
    },
    /// Gate is on, weights file was found, but no path was configured.
    /// Distinct from `WeightsNotFound` so ops dashboards can count
    /// "operator forgot to set the env var" separately from "operator
    /// set the env var but the file is missing".
    #[error("medusa heads path not configured (HALO_MEDUSA_HEADS_PATH unset)")]
    PathNotConfigured,
    /// Caller-side argument validation (shape, length). Returned from
    /// [`TreeVerifier`] when the tree shape the heads produce doesn't
    /// line up with the base model's logits batch.
    #[error("medusa: {0}")]
    BadInput(&'static str),
    /// The retrained-weights pass will surface parser errors from the
    /// on-disk head file through this variant. Currently unreachable —
    /// the loader stub only returns `WeightsNotFound` / `Disabled`.
    #[error("medusa weight loader: {0}")]
    LoaderError(String),
}

/// Runtime state for the Medusa speculative path.
///
/// Constructed once at router startup and kept behind the existing
/// `Arc<Mutex<Inner>>`. Two legal shapes:
///
/// * [`MedusaState::Disabled`] — feature gate off OR weights missing.
///   The scaffold returns this when `from_config` can't honour the
///   request. Server treats it as a no-op.
/// * [`MedusaState::Enabled`] — gate on AND weights present. Today this
///   variant holds a [`MedusaHeads`] + [`TreeVerifier`], but the live
///   forward pass has not landed yet. Attempting to dispatch through it
///   returns [`MedusaError::BadInput`] — scaffolding only.
// `Disabled` is a tag-only variant; `Enabled` holds a `Mutex<MedusaHeads>`
// + `TreeVerifier` (~hundreds of bytes). Boxing would require changing
// every `MedusaState::Enabled { heads, .. }` destructure pattern, which is
// an API ripple not in scope for the clippy gate flip.
// TODO(gap-p2): box the `Enabled` payload and update destructures.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum MedusaState {
    /// Speculative path is off for this router instance.
    Disabled,
    /// Speculative path is on; holds loaded heads + the verifier.
    Enabled {
        /// The four speculative heads. Wrapped in a `Mutex` because the
        /// device-side projection path threads host slices through a
        /// stack of per-cycle HIP scratch buffers held inside
        /// `MedusaHeads::device` — which requires `&mut self`. The
        /// outer router already serializes decode requests via the
        /// `Inner` mutex, so contention here is zero; the wrapper is
        /// purely a compile-time affordance to reconcile that `&mut`
        /// with the `Arc<MedusaState>` we share across tokio tasks.
        heads: Mutex<MedusaHeads>,
        /// Tree-attention verifier state. Currently empty; populated
        /// per decode step once the kernel is wired.
        verifier: TreeVerifier,
    },
}

impl MedusaState {
    /// Build the Medusa state from the env-derived config.
    ///
    /// Rules (encoded verbatim in the unit tests):
    ///
    /// 1. If `HALO_MEDUSA` is not `"1"`, return `Ok(Disabled)`. No
    ///    path is consulted, no file I/O is attempted. This is the
    ///    path every existing deployment takes.
    /// 2. If the gate is on and `cfg.medusa_heads_path` is `None`,
    ///    return `Err(PathNotConfigured)`.
    /// 3. If the gate is on and the path does not exist on disk,
    ///    return `Err(WeightsNotFound { path })`.
    /// 4. If the gate is on and the file exists, call
    ///    [`MedusaHeads::load`] which mmaps the file and parses the
    ///    `.h1b-medusa` header + per-head tensor offsets. The GPU
    ///    upload path lands in a follow-up pass; the parsed handle
    ///    lives CPU-side only.
    pub fn from_config(cfg: &MedusaConfig) -> Result<Self, MedusaError> {
        if !MedusaConfig::gate_enabled() {
            return Ok(MedusaState::Disabled);
        }
        let path = cfg
            .medusa_heads_path
            .clone()
            .ok_or(MedusaError::PathNotConfigured)?;
        if !Path::new(&path).exists() {
            return Err(MedusaError::WeightsNotFound { path });
        }
        let heads = MedusaHeads::load(&path, cfg)?;
        let verifier = TreeVerifier::new();
        Ok(MedusaState::Enabled {
            heads: Mutex::new(heads),
            verifier,
        })
    }

    /// Cheap check used at call sites that dispatch the Medusa path.
    /// `true` only for the `Enabled` variant.
    pub fn is_enabled(&self) -> bool {
        matches!(self, MedusaState::Enabled { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onebit_hip::{HipStream, RcppError, ternary_gemm_smallm};
    use std::sync::Mutex;

    /// Serialize env-var mutation across tests in this module. `std::env`
    /// is process-global and Medusa's gate check races with itself if
    /// `cargo test -p onebit-router` fans out across threads.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Test #1: the FFI wrapper arg validation.
    ///
    /// The small-M ternary GEMM is the Medusa kernel that turns heads +
    /// the backbone's hidden state into per-head logits. The wrapper in
    /// `1bit-hip` is scaffolding: on a real ROCm build it returns
    /// `Unsupported` after validating the shape contract, on CI hosts
    /// without ROCm it returns `Unsupported` via the stub. In both modes
    /// *shape-invalid* inputs must return `InvalidArg` / `Precondition`
    /// *before* touching the FFI boundary — that's the invariant the
    /// router relies on when it dispatches through this symbol.
    #[test]
    fn ffi_wrapper_arg_validation() {
        // M out of range (0, 17, 100). Kernel only handles M ∈ [1, 16].
        for bad_m in [0i32, 17, 100, -1] {
            let err = ternary_gemm_smallm(
                &[0i32; 64 * 64 / 16],
                &[0u16; 64 * 64 / 2],
                &mut [0u16; 64],
                bad_m,
                64,
                64,
                None,
            )
            .expect_err("bad M must be rejected");
            assert!(
                matches!(err, RcppError::Precondition(_)),
                "bad M={bad_m}: wrong error variant: {err:?}"
            );
        }

        // Keep `1 * K/2` and `1 * N` shape annotations so the M=1 intent is
        // obvious when scanning; clippy's identity_op would flatten them.
        #[allow(clippy::identity_op)]
        {
            // N not a multiple of 64.
            let err = ternary_gemm_smallm(
                &[0i32; 64 * 32 / 16],
                &[0u16; 1 * 64 / 2],
                &mut [0u16; 32],
                1,
                32, // not mod 64
                64,
                Some(HipStream::DEFAULT),
            )
            .expect_err("N%64 != 0 must be rejected");
            assert!(matches!(err, RcppError::Precondition(_)), "{err:?}");

            // K not a multiple of 64.
            let err = ternary_gemm_smallm(
                &[0i32; 64 * 32 / 16],
                &[0u16; 1 * 32 / 2],
                &mut [0u16; 64],
                1,
                64,
                32, // not mod 64
                None,
            )
            .expect_err("K%64 != 0 must be rejected");
            assert!(matches!(err, RcppError::Precondition(_)), "{err:?}");

            // Length mismatch: codes slice too short.
            let err = ternary_gemm_smallm(
                &[0i32; 4],          // far too short
                &[0u16; 1 * 64 / 2], // M*K/2 u16
                &mut [0u16; 64],     // M*N bf16
                1,
                64,
                64,
                None,
            )
            .expect_err("codes length mismatch must be rejected");
            assert!(matches!(err, RcppError::Precondition(_)), "{err:?}");

            // Valid shape: scaffolding returns `Unsupported` (no real dispatch).
            let result = ternary_gemm_smallm(
                &[0i32; 64 * 64 / 16],
                &[0u16; 1 * 64 / 2],
                &mut [0u16; 64],
                1,
                64,
                64,
                None,
            );
            assert!(
                matches!(result, Err(RcppError::Unsupported)),
                "scaffold must return Unsupported on valid shapes: {result:?}"
            );
        }
    }

    /// Test #2: heads-disabled pass-through.
    ///
    /// When the env gate is NOT set, `MedusaState::from_config` must
    /// return `Ok(Disabled)` regardless of whether the weights path
    /// exists. This is the invariant that keeps the feature off by
    /// default — every existing deployment's `forward_token` call site
    /// sees `state.is_enabled() == false` and dispatches the regular
    /// forward path.
    #[test]
    fn heads_disabled_pass_through() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024 env mutation; serialized via ENV_LOCK.
        unsafe {
            std::env::remove_var(HALO_MEDUSA_ENV);
        }

        // No path + no gate → Disabled.
        let cfg = MedusaConfig::default();
        let state = MedusaState::from_config(&cfg).expect("gate off must be Ok(Disabled)");
        assert!(!state.is_enabled(), "gate off must be disabled");

        // Even with a bogus path set, if the gate is off the path must
        // not be consulted — pointing at a guaranteed-nonexistent file
        // must still return Disabled.
        let cfg_with_missing = MedusaConfig {
            medusa_heads_path: Some(PathBuf::from("/nonexistent/medusa-heads.h1b")),
        };
        let state = MedusaState::from_config(&cfg_with_missing)
            .expect("gate off must short-circuit before file I/O");
        assert!(
            !state.is_enabled(),
            "gate off must stay disabled even with path"
        );

        // Explicit "0" must also count as off.
        unsafe {
            std::env::set_var(HALO_MEDUSA_ENV, "0");
        }
        let state = MedusaState::from_config(&cfg).expect("HALO_MEDUSA=0 must be Ok(Disabled)");
        assert!(!state.is_enabled(), "HALO_MEDUSA=0 must be disabled");

        // Clean up so other tests see unset.
        unsafe {
            std::env::remove_var(HALO_MEDUSA_ENV);
        }
    }

    /// Test #3: heads-enabled-but-no-weights returns a structured error.
    ///
    /// With `HALO_MEDUSA=1` and a configured path that does not exist on
    /// disk, `from_config` must return `Err(WeightsNotFound { path })` —
    /// not a panic, not `Disabled`, not `PathNotConfigured`. The error
    /// carries the path so the operator sees where we looked.
    /// Separately: gate on + no path at all must return
    /// `Err(PathNotConfigured)`.
    #[test]
    fn heads_enabled_but_no_weights_returns_structured_error() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024 env mutation; serialized via ENV_LOCK.
        unsafe {
            std::env::set_var(HALO_MEDUSA_ENV, "1");
        }

        // Case A: gate on, path unset → PathNotConfigured.
        let cfg = MedusaConfig::default();
        let err = MedusaState::from_config(&cfg)
            .expect_err("gate on + no path must error, not fall through");
        match err {
            MedusaError::PathNotConfigured => {}
            other => panic!("expected PathNotConfigured, got {other:?}"),
        }

        // Case B: gate on, path configured but missing → WeightsNotFound.
        let missing = PathBuf::from("/nonexistent/medusa/heads-test.h1b");
        let cfg = MedusaConfig {
            medusa_heads_path: Some(missing.clone()),
        };
        let err = MedusaState::from_config(&cfg)
            .expect_err("gate on + missing file must error, not fall through");
        match err {
            MedusaError::WeightsNotFound { path } => {
                assert_eq!(path, missing, "error must carry the path we searched");
            }
            other => panic!("expected WeightsNotFound, got {other:?}"),
        }

        // Clean up.
        unsafe {
            std::env::remove_var(HALO_MEDUSA_ENV);
        }
    }
}
