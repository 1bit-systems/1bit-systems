//! Host-side sampler dispatch — the 7th APU surface.
//!
//! # Why this module exists
//!
//! Strix Halo has 16 Zen5 cores sitting idle while the iGPU grinds the
//! ternary GEMV. In gen-1 (and the early gen-2 draft) the sampler —
//! repetition penalty, top-k partitioning, temperature softmax, top-p
//! renormalise, multinomial draw — ran on the **same** `spawn_blocking`
//! thread that drove `forward_token`. That pinned ~4-5% of every decode
//! step to a core the iGPU dispatch path wanted for kernel-launch bookkeeping.
//!
//! Moving the sampler onto its own rayon-owned Zen5 thread (and handing
//! logits across via a [`flume::bounded(1)`](flume::bounded) channel)
//! gives the GPU-dispatch thread a clear 1-deep overlap window: while
//! the CPU worker is computing the next sampled id, the GPU thread can
//! start staging the next `forward_token` launch (memset, embedding
//! lookup, first RMSNorm) before it actually needs the sampled token.
//!
//! # What ships in this module
//!
//! * [`SamplerMode`] — which dispatch path the router takes (`Inline` or
//!   `Cpu`). Read once at router construction from `HALO_SAMPLER`.
//! * [`sampler_mode_from_env`] / [`SAMPLER_MODE_ENV`] — the env parser
//!   + env-var name constant.
//! * [`cpu`] — the persistent Zen5 worker + bounded-channel handoff
//!   (see [`cpu::CpuSampler`]).
//!
//! # What the companion module [`crate::cpu_lane`] does today
//!
//! `cpu_lane` is a thin **backward-compatibility shim** that re-exports
//! [`cpu::CpuLane`] + [`SamplerMode`] / [`SAMPLER_MODE_ENV`] /
//! [`sampler_mode_from_env`] so `1bit-server`'s benches / cost-probe
//! tests (which import `onebit_router::cpu_lane::CpuLane`) keep compiling.
//! New code inside `1bit-router` should import from `crate::sampler::*`.
//!
//! # Rule A
//!
//! Pure Rust. `rayon` + `flume` are the only new deps; both are in
//! `Cargo.lock` already (rayon transitively via tokenizers, flume
//! added locally for the bounded handoff). No Python, no threadpool
//! beyond rayon, no new runtime.

pub mod cpu;

pub use cpu::{CpuLane, CpuLaneError, RESERVED_CORES_FOR_IGPU_COORD, default_thread_count};

/// Env-var name that selects between the inline sampler and the CPU
/// lane sampler offload.
///
/// Accepted values (case-insensitive): `inline` / `cpu` / `parallel`.
/// `parallel` is a legacy alias for `cpu` — gen-2 shipped with the
/// `parallel` spelling before the 2026-04-20 offload rename; we keep it
/// working so existing operator scripts don't break. Anything else
/// returns a parse error that names the accepted set.
///
/// The router consults this exactly once at construction — see
/// [`sampler_mode_from_env`] — so the knob is a process-level dial,
/// not a per-request one.
pub const SAMPLER_MODE_ENV: &str = "HALO_SAMPLER";

/// Which sampler dispatch path the router should take.
///
/// * [`SamplerMode::Inline`] — the legacy path. The sampler runs on the
///   same `tokio::task::spawn_blocking` thread that drove `forward_token`,
///   so one CPU core handles both the GPU dispatch and the host-side
///   sample. Retained for A/B comparison and rollback.
/// * [`SamplerMode::Cpu`] — **new default as of 2026-04-20**. The sampler
///   runs inside [`cpu::CpuLane`]'s rayon pool via [`cpu::CpuLane::sample`]
///   (semantics are bit-identical to `Inline` for the simple offload
///   path) or via the bounded-channel handoff in
///   [`cpu::CpuSampler`] when the pipeline is wired (see that struct's
///   docstring).
///
/// The mode is picked once at router construction — it's a process-wide
/// dial, not a per-request option. Expose it through the
/// [`SAMPLER_MODE_ENV`] env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplerMode {
    /// Sampler runs on whatever thread the caller is on. Fastest for
    /// single-request local decode — no cross-thread handoff per token.
    /// Measured 2026-04-23: 67 tok/s HTTP decode vs 43 tok/s on Cpu
    /// sampler for halo-1bit-2b at ctx=64 on gfx1151. **Default.**
    #[default]
    Inline,
    /// Sampler runs on the [`cpu::CpuLane`]'s rayon pool. Bit-identical
    /// output to `Inline`; different executing thread. Keep as an
    /// opt-in for concurrency-heavy serving where the rayon pool amortizes
    /// sampler cost across overlapping requests.
    Cpu,
}

impl SamplerMode {
    /// Case-insensitive parser. Accepts `inline` / `cpu` / `parallel`
    /// (the last a legacy alias for `cpu`); empty string resolves to
    /// the default (`Cpu`). Everything else errors with a message
    /// listing the accepted set so ops doesn't have to grep.
    pub fn parse_env(raw: &str) -> Result<Self, crate::BackendError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "inline" => Ok(SamplerMode::Inline),
            // Empty string → default (Cpu). Parity with
            // `sampler_mode_from_env`'s "unset" behaviour below.
            "" => Ok(SamplerMode::default()),
            // `parallel` is the pre-2026-04-20 spelling; keep it as an
            // alias so existing operator scripts (and the on-disk wiki
            // pages that reference `HALO_SAMPLER=parallel`) keep working.
            "cpu" | "parallel" => Ok(SamplerMode::Cpu),
            other => Err(crate::BackendError::Other(format!(
                "HALO_SAMPLER: unknown value {other:?}; accepted: inline | cpu | parallel"
            ))),
        }
    }

    /// Human-readable label for logs. Canonical spelling —
    /// `SamplerMode::Cpu` renders as `cpu`, not `parallel`.
    pub fn label(self) -> &'static str {
        match self {
            SamplerMode::Inline => "inline",
            SamplerMode::Cpu => "cpu",
        }
    }
}

/// Read [`SAMPLER_MODE_ENV`] out of the environment, tolerating
/// empty / unset.
///
/// Unset or empty → [`SamplerMode::default`] (currently [`SamplerMode::Cpu`]).
/// Any non-empty value is parsed through [`SamplerMode::parse_env`]; bad
/// values surface as [`crate::BackendError::Other`] and the caller is
/// expected to refuse to start rather than silently fall back. That
/// matches how `HALO_BACKEND` behaves so ops tooling sees a single
/// convention.
pub fn sampler_mode_from_env() -> Result<SamplerMode, crate::BackendError> {
    match std::env::var(SAMPLER_MODE_ENV) {
        Ok(raw) if !raw.is_empty() => SamplerMode::parse_env(&raw),
        _ => Ok(SamplerMode::default()),
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the `SamplerMode` enum + env parser. The rayon-pool
    //! side of the module (argmax parity, pipelined handoff) lives in
    //! `sampler::cpu::tests`.
    use super::*;
    use std::sync::Mutex;

    /// Serialize env mutation for this module. `std::env` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Default must be `Cpu` after the 2026-04-20 flip. Regression guard
    /// — any future accidental revert to `Inline` would silently erase
    /// the sampler offload for boxes that don't set `HALO_SAMPLER`.
    #[test]
    fn default_sampler_mode_is_cpu() {
        assert_eq!(SamplerMode::default(), SamplerMode::Cpu);
    }

    /// `inline` / `cpu` / `parallel` (alias) / empty all parse
    /// correctly; garbage errors with the accepted set listed.
    #[test]
    fn parse_env_accepts_all_spellings() {
        assert_eq!(
            SamplerMode::parse_env("inline").unwrap(),
            SamplerMode::Inline
        );
        assert_eq!(SamplerMode::parse_env("cpu").unwrap(), SamplerMode::Cpu);
        assert_eq!(SamplerMode::parse_env("CPU").unwrap(), SamplerMode::Cpu);
        assert_eq!(SamplerMode::parse_env("  Cpu\n").unwrap(), SamplerMode::Cpu);
        // Legacy alias — `parallel` still resolves to `Cpu`.
        assert_eq!(
            SamplerMode::parse_env("parallel").unwrap(),
            SamplerMode::Cpu
        );
        // Empty defaults.
        assert_eq!(SamplerMode::parse_env("").unwrap(), SamplerMode::default());
        // Garbage errors with message naming the accepted set.
        let err = SamplerMode::parse_env("nonsense").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("inline") && msg.contains("cpu") && msg.contains("parallel"),
            "error must list accepted values; got: {msg}"
        );
    }

    /// Labels are stable + canonical (no `parallel`, that's the legacy
    /// input spelling only).
    #[test]
    fn labels_are_canonical() {
        assert_eq!(SamplerMode::Inline.label(), "inline");
        assert_eq!(SamplerMode::Cpu.label(), "cpu");
    }

    /// `HALO_SAMPLER=cpu` parses to `Cpu`; unset → default (Cpu);
    /// `inline` explicitly picks the legacy path for A/B; garbage
    /// errors. Serialized via ENV_LOCK because `std::env` is
    /// process-global.
    #[test]
    fn sampler_mode_env_override_is_respected() {
        let _g = ENV_LOCK.lock().unwrap();

        // Unset → default (Cpu).
        // SAFETY: edition-2024 env mutation; single-threaded inside lock.
        unsafe {
            std::env::remove_var(SAMPLER_MODE_ENV);
        }
        assert_eq!(sampler_mode_from_env().unwrap(), SamplerMode::Cpu);

        // Empty → default (Cpu).
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "");
        }
        assert_eq!(sampler_mode_from_env().unwrap(), SamplerMode::Cpu);

        // Explicit `cpu`.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "cpu");
        }
        assert_eq!(sampler_mode_from_env().unwrap(), SamplerMode::Cpu);

        // A/B rollback value.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "inline");
        }
        assert_eq!(sampler_mode_from_env().unwrap(), SamplerMode::Inline);

        // Legacy alias.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "parallel");
        }
        assert_eq!(sampler_mode_from_env().unwrap(), SamplerMode::Cpu);

        // Bad value errors.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "nonsense");
        }
        let err = sampler_mode_from_env().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("inline") && msg.contains("cpu"),
            "error must list accepted values; got: {msg}"
        );

        // Clean up before releasing the lock.
        unsafe {
            std::env::remove_var(SAMPLER_MODE_ENV);
        }
    }
}
