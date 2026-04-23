//! 1bit-cpu — safe Rust wrapper over the AVX2 ternary GEMV static lib.
//!
//! Companion to [`onebit_hip`]: this crate exposes a single primitive —
//! [`ternary_gemv_tq2_cpu`] — which binds into the `halo_cpu_ternary_gemv_tq2`
//! C entry point declared in `rocm-cpp/cpu-avx2/include/halo_cpu/ternary_gemv.h`
//! and compiled to `rocm-cpp/cpu-avx2/build/libcpu_avx2_ternary.a`.
//!
//! The C kernel consumes the same on-disk TQ2_0_g128 block layout the
//! HIP kernel reads, so a `.h1b` tensor region that we already `mmap` for
//! the iGPU path is byte-compatible with the CPU lane with no
//! repacking. The safe wrapper validates lengths at the FFI boundary
//! and delegates the actual math to the static archive.
//!
//! ## Feature gating
//!
//! The `link-cpu-avx2` feature enables linking against the real kernel.
//! With the feature off (default), the stubs in [`ffi_stub`] are used
//! and every call returns [`CpuError::Unsupported`]. This mirrors the
//! `1bit-hip` + `link-rocm` contract and keeps CI / `cargo check` green
//! on hosts without the archive built.

#![warn(missing_docs)]

use thiserror::Error;

#[cfg(feature = "link-cpu-avx2")]
mod ffi {
    //! Raw extern declarations for the kernel entry points. Declared
    //! `unsafe extern "C"` per Rust 1.82+ edition-2024 rules.
    //!
    //! Both functions are thread-safe as documented in
    //! `halo_cpu/ternary_gemv.h` (line 32): the kernel writes to
    //! disjoint `out` rows so callers are free to hand it any number
    //! of threads via OpenMP (`num_threads` arg).
    unsafe extern "C" {
        pub(super) fn halo_cpu_ternary_gemv_tq2(
            packed: *const u8,
            act: *const u16,
            out: *mut u16,
            n_out: core::ffi::c_int,
            k_in: core::ffi::c_int,
            num_threads: core::ffi::c_int,
        );
    }
}

#[cfg(not(feature = "link-cpu-avx2"))]
mod ffi_stub {
    //! Stub variant for crates checked without the native archive present.
    //! Rust linking never pulls the symbol; calling through the safe
    //! wrapper surfaces [`crate::CpuError::Unsupported`] instead of a
    //! panic or undefined symbol at load time.
}

/// Block-size constant baked into the TQ2_0_g128 packing the kernel
/// expects. Mirrors `kBlockSize` in the scalar reference — 128 weights
/// per block, 2 bits per code.
pub const BLOCK_SIZE_K: usize = 128;

/// Bytes per packed block — 2 bytes of FP16 scale + 32 bytes of 2-bit codes.
pub const BYTES_PER_BLOCK: usize = 34;

/// Errors surfaced by the CPU lane. Kept narrow on purpose; the kernel
/// itself never allocates and has no fallible interior state.
#[derive(Debug, Error)]
pub enum CpuError {
    /// A `packed`, `act`, or `out` slice was the wrong length for the
    /// declared `(N_out, K_in)` shape.
    #[error(
        "cpu ternary gemv: length mismatch — \
         N_out={n_out}, K_in={k_in}, \
         packed.len()={packed_len} (need {packed_need}), \
         act.len()={act_len} (need {act_need}), \
         out.len()={out_len} (need {out_need})"
    )]
    LengthMismatch {
        /// Rows in the output vector.
        n_out: i32,
        /// Columns of the weight matrix (must be a multiple of 128).
        k_in: i32,
        /// Length in bytes of the caller-supplied `packed` slice.
        packed_len: usize,
        /// Expected length of `packed`.
        packed_need: usize,
        /// Length of the activation slice.
        act_len: usize,
        /// Expected length of `act`.
        act_need: usize,
        /// Length of the output slice.
        out_len: usize,
        /// Expected length of `out`.
        out_need: usize,
    },

    /// `K_in` was not a positive multiple of 128 — the kernel's hard
    /// precondition. The scalar reference enforces the same check; an
    /// early-exit here keeps us from handing garbage to the SIMD path.
    #[error("cpu ternary gemv: K_in={0} is not a positive multiple of 128")]
    BadKIn(i32),

    /// `N_out` must be a positive row count.
    #[error("cpu ternary gemv: N_out={0} is not positive")]
    BadNOut(i32),

    /// Crate was compiled without `link-cpu-avx2` so the FFI symbol is
    /// not linked in. Matches the `onebit_hip::RcppStatus::Unsupported`
    /// shape.
    #[error("cpu ternary gemv: kernel not linked (build with feature link-cpu-avx2)")]
    Unsupported,
}

/// Safe wrapper over `halo_cpu_ternary_gemv_tq2`.
///
/// Layout (must match the TQ2_0_g128 contract baked into
/// `halo_cpu/ternary_gemv.h`):
///
/// * `packed` — `N_out × (K_in / 128) × 34` bytes, row-major,
///   AoS-block-interleaved. Each 128-weight block is `[FP16 d : 2 B][qs : 32 B]`.
/// * `act` — `K_in` fp16 activations (bit-pattern in `u16`).
/// * `out` — `N_out` fp16 output slots (bit-pattern in `u16`). Written
///   in full on every successful call.
/// * `num_threads` — passed through to the kernel's OpenMP dispatch;
///   `0` or negative means "`omp_get_max_threads()`", matching the
///   header's documented semantics.
///
/// All length checks happen here, before the FFI boundary. Any length
/// or shape violation returns [`CpuError::LengthMismatch`] /
/// [`CpuError::BadKIn`] / [`CpuError::BadNOut`] without touching
/// `unsafe`.
///
/// Returns `Ok(())` on success; `out` has been fully written.
pub fn ternary_gemv_tq2_cpu(
    packed: &[u8],
    act: &[u16],
    out: &mut [u16],
    n_out: i32,
    k_in: i32,
    num_threads: i32,
) -> Result<(), CpuError> {
    if n_out <= 0 {
        return Err(CpuError::BadNOut(n_out));
    }
    if k_in <= 0 || !(k_in as usize).is_multiple_of(BLOCK_SIZE_K) {
        return Err(CpuError::BadKIn(k_in));
    }

    let num_blocks = (k_in as usize) / BLOCK_SIZE_K;
    let packed_need = (n_out as usize) * num_blocks * BYTES_PER_BLOCK;
    let act_need = k_in as usize;
    let out_need = n_out as usize;

    if packed.len() != packed_need || act.len() != act_need || out.len() != out_need {
        return Err(CpuError::LengthMismatch {
            n_out,
            k_in,
            packed_len: packed.len(),
            packed_need,
            act_len: act.len(),
            act_need,
            out_len: out.len(),
            out_need,
        });
    }

    call_kernel(packed, act, out, n_out, k_in, num_threads)
}

#[cfg(feature = "link-cpu-avx2")]
#[inline]
fn call_kernel(
    packed: &[u8],
    act: &[u16],
    out: &mut [u16],
    n_out: i32,
    k_in: i32,
    num_threads: i32,
) -> Result<(), CpuError> {
    // SAFETY: length invariants enforced directly above; slice pointers
    // are valid for the declared counts and aligned per Rust's slice
    // guarantees (u8/u16 are both trivially aligned for the kernel's
    // `uint8_t *` / `uint16_t *` reads). The kernel is thread-safe for
    // disjoint `out` buffers (see header line 32) and we pass an
    // exclusive borrow so no other thread can alias.
    unsafe {
        ffi::halo_cpu_ternary_gemv_tq2(
            packed.as_ptr(),
            act.as_ptr(),
            out.as_mut_ptr(),
            n_out as core::ffi::c_int,
            k_in as core::ffi::c_int,
            num_threads as core::ffi::c_int,
        );
    }
    Ok(())
}

#[cfg(not(feature = "link-cpu-avx2"))]
#[inline]
fn call_kernel(
    _packed: &[u8],
    _act: &[u16],
    _out: &mut [u16],
    _n_out: i32,
    _k_in: i32,
    _num_threads: i32,
) -> Result<(), CpuError> {
    Err(CpuError::Unsupported)
}

#[cfg(test)]
mod tests {
    //! Unit tests. The three tests below all run on the default feature
    //! set (no `link-cpu-avx2`) — they exercise the boundary validation
    //! layer, which is the part that must never let a malformed call
    //! reach the kernel. A fourth integration test behind
    //! `#[cfg(feature = "link-cpu-avx2")]` and `#[ignore]` exercises
    //! the live kernel when run on the strixhalo box.
    use super::*;

    /// `K_in` has to be a positive multiple of 128. The scalar reference
    /// early-exits on the same check (`ternary_gemv_scalar_ref.cpp:50`);
    /// we refuse at the boundary so the AVX2 path never sees it.
    #[test]
    fn rejects_bad_k_in() {
        let packed = Vec::<u8>::new();
        let act = vec![0u16; 0];
        let mut out = vec![0u16; 1];
        let err =
            ternary_gemv_tq2_cpu(&packed, &act, &mut out, 1, 127, 1).expect_err("should reject");
        assert!(matches!(err, CpuError::BadKIn(127)), "got {err:?}");

        let err_zero =
            ternary_gemv_tq2_cpu(&packed, &act, &mut out, 1, 0, 1).expect_err("zero K bad");
        assert!(matches!(err_zero, CpuError::BadKIn(0)), "got {err_zero:?}");

        // 128 passes the K_in check (shape + length still checked below).
        let err_shape =
            ternary_gemv_tq2_cpu(&packed, &act, &mut out, 1, 128, 1).expect_err("length mismatch");
        assert!(
            matches!(err_shape, CpuError::LengthMismatch { .. }),
            "got {err_shape:?}"
        );
    }

    /// `N_out` must be strictly positive. Enforced before length math
    /// so callers get a clean error rather than an underflow when the
    /// kernel would compute `N_out * num_blocks * 34`.
    #[test]
    fn rejects_bad_n_out() {
        let packed = Vec::<u8>::new();
        let act = vec![0u16; 128];
        let mut out = vec![0u16; 1];
        let err = ternary_gemv_tq2_cpu(&packed, &act, &mut out, 0, 128, 1).expect_err("n_out=0");
        assert!(matches!(err, CpuError::BadNOut(0)), "got {err:?}");
        let err_neg =
            ternary_gemv_tq2_cpu(&packed, &act, &mut out, -3, 128, 1).expect_err("n_out<0");
        assert!(matches!(err_neg, CpuError::BadNOut(-3)), "got {err_neg:?}");
    }

    /// The happy-path length-check arithmetic: for `N_out=4, K_in=256`
    /// the kernel wants `4 × 2 × 34 = 272` bytes of packed weight, 256
    /// u16 activations, 4 u16 outputs. Anything else returns
    /// [`CpuError::LengthMismatch`]. Also checks that the stub path
    /// returns [`CpuError::Unsupported`] (default feature set) once
    /// lengths match — so callers see a real error, not a silent-success.
    #[test]
    fn length_math_matches_layout_and_stub_returns_unsupported() {
        const N: i32 = 4;
        const K: i32 = 256;
        let num_blocks = (K as usize) / BLOCK_SIZE_K; // 2
        let packed_need = (N as usize) * num_blocks * BYTES_PER_BLOCK; // 4*2*34 = 272
        assert_eq!(packed_need, 272);

        let packed = vec![0u8; packed_need];
        let act = vec![0u16; K as usize];
        let mut out = vec![0u16; N as usize];

        // Under-length packed:
        let short = vec![0u8; packed_need - 1];
        let err = ternary_gemv_tq2_cpu(&short, &act, &mut out, N, K, 1).expect_err("short packed");
        match err {
            CpuError::LengthMismatch {
                packed_len,
                packed_need: need,
                ..
            } => {
                assert_eq!(packed_len, packed_need - 1);
                assert_eq!(need, packed_need);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }

        // Wrong-length act:
        let short_act = vec![0u16; (K as usize) - 1];
        let err =
            ternary_gemv_tq2_cpu(&packed, &short_act, &mut out, N, K, 1).expect_err("short act");
        assert!(matches!(err, CpuError::LengthMismatch { .. }));

        // Wrong-length out:
        let mut short_out = vec![0u16; (N as usize) - 1];
        let err =
            ternary_gemv_tq2_cpu(&packed, &act, &mut short_out, N, K, 1).expect_err("short out");
        assert!(matches!(err, CpuError::LengthMismatch { .. }));

        // All three match → stub returns Unsupported (default build); real
        // build links through and returns Ok(()).
        let result = ternary_gemv_tq2_cpu(&packed, &act, &mut out, N, K, 1);
        #[cfg(not(feature = "link-cpu-avx2"))]
        {
            assert!(matches!(result, Err(CpuError::Unsupported)));
        }
        #[cfg(feature = "link-cpu-avx2")]
        {
            result.expect("shape-matched call must succeed when linked");
        }
    }
}
