//! Integration tests for the Sherry 1.25-bit ternary GEMV wrappers.
//!
//! Three tests:
//!   1. [`length_validation`] — pure CPU; exercises the precondition
//!      boundary of [`sherry_pack`] and [`sherry_gemv_fp16`]. Runs on every
//!      host, including CI boxes without ROCm.
//!   2. [`sherry_roundtrip_k2560_n2560`] — `#[ignore]` round-trip against
//!      the scalar-reference launcher at the architect-pinned shape
//!      `K=2560, N=2560`. Tolerance is ≤4 bf16 ULP per architect's note
//!      (parallel-reduction order differs between the fast and scalar
//!      kernels, so exact equality is not expected).
//!   3. [`sherry_perf_probe_k2560_n2560`] — `#[ignore]` perf probe. Logs
//!      achieved GB/s (weight-byte traffic only — act/out are in the noise
//!      at these shapes). No assertion; this is a human-readable telemetry
//!      point for the rocprof follow-up.
//!
//! The GPU tests need `--features link-rocm` + a real gfx1151. Run with:
//!
//! ```text
//! cargo test -p onebit-hip --release --features link-rocm --ignored
//! ```

use onebit_hip::{
    DeviceBuffer, HipStream, RcppError, sherry_gemv_fp16, sherry_gemv_fp16_scalar_ref, sherry_pack,
    sherry_packed_bytes,
};

// ---------------------------------------------------------------------------
// 1. Length / precondition validation — CPU-only.
// ---------------------------------------------------------------------------

/// Every safe wrapper rejects malformed shapes BEFORE hitting the FFI
/// boundary. This runs on CI hosts without ROCm.
#[test]
fn length_validation() {
    // ---- sherry_pack ----
    // Non-multiple-of-4 input length.
    {
        let bad_in = vec![0i8; 7];
        let mut out = vec![0u8; 1];
        let err = sherry_pack(&bad_in, &mut out).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
    }
    // Wrong output length.
    {
        let good_in = vec![0i8; 32];
        // correct is 32 * 5 / 32 = 5; give the wrong count.
        let mut bad_out = vec![0u8; 4];
        let err = sherry_pack(&good_in, &mut bad_out).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
    }
    // Correct shapes with link-rocm off → Unsupported. With link-rocm on
    // the packer is host-only pure C and runs happily.
    {
        let good_in = vec![0i8; 32];
        let mut out = vec![0u8; sherry_packed_bytes(32)];
        assert_eq!(out.len(), 5);
        let res = sherry_pack(&good_in, &mut out);
        #[cfg(not(feature = "link-rocm"))]
        assert!(matches!(res, Err(RcppError::Unsupported)));
        #[cfg(feature = "link-rocm")]
        res.expect("host-only packer must succeed on valid shapes");
    }

    // ---- sherry_gemv_fp16 ----
    // Null device pointers are fine for the precondition-only path because
    // the wrapper short-circuits before dispatch. We only exercise the
    // shape validation here — the device pointers are never dereferenced.
    // SAFETY: we never pass these to a kernel; the precondition check
    // fires first and returns an error without touching the GPU.
    use onebit_hip::{DeviceMutPtr, DevicePtr};
    let packed: DevicePtr<u8> = unsafe { DevicePtr::new(core::ptr::null()) };
    let act: DevicePtr<u16> = unsafe { DevicePtr::new(core::ptr::null()) };
    let out: DeviceMutPtr<u16> = unsafe { DeviceMutPtr::new(core::ptr::null_mut()) };

    // K not a multiple of 32.
    let err = sherry_gemv_fp16(packed, act, out, 64, 31, None).unwrap_err();
    assert!(matches!(err, RcppError::Precondition(_)));

    // K = 0.
    let err = sherry_gemv_fp16(packed, act, out, 64, 0, None).unwrap_err();
    assert!(matches!(err, RcppError::Precondition(_)));

    // N = 0.
    let err = sherry_gemv_fp16(packed, act, out, 0, 64, None).unwrap_err();
    assert!(matches!(err, RcppError::Precondition(_)));

    // Scalar-ref variant validates the same contract.
    let err = sherry_gemv_fp16_scalar_ref(packed, act, out, 64, 31, None).unwrap_err();
    assert!(matches!(err, RcppError::Precondition(_)));
}

// ---------------------------------------------------------------------------
// 2. Round-trip vs scalar-ref at K=2560 / N=2560.
// ---------------------------------------------------------------------------

/// Shape constants — pinned by the task spec.
const RT_N_OUT: i32 = 2560;
const RT_K_IN: i32 = 2560;

/// Fast-kernel output must match the scalar-reference output to within
/// 4 bf16 ULP. Parallel reduction order differs, so exact equality is
/// not expected; the 4-ULP budget is the architect's note.
///
/// Ignored by default — needs real ROCm + `--features link-rocm`.
#[test]
#[ignore = "requires GPU + link-rocm feature; run with --ignored"]
fn sherry_roundtrip_k2560_n2560() {
    use half::f16;

    let n = RT_N_OUT as usize;
    let k = RT_K_IN as usize;

    // ---- Build ternary weights with Sherry's one-zero-per-group invariant.
    // Deterministic pseudo-random: for each group of 4, pick a zero_pos in
    // 0..4 via `(g % 4)`, and fill the other three lanes with signs derived
    // from a cheap LCG.
    let total_weights = n * k;
    let mut ternary: Vec<i8> = vec![0; total_weights];
    let mut seed: u32 = 0xC0FFEE;
    for row in 0..n {
        for grp in 0..(k / 4) {
            let zero_pos = ((row.wrapping_mul(31) + grp) & 0x3) as usize;
            for lane in 0..4 {
                if lane == zero_pos {
                    ternary[row * k + grp * 4 + lane] = 0;
                    continue;
                }
                // LCG: simple 32-bit Lehmer-style step.
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let s: i8 = if (seed >> 16) & 1 == 0 { 1 } else { -1 };
                ternary[row * k + grp * 4 + lane] = s;
            }
        }
    }

    // ---- Pack host-side.
    let row_bytes = sherry_packed_bytes(k);
    assert_eq!(row_bytes, k * 5 / 32);
    let mut packed_host: Vec<u8> = vec![0u8; n * row_bytes];
    for row in 0..n {
        let ter_row = &ternary[row * k..(row + 1) * k];
        let out_row = &mut packed_host[row * row_bytes..(row + 1) * row_bytes];
        sherry_pack(ter_row, out_row).expect("host packer");
    }

    // ---- Random fp16 activations in [-1.0, +1.0).
    let mut act_host: Vec<u16> = Vec::with_capacity(k);
    for i in 0..k {
        let phase = (i as f32) * 0.0137 + 0.5;
        let v = phase.sin() * 0.8;
        act_host.push(f16::from_f32(v).to_bits());
    }

    // ---- H2D copies.
    let mut d_packed = DeviceBuffer::<u8>::alloc(packed_host.len()).expect("alloc packed");
    d_packed.copy_from_slice(&packed_host).expect("H2D packed");
    let mut d_act = DeviceBuffer::<u16>::alloc(k).expect("alloc act");
    d_act.copy_from_slice(&act_host).expect("H2D act");

    let mut d_out_fast = DeviceBuffer::<u16>::alloc_zeroed(n).expect("alloc out fast");
    let mut d_out_ref = DeviceBuffer::<u16>::alloc_zeroed(n).expect("alloc out ref");

    // ---- Fast-kernel dispatch.
    sherry_gemv_fp16(
        d_packed.as_device_ptr(),
        d_act.as_device_ptr(),
        d_out_fast.as_device_mut_ptr(),
        RT_N_OUT,
        RT_K_IN,
        Some(HipStream::DEFAULT),
    )
    .expect("sherry_gemv_fp16 dispatch");

    // ---- Scalar-reference dispatch.
    sherry_gemv_fp16_scalar_ref(
        d_packed.as_device_ptr(),
        d_act.as_device_ptr(),
        d_out_ref.as_device_mut_ptr(),
        RT_N_OUT,
        RT_K_IN,
        Some(HipStream::DEFAULT),
    )
    .expect("sherry_gemv_fp16_scalar_ref dispatch");

    onebit_hip::device_synchronize().expect("sync");

    // ---- D2H.
    let mut out_fast: Vec<u16> = vec![0u16; n];
    let mut out_ref: Vec<u16> = vec![0u16; n];
    d_out_fast
        .copy_to_slice(&mut out_fast)
        .expect("D2H out fast");
    d_out_ref.copy_to_slice(&mut out_ref).expect("D2H out ref");

    // ---- Tolerance check: convert fp16 → bf16, compare integer bit
    // patterns, accept diffs up to 4 ULP. We cast both values to f32
    // first, then round to bf16 (top 16 bits of f32) before diffing,
    // because the architect's "≤4 bf16 ULP" tolerance is defined at the
    // bf16 precision grid.
    let mut max_ulp: u32 = 0;
    let mut max_idx: usize = 0;
    for i in 0..n {
        let a_f32 = f16::from_bits(out_fast[i]).to_f32();
        let b_f32 = f16::from_bits(out_ref[i]).to_f32();
        // f32 → bf16 via top-16-bit truncation.
        let a_bf16 = (a_f32.to_bits() >> 16) as u16;
        let b_bf16 = (b_f32.to_bits() >> 16) as u16;
        let ulp = (a_bf16 as i32 - b_bf16 as i32).unsigned_abs();
        if ulp > max_ulp {
            max_ulp = ulp;
            max_idx = i;
        }
    }
    assert!(
        max_ulp <= 4,
        "bf16 ULP budget exceeded: max={} (idx={}, fast={:?}, ref={:?})",
        max_ulp,
        max_idx,
        f16::from_bits(out_fast[max_idx]).to_f32(),
        f16::from_bits(out_ref[max_idx]).to_f32(),
    );
}

// ---------------------------------------------------------------------------
// 3. Perf probe — log GB/s at K=2560 / N=2560.
// ---------------------------------------------------------------------------

/// Warm-then-measure perf probe. Logs the achieved weight-bandwidth in GB/s.
/// No assertions — this is a human-readable telemetry point that pairs with
/// the rocprof trace in the Sherry tuning follow-up (see
/// `project_bitnet_rocprof_plan.md`).
///
/// Ignored by default — needs real ROCm + `--features link-rocm`.
#[test]
#[ignore = "requires GPU + link-rocm feature; run with --ignored"]
fn sherry_perf_probe_k2560_n2560() {
    use half::f16;
    use std::time::Instant;

    let n = RT_N_OUT as usize;
    let k = RT_K_IN as usize;
    let row_bytes = sherry_packed_bytes(k);

    // Populate a simple all-zero-pos-0 pattern — content doesn't matter for
    // a bandwidth probe, only the byte stream.
    let packed_host: Vec<u8> = vec![0x42u8; n * row_bytes];
    let act_host: Vec<u16> = (0..k)
        .map(|i| f16::from_f32((i as f32) * 1e-3).to_bits())
        .collect();

    let mut d_packed = DeviceBuffer::<u8>::alloc(packed_host.len()).expect("alloc packed");
    d_packed.copy_from_slice(&packed_host).expect("H2D packed");
    let mut d_act = DeviceBuffer::<u16>::alloc(k).expect("alloc act");
    d_act.copy_from_slice(&act_host).expect("H2D act");
    let mut d_out = DeviceBuffer::<u16>::alloc_zeroed(n).expect("alloc out");

    // Warm-up — let caches and kernel launch overhead settle.
    const WARMUP_ITERS: usize = 8;
    for _ in 0..WARMUP_ITERS {
        sherry_gemv_fp16(
            d_packed.as_device_ptr(),
            d_act.as_device_ptr(),
            d_out.as_device_mut_ptr(),
            RT_N_OUT,
            RT_K_IN,
            None,
        )
        .expect("warmup dispatch");
    }
    onebit_hip::device_synchronize().expect("warmup sync");

    // Timed burst.
    const TIMED_ITERS: usize = 256;
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        sherry_gemv_fp16(
            d_packed.as_device_ptr(),
            d_act.as_device_ptr(),
            d_out.as_device_mut_ptr(),
            RT_N_OUT,
            RT_K_IN,
            None,
        )
        .expect("timed dispatch");
    }
    onebit_hip::device_synchronize().expect("timed sync");
    let elapsed = start.elapsed();

    // Weight-bandwidth GB/s. One full read of the packed weight buffer per
    // iteration dominates the byte stream at these shapes.
    let weight_bytes = (n * row_bytes) as f64;
    let total_bytes = weight_bytes * TIMED_ITERS as f64;
    let gb = total_bytes / 1.0e9;
    let secs = elapsed.as_secs_f64();
    let gbs = gb / secs;
    let per_iter_us = (secs / TIMED_ITERS as f64) * 1.0e6;

    eprintln!(
        "sherry_gemv_fp16 @ N={}, K={}: {:.2} GB/s ({:.3} us/iter over {} iters)",
        RT_N_OUT, RT_K_IN, gbs, per_iter_us, TIMED_ITERS,
    );
}
