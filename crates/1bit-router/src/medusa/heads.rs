//! Four speculative heads that project the backbone's hidden state to
//! per-head logit rows.
//!
//! # Head shape (from `parrishcorcoran/MedusaBitNet-2B-4T`
//! `medusa_heads_step2000.pt`)
//!
//! | Field            | Value  |
//! |------------------|--------|
//! | Number of heads  | 4      |
//! | Per-head layers  | 1 residual SiLU-gated block + shared `lm_head` |
//! | Hidden dim       | 2560   |
//! | Output vocab     | 128256 (shared across heads, lives on backbone) |
//! | Per-head tensors | `w_in` + `w_out`, each fp16[2560, 2560] (25 MiB) |
//! | Total head file  | ~100 MiB fp16 |
//! | Acceptance rates | 63.0 / 29.0 / 11.1 / 4.6 % |
//!
//! Heads share the backbone's `lm_head`, so the vocab-projection path is
//! the existing LM-head GEMV. Each head only owns its `w_in` / `w_out`
//! matrices. The forward pass per head is (from the upstream README):
//!
//!   h_out  = h + W_out · SiLU(W_in · h)      // two fp16 GEMVs + silu
//!   logits = backbone.lm_head(h_out)           // existing fp16 GEMV
//!
//! # Device path (post-2026-04-20 rewrite)
//!
//! The old `project_*_host` methods decoded fp16 → f32 on the CPU and
//! ran a scalar matmul — 130 ms per 4-head cycle on the strixhalo box,
//! which wiped out speculative decoding's throughput win (67 → 12 tok/s
//! with `HALO_MEDUSA=1`). The new path uploads all 8 weight matrices
//! (4 heads × `W_in` + `W_out`) to device memory **once** at load time
//! and runs every per-head projection via the native `rcpp_fp16_gemv`
//! + `rcpp_silu_glu_fp16` kernels sequentially on the default HIP
//!   stream. The host never touches weights after load.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use onebit_hip::{DeviceBuffer, HipStream};

use super::MedusaConfig;
use super::MedusaError;
use super::loader::MedusaHeadsFile;

/// Number of speculative heads we plan to load. Pinned by the
/// `MedusaBitNet-2B-4T` artifact; changing this without also
/// retraining the heads is an ABI break.
pub const NUM_MEDUSA_HEADS: usize = 4;

/// Per-head hidden dimension. Matches the Microsoft `bitnet-b1.58-2B-4T`
/// backbone — a head whose dim doesn't match the backbone is a bug.
pub const MEDUSA_HIDDEN_DIM: usize = 2560;

/// A single speculative head — metadata only. Weight tensors live in
/// [`MedusaDeviceWeights`] (on-device) and the owning mmap in the
/// [`MedusaHeadsFile`].
#[derive(Debug)]
pub struct MedusaHead {
    /// Head index (0 = t+1, 1 = t+2, 2 = t+3, 3 = t+4).
    pub index: usize,
    /// Path the weights were loaded from. Shared across all four
    /// heads today — they live in one file per the upstream artifact.
    pub weights_path: PathBuf,
    /// Expected acceptance rate from the model card (0.0–1.0). Used
    /// for telemetry and sanity-checking the live measurements.
    pub expected_acceptance: f32,
}

/// Per-head acceptance rates from the `MedusaBitNet-2B-4T` model card
/// (Alpaca 52K training set). Ops dashboards compare live measurements
/// against these; a retrain may overwrite them.
const EXPECTED_ACCEPTANCE: [f32; NUM_MEDUSA_HEADS] = [0.630, 0.290, 0.111, 0.046];

/// Device-resident head weights + per-cycle scratch.
///
/// All 8 weight buffers (4 heads × W_in + W_out) and every scratch
/// buffer used by [`MedusaHeads::project_all_heads_device`] live here
/// so the projection itself is zero-allocation.
#[derive(Debug)]
pub struct MedusaDeviceWeights {
    /// fp16 `[hidden_dim, hidden_dim]` pre-SiLU projection per head.
    pub w_in: [DeviceBuffer<u16>; NUM_MEDUSA_HEADS],
    /// fp16 `[hidden_dim, hidden_dim]` post-SiLU projection per head.
    pub w_out: [DeviceBuffer<u16>; NUM_MEDUSA_HEADS],
    /// fp16 `[hidden_dim]` — backbone hidden state uploaded once per
    /// projection cycle.
    pub h_fp16: DeviceBuffer<u16>,
    /// f32 `[hidden_dim]` — output of `fp16_gemv(W_in, h)`.
    pub inner_f32: DeviceBuffer<f32>,
    /// fp16 `[hidden_dim]` — cast of `inner_f32`, then in-place silu.
    pub inner_fp16: DeviceBuffer<u16>,
    /// fp16 `[hidden_dim]` — silu(inner_fp16). Same shape as inner_fp16
    /// but a separate buffer so `silu_glu_fp16` (which forbids aliased
    /// up/y) stays legal.
    pub silu_fp16: DeviceBuffer<u16>,
    /// f32 `[hidden_dim]` — output of `fp16_gemv(W_out, silu_fp16)`.
    pub out_f32: DeviceBuffer<f32>,
    /// fp16 `[hidden_dim]` — cast of `out_f32`, pre-residual.
    pub out_fp16: DeviceBuffer<u16>,
    /// fp16 `[hidden_dim]` — final h_out = h + out_fp16, copied back
    /// to host after each head.
    pub h_out_fp16: DeviceBuffer<u16>,
    /// fp16 `[hidden_dim]` — all-ones gate vector, written once at
    /// upload time. Lets us reuse the fused `silu_glu_fp16` kernel as
    /// a plain-silu by setting gate = 1.0 (there's no standalone SiLU
    /// kernel in `librocm_cpp.so` today).
    pub ones_fp16: DeviceBuffer<u16>,
}

/// The four heads, loaded together.
#[derive(Debug)]
pub struct MedusaHeads {
    /// Per-head metadata (path, expected acceptance).
    pub heads: [MedusaHead; NUM_MEDUSA_HEADS],
    /// Path the heads were loaded from. Kept for error messages.
    pub source_path: PathBuf,
    /// The mmapped source file. `None` only for the `#[cfg(test)]`
    /// path-bookkeeping stub.
    pub file: Option<Arc<MedusaHeadsFile>>,
    /// Device-resident weights + per-cycle scratch. `None` for the
    /// scaffold/stub; `Some` once [`Self::load`] has successfully
    /// uploaded to the current HIP device.
    pub device: Option<MedusaDeviceWeights>,
}

impl MedusaHeads {
    /// Real loader: mmap + parse a `.h1b-medusa` file, upload all 8
    /// head-weight matrices to device memory, and return a handle
    /// ready to project. Allocates ~216 MiB of device memory
    /// (8 × 25 MiB weights + scratch).
    pub fn load(path: &Path, _config: &MedusaConfig) -> Result<Self, MedusaError> {
        let file = Arc::new(MedusaHeadsFile::open(path)?);
        let device = upload_device_weights(&file).map_err(|e| {
            MedusaError::LoaderError(format!(
                "medusa head device upload failed ({}): {}",
                path.display(),
                e,
            ))
        })?;
        let heads = std::array::from_fn(|i| MedusaHead {
            index: i,
            weights_path: path.to_path_buf(),
            expected_acceptance: EXPECTED_ACCEPTANCE[i],
        });
        Ok(Self {
            heads,
            source_path: path.to_path_buf(),
            file: Some(file),
            device: Some(device),
        })
    }

    /// The heads in index order (t+1, t+2, t+3, t+4). Convenience
    /// accessor so the verifier doesn't reach into the array directly.
    pub fn as_slice(&self) -> &[MedusaHead; NUM_MEDUSA_HEADS] {
        &self.heads
    }

    /// Device-side projection of all four heads against one backbone
    /// hidden state. Returns `[h_out_0, h_out_1, h_out_2, h_out_3]` as
    /// host-owned `Vec<u16>` fp16 arrays, matching the call site's
    /// existing contract.
    ///
    /// Per head we run:
    ///
    /// ```text
    ///   inner_f32  = fp16_gemv(W_in,  h)             // [hd] f32
    ///   inner_f16  = fp32_to_fp16(inner_f32)         // [hd] f16
    ///   silu_f16   = silu_glu_fp16(inner_f16, ones)  // [hd] f16 == silu(inner)
    ///   out_f32    = fp16_gemv(W_out, silu_f16)      // [hd] f32
    ///   out_f16    = fp32_to_fp16(out_f32)
    ///   h_out_f16  = h + out_f16   (copy h into h_out, then residual_add)
    ///   D→H copy to owned Vec<u16>
    /// ```
    ///
    /// Heads run sequentially on the default stream — 4 × 6 = 24
    /// kernel launches + 4 D→H memcpys + 1 H→D memcpy per cycle. At
    /// 2560 × 2560 fp16 GEMV the math is ~13 MB of weight bandwidth
    /// per head, i.e. ~100 µs on gfx1151 LPDDR5, so wall time is
    /// dominated by launch overhead (~2 µs × 24 = 48 µs) + GEMV work
    /// (~100 µs × 2 × 4 = 800 µs) + two D↔H copies (~30 µs). Target:
    /// **< 5 ms** per cycle, vs 130 ms for the retired host path.
    pub fn project_all_heads_device(
        &mut self,
        hidden: &[u16],
    ) -> Result<[Vec<u16>; NUM_MEDUSA_HEADS], MedusaError> {
        if hidden.len() != MEDUSA_HIDDEN_DIM {
            return Err(MedusaError::BadInput(
                "medusa heads: hidden state length must equal MEDUSA_HIDDEN_DIM (2560)",
            ));
        }
        let dev = self.device.as_mut().ok_or(MedusaError::BadInput(
            "medusa heads: no device weights (was this a scaffold stub?)",
        ))?;

        let hd = MEDUSA_HIDDEN_DIM as i32;
        let stream = HipStream::DEFAULT;

        // Upload hidden ONCE for this cycle.
        dev.h_fp16
            .copy_from_slice(hidden)
            .map_err(|e| MedusaError::LoaderError(format!("upload hidden: {e:?}")))?;

        let mut out: [Vec<u16>; NUM_MEDUSA_HEADS] = std::array::from_fn(|_| Vec::new());

        // `i` indexes four parallel arrays (w_in, w_out, out, format strings);
        // iter/zip would be a readability regression for kernel-launch code.
        #[allow(clippy::needless_range_loop)]
        for i in 0..NUM_MEDUSA_HEADS {
            // 1) inner_f32 = W_in[i] · h
            onebit_hip::fp16_gemv(
                dev.w_in[i].as_device_ptr(),
                dev.h_fp16.as_device_ptr(),
                dev.inner_f32.as_device_mut_ptr(),
                hd,
                hd,
                stream,
            )
            .into_result()
            .map_err(|e| MedusaError::LoaderError(format!("fp16_gemv W_in[{i}]: {e:?}")))?;

            // 2) inner_fp16 = cast(inner_f32)
            onebit_hip::fp32_to_fp16(
                dev.inner_f32.as_device_ptr(),
                dev.inner_fp16.as_device_mut_ptr(),
                hd,
                stream,
            )
            .into_result()
            .map_err(|e| MedusaError::LoaderError(format!("fp32_to_fp16 inner[{i}]: {e:?}")))?;

            // 3) silu_fp16 = silu(inner_fp16) * 1.0  (ones gate).
            onebit_hip::silu_glu_fp16(
                dev.inner_fp16.as_device_ptr(),
                dev.ones_fp16.as_device_ptr(),
                dev.silu_fp16.as_device_mut_ptr(),
                hd,
                stream,
            )
            .into_result()
            .map_err(|e| MedusaError::LoaderError(format!("silu_glu[{i}]: {e:?}")))?;

            // 4) out_f32 = W_out[i] · silu_fp16
            onebit_hip::fp16_gemv(
                dev.w_out[i].as_device_ptr(),
                dev.silu_fp16.as_device_ptr(),
                dev.out_f32.as_device_mut_ptr(),
                hd,
                hd,
                stream,
            )
            .into_result()
            .map_err(|e| MedusaError::LoaderError(format!("fp16_gemv W_out[{i}]: {e:?}")))?;

            // 5) out_fp16 = cast(out_f32)
            onebit_hip::fp32_to_fp16(
                dev.out_f32.as_device_ptr(),
                dev.out_fp16.as_device_mut_ptr(),
                hd,
                stream,
            )
            .into_result()
            .map_err(|e| MedusaError::LoaderError(format!("fp32_to_fp16 out[{i}]: {e:?}")))?;

            // 6) h_out = h (seed), then residual_add(h_out, out_fp16).
            //    `residual_add_fp16(y, src, N)` does y[i] += src[i],
            //    so we D→D copy h_fp16 into h_out_fp16 first.
            dev.h_out_fp16
                .copy_from_device(&dev.h_fp16)
                .map_err(|e| MedusaError::LoaderError(format!("D→D copy h→h_out[{i}]: {e:?}")))?;
            onebit_hip::residual_add_fp16(
                dev.h_out_fp16.as_device_mut_ptr(),
                dev.out_fp16.as_device_ptr(),
                hd,
                stream,
            )
            .into_result()
            .map_err(|e| MedusaError::LoaderError(format!("residual_add[{i}]: {e:?}")))?;

            // 7) D→H copy into the owned host vector for this head.
            //    `copy_to_slice` uses blocking `hipMemcpy` on the null
            //    stream which implicitly serializes against every prior
            //    kernel on that stream, so no explicit sync is needed.
            let mut host = vec![0u16; MEDUSA_HIDDEN_DIM];
            dev.h_out_fp16
                .copy_to_slice(host.as_mut_slice())
                .map_err(|e| MedusaError::LoaderError(format!("D→H h_out[{i}]: {e:?}")))?;
            out[i] = host;
        }

        Ok(out)
    }
}

/// Upload all 8 weight buffers and pre-fill scratch from an open
/// [`MedusaHeadsFile`]. Returns the owned [`MedusaDeviceWeights`].
///
/// Allocation pattern:
///   * 8 × (hd²) fp16 weight buffers — 8 × ~13 MB = ~100 MB.
///   * 7 scratch buffers of length `hd` (fp16 or f32) — negligible.
///   * Ones gate populated from a stack-allocated source.
fn upload_device_weights(
    file: &MedusaHeadsFile,
) -> Result<MedusaDeviceWeights, onebit_hip::RcppError> {
    let hd = MEDUSA_HIDDEN_DIM;

    // Weights: alloc then copy_from_slice — copies the borrowed mmap
    // view into device memory once. No decoding, no unpacking; the
    // fp16 bit pattern is identical on disk and on device.
    let mut w_in: [Option<DeviceBuffer<u16>>; NUM_MEDUSA_HEADS] = [None, None, None, None];
    let mut w_out: [Option<DeviceBuffer<u16>>; NUM_MEDUSA_HEADS] = [None, None, None, None];

    for i in 0..NUM_MEDUSA_HEADS {
        let view = file
            .head(i)
            .map_err(|_| onebit_hip::RcppError::Precondition("medusa head view out of range"))?;
        let mut bw_in: DeviceBuffer<u16> = DeviceBuffer::alloc(hd * hd)?;
        bw_in.copy_from_slice(view.w_in)?;
        w_in[i] = Some(bw_in);

        let mut bw_out: DeviceBuffer<u16> = DeviceBuffer::alloc(hd * hd)?;
        bw_out.copy_from_slice(view.w_out)?;
        w_out[i] = Some(bw_out);
    }

    // Scratch buffers. Alloc zeroed to keep them deterministic before
    // the first kernel writes (helps debugging — reads-before-writes
    // show as zero, not random).
    let h_fp16 = DeviceBuffer::alloc_zeroed(hd)?;
    let inner_f32 = DeviceBuffer::alloc_zeroed(hd)?;
    let inner_fp16 = DeviceBuffer::alloc_zeroed(hd)?;
    let silu_fp16 = DeviceBuffer::alloc_zeroed(hd)?;
    let out_f32 = DeviceBuffer::alloc_zeroed(hd)?;
    let out_fp16 = DeviceBuffer::alloc_zeroed(hd)?;
    let h_out_fp16 = DeviceBuffer::alloc_zeroed(hd)?;

    // All-ones gate. fp16 1.0 = 0x3C00.
    let mut ones_fp16 = DeviceBuffer::alloc(hd)?;
    let ones_host = vec![half::f16::from_f32(1.0).to_bits(); hd];
    ones_fp16.copy_from_slice(&ones_host)?;

    // Option → array-of-buffers. `map` + `expect` is infallible here
    // because we wrote every slot in the loop above.
    let w_in: [DeviceBuffer<u16>; NUM_MEDUSA_HEADS] =
        w_in.map(|o| o.expect("w_in slot populated above"));
    let w_out: [DeviceBuffer<u16>; NUM_MEDUSA_HEADS] =
        w_out.map(|o| o.expect("w_out slot populated above"));

    Ok(MedusaDeviceWeights {
        w_in,
        w_out,
        h_fp16,
        inner_f32,
        inner_fp16,
        silu_fp16,
        out_f32,
        out_fp16,
        h_out_fp16,
        ones_fp16,
    })
}

#[cfg(test)]
mod tests {
    use super::super::loader::{
        MEDUSA_DTYPE_FP16, MEDUSA_FORMAT_VERSION, MEDUSA_HEADER_BYTES, MEDUSA_MAGIC,
        MEDUSA_RESIDUAL_LAYERS, MedusaHeadsFile,
    };
    use super::*;
    use std::io::Write;

    /// Shape smoke test: construct a synthetic `.h1b-medusa` handle and
    /// check that `MedusaHead` metadata comes out in the right order and
    /// the mmap-backed file view is populated.
    ///
    /// Does NOT upload to device — the host CI runners don't have a
    /// HIP device. Device-path coverage lives in the `#[ignore]`d
    /// `medusa_bench` integration test on the strixhalo box.
    #[test]
    fn head_metadata_order_matches_expected_acceptance() {
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&MEDUSA_MAGIC).unwrap();
        tmp.write_all(&MEDUSA_FORMAT_VERSION.to_le_bytes()).unwrap();
        tmp.write_all(&num_heads.to_le_bytes()).unwrap();
        tmp.write_all(&hidden_dim.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_RESIDUAL_LAYERS.to_le_bytes())
            .unwrap();
        tmp.write_all(&MEDUSA_DTYPE_FP16.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 40]).unwrap();
        let hd = hidden_dim as usize;
        let per_head = 2 * hd * hd * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;
        tmp.as_file_mut().set_len(total as u64).unwrap();
        let path = tmp.path().to_path_buf();

        // Scaffold handle: mmap the file but leave `device = None`.
        // `MedusaHeads::load` would attempt a device upload that fails
        // on hosts without ROCm, so we build the handle manually here
        // to keep the test host-portable.
        let file = MedusaHeadsFile::open(&path).expect("synthetic file opens");
        let heads = MedusaHeads {
            heads: std::array::from_fn(|i| MedusaHead {
                index: i,
                weights_path: path.clone(),
                expected_acceptance: EXPECTED_ACCEPTANCE[i],
            }),
            source_path: path,
            file: Some(Arc::new(file)),
            device: None,
        };

        for (i, h) in heads.as_slice().iter().enumerate() {
            assert_eq!(h.index, i);
            assert!((h.expected_acceptance - EXPECTED_ACCEPTANCE[i]).abs() < 1e-6);
        }
        assert!(heads.file.is_some());
        assert!(heads.device.is_none());
    }

    /// Bad-input guard: calling the device path without a device
    /// upload (e.g. on a scaffold stub) must return `BadInput`, not
    /// panic.
    #[test]
    fn project_without_device_weights_is_bad_input() {
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&MEDUSA_MAGIC).unwrap();
        tmp.write_all(&MEDUSA_FORMAT_VERSION.to_le_bytes()).unwrap();
        tmp.write_all(&num_heads.to_le_bytes()).unwrap();
        tmp.write_all(&hidden_dim.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_RESIDUAL_LAYERS.to_le_bytes())
            .unwrap();
        tmp.write_all(&MEDUSA_DTYPE_FP16.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 40]).unwrap();
        let hd = hidden_dim as usize;
        let per_head = 2 * hd * hd * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;
        tmp.as_file_mut().set_len(total as u64).unwrap();
        let path = tmp.path().to_path_buf();
        let file = MedusaHeadsFile::open(&path).expect("synthetic file opens");
        let mut heads = MedusaHeads {
            heads: std::array::from_fn(|i| MedusaHead {
                index: i,
                weights_path: path.clone(),
                expected_acceptance: EXPECTED_ACCEPTANCE[i],
            }),
            source_path: path,
            file: Some(Arc::new(file)),
            device: None,
        };

        let hidden = vec![0u16; MEDUSA_HIDDEN_DIM];
        let err = heads
            .project_all_heads_device(&hidden)
            .expect_err("no device weights must be BadInput");
        match err {
            MedusaError::BadInput(_) => {}
            other => panic!("expected BadInput, got {other:?}"),
        }
    }

    /// Wrong-length hidden must be rejected before any device call.
    #[test]
    fn project_rejects_wrong_length_hidden() {
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&MEDUSA_MAGIC).unwrap();
        tmp.write_all(&MEDUSA_FORMAT_VERSION.to_le_bytes()).unwrap();
        tmp.write_all(&num_heads.to_le_bytes()).unwrap();
        tmp.write_all(&hidden_dim.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_RESIDUAL_LAYERS.to_le_bytes())
            .unwrap();
        tmp.write_all(&MEDUSA_DTYPE_FP16.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 40]).unwrap();
        let hd = hidden_dim as usize;
        let per_head = 2 * hd * hd * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;
        tmp.as_file_mut().set_len(total as u64).unwrap();
        let path = tmp.path().to_path_buf();
        let file = MedusaHeadsFile::open(&path).expect("synthetic file opens");
        let mut heads = MedusaHeads {
            heads: std::array::from_fn(|i| MedusaHead {
                index: i,
                weights_path: path.clone(),
                expected_acceptance: EXPECTED_ACCEPTANCE[i],
            }),
            source_path: path,
            file: Some(Arc::new(file)),
            device: None,
        };

        // Off-by-one hidden.
        let bad = vec![0u16; MEDUSA_HIDDEN_DIM - 1];
        let err = heads
            .project_all_heads_device(&bad)
            .expect_err("short hidden must be rejected");
        match err {
            MedusaError::BadInput(_) => {}
            other => panic!("expected BadInput, got {other:?}"),
        }
    }
}
