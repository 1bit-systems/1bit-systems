# 1bit-image.cpp — our own diffusion stack

Status: design, 2026-04-23.

## Why fork

sd.cpp is an excellent general engine but four things hold halo-ai back on it today:

1. **Wan 2.2 TI2V-5B patch_embedding is 5D** (spatiotemporal). ggml's tensor dim ceiling is 4. Our build chokes on load. Upstream hasn't patched it.
2. **Zero ternary DiT support.** TerDiT / RobuQ papers exist, no shipped engine implements their weight scheme. Greenfield per `docs/research-digest-20260423.md`.
3. **No native HIP ternary dispatch.** Our `rocm-cpp/ternary_gemv_*.hip` kernels live next door but upstream sd.cpp uses ggml CUDA kernels via HIP compat — no ternary path.
4. **Same "kyuz0 containerizes it" problem** on the image lane as on the LLM lane. We stay alone on 1.58-bit + native HIP + multimodal if we fork.

## Scope for 0.2.x

### Stage 1 — fork + patch (1-2 wk)

- Rename `bong-water-water-bong/stable-diffusion.cpp` fork surface to `1bit-image.cpp` (file-level clone, not a fresh cut — preserve commit history + git log continuity).
- Land 5D patch_embedding loader in `model.cpp` (whitelisted tensor name, reshape-on-load, remember original rank for prefill math).
- Wire our `ternary_gemv_halo.hip` into `ggml-hip/` as a TQ2 path for DiT linear layers (same design as `docs/halo-hip-ternary-port.md` but applied to the diffusion transformer blocks rather than the LLM).
- Publish first release as a separate artifact in the `1bit install` registry:
  ```toml
  [component.image-engine]
  binary = "1bit-image-server"   # was sd-server
  ```
  with `sd-server` kept as a back-compat symlink.

### Stage 2 — ternary DiT retrain (Run 7, ~4 wk on 2× H200 NVL after Run 6)

- Pick base: Wan 2.2 TI2V-5B (Apache-2.0 weights) or FLUX.1-dev. 5B for first iteration.
- QAT recipe stacks:
  - **TerDiT stability fix** — RMSNorm after adaLN MLP (paper-proven).
  - **RobuQ W1.58A2** — Hadamard transform on activations for int2 quantization.
  - Keep VAE + text encoder FP16 (same split as BitTTS keeps codec FP16).
- Pod: 2× H200 NVL DDP. Budget ~$500 @ $3.39/hr × 75h.
- Deliverable: `bong-water-water-bong/1bit-image-wan22-ternary` HF repo with TQ2-encoded DiT + FP16 VAE / text enc.

### Stage 3 — native HIP ternary kernel (parallel with stage 2)

- Extend `ternary_gemv_halo.hip` variants:
  - Batch-dim tiling for diffusion workloads (DiT does `B × seq × d` not `1 × seq × d`).
  - Activation i2 path (matches RobuQ).
  - Scale quant layout matching the retrained weights.
- Benchmark on strixhalo: compare fp16 Wan 2.2 vs ternary Wan 2.2 on 512² / 768² / 5s clips.

### Stage 4 — 1bit-image.npu (stretch, after ship-gate closes)

- Author a tile-mapped ternary DiT kernel in MLIR-AIE for Strix Halo XDNA2 NPU.
- Same IRON + Peano + libxrt pattern as `bitnet_gemm`; reuse the kernel structure.
- Two layers deep: LLM ternary GEMV on NPU + diffusion ternary GEMM on NPU. First-public either way.

## Naming

- Repo: `bong-water-water-bong/1bit-image.cpp` (renamed fork)
- Workspace component: `[component.image-engine]` (unchanged; just re-binds the binary)
- Binary: `1bit-image-server` (was `sd-server`)
- Library: `libonebit-image.so` (was `libstable-diffusion.so`)
- Back-compat symlinks for `sd-server` / `sd-cli` / etc.

Rule A, B, C, E all hold. Rule B kernel work lives in rocm-cpp (no change). Rule E NPU lane uses IRON at compile-time (no change).

## Related

- `docs/halo-hip-ternary-port.md` — LLM ternary → ggml-hip port design
- `project_wan22_5d_blocker.md` — today's upstream gap (drives Stage 1)
- `project_npu_unlocked.md` — NPU lane status (Stage 4 enabler)
- `docs/research-digest-20260423.md` — ternary DiT greenfield evidence
