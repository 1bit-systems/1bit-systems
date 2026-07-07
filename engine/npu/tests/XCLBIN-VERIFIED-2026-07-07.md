# XCLBIN GEMM Verified Bit-Exact — Bug Is Host-Side

**Date:** 2026-07-07  **Verdict:** The deployed INT8 xclbins are **correct**.
The incoherent-output bug is **host-side** in `npu_engine_universal.cpp`, NOT an
"xclbin kernel precision issue." The current diagnosis (commit `042808434`
"Output still non-coherent due to xclbin kernel precision issues") is **wrong**.
This misdiagnosis is why 7+ rounds of fixes did not resolve coherence.

## What was verified

A new INT8-GEMM oracle (`tests/test_gemm_i32.cpp`) loads each deployed xclbin,
feeds deterministic int8 A+B patterns, runs the `MLIR_AIE` kernel, and compares
the int32 output to a CPU int8×int8→int32 reference GEMM.

**All 4 deployed Qwen3-0.6B xclbins PASS bit-exact (0 / 10000 errors, maxdiff 0):**

| xclbin | shape (M×K×N) | result |
|--------|-------------|--------|
| `final_i8_QKV_qwen3_0_6b.xclbin` | 128×1024×4096 | ✅ PASS 0/10000 |
| `final_i8_O_qwen3_0_6b.xclbin`   | 128×2048×1024 | ✅ PASS 0/10000 |
| `final_i8_GU_qwen3_0_6b.xclbin`  | 128×1024×6144 | ✅ PASS 0/10000 |
| `final_i8_D_qwen3_0_6b.xclbin`   | 128×3072×1024 | ✅ PASS 0/10000 |

xclbins: `/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_*_qwen3_0_6b.xclbin`
insts:   `/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_*_qwen3_0_6b.txt`

## Why the prior `test_gemm_i8` (int16) "failed"

`test_gemm_i8.cpp` reads the output BO as `int16_t` but the kernel writes **int32**
(see commit `386e40a07 fix: correct NPU output type from int16 to int32`). Reading
an int32 as two int16 produces the "scrambled with -1/0 on odd indices" pattern
that *looked* like a kernel bug but is purely a test-side type error. The engine
reads int32 correctly, so its GEMM results are correct.

## Reproduction

```bash
cd /home/bcloud/npu-sandbox/npu-infer
TC=/home/bcloud/torch2aie/toolchain
g++ -std=gnu++17 -O3 -o tools/test_gemm_i32 tools/test_gemm_i32.cpp \
    -I$TC/xrt/include -I/usr/include -L$TC/xrt/lib64 -lxrt_coreutil -luuid -lm
sudo LD_LIBRARY_PATH=$TC/xrt/lib64 ./tools/test_gemm_i32 \
    /home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_QKV_qwen3_0_6b.xclbin \
    /home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_QKV_qwen3_0_6b.txt 128 1024 4096
# → PASS — xclbin GEMM correct (int32), 0/10000 errors
```

## Confirmed-correct host elements (verified against config.json)

- dims: H=1024 NC=28 NH=16 NKV=8 HD=128 IM=3072 NV=151936 ✓
- rope_theta = 1000000 (matches engine default) ✓
- tie_word_embeddings = True → engine uses emb_f32 for lm_head ✓
- QKV split offsets: qkv_k_offset=NH*HD=2048, qkv_v_offset=3072, qkv_total=4096 ✓ (consistent with build_xclbins.sh QKV N=4096 and the transpose_pack concatenation order)
- GEMM quantize→rescale math in `I8Ctx::go()` / `packB()` is algebraically correct.

## Known host-side bug (separate from coherence, found during this session)

`ri(HD, cfg.rope_theta, 4096)` precomputes RoPE tables only up to **4096**
positions, but `config.json max_position_embeddings = 40960`. Latent bug for
long contexts; not the short-prompt coherence bug. Fix: raise the cap to
`max_position_embeddings` (read from config.json).

## Where the coherence bug actually lives (next steps)

The bug is in `npu_engine_universal.cpp`'s transformer math around a verified
GEMM. Highest-probability suspects, in order:

1. **q_norm / k_norm application order vs RoPE** — Qwen3 applies q_norm/k_norm
   (RMSNorm per head) to Q/K. Verify the order matches HF Qwen3: is RMSNorm
   applied over the full hidden dim or per-head_dim? Is it applied before or
   after RoPE? The engine applies RMSNorm-over-HD then RoPE. Confirm against
   `transformers` Qwen3 modeling.
2. **Attention / causal mask / GQA** in `attn_omp` — the v12 batched path's
   `max_pos = sp+b+1` was recently touched; verify single-token boot path is
   consistent.
3. **Residual add ordering** — `sb_data` save/restore around `rn_c` looks
   correct but is fragile; verify no aliasing.`h_b` and `sb_data` are separate buffers — OK.
4. **lm_head / sampling** — `lm_topk_omp` samples with `rand()` (non-deterministic,
   unseeded); for a correctness bisect, fix temperature=0 / argmax first.

**Definitive next diagnostic:** build a pure-float CPU forward of Qwen3-0.6B
(dequant all weights to f32, run the transformer in float with the SAME
RoPE/norm/attention code) and diff the engine's per-layer hidden state against
it. The first layer where the diff diverges pinpoints the bug. This was never
built — it is the missing oracle that the blocker doc's "standalone INT8
xclbins NOT re-verified" was a proxy for. We now know the xclbins are fine;
the CPU-float-vs-engine bisect is the real unlock.

## UPDATE 2026-07-07 (later): real bug #1 found + fixed — Q4NX unsigned nibbles

Building a CPU float reference (`tests/npu_engine_f32ref.cpp`, mirrors
`npu_engine_universal.cpp`'s transformer math verbatim but swaps the INT8-xclbin
GEMM for a float matmul on dequantized weights) revealed the **actual** root cause:

`src/dequant_q4nx.c` was converting the 4-bit nibbles to **signed** (`if val>=8:
val-=16`), but the Q4NX format uses **unsigned 0..15** nibbles. Verified bit-exact:
dequantizing with unsigned nibbles reproduces HF Qwen3-0.6B
`safetensors` q_proj L0 weights to within max-abs 0.18 (ratio mean 0.95), vs the
signed version at 2x mis-scaled (max diff 1.4, ratio mean -1.7).

With the signed→unsigned fix the f32ref's hidden-state norm **stops exploding**:
previously layer-0 hidden norm was 3.8 *million* (input embedding norm 0.38);
after the fix it is 14.8 (HF layer-0 hidden norm 8.4). Still diverges from HF
post-layer-0 (4/8 wrong signs at h[0..7]) → there is a **second bug** in the
layer-0 attention/FFN/residual path that the f32ref-vs-HF bisect has not yet
isolated. Both bugs are host-side; both are reproduced in the f32ref (which
shared the dequant). **The native NPU engine rebuilt with just the unsigned fix
is still incoherent** (first decode token 10185, garbage), confirming the second
bug needs fixing too.

Diff against HF (4.57 GB torch, transformers) via `tests/hf_compare.py` flow.
HF post-layer-0 last-token h[0..7] =
`0.4501 -0.1092 0.0842 -1.6524 0.1054 0.204 0.0688 -0.7409`
f32ref post-layer-0            h[0..7] =
`-0.4528 0.3035 1.2203 -0.9953 0.1854 -0.5901 -0.3046 -0.1363`

### Next-step suspects for bug #2 (layer-0 attention path)
1. **QKV concat layout** — f32ref/universal pack Q/K/V into a fused buffer
   `[Q | K | V]` per token; offsets `qkv_k_offset=2048`, `qkv_v_offset=3072`.
   verify against HF which keeps `q_proj`, `k_proj`, `v_proj` separate.
2. **Attention scaling** — f32ref's `attn_omp` divides scores by `sqrt(HD)`;
   HF Qwen3 uses `self.scaling = head_dim**-0.5` applied identically — OK.
3. **RoPE convention** — `ra()` uses half-split `rotate_half`. Verify HF's
   `apply_rotary_pos_emb` matches the element-pairing exactly.
4. **Residual add sign / save-restore** — `sb_data` save before `rn_c`.
5. **Causal mask in prefill** — `sp+pi+1`.

### Artifacts
- `tests/npu_engine_f32ref.cpp` + binary — float oracle (mirrors universal's math).
- `src/dequant_q4nx.c` — FIXED (unsigned nibbles). Affects ALL engines that
  call `dequant_i8_to_float_ex`; rebuild with `cd engine/npu && gcc -c -O3 -o
  build/dequant_q4nx.o src/dequant_q4nx.c && bash build_npu.sh`.

## Throughput reality check

A live chat run of `npu_engine_universal` on Qwen3-0.6B ("What is 2+2?") measured
**~187 ms/tok = 5 tok/s and crashed (`free(): invalid size`)**. The "291 tok/s"
figure is a raw batched-GEMM benchmark, not chat-mode throughput. Reaching the
"273 tok/s coherent" target therefore requires (a) fixing the host coherence
bug, (b) fixing the crash, and (c) closing the 5→273 tok/s chat-mode gap. The
xclbin rebuild path is NOT the blocker.