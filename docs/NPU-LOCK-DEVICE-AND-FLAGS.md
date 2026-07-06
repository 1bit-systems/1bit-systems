# NPU Lock Device & Complete Flags Reference

**Generated:** 2026-07-06 (updated for fused layer engine at 291 tok/s)  
**Scope:** All NPU, GPU (ZINC), fused engine, and tooling flags, switches, env vars, and locking layers.  
**Primary engine:** Fused layer engine at 291 tok/s (3.4 ms/tok, 38 KB). C++ v12 (97 tok/s) is fallback.

---

## 1. NPU Lock Device — 6-Layer Gating Stack

The "NPU lock device" is not a single lock — it is a multi-layered gating stack
controlling access to the AMD XDNA2 NPU (Ryzen AI Max+ 395 on Strix Halo).

### Layer Θ: Firmware Lock (AGESA/BIOS)

**What:** AMD ships the XDNA2 NPU locked on consumer Strix Halo silicon.
AGESA firmware returns a gated status that prevents driver access.

**Bypass:** SmokelessRuntimeEFIPatcher (SREP) v0.20 — UEFI runtime patcher  
**Mechanism:** Single `FastPatch` flips the lockdown check:

| Hex Before | Hex After  | Meaning               |
|------------|------------|-----------------------|
| `32 C0`    | `B0 01`    | `xor eax,eax`→`mov al,1` |
| (rest unchanged) | | Epilogue + ret |

**Files (USB, FAT32):**
- `BOOTX64.efi` — EFI bootloader
- `SmokelessRuntimeEFIPatcher(020).efi` — SREP v0.20
- `Oniguruma.efi` — Regex library
- `SREP_Config.cfg` — Patch config (one FastPatch directive)
- `startup.nsh` — UEFI shell boot script

**After unlock:** `xrt-smi examine` shows `RyzenAI-npu5` at PCIe `0000:c6:00.1`

### Layer 1: Firmware Column Limit

**Lock:** NPU firmware `npu.sbin.1.1.2.65` only permits **8 of 40** physical AIE columns
**Where:** Encrypted ARM64 text section within `npu.sbin` (RSA-4096 signed) — not user-patchable
**Symptoms:**
- Kernel driver allows 40 columns (`aie2_max_col=128` in sysfs)
- At `MSG_OP_CREATE_CONTEXT`, firmware validates `num_col` against its own limit and rejects >8
- Caps NPU at ~31 TFLOPS BFP16 vs 50 TOPS INT8 theoretical
**Status:** BLOCKED — requires AMD firmware update (alternative: older fw `1.0.0.166` has no column validation but lacks features)

### Layer 2: XRT Device Lock

**Device node:** `/dev/accel/accel0` (char 261:0, `root:render`, user `bcloud` has `rw-`)
**XRT version:** 2.21.75 (userspace library `libxrt_coreutil`)
**Kernel driver:** `amdxdna.ko` v7.0.0-27-generic
**NPU firmware:** `npu.sbin.1.1.2.65` (430KB, at `/lib/firmware/amdnpu/17f0_11/`)

**XRT API sequence:**
```cpp
xrt::device dev(0);                        // xrtDeviceOpen — opens /dev/accel/accel0
dev.register_xclbin(xclbin);               // xrtDeviceLoadXclbinHandle — loads xclbin
xrt::hw_context hc(dev, uuid);             // xrtHwContextCreate — firmware validates column count
xrt::kernel k(hc, "MLIR_AIE");             // xrtKernelOpen — opens named kernel
xrt::bo bo(dev, size, flags, group);       // xrtBOAlloc — allocates buffer objects
bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);         // xrtBOSync — DMA to/from device
auto run = k(3, instr_bo, count, a, w, c); // xrtKernelRun — launch kernel
run.wait();                                 // xrtRunWait — wait for completion
```

**Zig wrapper:** `engine/npu/src/xrt.zig` — `XrtDevice`, `XrtBuffer`, `XrtKernel`, `XrtRun`, `XrtHwContext`

### Layer 3: Engine Init Lock (Zig/C++)

Each inference engine locks the NPU at init:
- **Fused layer engine** (`npu_engine_fused.cpp`): `xrt::device dev(0)`, one kernel context, one xclbin dispatch per layer
- **Zig engine** (`main.zig`): `XrtDevice.open(0)` during `NpuEngine.init()`
- **C++ engines (v12)**: `xrt::device dev(0)` at startup, 4 kernel contexts
- **Fused layer**: 38 KB binary, single xclbin call per layer (QKV→attention→O→GU→SiLU→D on NPU), no CPU attention
- **Legacy (v12)**: 4 kernel contexts per engine (QKV, O, GU/G, D), 112 xclbin dispatches per batch

### Layer 4: Cross-Process GPU Lock

**File:** `engine/gpu/src/gpu/process_lock.zig`  
**Mechanism:** Filesystem advisory lock  
**Path pattern:** `/tmp/zinc-gpu-{backend}-{device_index}.lock`  
**Backends:** `vulkan`, `metal` (enum `Backend`)  
**Mode:** Non-blocking exclusive (`lock_nonblocking = true`)  
**Error:** `GpuAlreadyReserved` if another process holds the lock  
**Used by:** `engine/gpu/src/main.zig`, `model_manager.zig`, `model_manager_metal.zig`

### Layer 5: Thread-Safety Mutexes (LD_PRELOAD Hooks)

Three LD_PRELOAD libraries capture FLM's xrt::bo allocations. All use `std::mutex`:

| File | Mutex | Protected Data |
|------|-------|---------------|
| `engine/npu/src/hook_bo.cpp` | `static std::mutex mtx` | `captured_bos` vector, `seen_impls` set |
| `engine/npu/src/flm_bo_capture.cpp` | `static std::mutex mtx` | `g_weight_bos` list |
| `engine/npu/src/hook_xrt_bo.cpp` | `static std::mutex bo_mutex` | `captured_bos` vector |

### Layer 6: Vulkan Device Mutexes (ggml-rocm backend)

In `zaya-llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
- `device->compile_mutex` (`std::mutex`) — shader compilation serialization
- `device->mutex` (`std::recursive_mutex`) — device-wide operations
- `device->pinned_memory_mutex` (`std::shared_mutex`) — pinned staging buffer access
- `device->compile_cv` (`std::condition_variable`) — compile wait notification

---

## 2. NPU Engine — Environment Variables

| Variable | Default | Used In | Purpose |
|----------|---------|---------|---------|
| `MODEL_PATH` | `~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx` | `main.zig` | Path to Q4NX model file |
| `MODEL_TAG` | auto-detected from path | `main.zig` | Model architecture tag (e.g. `qwen3_0_6b`) |
| `XCLBIN_DIR` | `/home/bcloud/npu-sandbox/npu-infer/build/int8` | `main.zig` | Directory with xclbin + insts files |
| `TOKENS` | `32` | `main.zig` | Max tokens to generate |
| `KV_PAGES` | `1024` | `main.zig` | Total KV cache pages |
| `NPU_MODEL_PATH` | engine default paths | `q4nx_stream.cpp`, test files | Q4NX model (C++ engines) |
| `NPU_XCLBIN_DIR` | `/home/bcloud/npu-sandbox/npu-infer/build/int8` | `npu_engine_mt.cpp`, `npu_engine_all.cpp`, `npu_engine_spec_v3.cpp`, `bench_gemm.cpp`, `q4nx_stream.cpp` | XCLBIN dir override |
| `NPU_GEN` | 0 (off) | `npu_engine_mt.cpp` | Auto-regressive decode count after prefill (0-64) |
| `NPU_TEMP` | 0.8 | `npu_engine_mt.cpp` | Sampling temperature for decode |
| `NPU_SPEC` | 0 | `npu_engine_spec_v3.cpp` | Speculative decode block size (0=off, 1-10) |
| `NPU_FEAT_OUT` | (none) | `npu_engine_spec_v3.cpp` | Feature dump output path |
| `FUSED_XCLBIN_DIR` | `/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127` | `npu_engine_fused.cpp` | Fused layer xclbin + instruction file directory |
| `FUSED_WEIGHTS_DIR` | `/home/bcloud/npu-sandbox/npu-infer/build/int8` | `npu_engine_fused.cpp` | Fused weight binary files directory |
| `FLM_MODEL` | `qwen3:0.6b` | `npu-gpu-cpud.py` | FLM model name for daemon proxy |
| `FLM_BIN` | `/usr/bin/flm` | `npu-gpu-cpud.py` | FLM binary path for daemon proxy |

---

## 3. NPU Engine — Command-Line Arguments

| Engine Binary | Syntax | Notes |
|---------------|--------|-------|
| `npu_engine_cb` | `<npt>` `<ng>` | Prefill tokens (1-9), decode steps |
| `npu_engine_mt` | `<model.q4nx>` `<token1>` `<token2>` `...` | Model + up to 256 token IDs |
| `npu_engine_all` | `<model.q4nx>` `[decode_tokens]` | Model + optional decode count |
| `npu_engine_spec_v3` | `<model.q4nx>` `[decode_tokens]` | Same; `NPU_SPEC` env enables speculative |
| `npu_engine_fused` | `<model.q4nx>` `[decode_tokens]` | Fused layer engine (291 tok/s). Uses `FUSED_XCLBIN_DIR` + `FUSED_WEIGHTS_DIR` env vars |
| `wan_engine` | `<weights.bin>` `<meta.json>` `[layer=0]` | WAN 2.1 video model test |
| `q4nx_stream` | `<model.q4nx>` `[output_dir]` | Weight stream packer |
| `bench_gemm` | (none) | Uses `NPU_XCLBIN_DIR` env |
| `detokenize` | `<tokenizer.json>` | Decode token IDs from stdin |
| `tokenize` | `<tokenizer.json>` | Encode text to token IDs (stdin) |
| `gguf_to_q4nx` | `<input.gguf>` `<output.q4nx>` | GGUF → Q4NX conversion |
| `bitnet_to_q4nx` | `<input.gguf>` `<output.q4nx>` | BitNet GGUF → Q4NX conversion |
| `zig engine` (main.zig) | (env vars only) | `MODEL_PATH`, `XCLBIN_DIR`, `TOKENS`, `KV_PAGES`, `MODEL_TAG` |

---

## 4. Fused Engine — CLI & Dispatch Policies

### Command-Line Flags (`engine/fusion/main.zig`)

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--policy` | see below | `auto` | Dispatch policy |
| `-n`, `--max-tokens` | integer | `64` | Max tokens to generate |
| `--list-policies` | flag | off | List all dispatch policies |
| `-h`, `--help` | flag | off | Show help |

### Dispatch Policies (`engine/fusion/dispatcher.zig`)

| Policy | Attention | FFN | QKV | Description |
|--------|-----------|-----|-----|-------------|
| `npu_only` | NPU | NPU | NPU | All on NPU (no GPU) |
| `gpu_only` | GPU | GPU | GPU | All on GPU (no NPU) |
| `layer_by_layer` | even→NPU, odd→GPU | same | same | Round-robin per layer |
| `attention_on_npu` | **NPU** | GPU | GPU | NPU edge_attention |
| `ffn_on_npu` | GPU | **NPU** | GPU | NPU INT8 GEMM for FFN |
| `qkv_on_npu` | GPU | GPU | **NPU** | NPU for QKV only |
| `prefill_npu_decode_gpu` | GPU(decode) | GPU | NPU(prefill) | Prefill on NPU, decode on GPU |
| `auto` | GPU | **NPU** | **NPU** | Default auto-tuned |

### Fused Engine Defaults

- **Model:** same as NPU engine (`MODEL_PATH` env)
- **XCLBIN dir:** same (`XCLBIN_DIR` env)
- **KV pages:** 1024
- **Max parallel:** 4
- **Port:** 8080 (HTTP), proxies to FLM at 127.0.0.1:52625

---

## 5. GPU Engine (ZINC) — Command-Line Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-m`, `--model` | path | — | GGUF model path |
| `-d`, `--device` | u32 | auto | Vulkan/CUDA device index |
| `--shader-dir` | path | `zig-out/share/zinc/shaders` | Shader directory |
| `--iterations` | u32 | 200 | Timed iterations per benchmark case |
| `--warmup` | u32 | 25 | Warmup iterations per case |
| `--working-set` | u32 | 16 | Buffer set rotation count |
| `--case` | string | all | Filter benchmark case |
| `--json` | flag | off | JSON output |
| `-h`, `--help` | flag | off | Show help |

---

## 6. GPU Engine (ZINC) — Environment Variables

### Kernel Backend Selection
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_BATCHED_TC` | bool toggle | Enable tensor core path (default: on) |
| `ZINC_BATCHED_TC_PLAIN` | flag | Use plain TC, no Q6 |
| `ZINC_BATCHED_TC_NOQ6` | flag | Disable Q6 TC variant |
| `ZINC_BATCHED_TC_M128` | flag | Enable M=128 TC variant |
| `ZINC_BATCHED_TC_M64` | flag | Enable M=64 TC variant |
| `ZINC_BATCHED_TC_SHAREA` | flag | Enable shared-A TC variant |
| `ZINC_BATCHED_TC_NORMF16` | flag | Enable f16 norm TC variant |
| `ZINC_BATCHED_TC_Q6_LOWSMEM` | flag | Low-smem Q6 TC variant |
| `ZINC_BATCHED_TC_M128_LOWSMEM` | flag | Low-smem M128 TC variant |
| `ZINC_BATCHED_CUBLAS` | bool toggle | Enable cuBLAS (default: on) |
| `ZINC_BATCHED_CUBLAS_NOQ5` | flag | Disable Q5 cuBLAS |
| `ZINC_BATCHED_CUBLAS_NOQ6` | flag | Disable Q6 cuBLAS |
| `ZINC_BATCHED_CUBLAS_NOQ8` | flag | Disable Q8 cuBLAS |
| `ZINC_BATCH_B1_MATVEC` | bool toggle | Batch-1 matvec (default: on) |
| `ZINC_BATCH_MROW` | bool toggle | M-row batching (default: on) |
| `ZINC_BATCH_MOE_SHARED` | bool toggle | Batched MoE shared (default: on) |
| `ZINC_BATCHED_EXPERTS_GROUPED` | flag | Grouped expert dispatch |
| `ZINC_CUBLAS_MIN_T` | u32 | Min tokens for cuBLAS fallback |
| `ZINC_SERVE_CUBLAS` | bool toggle | cuBLAS in serving (default: on) |
| `ZINC_SERVE_CUBLAS_MINB` | u32 | Min batch for serving cuBLAS |
| `ZINC_SERVE_WCACHE` | bool toggle | Weight cache in serving (default: off) |
| `ZINC_SERVE_WCACHE_MB` | u32 | Weight cache size in MB |
| `ZINC_BATCH_GRAPH` | string | CUDA graph mode for batched ops |
| `ZINC_CUDA_GRAPH` | string | CUDA graph mode for non-batched |
| `ZINC_DP4A_WAVE32` | string | Force DP4A wave32 mode |

### Prefill Variants
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_PREFILL_TC` | bool toggle | Enable TC prefill (default: on) |
| `ZINC_PREFILL_Q8` | flag | Force Q8 prefill |
| `ZINC_PREFILL_F16` | flag | Force f16 prefill |
| `ZINC_PREFILL_DP4A` | flag | Force DP4A prefill |
| `ZINC_PREFILL_LOWSMEM` | bool toggle | Low-smem prefill (default: on) |
| `ZINC_PREFILL_Q8_TC` | bool toggle | Q8 TC prefill (default: on) |
| `ZINC_PREFILL_GATE_UP_SWIGLU` | bool toggle | Fused gate-up+SiLU in prefill (default: off) |
| `ZINC_PREFILL_SKIP` | flag | Skip prefill (debug) |
| `ZINC_PREFILL_PROFILE` | flag | Profile prefill timings |
| `ZINC_PRE_DEQUANT` | flag | Pre-dequantize weights |
| `ZINC_BATCHED_PREFILL` | string | Batched prefill mode string |
| `ZINC_INTEL_BATCHED_PREFILL` | string | Intel batched prefill mode |
| `ZINC_INTEL_BATCHED_PREFILL_CHUNK` | u32 | Intel batched prefill chunk (default: 96) |

### MoE Router & Expert Paths
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_MOE_TC` | bool toggle | Enable TC MoE (default: on for decode, off for prefill) |
| `ZINC_MOE_DOWN_TC` | bool toggle | TC MoE down-proj (default: on) |
| `ZINC_MOE_DOWN_Q6K_TC` | bool toggle | Q6K TC MoE down (default: on) |
| `ZINC_MOE_KPAR` | u32 | MoE K-dimension parallelism |
| `ZINC_MOE_Q5K_KPAR` | u32 | Q5K MoE K-dim parallelism |
| `ZINC_MOE_FUSED_GATE_UP` | bool toggle | Fused gate+up (default: on) |
| `ZINC_MOE_FUSED_GATE_UP_SWIGLU` | bool toggle | Fused gate+up+SiLU |
| `ZINC_FUSE_MOE_DOWN_ACC` | bool toggle | Fuse MoE down accumulator |
| `ZINC_MOE_NORM_COMBINE` | flag | Fuse norm with combine |
| `ZINC_ATTN_MOE_NORM` | flag | Fuse attention + MoE norm |
| `ZINC_Q4K_BATCH_KPAR` | u32 | Q4K batch K-parallelism |
| `ZINC_GEMMA_DENSE_DECODE_DP4A` | flag | DP4A for Gemma dense decode |
| `ZINC_GEMMA_MOE_GROUPED_PREFILL` | bool toggle | Grouped MoE prefill (default: on) |
| `ZINC_GEMMA_MOE_TOPK` | u32 | Gemma MoE top-K override |
| `ZINC_GEMMA_MOE_PREFILL_TOPK` | u32 | Gemma MoE prefill top-K |
| `ZINC_QWEN36_MOE_TOPK` | u32 | Qwen3.6 MoE top-K override |
| `ZINC_QWEN36_MOE_PREFILL_TOPK` | u32 | Qwen3.6 MoE prefill top-K |
| `ZINC_QWEN36_MOE_PREFILL_TOPK_GUARD` | u32 | Guard value for prefill top-K |
| `ZINC_FUSED_RMS_ROUTER` | flag | Fuse RMSNorm + router |
| `ZINC_FUSED_SSM_AB` | flag | Fuse SSM A/B matmuls |
| `ZINC_FUSED_SSM_QKV_Z` | flag | Fuse SSM QKV+Z projections |
| `ZINC_FUSED_DENSE_FFN` | flag | Fuse dense FFN layers |
| `ZINC_FUSED_OPROJ_MERGE` | flag | Fuse O-proj merge |
| `ZINC_FUSED_QK_KV` | flag | Fuse QK and KV steps |
| `ZINC_COUNT_EXPERTS_PREFILL` | flag | Count active experts per layer |
| `ZINC_TOPK_V1` | flag | Use v1 top-k kernel |

### Model-Specific Tuning (Qwen)
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_QWEN_DENSE_FFN_DP4A` | string | DP4A mode for Qwen dense FFN |
| `ZINC_QWEN_MOE_BATCHED` | bool toggle | Batched MoE for Qwen (default: on) |
| `ZINC_QWEN36_27B_BM64_DOWN_ACC` | bool toggle | BM64 down accumulator (default: on) |
| `ZINC_QWEN36_27B_MMQ64_DOWN` | bool toggle | MMQ64 down (default: off) |
| `ZINC_QWEN36_27B_FUSE_RMS_QUANT` | string | Fuse RMSNorm+quantize mode |
| `ZINC_QWEN36_27B_FULL_ATTN_BATCHED` | string | Full attention batching mode |
| `ZINC_QWEN36_27B_SSM_PREFILL_PROJ` | string | SSM prefill projection mode |
| `ZINC_QWEN36_27B_SSM_BATCHED_DELTA` | string | SSM batched delta mode |
| `ZINC_QWEN36_27B_DENSE_PREFILL` | string | Dense prefill mode |
| `ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS` | u32 | Dense prefill layer count |
| `ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT` | u32 | Dense prefill segment size |
| `ZINC_QWEN36_27B_PREFIX_TAIL_PIPELINE` | string | Prefix-tail pipeline mode |
| `ZINC_QWEN36_27B_BATCH_FUSED_GATEUP` | flag | Batched fused gate-up |
| `ZINC_QWEN36_27B_Q6_DOWN_MUL_MM` | flag | Q6 down with mul+mm |
| `ZINC_QWEN36_27B_Q5_SSM_OUT_MUL_MM` | flag | Q5 SSM out with mul+mm |
| `ZINC_QWEN36_27B_DP4A_DOWN` | string | DP4A down-proj mode |
| `ZINC_QWEN36_27B_PREFILL_VALIDATE` | flag | Validate dense prefill |
| `ZINC_QWEN36_27B_PREFILL_VALIDATE_TOKENS` | u32 | Validation token count |
| `ZINC_QWEN36_27B_PREFILL_VALIDATE_LAYER` | u32 | Validation layer filter |
| `ZINC_QWEN36_27B_DENSE_FUSED_ROW1` | flag | Fuse row 1 of dense |

### Model-Specific Tuning (Gemma)
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_GEMMA_BATCHED_PREFILL` | string | Gemma batched prefill mode |
| `ZINC_GEMMA4_ATTN_SCALE_DEFAULT` | flag | Override attn scale for Gemma4 |

### SSM (State Space Model) Control
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_SSM_PROFILE` | flag | Profile SSM timing |
| `ZINC_SSM_CHUNKED` | string | Chunked SSM mode (=="1" to enable) |
| `ZINC_SSM_WARP` | bool toggle | Warp-level SSM (default: on) |
| `ZINC_SSM_DELTA_COLS8` | flag | SSM delta with cols=8 |
| `ZINC_SSM_DELTA_NORMED_QK` | flag | SSM delta with normed QK |
| `ZINC_FUSED_SSM_AB` | flag | Fuse SSM A/B matmuls |
| `ZINC_FUSED_SSM_QKV_Z` | flag | Fuse SSM QKV+Z |
| `ZINC_QWEN36_27B_SSM_PREFILL_PROJ` | string | SSM prefill projection mode |
| `ZINC_QWEN36_27B_SSM_BATCHED_DELTA` | string | SSM batched delta mode |

### Attention Control
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_FA_SPLIT_K` | string | Flash attention split-K |
| `ZINC_FA_PROFILE_LAYER` | u32 | Profile FA at specific layer |
| `ZINC_BATCH_ATTN` | string | Batched attention mode |
| `ZINC_FULL_ATTN_INTERVAL` | u32 | Full attention interval |
| `ZINC_FUSED_QK_KV` | flag | Fuse QK + KV steps |
| `ZINC_FUSED_OPROJ_MERGE` | flag | Fuse O-proj merge |

### LM Head Control
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_MUL_MM_LM_HEAD` | flag | Use mul+mm for LM head |
| `ZINC_MUL_MM_PROJ` | flag | Use mul+mm for projections |
| `ZINC_Q8_WIDE_LM_HEAD` | flag | Q8 wide LM head kernel |
| `ZINC_Q8_BATCH_LM_HEAD` | flag | Q8 batched LM head |
| `ZINC_Q8_1_LM_HEAD` | flag | Q8_1 LM head kernel |
| `ZINC_Q4K_LM_HEAD_DP4A` | flag | Q4K DP4A LM head |
| `ZINC_Q8_SPEC_DMMV` | flag | Q8 speculative DMMV |

### Metal-Specific
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_METAL_Q8_TG_SIZE` | u32 | Metal Q8 threadgroup size override |
| `ZINC_METAL_QWEN_SSM_REPACKED_Q8_QUAD` | bool toggle | Repacked Q8 quad SSM (default: on) |
| `ZINC_METAL_QWEN_SSM_REPACKED_Q8_PRIVATE` | bool toggle | Repacked Q8 private SSM (default: on) |
| `ZINC_METAL_QWEN_ROUTER_REPACKED_Q8_PRIVATE` | bool toggle | Repacked Q8 private router (default: on) |
| `ZINC_METAL_GEMMA_Q4K_GEGLU_VALIDATE` | flag | Validate Gemma Q4K GEGLU |
| `ZINC_METAL_GEMMA_Q4K_GEGLU_VALIDATE_SCAN` | flag | Scan validation only |
| `ZINC_METAL_GEMMA_Q4K_GEGLU_VALIDATE_LAYER` | u32 | Validate specific layer |
| `ZINC_METAL_GEMMA_Q4K_GEGLU_VALIDATE_TOKENS` | u32 | Validate token count |
| `ZINC_METAL_GEMMA_Q4K_GEGLU_PROFILE_SCAN` | flag | Profile scan ops |
| `ZINC_METAL_QWEN27B_Q4K_SWIGLU_VALIDATE_LAYER` | u32 | Qwen 27B validate layer |
| `ZINC_METAL_QWEN36_27B_Q4K_SWIGLU_VALIDATE_LAYER` | u32 | Qwen3.6 27B validate layer |
| `ZINC_METAL_QWEN27B_Q4K_SWIGLU_VALIDATE_TOKEN` | u32 | Qwen 27B validate token |
| `ZINC_METAL_QWEN36_27B_Q4K_SWIGLU_VALIDATE_TOKEN` | u32 | Qwen3.6 27B validate token |

### Zinc RT (Runtime Engine)
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_RT_TIER` | string | Runtime tier override |
| `ZINC_RT_MAX_DECODE_TOKENS` | u32 | Max decode tokens limit |
| `ZINC_RT_LM_HEAD_ROWS` | u32 | LM head rows override |
| `ZINC_RT_CPU_WORKERS` | u32 | CPU worker thread count |
| `ZINC_RT_FAST_POOL` | bool toggle | Enable fast pool (default: on) |
| `ZINC_RT_DIRECT_DECODE_FULL_SLICE` | string | Direct decode full slice |
| `ZINC_RT_DIRECT_DECODE_SLICE_CADENCE` | string | Direct decode slice cadence |
| `ZINC_RT_DIRECT_PREFILL_MODEL_SLICE` | string | Direct prefill model slice |
| `ZINC_RT_DIRECT_LM_HEAD_DECODE_CADENCE` | string | Direct LM head decode cadence |
| `ZINC_RT_DIRECT_ROUTER_DECODE` | string | Direct router decode |
| `ZINC_RT_DIRECT_ROUTER_DECODE_CADENCE` | string | Direct router decode cadence |
| `ZINC_RT_DIRECT_ROUTER_TRUST_AFTER_SUCCESSES` | u32 | Router trust threshold |
| `ZINC_RT_DIRECT_SSM_Q8_ROW_RANGE_MAX_SUCCESSES` | u32 | SSM Q8 success threshold |
| `ZINC_RT_DIRECT_SSM_Q8_TRUST_AFTER_SUCCESSES` | u32 | SSM Q8 trust threshold |

### Debug / Diagnostics
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_DEBUG` | flag | Enable debug mode |
| `ZINC_LAYER_DIAG` | flag | Layer diagnostics on |
| `ZINC_MOE_DEBUG` | flag | MoE debug output (layers 0,3,39) |
| `ZINC_CAPTURE_ROUTING` | flag | Capture routing decisions |
| `ZINC_CAPTURE_FFN_INPUT` | flag | Capture FFN input tensors |
| `ZINC_A3B_VALIDATE` | flag | Validate A3B outputs |
| `ZINC_A3B_PRODUCTION` | flag | Production A3B mode |
| `ZINC_OFFLOAD_MOE_EXPERTS` | string | MoE expert offload policy |

### KV Cache Control
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_KV_EVICTION_POLICY` | string | Eviction policy: `h2o_attention_score`, `lru`, `fifo` |
| `ZINC_SCHED_EOS` | string | Scheduler EOS token override |
| `ZINC_FULL_ATTN_INTERVAL` | u32 | Full attention recompute interval |

### Wide GEMM / Quantized
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_WIDE_GEMM` | flag | Enable wide GEMM kernel |
| `ZINC_Q4K_Q8_1_DMMV` | flag | Q4K Q8_1 DMMV |
| `ZINC_Q8_SPEC_DMMV` | flag | Q8 speculative DMMV |

### Display / Terminal
| Variable | Type | Effect |
|----------|------|--------|
| `NO_COLOR` | flag | Disable color output |
| `FORCE_COLOR` | flag | Force color output |
| `CLICOLOR_FORCE` | flag | Force ANSI color |
| `TERM` | string | Terminal type for color detection |
| `RADV_PERFTEST` | string | Vulkan RADV perf test flags |

### Tool Calling
| Variable | Type | Effect |
|----------|------|--------|
| `ZINC_TOOL_CALLING` | string | Tool calling mode override |

---

## 7. NPU XCLBIN Generator Flags (Python — MLIR)

**File:** `engine/npu/xclbins/n1_core_i8_v2.py` (argparse-based)

| Flag | Type | Description |
|------|------|-------------|
| `--md` | int | M dimension (tile rows, e.g. 128) |
| `--kd` | int | K dimension (in features) |
| `--nd` | int | N dimension (out features) |
| `--output` | str | Output xclbin prefix |
| `--out-dtype` | str | Output dtype (`i16`, `i32`) |

Same flags for `n1_core_i8_wan_q.py` (WAN video model variant).

---

## 8. Model Config Fields (`model_config.h` / `model_reader.zig`)

### `ModelConfig` (C++ struct)

| Field | Type | Description |
|-------|------|-------------|
| `H` | int | Hidden dimension |
| `NC` | int | Number of transformer layers |
| `NH` | int | Number of attention heads |
| `NKV` | int | Number of KV heads |
| `HD` | int | Head dimension (typically 128) |
| `IM` | int | FFN intermediate dimension |
| `NV` | int | Vocabulary size |
| `GQA` | int | Grouped-query attention ratio (NH/NKV) |
| `AW` | int | Attention windows (default: 4) |
| `WQH` | int | Working query heads (NH/AW) |
| `WKVH` | int | Working KV heads (NKV/AW) |
| `XM` | int | Max tile dimension for NPU GEMM (default: 128) |
| `qkv_k_offset` | int | K-start offset in QKV output (NH*HD) |
| `qkv_v_offset` | int | V-start offset (NH*HD + NKV*HD) |
| `qkv_total` | int | Total QKV output dim (NH*HD + 2*NKV*HD) |
| `xclbin_qkv_k` | int | QKV xclbin K dimension (=H) |
| `xclbin_qkv_n` | int | QKV xclbin N dimension (=qkv_total) |
| `xclbin_o_k` | int | O xclbin K dimension (=NH*HD) |
| `xclbin_o_n` | int | O xclbin N dimension (=H) |
| `xclbin_g_k` | int | G xclbin K dimension (=H) |
| `xclbin_g_n` | int | G xclbin N dimension (=IM) |
| `xclbin_u_k` | int | U xclbin K dimension (=H) |
| `xclbin_u_n` | int | U xclbin N dimension (=IM) |
| `xclbin_gu_k` | int | GU (combined) K dimension (=H) |
| `xclbin_gu_n` | int | GU (combined) N dimension (=IM*2) |
| `xclbin_d_k` | int | D (down) K dimension (=IM) |
| `xclbin_d_n` | int | D (down) N dimension (=H) |
| `has_q_norm` | bool | Has Q-norm weights |
| `has_k_norm` | bool | Has K-norm weights |
| `has_rope_freqs_file` | bool | Has file-based RoPE freqs |
| `has_lm_head` | bool | Has separate lm_head (vs tied embeddings) |
| `gu_split` | bool | Gate/Up split into separate xclbins |
| `rope_theta` | float | RoPE theta (default: 1,000,000) |
| `rope_factor` | float | RoPE scaling factor (default: 1.0) |
| `model_tag` | string | Model architecture tag |

---

## 9. XRT & System Commands

| Command | Purpose |
|---------|---------|
| `xrt-smi examine` | Show NPU device, firmware, XRT version |
| `xrt-smi validate` | Run NPU validation tests (TOPS benchmark) |
| `echo N > /sys/module/amdxdna/parameters/aie2_max_col` | Override max columns (firmware overrides) |
| `ls -la /dev/accel/accel0` | NPU device node (char 261:0) |
| `ls -la /dev/dri/renderD128` | GPU render node |
| `groups` | Check membership in `render` group |

---

## 10. Quick Reference Card

### Most-Used Env Vars for Daily Work

```bash
# NPU engine: specify model + xclbin
export MODEL_PATH=/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx
export XCLBIN_DIR=/home/bcloud/npu-sandbox/npu-infer/build/int8
export TOKENS=64
export KV_PAGES=1024

# Multi-token + decode
./npu_engine_mt "$MODEL_PATH" 151643 872 198 11852 151644
NPU_GEN=32 NPU_TEMP=0.9 ./npu_engine_mt "$MODEL_PATH" 151643 872

# Speculative decode
NPU_SPEC=5 ./npu_engine_spec_v3 "$MODEL_PATH" 32

# Fused layer engine (291 tok/s — 3x v12)
export FUSED_XCLBIN_DIR=/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127
export FUSED_WEIGHTS_DIR=/home/bcloud/npu-sandbox/npu-infer/build/int8
./npu_engine_fused "$MODEL_PATH" 64

# Fused engine (dispatch policy)
./fused-engine --policy auto "Hello, world!"
./fused-engine --list-policies
./fused-engine --policy ffn_on_npu --max-tokens 128

# GPU engine
zinc --model model.gguf --check
zinc --model model.gguf --profile
ZINC_BATCHED_PREFILL=1 zinc --model model.gguf --check

# LD_PRELOAD hooks (BO capture)
LD_PRELOAD=./hook_bo.so flm serve qwen3:0.6b

# XCLBIN generator
python3 n1_core_i8_v2.py --md 128 --kd 1024 --nd 4096 --output final_i8_QKV_v
```

### SREP Unlock (one-time per boot)
```bash
# On USB (FAT32):
cp BOOTX64.efi SmokelessRuntimeEFIPatcher\(020\).efi Oniguruma.efi SREP_Config.cfg startup.nsh /usb/
# Boot from USB → SREP applies patch at runtime → reboot → NPU unlocked
```
