# FLM Fused XCLBIN Port — Qwen3-0.6B

## Status: ✅ Kernels recompiled, contract created, MLIR generator pending

### Step 1: Recompile Kernels ✅ DONE

All 5 kernels recompiled for Qwen3-0.6B with Chess (xchesscc):

| Kernel | Object | Size | Status |
|--------|--------|------|--------|
| main_projection (GEMM) | `main_projection_q4nx_06b.o` | 80KB | ✅ |
| edge_attention (4 heads/window) | `edge_attention_06b.o` | 37KB | ✅ |
| postprocess_qkv (Q records=4) | `postprocess_qkv_06b.o` | 34KB | ✅ |
| full_vector_station (RMSNorm) | `full_vector_station_06b.o` | 20KB | ✅ |
| swiglu (LUT-based) | `swiglu_06b.o` | 7KB | ✅ |

**Locations:**
- Sources: `/home/bcloud/torch2aie/examples/qwen3-decode-layer/*_06b.{cc,h}`
- Objects: `/home/bcloud/torch2aie/build/qwen3_decode_layer_objects_06b/*.o`
- Constants: `qwen3_constants_06b.h` (H=1024, IM=3072, NH=16, NKV=8)

### What FLM's fused design does

One xclbin per transformer layer. Single XRT dispatch. 5 kernels on 38 AIE tiles:

```
QKV GEMM → postprocess_qkv → attention → O GEMM → norm → GU GEMM → SiLU → D GEMM
│          │                  │          │        │      │         │      │
│          │                  │          │        │      │         │      │
16 Main16   c1r3 tile        8 Shape     c1r2    16      c6r2    16     c1r2
cores       (row 1 col 3)    tiles      tile     Main16  tile    Main16 tile
(cols 2-5,                    (edge)     (edge)  cores   (row 6  cores  (row 1
 rows 2-5)                                        reuse)  col 2)  reuse) col 2)
```

**Design files live at:** `/home/bcloud/torch2aie/examples/qwen3-decode-layer/`

**Compiled kernels:** `/home/bcloud/torch2aie/build/qwen3_decode_layer_objects/`

### Step 1: Recompile Kernels for Qwen3-0.6B ✅ Partial

| Kernel | Source | Object | Status | Changes Needed |
|--------|--------|--------|--------|---------------|
| edge_attention (0.6B) | `edge_attention.cc` | `edge_attention_06b.o` | ✅ Done | kHeads=4, GQA=2, kContext=16 |
| edge_attention (0.6B, all heads) | needs all 16 heads | TBD | ❌ Need | FLM uses 4 windows × 8 Q heads each. 0.6B needs 8 windows × 2 Q heads OR 4 windows × 4 Q heads |
| postprocess_qkv | `postprocess_qkv.cc` | TBD | ❌ Need | Q=2048bf16 (from 4096), K=V=1024bf16 (same) |
| full_vector_station | `full_vector_station.cc` | TBD | ❌ Need | H=1024 (from 4096), residual add size changes |
| swiglu | `swiglu.cc` | `swiglu.o` | ✅ Done | Same for all models (IM-independent LUT) |
| qwen3_decode_kernels (main16) | `qwen3_decode_kernels.cc` | TBD | ❌ Need | Q_body_records=4 (from 8), O_body_records=2 (from 8) |

### Step 2: Modify Contract ✅ Created

`contract_06b.py` — `/home/bcloud/torch2aie/examples/qwen3-decode-layer/contract_06b.py`

Key dimension changes:
| Constant | 8B | 0.6B |
|----------|-----|------|
| HIDDEN_DIM | 4096 | **1024** |
| INTERMEDIATE_DIM | 12288 | **3072** |
| NUM_Q_HEADS | 32 | **16** |
| NUM_KV_HEADS | 8 | 8 |
| GQA_RATIO | 4 | **2** |
| Q output dim | 4096 | **2048** |
| O output dim | 4096 | **1024** |
| Down output dim | 4096 | **1024** |
| C1R2_PACKET_DWORDS | 2049 | **513** |
| ATTENTION_PACKET_DWORDS | 2048 | **1024** |
| SHAPE_WINDOW_DWORDS | 512 | **256** |

### Step 3: Modify MLIR Generator

File: `cases/full_layer_engine_generate.py`

Needs to use `contract_06b` imports instead of `contract`. All dimension-dependent
string templates (packet sizes, buffer sizes, BD lengths) auto-derived from contract.
Main16Buffer layout stays the same. Tile grid stays the same.

### Step 4: Build + Validate

```bash
cd /home/bcloud/torch2aie/examples/qwen3-decode-layer
python design.py --current-token 9  # Generate QKV-only prefix MLIR
python run_full_layer.py --current-token 9 --model-path /path/to/Qwen3-0.6B-NPU2
```

### Step 5: Integrate into Engine

Once xclbin compiles: replace 112 engine dispatches with 28 fused dispatches.
New engine: `npu_engine_fused.cpp`.

Expected: ~5ms/tok batch step at M=32. Effective ~3ms/tok = 333 tok/s.

### Why This Takes Weeks

1. **5 kernels need recompilation** with Chess toolchain for new loop dimensions
2. **MLIR generator** is 3407 lines generated from 800+ lines of Python string templates
3. **ABI verification** — 50+ validation checks in the MLIR output
4. **Numerical validation** — CPU reference must match NPU output to 0.01 tolerance
5. **Model weight format** — Q4NX Compact format, not our current INT8 flat weights

### What We Have Ready

- ✅ Contract for 0.6B dimensions
- ✅ Qwen3-0.6B edge attention kernel (4 Q heads per window, Chess-compiled)
- ✅ NPU attention kernel (Peano, attn_scalar.o)
- ✅ SiLU kernel (Peano, silu_gate.o)
- ✅ Qwen3-0.6B model weights (Q4NX format at `/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/`)
- ✅ Chess compiler license (through 2027.06)
- ✅ FLM design source code studied and documented
- ✅ Working single-dispatch GEMM xclbins (fallback if fused build fails)

### Fallback Path

If fused xclbin doesn't compile: continue with M=32 batch approach (v12).
Already 97 tok/s, beating FLM Kraken Point. LM head on NPU already done.
Next: Qwen3-1.7B + Llama support via the other agent's xclbins.
