# Fix Output Correctness — Hybrid GPU/CPU Forward Pass Bug

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the RocmInferenceEngine output correctness bug where all generated tokens are identical (82291 repeated) due to CPU-side forward pass error(s).

**Architecture:** The engine uses GPU GEMM kernels for matrix multiplies but CPU-side operations for everything else (RMSNorm, RoPE, Q/K norm, attention softmax and mixing, SiLU activation, residual connections). The forward pass in `forward_layer()` is the suspect — GPU GEMM is verified accurate. We'll add instrumentation to compare hidden states layer-by-layer, commit fixes as they're found.

**Tech Stack:** C++17, HIP, ROCm 7.2.4, Radeon 8060S (gfx1151), `.q4nx` model format

## Global Constraints

- Compile with system `hipcc` (ROCm 7.2.4 runtime at `/opt/rocm/lib`, includes `-I/opt/rocm/include`)
- Run with `LD_LIBRARY_PATH=/opt/rocm/lib`
- Must call `hipSetDevice(0)` before `hipGetDeviceCount`
- All weights are FP32 in engine (BF16→FP32 conversion at load time)
- GPU GEMM kernels (`gemv_kernel`, `gemm_kernel`) are verified working
- Do NOT modify the weight loading code — BF16 fix is already applied

---

### Task 1: Add Layer-By-Layer Diagnostic Output to Dump Hidden States

**Files:**
- Modify: `src/rocm_engine.cpp` (in `forward_layer()` and `generate()`)

**Interfaces:**
- Consumes: `RocmInferenceEngine::forward_layer(int l, int pos)` — existing method
- Produces: Per-layer hidden state dumps for comparison against NPU reference

**Context:** Before fixing anything, we need to see *where* the hidden state blows up. Add printf-based dumping after each key step in layer 0 and layer 1 at position 0. Compare against a Python reference forward pass (Task 2).

- [ ] **Step 1: Add debug print after each sub-operation in layer 0**

In `forward_layer()`, add `if (l == 0 && pos == 0)` guards to print:
- hidden_ after input RMSNorm (first 8 values, last 8 values, min, max, mean, norm)
- q_out_ after QKV GEMM (first 32 values of Q[0], Q[7], K[0], V[0])
- q_out_ after Q norm (first 8 values of each head)
- q_out_ after RoPE (first 8 values of each head)
- attn_scores_ before softmax (all values for pos=0)
- attn_out_ after attention (first 32 values)
- o_out_ after O projection GEMM (first 32 values)
- hidden_ after residual add (first 32 values)
- hidden_ after post-attention RMSNorm
- gate_ after GU GEMM (first 32 values of gate[0..IM-1] and gate[IM..2*IM-1])
- silu_out_ (first 32 values)
- down_ after down projection GEMM  
- hidden_ after final residual add

Print format for each: `"[DBG L%d P%d] hidden after X: min=%f max=%f mean=%f norm=%f first8=[%f,%f,...]"`

- [ ] **Step 2: Add debug print in generate() for prefill and decode**

In `generate()`, add printing of hidden_ after the RMSNorm/final_norm and before GEMM with lm_head:
- "hidden before final norm: min=%f max=%f mean=%f norm=%f first8=..."
- "hidden after final norm: min=%f max=%f mean=%f norm=%f first8=..."  
- "logits[0..7]: %f %f %f %f %f %f %f %f ... logits[NV-1]: %f ..."
- "logits min=%f max=%f mean=%f top5_tokens: ..."

- [ ] **Step 3: Build and run**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -g \
    -I../include -I/opt/rocm/include \
    -c ../src/rocm_engine.cpp -o rocm_engine.o
hipcc -o rocm_engine rocm_engine.o dequant_q4nx.o -lm
LD_LIBRARY_PATH=/opt/rocm/lib timeout 120 ./rocm_engine 2>&1 | tee /tmp/debug_layer0.log
```

Expected: Get detailed per-op stats for layer 0. Output will tell us where hidden state goes wrong.

- [ ] **Step 4: Commit**

```bash
git add src/rocm_engine.cpp
git commit -m "debug: add layer-by-layer hidden state diagnostics"
```

---

### Task 2: Write Python Reference Forward Pass for Layer 0

**Files:**
- Create: `tools/ref_debug.py`

**Interfaces:**
- Produces: Reference hidden states for position 0, layer 0 of Qwen3-0.6B
- Consumes: Same `.q4nx` model file, weight format understood from C++ code

**Context:** We need a correct reference implementation of the Qwen3-0.6B forward pass (just layer 0, no MLP, just attention) to compare against our C++ engine's output. Python with numpy gives us a trusted baseline.

- [ ] **Step 1: Write reference layer-0 forward pass in Python**

Create `tools/ref_debug.py` that:
1. Loads model.q4nx using the same mmap approach (or just the C++ binary's weight dump)
2. Reads embed_tokens for token 151643
3. Runs: RMSNorm → QKV GEMM → split → Q/K norm → RoPE → scaled dot-product attention (for position 0, single token) → O projection → residual add
4. Prints hidden states after each step in the same format as Task 1

Actually — let's take a better approach. Instead of duplicating the weight loader, let's pipe the C++ engine's data out and compare against numpy.

Use the C++ engine to dump the weights for layer 0:

```python
import numpy as np
import struct

def bf16_to_f32(bf16_data):
    """Convert bf16 bytes to float32 numpy array"""
    arr = np.frombuffer(bf16_data, dtype=np.uint16)
    return arr.astype(np.int32).astype(np.float32) * (1.0 / 2**16)
    # Actually: bf16 bits = uint16 → shift left 16 bits → reinterpret as float32
    conv = arr.astype(np.uint32) << 16
    return conv.view(np.float32)
```

But actually — the simplest approach is to dump the C++ layer-0 weights to a binary file, then run the numpy forward pass.

- [ ] **Step 2: Add weight dump option to C++ engine**

Add a `--dump-weights` flag or just add unconditional printf of a few parameters to verify against Python. Better: write a small Python script that uses subprocess to run the engine and parses the debug output.

Actually the cleanest approach: create a standalone Python reference forward pass that loads the same `.q4nx` file and runs layer 0. We already know the weight format from `load_model()`.

```python
#!/usr/bin/env python3
"""Reference forward pass for Qwen3-0.6B layer 0, for debugging"""
import numpy as np
import struct
import os

MODEL_PATH = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
H = 1024
NH = 16
NKV = 8
HD = 128
GQA = 2
EPS = 1e-6

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = f.read()
    hdr_len = struct.unpack('<Q', data[:8])[0]
    json_hdr = data[8:8+hdr_len]
    data_start = 8 + hdr_len
    return data, data_start, json_hdr

def find_offset(json_hdr, name):
    """Find data_offsets[0] for a weight name in the JSON header"""
    idx = json_hdr.find(f'"{name}"')
    if idx < 0:
        return None
    # Find "data_offsets": [XX, YY]
    do_idx = json_hdr.find(b'"data_offsets"', idx)
    br_idx = json_hdr.find(b'[', do_idx)
    end_idx = json_hdr.find(b',', br_idx)
    return int(json_hdr[br_idx+1:end_idx])

def bf16_to_f32(bf16_bytes):
    """Convert BF16 bytes to float32 numpy array"""
    u16 = np.frombuffer(bf16_bytes, dtype=np.uint16).copy()
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def dequant_i8(data_off, data, nrows, ncols):
    """Dequantize INT8 weights: each row has scale + data"""
    # Format (from dequant_i8_to_float.c):
    # For each row: float scale, then ncols uint8 values
    rows = []
    offset = data_off
    for r in range(nrows):
        scale = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        vals = np.frombuffer(data[offset:offset+ncols], dtype=np.uint8).astype(np.float32)
        offset += ncols
        vals = (vals - 128.0) * scale
        rows.append(vals)
    return np.array(rows)  # (nrows, ncols)

def rms_norm(x, w):
    """RMSNorm: x = x / sqrt(mean(x^2) + eps) * w"""
    ss = np.mean(x.astype(np.float64)**2)
    ir = 1.0 / np.sqrt(ss + EPS)
    return x * ir * w

def rope(x, pos, head_dim=HD):
    """Apply RoPE to x (NH x HD)"""
    result = x.copy()
    for h in range(x.shape[0]):
        for d in range(0, head_dim, 2):
            theta = 1.0 / (1000000.0 ** (d / head_dim))
            angle = pos * theta
            c, s = np.cos(angle), np.sin(angle)
            a, b = x[h, d], x[h, d+1]
            result[h, d] = a*c - b*s
            result[h, d+1] = b*c + a*s
    return result

def qk_norm(x, w):
    """QK norm: x = x / sqrt(mean(x^2) + eps) * w"""
    ss = np.mean(x.astype(np.float64)**2)
    ir = 1.0 / np.sqrt(ss + EPS)
    return x * ir * w

def main():
    data, data_start, json_hdr = load_model()
    
    # Find layer 0 offsets
    q_off = find_offset(json_hdr, b'model.layers.0.self_attn.q_proj.weight')
    k_off = find_offset(json_hdr, b'model.layers.0.self_attn.k_proj.weight')
    v_off = find_offset(json_hdr, b'model.layers.0.self_attn.v_proj.weight')
    o_off = find_offset(json_hdr, b'model.layers.0.self_attn.o_proj.weight')
    in_off = find_offset(json_hdr, b'model.layers.0.input_layernorm.weight')
    pa_off = find_offset(json_hdr, b'model.layers.0.post_attention_layernorm.weight')
    qn_off = find_offset(json_hdr, b'model.layers.0.self_attn.q_norm.weight')
    kn_off = find_offset(json_hdr, b'model.layers.0.self_attn.k_norm.weight')
    
    print(f"Offsets: q={q_off} k={k_off} v={v_off} o={o_off} in={in_off} pa={pa_off} qn={qn_off} kn={kn_off}")
    
    # Load embedding for token 151643
    embed_bf16 = data[data_start:data_start + 151936 * H * 2]  # BF16
    embed = bf16_to_f32(embed_bf16).reshape(151936, H)
    token_id = 151643
    hidden = embed[token_id].copy()
    print(f"Embedding token {token_id}: min={hidden.min():.6f} max={hidden.max():.6f} mean={hidden.mean():.6f} norm={np.sqrt(np.mean(hidden**2)):.6f}")
    print(f"  first8: {hidden[:8]}")
    
    # Load layer 0 weights
    q_w = dequant_i8(data_start + q_off, data, H, 256)  # H x 256 (NH*HD)
    k_w = dequant_i8(data_start + k_off, data, H, 128)  # H x 128 (NKV*HD/2? no, NKV*HD=1024)
    v_w = dequant_i8(data_start + v_off, data, H, 128)
    o_w = dequant_i8(data_start + o_off, data, 2048, H)  # 2048 x H
    in_w = bf16_to_f32(data[data_start+in_off:data_start+in_off+H*2])
    qn_w = bf16_to_f32(data[data_start+qn_off:data_start+qn_off+HD*2])
    kn_w = bf16_to_f32(data[data_start+kn_off:data_start+kn_off+HD*2])
    
    print(f"\n=== LAYER 0 ===")
    
    # Step 1: Input RMSNorm
    hidden_norm = rms_norm(hidden, in_w)
    print(f"After input RMSNorm: min={hidden_norm.min():.6f} max={hidden_norm.max():.6f} norm={np.sqrt(np.mean(hidden_norm**2)):.6f}")
    print(f"  first8: {hidden_norm[:8]}")
    
    # Step 2: QKV GEMM
    # Q = hidden @ Q^T (256 x H in file = Q weight is H x 256)
    q = hidden @ q_w  # (1024,) @ (1024, 256) = (256,) ... wait
    # Actually q_w is (H, 256) so hidden @ q_w = (H,) @ (H, 256) = (256,) = NH*HD = 16*128
    # K = hidden @ k_w = (1024,) @ (1024, 1024) = (1024,) wait k_w is (H, 128)?
    # Let me check: k_proj weight should be H x (NKV*HD) = 1024 x 1024
    # But k_off found NKV*HD=128? No that's per-row.
    # The INT8 format: each row has scale + ncols uint8 values.
    # q_proj: NH*HD = 2048 cols per row. ncols = 2048. But we passed 256 to dequant...
    # Actually from load_model: dequant_i8(data + data_start + offsets[l].qp, 256, &qr, &unused)
    # 256 is the ROW count (not col). qr returns actual rows read.
    # So q_w shape should be (256, H) transposed? Let me re-check the C++ code...
    # In load_model: qr, kr, vr - these are row counts from dequant_i8.
    # qr = NH * HD? Actually wait - dequant_i8's second arg is described differently...
    
    # Looking at C++: dequant_i8(data + ..., 256, &qr, &unused) - 256 is nrows
    # The dequant function: each row has scale + data[ncols]. The second param is nrows.
    # So q_proj = NH * HD / 128 * H? That doesn't make sense.
    # Actually looking more carefully at the qkv fusion code:
    # for k in range(H): memcpy(layers[l].qkv[k * tqkv], &qw[k * qr], qr * 4)
    # So qw[k] has qr elements = NH * HD = 2048. qr = 2048.
    # qw is (H, qr) = (1024, 2048) float array.
    # But dequant_i8 returned 256 rows... this must be 256 rows of 2048/256=8 sub-blocks?
    
    # I need to check the dequant function. Let me look at how the encoding works.
    print("Need to check dequant_i8_to_float format...")
    print(f"\n  q_proj dequant rows={len(q_w)} cols={q_w.shape[1] if len(q_w.shape)>1 else '?'}")

if __name__ == '__main__':
    main()
```

This is getting complex. Let me take a different, more direct approach:

- [ ] **Step 2 (revised): Write a simpler reference — dump layer-0 hidden state from the C++ engine and compare against a hand-computed layer 0**

Actually the most efficient path is:

**Dump the C++ engine's intermediate values to a file, then write a Python script to load that dump and compare against a numpy reference for the SAME operations.**

Or even better — make the C++ engine dump the QKV weight values and hidden state vectors, then run the same operations in numpy.

Let me simplify: output the intermediate values in a structured format from the C++ engine, then check them in numpy.

- [ ] **Step 2: Add structured dump mode to engine**

Add `--debug-layer 0` flag that dumps all intermediate tensors to a binary file for analysis.

- [ ] **Step 3: Build and run debug mode, capture dump**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O0 -g -I../include -I/opt/rocm/include \
    -c ../src/rocm_engine.cpp -o rocm_engine.o
hipcc -o rocm_engine rocm_engine.o dequant_q4nx.o -lm
LD_LIBRARY_PATH=/opt/rocm/lib timeout 120 ./rocm_engine --debug-layer 0 2>&1 | tee /tmp/debug_full.log
```

---

### Task 3: Identify and Fix Forward Pass Bug(s)

**Files:**
- Modify: `src/rocm_engine.cpp`

**Interfaces:**
- Consumes: Debug output from Task 1 + Python reference from Task 2
- Produces: Fixed `forward_layer()` and supporting functions

**Context:** With the debug output and reference, we'll identify the exact operation where the hidden state diverges. Likely candidates based on the symptom (uniform logits, hidden state magnitude growing through layers):

1. **RMSNorm** — checks: `sqrt(mean(x²) + eps)` — C++ uses `sqrtf((float)(ss / n) + 1e-6f)` — this computes `sqrt(mean(x²) + eps)` which is correct. But `ss` is computed as `double sum of x[i]²` then cast to float for division. This should be fine. 
   - **Bug check**: The norm weights w[i] multiply into result. If w is wrong (e.g., still reading BF16 as float32 before the fix), x blows up. But the BF16 fix is applied. Let's verify w values.
   - **Bug check**: `x[i] = 0.0f` for non-finite — this clips NaN but may mask a real problem.

2. **QKV GEMM** — GPU kernel, should be accurate. But check: is the fused QKV correct?
   - C++: `gemm(hidden, d_qkv, q_out, 1, H, 4096)` where H=1024, 4096=NH*HD+2*NKV*HD=2048+1024+1024
   - But wait: `d_qkv` is dimension `H × 4096`? Let's check `load_model()`:
     - `qr = qr_actual`, `kr`, `vr` from dequant
     - `tqkv = qr + kr + vr` (total columns per row)
     - `layers[l].qkv.resize(H * tqkv)` = H × tqkv
   - `tqkv` should equal 4096 (2048+1024+1024). Is it? **Check this.**
   - **Potential bug**: If dequant returns wrong dimensions, the QKV weight matrix could be misaligned.

3. **Q/K norm** — `q_out_[h*HD]` for each head — correct dimension. Norm uses `sqrt(mean(q²) + eps)` which matches Qwen3's QK norm implementation. But wait — Qwen3's QK norm is:
   ```
   q = q / RMS(q) * q_norm_weight  (but Qwen3 QK norm might be LayerNorm, not RMSNorm)
   ```
   **Check**: Is Qwen3's QK norm LayerNorm (with bias) or RMSNorm (no bias)? The `.q4nx` file has only `weight` for q_norm and k_norm, no bias. So RMSNorm is correct. ✓

4. **Attention score scale** — `sqrtf(HD)` = sqrt(128) = 11.31. This is correct for scaled dot-product attention. ✓

5. **Softmax** — The clamp to ±80.0 is correct for float32 exp. The fallback for sum≤0 is reasonable. But check: is `x[i] = expf(x[i] - mx)` actually computing softmax correctly? Yes: `softmax(x_i) = exp(x_i - max_x) / sum(exp(x_j - max_x))`. ✓

6. **Residual connection** — `hidden[i] = residual[i] + o_out[i]` — correct. ✓

7. **FFN SiLU** — `silu(x) = x / (1 + exp(-x))` — correct. `silu_out[i] = silu(gate[i]) * gate[IM + i]` — correct SwiGLU. ✓

8. **Final norm → lm_head** — The big one. `gemm(hidden, d_lm_head, logits, 1, H, NV)` where NV=151936. 
   - `d_lm_head` size: `H * NV = 1024 * 151936 = 155,582,464 floats = 622 MB`
   - This fits in 32GB VRAM. But the GPU kernel: `gemv_kernel` with `gridDim = (NV + 255) / 256 = 594` blocks. Each block: 256 threads, K=1024 iterations.
   - This should work. But let's check: `d_lm_head` upload in `gpu_init()`:
     ```cpp
     if (!gm(d_lm_head_, (size_t)H * NV * 4)) return false;
     if (!up(d_lm_head_, lm_head_.data(), (size_t)H * NV)) return false;
     ```
   - `lm_head_` is `H * NV` floats — correct.
   - But `lm_head_` is loaded in `load_model()`:
     ```cpp
     for (int k = 0; k < H; k++)
         for (int n = 0; n < NV; n++)
             lm_head_[k * NV + n] = lm[n * H + k];
     ```
   - This transposes from (NV × H) to (H × NV). The GEMM expects (K×N) = (H×NV) layout. **But wait**: `gemv_kernel` does:
     ```cpp
     for (int k = 0; k < K; k++)
         sum += A[k] * B[(size_t)k * N + col];
     ```
   - So C[col] = sum_k A[k] * B[k][col], where B has shape (K, N). 
   - A = hidden (1×H), B = d_lm_head → should be (H, NV). 
   - `lm_head_[k * NV + n]` = element at row=k, col=n → B[k][n]. ✓ Correct.

9. **Check the QKV weight loading more carefully**:
   ```cpp
   float* qw = dequant_i8_to_float(data + data_start + offsets_[l].qp, 256, &qr, &unused);
   ```
   Wait, the second param is `256` — that's the ncols parameter, not nrows. Let me re-check the signature from `common.h`:
   ```cpp
   extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);
   ```
   The params are: (data_ptr, ncols, &nrows_out, &ncols_out).
   
   So `dequant_i8_to_float(data + ..., 256, &qr, &unused)` means: each row has 256 values. qr = number of rows.
   
   But q_proj weight is H × (NH*HD) = 1024 × 2048. If ncols=256 and nrows=qr, then qr should be 2048/256 * 1024... no wait.
   
   Actually from the C++ code:
   ```cpp
   int tqkv = qr + kr + vr;
   layers_[l].qkv.resize((size_t)H * tqkv);
   for (int k = 0; k < H; k++) {
       memcpy(&layers_[l].qkv[k * tqkv], &qw[k * qr], qr * 4);
       memcpy(&layers_[l].qkv[k * tqkv + qr], &kw[k * kr], kr * 4);
       ...
   ```
   
   So `qw` is `(H, qr)` and `kw` is `(H, kr)` and `vw` is `(H, vr)`.
   
   For q_proj: qr should be NH*HD = 2048. 
   For k_proj: kr should be NKV*HD = 1024.
   For v_proj: vr should be NKV*HD = 1024.
   Total: tqkv = 2048 + 1024 + 1024 = 4096. ✓
   
   But `dequant_i8_to_float(data + ..., 256, &qr, &unused)` with ncols=256 and nrows=? ...
   
   Actually the dequant function returns `H × (NH*HD/ncols * ncols)` = `1024 × (2048/256 * 256)` = `1024 × 2048`.
   
   Wait, the function prototype: `dequant_i8_to_float(data, ncols_per_chunk, &nrows, &cols)`.
   
   Let me check by looking at how it's called for k_proj:
   ```cpp
   float* kw = dequant_i8_to_float(data + data_start + offsets_[l].kp, 128, &kr, &unused);
   ```
   k_proj: ncols=128, kr should be H * (NKV*HD/128) = 1024 * 8 = 8192 rows? No...
   
   Actually if each row = 128 elements, and k_proj is 1024 × 1024, then the file has 1024 rows of 1024 elements each, split into 1024/128 = 8 chunks per row = 8192 "sub-rows". So kr = 8192.
   
   Then `kw[k * kr]` with kr=8192 accesses element k*8192 which is in a completely different row. This doesn't make sense.
   
   **SUSPECT**: Let me re-read the dequant function. Let me look at its source or verify.

- [ ] **Step 1 (revised for Task 3): Verify dequant_i8_to_float returns correct dimensions**

We need to check: Does `dequant_i8_to_float(ptr, 256, &qr, &cols_out)` return a flat array of `qr * cols_out` values where qr is nrows-output and cols_out is the actual ncols?

Or does the second parameter mean something different?

Let me look at the existing NPU engine code that uses it correctly.

```bash
grep -n "dequant_i8_to_float" /home/bcloud/npu-sandbox/npu-infer/src/*.cpp /home/bcloud/npu-sandbox/npu-infer/tools/*.cpp
```

- [ ] **Step 2: Once dequant is understood, add printf to check qr, kr, vr values**

```cpp
printf("[DBG] L%d qr=%d kr=%d vr=%d tqkv=%d\n", l, qr, kr, vr, qr+kr+vr);
printf("[DBG] L%d qw[0..7]=%f %f %f %f %f %f %f %f (last qw row first col=%f)\n", 
       l, qw[0], qw[1], qw[2], qw[3], qw[4], qw[5], qw[6], qw[7],
       qw[(H-1)*qr]);
```

- [ ] **Step 3: Based on findings from Steps 1-2, fix the bug(s)**

Potential fixes based on likely root causes:

**If dequant dimensions are wrong:**
Fix the qkv fusion loop:
```cpp
// Current (likely wrong for certain dequant layouts):
memcpy(&layers_[l].qkv[k * tqkv], &qw[k * qr], qr * 4);
// qw is a flat array of (H * qr/ncols_per_row * ncols_per_row?) elements
// We need to figure out the actual layout
```

**If RMSNorm is wrong:**
The current implementation:
```cpp
void rms_norm_f(float* x, const float* w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] = x[i] * ir * w[i];
}
```
The fix: Use `1e-6f` constant. Wait — Qwen2/V3 uses `1e-6`. Let's verify:
- Qwen3 code: `rms_norm` with eps=1e-6. ✓

**If attention scores are computed incorrectly:**
Check: `for (int d = 0; d < HD; d++) s += (double)hq[d] * hk[p * NKV * HD + d];`
The hk access for position p: `hk[p * NKV * HD + d]` — this accesses the K cache which stores NKV heads × HD per position.
But `hk` is `&kv.k[kvh * HD]` where kvh = h / GQA. So this points to the start of the head kvh's data within the current position.
Then `hk[p * NKV * HD + d]` — wait, `hk` is already offset to the right head. So `hk[p * NKV * HD + d]` reads from the wrong position!
- `hk = &kv.k[kvh * HD]` — this is the raw pointer into kv.k for head kvh at position 0.
- `hk[p * NKV * HD + d]` — this accesses position p, head kvh. ✓ Correct! Because hk is already the base pointer for that head, and p * NKV * HD jumps to position p.

Let me double-check: `kv.k` is a flat array of size `MAX_POS * NKV * HD`. Element at position `p`, head `kvh`, dim `d` = `kv.k[p * NKV * HD + kvh * HD + d]`.
- `hk = &kv.k[kvh * HD]` → `hk[x] = kv.k[kvh * HD + x]`
- `hk[p * NKV * HD + d]` = `kv.k[kvh * HD + p * NKV * HD + d]` 
- But that should be `kv.k[p * NKV * HD + kvh * HD + d]`!
- **BUG!**: `hk[p * NKV * HD + d]` = `kv.k[kvh * HD + p * NKV * HD + d]`, but the correct access is `kv.k[p * NKV * HD + kvh * HD + d]`.
- These are only equal if `kvh = 0` (head 0). For `kvh > 0`, this reads from the wrong memory region!
- Similarly for V: `hv[p * NKV * HD + d]` where `hv = &kv.v[kvh * HD]`.

**Wait, let me recompute:**
- `kv.k` layout: `[pos0_head0_d0, pos0_head0_d1, ..., pos0_head0_d127, pos0_head1_d0, ..., pos0_head7_d127, pos1_head0_d0, ...]`
- So element at (pos p, head kvh, dim d) = `kv.k[p * NKV * HD + kvh * HD + d]`
- `hk = &kv.k[kvh * HD]` → `hk[x] = kv.k[kvh * HD + x]`
- To access (p, kvh, d): we want `kv.k[p * NKV * HD + kvh * HD + d]` = `hk[p * NKV * HD + d]`? Let's check:
  - `hk[p * NKV * HD + d]` = `kv.k[kvh * HD + p * NKV * HD + d]`
  - But we want `kv.k[p * NKV * HD + kvh * HD + d]`
  - These are NOT the same! `kvh * HD + p * NKV * HD ≠ p * NKV * HD + kvh * HD` — wait, they ARE the same (commutative addition).
  - Actually: `kvh * HD + p * NKV * HD = p * NKV * HD + kvh * HD`. ✓ Yes, addition is commutative. So the access IS correct.

OK so the attention score access pattern is correct. Let me focus on what's most likely wrong.

**Most likely bug**: The dequant_i8 weight layout for QKV fusing. Let me trace through more carefully.

The `dequant_i8_to_float` function: from looking at the signature `(const uint8_t* data, int ncols_per_chunk, int* nrows, int* ncols_out)`, the second param is the number of int8 values per "chunk"