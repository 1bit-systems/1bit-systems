# GPU-Optimized Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the CPU-bound ROCm inference engine (106 ms/tok) into a GPU-resident engine where all ops run as HIP kernels, eliminating all PCIe transfers between layers.

**Architecture:** Two-phase approach: (A) profile to get hard numbers on the bottleneck, (C) full GPU residency — keep hidden state on-device across all layers, port all ops (RMSNorm, SiLU, RoPE, QK norm, attention) to HIP kernels as part of that transition.

**Tech Stack:** HIP/CUDA-like kernels, ROCm 7.2.4, gfx1151 (Radeon 8060S), single-file engine (`src/rocm_engine.cpp`)

## Global Constraints

- No Python, no Rust — pure C++/HIP only
- No hipBLAS/rocBLAS dependency — raw HIP kernels only
- System ROCm 7.2.4 from /opt/rocm, hipcc from system
- Target gfx1151 (Radeon 8060S, 20 CUs, RDNA 3.5)
- All weights as float32, already on GPU (d_* buffers)
- Run with LD_LIBRARY_PATH=/opt/rocm/lib
- hipSetDevice(0) already done in init
- Prompt: 9 tokens, generate 4 max output
- Each step produces a working, testable binary

## File Structure

| File | Role |
|------|------|
| `src/rocm_engine.cpp` | Engine — kernels, host code, main() |
| `include/rocm_engine.h` | Class declaration |

All changes are to these two files. No new files.

---

## Task A: Profile the Hot Path

**Files:**
- Modify: `src/rocm_engine.cpp`
- Modify: `include/rocm_engine.h`

**Approach:** Add `std::chrono` timers around each section of `forward_layer()` and the generate loop. Print per-operation cumulative times at the end.

- [ ] **Step 1: Add profiling struct to header**

In `include/rocm_engine.h`, add to the engine class:

```cpp
    // Profiling accumulators (all times in microseconds)
    struct {
        uint64_t rms_norm_us = 0;
        uint64_t gemm_us = 0;        // includes H2D + kernel + D2H
        uint64_t qk_norm_us = 0;
        uint64_t rope_us = 0;
        uint64_t attention_us = 0;    // scores + softmax + weighted sum
        uint64_t silu_us = 0;
        uint64_t elementwise_us = 0;  // residual add
    } perf_;
```

- [ ] **Step 2: Add PerfTimer helper**

In `src/rocm_engine.cpp`, near the top after includes:

```cpp
// Profile: accumulate elapsed time in microseconds
struct PerfTimer {
    std::chrono::steady_clock::time_point start;
    uint64_t& accum;
    PerfTimer(uint64_t& a) : start(std::chrono::steady_clock::now()), accum(a) {}
    ~PerfTimer() {
        auto end = std::chrono::steady_clock::now();
        accum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};
```

- [ ] **Step 3: Wrap each section in forward_layer with PerfTimer**

Add `{ PerfTimer pt(perf_.rms_norm_us); ... }` blocks around:

```cpp
// 1. Input RMSNorm
{ PerfTimer pt(perf_.rms_norm_us);
  rms_norm_f(hidden_.data(), layer.in_norm.data(), H); }

// 2. QKV GEMM
{ PerfTimer pt(perf_.gemm_us);
  gemm(hidden_.data(), layers_[l].d_qkv, q_out_.data(), 1, H, 4096); }

// 3. QK Norm (the for h in 0..NH loop with sqrt+multiply)
{ PerfTimer pt(perf_.qk_norm_us);
  for (int h = 0; h < NH; h++) { ... } }

// 4. RoPE (both Q and K loops)
{ PerfTimer pt(perf_.rope_us);
  for (int h = 0; h < NH; h++) { ... }
  for (int h = 0; h < NKV; h++) { ... } }

// 5. Attention (scores + softmax + weighted sum per head)
{ PerfTimer pt(perf_.attention_us);
  for (int h = 0; h < NH; h++) { ... } }

// 6. O projection GEMM
{ PerfTimer pt(perf_.gemm_us);
  gemm(attn_out_.data(), layers_[l].d_o, o_out_.data(), 1, NH * HD, H); }

// 7. Residual add after O
{ PerfTimer pt(perf_.elementwise_us);
  for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + o_out_[i]; }

// 8. Post-attn RMSNorm
{ PerfTimer pt(perf_.rms_norm_us);
  rms_norm_f(hidden_.data(), layer.pa_norm.data(), H); }

// 9. GU GEMM
{ PerfTimer pt(perf_.gemm_us);
  gemm(hidden_.data(), layers_[l].d_gu, gate_.data(), 1, H, 6144); }

// 10. SiLU gate*up
{ PerfTimer pt(perf_.silu_us);
  for (int i = 0; i < IM; i++) { ... } }

// 11. Down projection GEMM
{ PerfTimer pt(perf_.gemm_us);
  gemm(silu_out_.data(), layers_[l].d_down, down_.data(), 1, IM, H); }

// 12. Residual add after Down
{ PerfTimer pt(perf_.elementwise_us);
  for (int i = 0; i < H; i++) hidden_[i] = residual_[i] + down_[i]; }
```

- [ ] **Step 4: Print profile at end of generate()**

Before the return, after the existing timing output:

```cpp
    uint64_t total_perf =
        perf_.rms_norm_us + perf_.gemm_us + perf_.qk_norm_us +
        perf_.rope_us + perf_.attention_us + perf_.silu_us + perf_.elementwise_us;
    printf("\n=== Perf Profile (%d layers x %d tokens) ===\n", NC, pos);
    printf("  RMSNorm:    %10llu us (%5.1f%%)\n", perf_.rms_norm_us, 100.0 * perf_.rms_norm_us / total_perf);
    printf("  GEMM:       %10llu us (%5.1f%%)\n", perf_.gemm_us, 100.0 * perf_.gemm_us / total_perf);
    printf("  QK Norm:    %10llu us (%5.1f%%)\n", perf_.qk_norm_us, 100.0 * perf_.qk_norm_us / total_perf);
    printf("  RoPE:       %10llu us (%5.1f%%)\n", perf_.rope_us, 100.0 * perf_.rope_us / total_perf);
    printf("  Attention:  %10llu us (%5.1f%%)\n", perf_.attention_us, 100.0 * perf_.attention_us / total_perf);
    printf("  SiLU:       %10llu us (%5.1f%%)\n", perf_.silu_us, 100.0 * perf_.silu_us / total_perf);
    printf("  Elementwise:%10llu us (%5.1f%%)\n", perf_.elementwise_us, 100.0 * perf_.elementwise_us / total_perf);
    printf("  Total:      %10llu us (%5.1f%% of %llu ms)\n",
           total_perf, 100.0 * total_perf / (1000 * (uint64_t)last_token_ms_), (uint64_t)last_token_ms_ * 1000);
```

- [ ] **Step 5: Build, run, record**

```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -I../include -I/opt/rocm/include \
    -c ../src/rocm_engine.cpp -o rocm_engine.o && \
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -I../include -I/opt/rocm/include \
    rocm_engine.o dequant_q4nx.o -lm -o rocm_engine && \
LD_LIBRARY_PATH=/opt/rocm/lib ./rocm_engine /tmp/qwen3_raw 2>&1
```

Record the profile output. Then commit.

```bash
git add src/rocm_engine.cpp include/rocm_engine.h
git commit -m "perf: add per-operation profiling instrumentation

Results on Radeon 8060S (~XX ms/tok, 28 layers, 4 tokens):
  RMSNorm:    XXX us (YY%)
  GEMM:       XXX us (YY%)
  QK Norm:    XXX us (YY%)
  RoPE:       XXX us (YY%)
  Attention:  XXX us (YY%)
  SiLU:       XXX us (YY%)
  Elementwise:XXX us (YY%)

Hot path identified: [bottleneck name] dominates
Next: full GPU residency"
```

---

## Task C: Full GPU Residency

**Files:**
- Modify: `src/rocm_engine.cpp`
- Modify: `include/rocm_engine.h`

**Goal:** Keep hidden state on GPU across all layers. Eliminate all H2D/D2H transfers between layers. Only transfer input embedding lookup and final logit sampling.

### Sub-step C1: Upload norm weights to GPU

Add GPU pointers to `LayerWeights` struct (in `rocm_engine.h`):

```cpp
float* d_in_norm = nullptr;
float* d_pa_norm = nullptr;
float* d_q_norm = nullptr;
float* d_k_norm = nullptr;
```

In `gpu_init()` (in `rocm_engine.cpp`), after uploading QKV/O/GU/Down for each layer:

```cpp
    auto& lw = layers_[l];
    gm(lw.d_in_norm, H * 4);
    gm(lw.d_pa_norm, H * 4);
    gm(lw.d_q_norm, HD * 4);
    gm(lw.d_k_norm, HD * 4);
    up(lw.d_in_norm, lw.in_norm.data(), H);
    up(lw.d_pa_norm, lw.pa_norm.data(), H);
    up(lw.d_q_norm, lw.q_norm.data(), HD);
    up(lw.d_k_norm, lw.k_norm.data(), HD);
```

Also upload `final_norm_` and `embed_table_` to GPU (they're already on device from prior code, verify).

### Sub-step C2: Add GPU-side temporary buffers

Need device-side buffers for: residual, Q/K/V split, attention scores, attention output, O output, gate, SiLU output, down output, logits.

Add to header:

```cpp
float *d_residual_ = nullptr;
float *d_q_ = nullptr;     // NH * HD (2048)
float *d_k_ = nullptr;     // NKV * HD (1024)
float *d_v_ = nullptr;     // NKV * HD (1024)
float *d_attn_scores_ = nullptr; // max_ctx
float *d_attn_out_ = nullptr;    // NH * HD
float *d_o_out_ = nullptr;       // H
float *d_gate_ = nullptr;        // IM (3072)
float *d_up_ = nullptr;          // IM (3072)
float *d_silu_out_ = nullptr;    // IM
float *d_down_ = nullptr;        // H
float *d_logits_ = nullptr;      // NV (151936)
```

Allocate in `gpu_init()`:

```cpp
gm(d_residual_, H * 4);
gm(d_q_, NH * HD * 4);
gm(d_k_, NKV * HD * 4);
gm(d_v_, NKV * HD * 4);
gm(d_attn_scores_, max_seq_len_ * 4);
gm(d_attn_out_, NH * HD * 4);
gm(d_o_out_, H * 4);
gm(d_gate_, IM * 4);
gm(d_up_, IM * 4);
gm(d_silu_out_, IM * 4);
gm(d_down_, H * 4);
gm(d_logits_, NV * 4);
```

### Sub-step C3: HIP kernels for all CPU ops

All kernels take GPU pointers and operate on-device. No host memory access.

**RMSNorm kernel:**

```cpp
__global__ void rms_norm_kernel(float* x, const float* w, int n, float eps) {
    int tid = threadIdx.x;
    int W = blockDim.x;  // single wavefront (64)
    float sum = 0.0f;
    for (int i = tid; i < n; i += W) sum += x[i] * x[i];
    for (int offset = W / 2; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    float inv_rms = rsqrtf(sum / (float)n + eps);
    for (int i = tid; i < n; i += W) x[i] = x[i] * inv_rms * w[i];
}
```

Launch: `rms_norm_kernel<<<1, 64>>>(d_hidden_, lw.d_in_norm, H, 1e-6f)`

**Elementwise add (residual) kernel:**

```cpp
__global__ void add_kernel(float* a, const float* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] += b[i];
}
```

**QK Norm kernel (head-wise RMSNorm with per-head weight):**

```cpp
__global__ void qk_norm_kernel(float* q, float* k, const float* q_w, const float* k_w,
                                int nh, int nkv, int hd) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    float* hq = q + h * hd;
    float sq = 0.0f;
    for (int d = tid; d < hd; d += blockDim.x) sq += hq[d] * hq[d];
    for (int off = blockDim.x/2; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, off);
    float iq = rsqrtf(sq / (float)hd + 1e-6f);
    for (int d = tid; d < hd; d += blockDim.x) hq[d] *= iq * q_w[d];

    if (h % (nh / nkv) == 0) {
        int kvh = h / (nh / nkv);
        float* hk = k + kvh * hd;
        float sk = 0.0f;
        for (int d = tid; d < hd; d += blockDim.x) sk += hk[d] * hk[d];
        for (int off = blockDim.x/2; off > 0; off >>= 1)
            sk += __shfl_xor_sync(0xFFFFFFFF, sk, off);
        float ik = rsqrtf(sk / (float)hd + 1e-6f);
        for (int d = tid; d < hd; d += blockDim.x) hk[d] *= ik * k_w[d];
    }
}
```

Launch: `qk_norm_kernel<<<NH, 64>>>(d_q_, d_k_, q_norm, k_norm, NH, NKV, HD)`
(Need to pass d_q_norm and d_k_norm pointers for the current layer)

**RoPE kernel:**

```cpp
__global__ void rope_kernel(float* q, float* k, const float* cos_t, const float* sin_t,
                             int pos, int nh, int nkv, int hd) {
    int h = blockIdx.x;
    const float* c = cos_t + (size_t)pos * hd;
    const float* s = sin_t + (size_t)pos * hd;
    if (h < nh) {
        float* hq = q + h * hd;
        for (int d = threadIdx.x * 2; d < hd; d += blockDim.x * 2) {
            float a = hq[d], b = hq[d+1];
            hq[d] = a * c[d] - b * s[d];
            hq[d+1] = b * c[d] + a * s[d];
        }
        if (h < nkv) {
            float* hk = k + h * hd;
            for (int d = threadIdx.x * 2; d < hd; d += blockDim.x * 2) {
                float a = hk[d], b = hk[d+1];
                hk[d] = a * c[d] - b * s[d];
                hk[d+1] = b * c[d] + a * s[d];
            }
        }
    }
}
```

Launch: `rope_kernel<<<max(NH, NKV), 64>>>(d_q_, d_k_, d_rope_cos_, d_rope_sin_, pos, NH, NKV, HD)`

**Attention kernel (scaled dot-product + softmax + weighted sum):**

```cpp
__global__ void attention_kernel(float* q, float* k_cache, float* v_cache,
                                  float* out, int nh, int nkv, int n_groups,
                                  int hd, int ctx_len) {
    int h = blockIdx.x;
    int kvh = h / n_groups;
    int tid = threadIdx.x;
    int W = blockDim.x;

    float* hq = q + h * hd;
    float* hk_base = k_cache + kvh * hd;
    float* hv_base = v_cache + kvh * hd;

    extern __shared__ float scores[];

    // Compute attention scores
    for (int p = tid; p < ctx_len; p += W) {
        double s = 0.0;
        float* hkp = hk_base + p * nkv * hd;
        for (int d = 0; d < hd; d++) s += (double)hq[d] * hkp[d];
        scores[p] = (float)(s / sqrtf((float)hd));
    }
    __syncthreads();

    // Softmax (in shared memory)
    float mx = scores[0];
    for (int i = 1; i < ctx_len; i++) if (scores[i] > mx) mx = scores[i];
    double sum = 0.0;
    for (int i = 0; i < ctx_len; i++) {
        scores[i] = __expf(scores[i] - mx);
        sum += scores[i];
    }
    float inv_sum = 1.0f / (float)sum;
    for (int i = 0; i < ctx_len; i++) scores[i] *= inv_sum;
    __syncthreads();

    // Weighted sum of V
    for (int d = tid; d < hd; d += W) {
        float acc = 0.0f;
        for (int p = 0; p < ctx_len; p++) {
            acc += scores[p] * hv_base[p * nkv * hd + d];
        }
        out[h * hd + d] = acc;
    }
}
```

Launch: `attention_kernel<<<NH, 64, ctx_len * 4>>>(...)`

(Shared memory = ctx_len * sizeof(float). For prefill=9 ctx, that's 36 bytes.)

**SiLU × up kernel:**

```cpp
__global__ void silu_mul_kernel(const float* gate, const float* up,
                                  float* out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float g = gate[i];
        if (!isfinite(g)) g = 0.0f;
        float s = g / (1.0f + __expf(-g));
        out[i] = s * up[i];
    }
}
```

Launch: `silu_mul_kernel<<<(IM+255)/256, 256>>>(d_gate_, d_gate_ + IM, d_silu_out_, IM)`

**Embedding lookup kernel:**

```cpp
__global__ void embed_lookup_kernel(const float* embed, int token, float* out, int H) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < H) out[i] = embed[(size_t)token * H + i];
}
```

**KV cache append kernel:**
Not needed — just use `hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice)` which is fast on unified memory.

### Sub-step C4: gemm_dev() — GEMM without host transfers

Add a new method that assumes A is on-device and writes C to device:

```cpp
void RocmInferenceEngine::gemm_dev(float* d_A, float* d_B, float* d_C, int M, int K, int N) {
    hipGetLastError();
    if (M == 1) {
        int bs = 256;
        int gs = (N + bs - 1) / bs;
        hipLaunchKernelGGL(gemv_kernel, gs, bs, 0, 0, d_A, d_B, d_C, K, N);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0, d_A, d_B, d_C, M, K, N);
    }
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) fprintf(stderr, "gemm_dev kernel: %s\n", hipGetErrorString(e));
}
```

### Sub-step C5: Rewrite forward_layer to stay on GPU

New signature and logic:

```cpp
void RocmInferenceEngine::forward_layer(int l, int pos) {
    auto& lw = layers_[l];
    auto& kv = kv_cache_[l];
    int dbg = DEBUG_DUMP && l == 0 && pos == 0;

    // Save residual: d_residual_ = d_hidden_
    hipMemcpy(d_residual_, d_hidden_, H * 4, hipMemcpyDeviceToDevice);

    // 1. Input RMSNorm: d_hidden_ ← RMSNorm(d_hidden_, lw.d_in_norm)
    rms_norm_kernel<<<1, 64>>>(d_hidden_, lw.d_in_norm, H, 1e-6f);

    // 2. QKV GEMM: d_q_ = QKV project (d_hidden_ × lw.d_qkv)
    gemm_dev(d_hidden_, lw.d_qkv, d_q_, 1, H, 4096);

    // 3. Split Q/K/V:   Q=d_q_[0:2048], K=d_q_[2048:3072], V=d_q_[3072:4096]
    hipMemcpy(d_k_, d_q_ + 2048, NKV * HD * 4, hipMemcpyDeviceToDevice);
    hipMemcpy(d_v_, d_q_ + 3072, NKV * HD * 4, hipMemcpyDeviceToDevice);

    // 4. QK Norm: in-place on d_q_ (first NH*HD elems) and d_k_
    qk_norm_kernel<<<NH, 64>>>(d_q_, d_k_, lw.d_q_norm, lw.d_k_norm, NH, NKV, HD);

    // 5. RoPE: in-place on d_q_ and d_k_
    rope_kernel<<<max(NH, NKV), 64>>>(d_q_, d_k_, d_rope_cos_, d_rope_sin_, pos, NH, NKV, HD);

    // 6. Append to KV cache (device-to-device copy)
    hipMemcpy(kv.d_k + pos * NKV * HD, d_k_, NKV * HD * 4, hipMemcpyDeviceToDevice);
    hipMemcpy(kv.d_v + pos * NKV * HD, d_v_, NKV * HD * 4, hipMemcpyDeviceToDevice);
    kv.len = pos + 1;
    int ctx_len = kv.len;

    // 7. Attention: scores + softmax + weighted sum
    attention_kernel<<<NH, 64, ctx_len * 4>>>(d_q_, kv.d_k, kv.d_v,
        d_attn_out_, NH, NKV, NH/NKV, HD, ctx_len);

    // 8. O projection: d_o_out_ = d_attn_out_ × lw.d_o
    gemm_dev(d_attn_out_, lw.d_o, d_o_out_, 1, NH * HD, H);

    // 9. Residual add: d_hidden_ = d_residual_ + d_o_out_
    add_kernel<<<(H+255)/256, 256>>>(d_residual_, d_o_out_, H);
    // d_hidden_ = d_residual_ (now holds hidden + o_out)
    hipMemcpy(d_hidden_, d_residual_, H * 4, hipMemcpyDeviceToDevice);

    // 10. Post-attn RMSNorm: d_hidden_ ← RMSNorm(d_hidden_, lw.d_pa_norm)
    rms_norm_kernel<<<1, 64>>>(d_hidden_, lw.d_pa_norm, H, 1e-6f);

    // 11. Save residual
    hipMemcpy(d_residual_, d_hidden_, H * 4, hipMemcpyDeviceToDevice);

    // 12. GU GEMM: d_gate_ = d_hidden_ × lw.d_gu  (output: 6144 = gate|up)
    gemm_dev(d_hidden_, lw.d_gu, d_gate_, 1, H, 6144);

    // 13. SiLU(gate) × up: d_silu_out_ = silu(d_gate_[0:IM]) × d_gate_[IM:2*IM]
    silu_mul_kernel<<<(IM+255)/256, 256>>>(d_gate_, d_gate_ + IM, d_silu_out_, IM);

    // 14. Down projection: d_down_ = d_silu_out_ × lw.d_down
    gemm_dev(d_silu_out_, lw.d_down, d_down_, 1, IM, H);

    // 15. Residual add: d_hidden_ = d_residual_ + d_down_
    add_kernel<<<(H+255)/256, 256>>>(d_residual_, d_down_, H);
    hipMemcpy(d_hidden_, d_residual_, H * 4, hipMemcpyDeviceToDevice);
}
```

### Sub-step C6: Update generate() for GPU residency

```cpp
int RocmInferenceEngine::generate(const int* input_tokens, int num_input_tokens,
                                    int* output_tokens, int max_output_tokens) {
    auto gen_start = std::chrono::steady_clock::now();

    int pos = 0;
    for (int pi = 0; pi < num_input_tokens; pi++) {
        // Embedding lookup directly to GPU
        embed_lookup_kernel<<<(H+255)/256, 256>>>(d_embed_, input_tokens[pi], d_hidden_, H);
        for (int l = 0; l < NC; l++) forward_layer(l, pos);
        pos++;
    }

    int ngen = 0;
    for (int g = 0; g < max_output_tokens; g++) {
        auto ts = std::chrono::steady_clock::now();

        // Final RMSNorm on GPU
        rms_norm_kernel<<<1, 64>>>(d_hidden_, d_final_norm_, H, 1e-6f);

        // LM head GEMM on GPU
        gemm_dev(d_hidden_, d_lm_head_, d_logits_, 1, H, NV);

        // Copy logits to host for argmax (only ~600KB)
        std::vector<float> logits(NV);
        hipMemcpy(logits.data(), d_logits_, NV * 4, hipMemcpyDeviceToHost);

        // Argmax
        int best = 0;
        float mx = logits[0];
        for (int i = 1; i < NV; i++) {
            if (logits[i] > mx) { mx = logits[i]; best = i; }
        }

        auto te = std::chrono::steady_clock::now();
        last_token_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();

        output_tokens[ngen++] = best;
        printf("[ROCm]  [%d] tok=%d (%lld ms)\n", g, best, (long long)last_token_ms_);
        if (g >= max_output_tokens - 1) break;

        // Next token: embed on GPU
        embed_lookup_kernel<<<(H+255)/256, 256>>>(d_embed_, best, d_hidden_, H);
        for (int l = 0; l < NC; l++) forward_layer(l, pos);
        pos++;
    }

    // Print profile (from Task A)
    // ...

    gen_end = std::chrono::steady_clock::now();
    printf("\nOutput tokens:");
    for (int i = 0; i < ngen; i++) printf(" %d", output_tokens[i]);
    printf("\n");
    printf("=== Done (%.0f ms/tok) ===\n", avg_token_ms());
    return ngen;
}
```

### Sub-step C7: KV cache on GPU

Need `d_k` and `d_v` in `KVCache`. Add to header:

```cpp
struct KVCache {
    float* d_k = nullptr;
    float* d_v = nullptr;
    int len = 0;
    int cap;
};
```

Allocate in `gpu_init()`:

```cpp
for (int l = 0; l < NC; l++) {
    auto& kv = kv_cache_[l];
    kv.cap = max_seq_len_;
    gm(kv.d_k, (size_t)max_seq_len_ * NKV * HD * 4);
    gm(kv.d_v, (size_t)max_seq_len_ * NKV * HD * 4);
}
```

### Sub-step C8: Upload rope tables to GPU

Already have `d_rope_cos_` and `d_rope_sin_` from the existing code (check in `gpu_init()`).

### Sub-step C9: Remove unused host buffers

After verifying correctness, can remove the host-side `q_out_`, `k_out_`, `v_out_`, `attn_scores_`, `attn_out_`, `o_out_`, `gate_`, `silu_out_`, `down_` vectors (or keep for optional debug dump via D2H copy).

### Sub-step C10: Build, test, verify

```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -I../include -I/opt/rocm/include \
    -c ../src/rocm_engine.cpp -o rocm_engine.o && \
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -I../include -I/opt/rocm/include \
    rocm_engine.o dequant_q4nx.o -lm -o rocm_engine && \
LD_LIBRARY_PATH=/opt/rocm/lib timeout 120 ./rocm_engine /tmp/qwen3_raw
```

Expected: same output tokens as before (`32313 11 11 1077`) but much faster (<10-15 ms/tok).

### Sub-step C11: Commit

```bash
git add src/rocm_engine.cpp include/rocm_engine.h
git commit -m "perf: full GPU residency for inference engine

All ops now run as HIP kernels on-device:
- RMSNorm, QK Norm, RoPE, Attention, SiLU, elementwise ops
- Hidden state stays on GPU across all 28 layers
- Only 2 host transfers per token: embedding lookup and logit sampling
- KV cache lives entirely on GPU

Speedup: ~106 ms/tok → ~X ms/tok (1 H2D + 1 D2H per token vs 112 before)"
```

---

## Expected Speedup

| Metric | Before | After |
|--------|--------|-------|
| H2D/D2H per token | 112 transfers | 2 transfers |
| Kernel launches per token | 112 | ~35 (4 GEMM + 4-5 small kernels) |
| Est. ms/tok | ~106 | **~5-15** |
