# 1bit.systems Benchmarks — July 5, 2026 (verified)

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo)  
- NPU: XDNA 2, 32 AIE2P tiles  
- GPU: Radeon 8060S (RADV, 32 CUs, 256 GB/s, Vulkan)  
- CPU: Zen 5, 16C/32T  
**OS**: Ubuntu 26.04 LTS, Kernel 7.0.0-27-generic  

## Production Daemon (C++23, 110 KB)

| Backend | Model | TTFT | Decode | Verified |
|---------|-------|------|--------|----------|
| FLM proxy | Qwen3-0.6B | **529 ms** | **94.4 tok/s** | ✅ Correct output |

The `npu-gpu-cpud` C++23 daemon proxies to FastFlowLM for production inference.
Zero Python dependency. Replaces the previous Python daemon.

## GPU (llama.cpp) — 1-Bit Models

| Model | BPW | Size | Decode |
|-------|-----|------|--------|
| Qwen2 0.5B IQ1_S | 1.06 | 296 MB | **381 tok/s** |
| Qwen3.5-0.8B Q1_0 | 1.25 | 268 MB | **312 tok/s** |
| gemma-2-2b IQ1_S | 1.06 | 788 MB | **158 tok/s** |
| gemma3 4B IQ1_S | 1.06 | 1.05 GB | **122 tok/s** |
| Nemo 8B IQ1_S | 1.06 | 1.97 GB | **79 tok/s** |
| Qwen3.5-9B Q1_0 | 1.25 | 1.82 GB | **70 tok/s** |

## GPU (ZINC Vulkan)

| Model | Decode |
|-------|--------|
| Bonsai-1.7B-F16 | **22 tok/s** |

## C++ NPU Engine (Open Source)

| Engine | Decode | Status |
|--------|--------|--------|
| Universal (5-model) | 28 tok/s | ✅ Coherent — AIE micro-tiling fix |
| Fused xclbin | 4 tok/s | ✅ Validated (max_abs=0.0078 vs oracle), slow |

The standalone INT8 xclbins have host-side math fixes applied (8 rounds) but
end-to-end coherent output has not been re-validated. The fused xclbin path
is numerically verified but runs at 4 tok/s.

---

*Benchmarks verified on-device July 5, 2026. Production daemon: C++23, 110 KB.*
