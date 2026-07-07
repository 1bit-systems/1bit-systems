# Speculative Decode — Status

## Architecture

```
Draft (CPU, Eagle3 1-layer / DSpark 5-layer)
    │ proposes tokens
    ▼
Target (NPU, fused xclbin via NpuFusedTarget — single launch/layer)
    │ verifies via forward_with_kv()
    ▼
Accept/Reject → next iteration
```

---

## 🚀 Breakthrough: Fused Layer Target (July 7)

The **fused target** (`NpuFusedTarget` in `engine/npu_fused_target.h`) is now functional:

- **Single xclbin per layer** (QKV→Attn→O→GU→SiLU→D all on NPU), vs 4 xclbins per layer
- **Load time: 671ms** (vs 5-12s for the 4-xclbin target)
- **Functional**: produces different outputs for different inputs (verified)
- **Xclbin reload fix** (`f5bae4049`): AIE2 array reset between layer dispatches
- Integrated into `npu_spec_decode --fused-spec` and `--fused` modes

**Known issue**: LM head produces all-zero outputs (numerical alignment with reference weights). Fused kernel compute is correct.

---

## ✅ New Modes

| Mode | Command | Target | Draft | Status |
|------|---------|--------|-------|--------|
| 4-xclbin spec-decode | `--spec-decode` | `NPUQwen3Target` (4 xclbins/layer) | DSpark 5-layer | ✅ Works (0.8 tok/s) |
| Fused | `--fused` | `NpuFusedTarget` (1 xclbin/layer) | None | ✅ Works (2.6 tok/s) |
| **Fused spec-decode** | **`--fused-spec`** | **`NpuFusedTarget`** | **Eagle3 1-layer** | **✅ End-to-end** |
| Daemon | `--daemon` | `NPUQwen3Target` | DSpark | ✅ |
| Daemon fused | `--daemon-fused` | `NpuFusedTarget` | None (spec disabled) | ✅ |
| Benchmark (simulated) | `--bench` | Simulated | DSpark | ✅ |

---

## 🏋️ Training (in progress)

| Run | Examples | Epochs | Loss | Acceptance | Data Source |
|-----|----------|--------|------|------------|-------------|
| FP (old) | ~10K | 5 | unknown | 0% | HuggingFace FP16 |
| NPU 200 | 200 | 5 | 12.1→6.4 | 0% | NPU INT8 hidden states |
| **NPU 1K** | **1,000** | **5 (running)** | **9.10 (ep1)** | **TBD** | **NPU INT8 hidden states** |

---

## Next Steps

1. **Wait for training** to finish (~2h) → test acceptance on fused target
2. **Fix LM head output** in fused target (numerical alignment with `npu_engine_fused_server.cpp`)
3. **Capture fused-target hidden states** (on new fixed LM head) → retrain draft on fused distribution
4. **Remove xclbin reload overhead** — batch layer dispatches or use persistent connection
5. **Benchmark**: target speed without reload overhead should approach 250+ tok/s
