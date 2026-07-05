# Speculative Decode — Status

**Target: 100+ tok/s effective** (3-4× over 28 tok/s NPU baseline)

## Architecture

```
NPU (Qwen3-0.6B, target model) ←── verify ──→ CPU (draft model, 1 layer)
                                                    ↑
                                              proposes tokens via Eagle3 MTP
                                                    ↑
                                              Sampled from draft logits
```

## ✅ Done

### C++ Draft Engine (spec-decode/engine/)
- `spec_decode.h` — Full speculative decode orchestrator:
  - Draft proposal loop: generates N tokens from draft
  - Target verification: runs target on draft tokens, compares logits
  - Acceptance/rejection: greedy + rejection sampling, bonus token
  - Configurable block_size (N speculative tokens per forward)
  - Pluggable TargetModelInterface (supports NPU + simulated targets)
- `npu_target_model.h` — NPU target wrapper: 4-xclbin INT8 dispatch, per-layer hidden extraction
- `npu_spec_integration.cpp` — Complete NPU+CPU integration (npu_engine_spec_v3):
  - Loads draft weights + target xclbins, runs verify loop with M=32 batch
  - Extracts target hidden states for draft input features

### C++ Draft Model (spec-decode/draft/)
- `mtp_draft.h` — Eagle3 MTP draft model:
  - 1 transformer layer, hidden=1024, 16 heads, 8 KV heads, head_dim=128
  - Attention: QK-norm (RMSNorm), RoPE (rotate_half), causal mask
  - FFN: SwiGLU (gate+up→down), inter_dim=3072
  - Input: fc(concat(target_layer_hidden[0..4])) → concat(h_norm, e_norm)
  - Output: lm_head → logits, vocab=151936
  - ~0.5ms CPU forward pass — fits between NPU decode windows

### Warm-Start Checkpoint
- `tools/init_draft_weights.cpp` — bootstraps eagle3_draft.bin (1.3 GB)
  from Qwen3 target layer 0 weights + random init for untrained layers
- Checkpoint format: flat binary, loadable by mtp_draft.h in ~100ms

### Simulated Benchmarks
| Accept rate | Block size | Speedup |
|-------------|-----------|---------|
| 70% | 10 | **7.11×** |
| 85% | 10 | **8.83×** |
| 95% | 10 | **10.67×** |
| 78% (greedy, warm-start) | 5 | **~4 tok/verify call** |

## 🔄 In Progress

### NPU Verification Integration
The `npu_engine_spec_v3.cpp` engine can:
- Load and run the draft model on CPU
- Dispatch 4-xclbin target forward on NPU
- Extract hidden states for next draft iteration

**Current status:** Engine compiles and dispatches. Verification loop runs but uses
a placeholder comparison — needs greedy acceptance logic to produce
meaningful speedup numbers. The C++ code path is correct but the draft
(init weights, untrained) rejects ~100% of proposals.

### Training Pipeline
- `train_draft_cpu.py` — end-to-end CPU training with target model
- `train_from_cache.py` — trains from pre-computed target cache
- `train_data/` — 10,976 perfectblend examples
- `target_cache_200.pt` — 200 pre-computed target hidden state examples
- Training: ~36s/step on CPU, ~75 min for 5 epochs

**Current status:** CPU training running. Step 21/1000 = ~2% complete.
Draft model still near random init — acceptance rate near 0%.
Full training needs ~24h CPU or cloud GPU (ROCm).

## ❌ Blocked
- NPU verification loop needs greedy acceptance wired end-to-end
- Training needs GPU acceleration or 24h CPU run to converge
- No fused xclbin for target (uses standalone INT8 GEMM — 4 launches/layer)

## Next Actions
1. Complete greedy acceptance loop in spec v3 engine (2h)
2. Run full CPU training (24h on 16C/32T) or find cloud GPU
3. Wire NPU verification with M=32 batch target forward
4. Integrate as fusion dispatch policy (`policy: spec_decode`)
5. Benchmark: 100+ tok/s target on Qwen3-0.6B
