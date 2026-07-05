# Speculative Decode — Status

## ✅ Done
- C++ MTP draft model (mtp_draft.h) — Eagle3, 1 transformer layer, CPU
- Speculative decode loop (spec_decode.h) — draft → verify → accept/reject
- Warm-start checkpoint — bootstrapped from Qwen3 target layer 0 (eagle3_draft.bin, 1.3 GB)
- NPU spec engine built (npu_engine_spec_v3) — loads xclbins, runs draft
- Simulated bench: up to 10.67x speedup (spec_decode_bench)
- Training data: 10,976 perfectblend examples at train_data_10k/
- DeepSpec installed at ~/DeepSpec

## 🔄 In Progress
- NPU verification step: run target forward on draft tokens, compare logits
- M=32 batch verification: verify all draft tokens in one NPU forward
- Acceptance loop: reject mismatches, accept matches, bonus token

## ❌ Blocked
- Draft model training: needs GPU (ROCm on this system or cloud)
- Target cache extraction: needs GPU to run DeepSpec

## Next Actions
1. Train Eagle3 draft (needs cloud GPU or ROCm access)
2. Wire NPU verification into spec v3 engine
3. Integrate as dispatcher policy in engine/fusion/
