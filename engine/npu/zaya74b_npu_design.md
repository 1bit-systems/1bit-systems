# Zaya1-74B NPU Inference Design

## Model Config

| Property | Value |
|----------|-------|
| Architecture | Dense transformer (ONEBP_DENSE) |
| hidden_size (H) | 4096 |
| num_layers (L) | 120 |
| num_attention_heads | 16 |
| num_kv_heads | 2 (MQA) |
| head_dim (HD) | 128 |
| intermediate_size (IM) | 4096 |
| vocab_size | 262147 |
| 1BP size | 739 MB |
| Est. params | ~74B (1-bit compressed) |

## NPU GEMM dimensions per layer

| Op | Input | Weight | Output | MACs |
|----|-------|--------|--------|------|
| QKV | H=4096 | [2560, 4096] | qkv_n=2560 | 10.5M |
| O | NH*HD=2048 | [4096, 2048] | H=4096 | 8.4M |
| G (gate) | H=4096 | [4096, 4096] | IM=4096 | 16.8M |
| U (up) | H=4096 | [4096, 4096] | IM=4096 | 16.8M |
| D (down) | IM=4096 | [4096, 4096] | H=4096 | 16.8M |
| **Per layer** | | | | **69.1M MACs** |
| **All 120 layers** | | | | **8.3B MACs** |

## xclbin compatibility analysis

The existing xclbins use tile sizes of 512 (output) × 256 (input) with quantized int8.

### `final_i8_D_llama.xclbin`
- Llama has H=4096, IM=11008. The D (down) xclbin for Llama handles IM→H at those dims.
- For Zaya-74B: IM=4096, H=4096 — **different dimensions than Llama**. 
- **Tile reuse**: The D kernel does 512-wide output tiles from 256-wide input chunks.
  - Zaya-74B D: 4096/512 = 8 body records × 4096/256 = 16 chunks/record.
  - Llama D: 4096/512 = 8 body records × 11008/256 = 43 chunks/record.
  - Different chunk counts → different instruction sequences.
- **Verdict**: New instruction file (`insts_i8_D_zaya_74b.txt`) needed. The xclbin *binary* (AIE graph) may be reusable if tile size = 512×256, but the instruction sequence changes.

### `final_i8_QKV_llama.xclbin`
- Llama QKV: H=4096 → qkv_n=128*32+2*128*8=6144.
- Zaya-74B QKV: H=4096 → qkv_n=16*128+2*2*128=2560.
- Different QKV output dimension → different body record count.
- **Verdict**: Needs new instruction file.

### `final_i8_GU_*.xclbin`
- For Zaya-74B: G and U both map H=4096 → IM=4096.
- The GU xclbin for models with IM=4096 would work if the G/U split ratio matches.
- **Verdict**: Needs investigation — may need instruction tuning.

## NPU alone: throughput estimate

At 50 µs per GEMM (NPU v12 measured throughput for H=2048):
- QKV: 50 µs
- O: 50 µs
- GU: 50 µs (fused? or 2×50µs?)
- D: 50 µs
- Attention: CPU fallback ~100 µs (at seq_len=1)
- RoPE/norm/residual: ~20 µs CPU
- **Per layer**: ~270 µs
- **120 layers**: ~32 ms
- **Decode throughput**: ~31 tok/s (NPU only)

With CPU attention (no NPU attn xclbin):
- Add ~100 µs for CPU attention
- **Total**: ~37 ms → ~27 tok/s

With NPU attention xclbin:
- Attention on NPU: ~20 µs
- **Total**: ~29 ms → ~34 tok/s

## CPU+NPU hybrid breakdown

```
Per token (greedy decode, seq_len=1):
  for each of 120 layers:
    RMS norm          CPU      5 µs
    QKV GEMM          NPU     50 µs
    Q/K norm + RoPE   CPU      5 µs
    KV cache store    CPU      2 µs
    Attention         CPU    100 µs  ← bottleneck for long sequences
    O GEMM            NPU     50 µs
    Residual add      CPU      2 µs
    RMS norm          CPU      5 µs
    GU GEMM           NPU     50 µs
    SiLU gate         CPU      5 µs
    D GEMM            NPU     50 µs
    Residual add      CPU      2 µs
    ───────────────────────────────
    Per layer:                276 µs  (118 µs NPU + 126 µs CPU + 32 µs sync)
  120 layers:                 33 ms
  Final norm + LM head:       10 ms
  ───────────────────────────────
  Total per token:            43 ms  → ~23 tok/s
```

## Key bottleneck

**CPU attention becomes the bottleneck at longer sequences.** At seq_len=1024, attention costs ~5ms per layer → 600ms per token → 1.7 tok/s. The `final_i8_KV_v.xclbin` + `final_i8_ATTN_v.xclbin` (NPU attention) is essential for long-context performance.

## Implementation plan

### Phase 1: CPU+NPU hybrid (this session)
Runs Zaya-74B with:
- NPU GEMM for QKV, O, GU, D using existing xclbins + reshaped instruction files
- CPU for RoPE, RMS norm, SiLU, residual, attention
- Target: ~15-20 tok/s at seq_len=1

### Phase 2: NPU attention
Add NPU attention xclbin (KV+ATTN):
- Reuse `final_i8_KV_v.xclbin` and `final_i8_ATTN_v.xclbin` with Zaya-74B dimensions
- Target: ~25-30 tok/s at seq_len=1, ~15 tok/s at seq_len=2048

### Phase 3: Pipelined decode
Use the fused pipeline (engine/fusion/zero_copy/) to overlap CPU and NPU work:
- GPU attention on slot A ∥ NPU FFN on slot B
- Target: ~30-35 tok/s at any seq_len
