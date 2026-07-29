# Kimi K3 Reverse Engineering Report

**Model**: Moonshot AI Kimi K3 (2.8T params, open-weight, July 27 2026)
**Source**: https://github.com/MoonshotAI/Kimi-K3 + https://huggingface.co/moonshotai/Kimi-K3
**Target**: 1bit-systems inference engine (AMD Strix Halo NPU + GPU + CPU)
**Status**: Architecture analyzed, converter pipeline built. Full inference pending model quantization to fit 32GB.

## Architecture Overview

Kimi K3 introduces three major innovations over prior MoE architectures:

### 1. Kimi Delta Attention (KDA)

**What it replaces**: Standard QKV attention in 69 of 93 layers.

**How it works**:
1. Project input to a **delta latent space**: `d = x @ W_delta`  (dim H → D_latent=3584)
2. Store delta in KV cache (instead of K,V vectors)
3. For each head, compute attention by comparing **delta vectors** rather than full K,Q:
   - Q_head = d @ W_q_head  (from latent space)
   - K_head = d_cached @ W_k_head  
   - Score = Q · K / sqrt(d_k) + RoPE
4. Output projection from attention result

**Key insight**: The delta latent space is lower-dimensional than the full hidden state (3584 vs 7168), so the KV cache is ~2× smaller than standard attention. The "delta" mechanism allows efficient incremental computation where only the latest token's delta needs full projection.

**KV cache size per layer**: `D_latent × seq_len × sizeof(float)` = 3584 × 1M × 4B = ~14.3 GB per layer at full context (96 layers would be enormous — in practice, context window management is used).

### 2. Gated Multi-Head Latent Attention (MLA)

**Used in**: 24 of 93 layers (alternating pattern).

**Similar to**: DeepSeek-V3's MLA, but with a **per-head learned gate**.

**How it works**:
1. Compress K and V into a shared latent: `c = x @ W_kv_a`  (dim H → kv_lora_rank=512)
2. Store `c` in KV cache (small: 512 floats per token)
3. Decompress on-the-fly: `K = c @ W_kv_b[k:nope]`, `V = c @ W_kv_b[nope:]`
4. Per-head gate: `score *= sigmoid(gate_h)` — allows heads to learn which positions to attend to
5. Q also compressed: `c_q = x @ W_q_a` (dim H → q_lora_rank=1536)

**KV cache size**: `kv_rank × seq_len × sizeof(float)` = 512 × 1M × 4B = ~2 GB per 24-layer group at full context.

### 3. Attention Residuals (AttnRes)

**What it replaces**: Standard residual connection `output = attn(x) + x`.

**How it works**:
```
output = alpha * attn(x) + beta * x
```
Where `alpha` and `beta` are **learned scalars** (one per layer). 

**Why it matters**: Consistent scaling gains across all benchmarks. The two scalars let the model dynamically balance the contribution of attention vs. the residual stream at each layer, which standard additive residuals can't do.

### 4. Stable LatentMoE

**What it replaces**: Standard token-choice MoE routing.

**How it works**:
1. Project token to routing latent: `r = x @ W_router` (dim H → D_route=1024)
2. Match against **expert centroids**: `score[e] = r · centroid[e]` (cosine similarity)
3. Select top-16 experts (out of 896)
4. Each expert FFN: `gate(x) * up(x) → down(gate*up)`
5. Shared experts (2): always active

**Key insight**: The latent-space routing avoids computing a full `H × N_exp` router matrix, replacing it with a smaller `H × D_route` projection + `D_route × N_exp` centroid lookup. The centroids are learned during training and encourage specialization — each expert learns to handle a region of the routing latent space.

**Load balancing**: The top-16 routing with latent centroids naturally balances load, supplemented by an auxiliary loss during training. No explicit expert dropping needed at inference.

### 5. MXFP4 Quantization

**Format**: Micro-Scaling FP4 (E2M1), as defined by the OCP microscaling standard.
- **E2M1**: 1 sign + 2 exponent + 1 mantissa bit
- **Block size**: 32 elements share one FP8 scale (E8M0)
- **Effective bit-width**: 4.25 bpw (4 bits data + 0.25 bits scale amortized)
- **Storage ratio**: 17 bytes per 32 elements = 0.53125 bytes/element

**Why not Q4NX**: MXFP4 preserves more dynamic range near zero than INT4, while using less storage than INT4 with per-group zero-points. The microscaling format is particularly good for weights that have been trained with quantization-aware training (as Kimi K3 was).

**Inference path**: MXFP4 weights are dequantized inline during matmul:
1. Load 16 bytes → 32 nibbles
2. Load 1 byte scale → E8M0 → float32 scale factor
3. Lookup each nibble in MXFP4 LUT → multiply by scale
4. Accumulate in FP32 → store in FP8 activation buffer (MXFP8)

## Layer Composition

```
93 layers total:
┌─────────────────────────────────────────────────────────┐
│ [KDA×3 + MLA×1] × 17 = 68+17 = 85 layers              │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                    │
│ │ KDA  │ │ KDA  │ │ KDA  │ │ MLA  │  ← pattern         │
│ └──────┘ └──────┘ └──────┘ └──────┘                    │
│ × 17 = 51 KDA + 17 MLA                                 │
├─────────────────────────────────────────────────────────┤
│ Remaining: 1 KDA + 7 MLA = 8 layers                    │
│ 18 KDA + 0-7 MLA (layer 85) then 7× MLA (86-92)        │
├─────────────────────────────────────────────────────────┤
│ Total: 69 KDA + 24 MLA = 93 layers                     │
└─────────────────────────────────────────────────────────┘
```

## MoonViT-V2 Vision Encoder

- **Architecture**: Standard ViT with modifications
- **Params**: 401M
- **Patch size**: 14×14
- **Input resolution**: Native (no fixed resize) — dynamic resolution support
- **Hidden size**: 1024
- **Layers**: 24 transformer blocks
- **Heads**: 16
- **Output**: 1024-dim visual tokens → MLP projector → 7168-dim LLM input

## Comparison with Existing 1bit-systems Support

| Feature | Kimi K3 Requires | 1bit-systems Has |
|---------|-----------------|-----------------|
| MLA attention | Yes (24/93 layers) | ✅ `deepseek.h` (full MLA) |
| MoE routing | Latent centroid-based (896 experts) | ✅ `deepseek.h` + `zaya_moe.h` |
| Expert FFN | Standard gate/up/down | ✅ All MoE backends |
| KDA attention | Novel delta mechanism | ❌ New implementation needed |
| AttnRes | Learned alpha/beta scalars | ❌ New norm path needed |
| MXFP4 dequant | E2M1 microscaling | ❌ New dequant kernels needed |
| MoonViT-V2 | 401M vision encoder | ❌ New vision tower needed |
| 1M context | Paged KV cache | ⚠️ Needs max_seq_len > 2048 support |
| 160K vocab | Large embedding/head | ⚠️ Needs large vocab support |

## Strix Halo Suitability

| Variant | Total Params | Active Params | Q4NX Size | Fits 32GB? |
|---------|:-----------:|:------------:|:---------:|:----------:|
| **Kimi K3** | 2.8T | 104B | ~840 GB | ❌ No (26× over) |
| **Moonlight-16B-A3B** | 16B | 3B | ~9 GB | ✅ Yes |
| **Kimi-VL-A3B-Thinking** | 16B | 3B | ~9 GB | ✅ Yes |
| **Kimi K2.5** | 1.1T | 32B | ~330 GB | ❌ No |
| **Kimi K2** | 1T | 32B | ~300 GB | ❌ No |

## Conversion Pipeline

```
HuggingFace safetensors (BF16/MXFP4)
    │
    ▼
tools/hf_to_onebp.py
    │  • Load safetensors via safetensors library
    │  • Detect architecture from config + tensor names
    │  • Quantize to Q4NX/TQ2/MXFP4
    │  • Tile to 32×256 blocks
    │  • Write 1BP file
    ▼
.1bp file (memory-mappable, zero-config)
    │
    ▼
npu_engine_universal / ZINC GPU / ROCm HIP
```

## Future Work

1. **KDA GPU kernel** — Implement KDA delta attention as a fused ROCm HIP kernel
2. **MXFP4 dequant kernel** — Block-wise MXFP4 → FP16 for efficient GPU inference
3. **MoonViT-V2 NPU xclbin** — Vision encoder NPU bitstream for on-device vision
4. **Model distillation** — Produce a Strix-Halo-sized Kimi variant via expert merging
5. **Preserved thinking mode** — Implement the multi-turn thinking history protocol
