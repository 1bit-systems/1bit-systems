# Architecture

## Inference Pipeline

```
Token → Embedding → [Layer ×28] → LM Head → Sample → Token
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    RMS Norm       Attention        MLP (SiLU)
         │               │               │
    QKV GEMM      Q·K^T softmax    GU GEMM
    (INT8 xclbin)  + V weighted    (INT8 xclbin)
         │               │               │
         └───────────────┼───────────────┘
                         │
                    D GEMM
                    (INT8 xclbin)
                         │
                    Residual + Norm
```

## INT8 Quantization

Weights are dequantized from Q4NX (4-bit with per-group BF16 scales) to float32,
then re-quantized to symmetric INT8 using per-tensor max-abs scaling:

```
q(x) = round(x / scale), scale = max(|x|) / 127
```

This is done once at startup. The INT8 weights are uploaded to pre-allocated
per-layer BOs and never copied during inference.

Activations are quantized on-the-fly before each GEMM:
```
A_int8 = round(A_float * 127/5.0)
```
The fixed scale of 5.0/127.0 was empirically validated on Strix Halo.

## NPU Context Architecture

```
xrt::device (1 instance)
  ├── xclbin: QKV (128×1024×4096, group_id_B=4)
  │   ├── layerB[0..27] — pre-loaded INT8 weight BOs
  │   └── bA, bC — activation/output scratch BOs
  ├── xclbin: O   (128×2048×1024, group_id_B=4)
  │   └── layerB[0..27]
  ├── xclbin: GU  (128×1024×6144, group_id_B=4)
  │   └── layerB[0..27]
  ├── xclbin: D   (128×3072×1024, group_id_B=4)
  │   └── layerB[0..27]
  └── xclbin: ATTN ×4 (per-window, optional)
```

All 4 GEMM contexts are created at startup and kept alive.
NPU2 supports multiple concurrent hw_contexts — we've verified 8 alive at once.

## Key Fixes

### K-Interleaving Bug

The original INT8 MLIR generator used a shared A ObjectFifo with round-robin
distribution. With 8 cores and depth=2, each core only saw 1/8 of the K-dimension
— producing 394% error on random data.

**Fix**: Added `dataReuse` annotations (`dimensionsToStream`/`dimensionsFromStream`)
to the A_L2L1 ObjectFifo, switching DMA from round-robin to broadcast mode.
All 8 cores now receive the same A data. Verified correct on random data.

### BFP16 Precision Collapse

BFP16 GEMMs with Chess-compiled xclbins produced correct matmul results but
accumulated precision loss over 28 layers — the hidden state collapsed to a
fixed point and the model repeated the same token.

**Fix**: Switched to INT8 quantization, which avoids the double-quantization
(Q4NX→BFP16→hardware) in the BFP16 pipeline. INT8 weights are directly
quantized from float32, preserving per-value precision.

### NPU2 Context Limit

Multiple simultaneous hw_contexts were thought to cause ERT state=8 timeouts.

**Fix**: The limitation was firmware-version-dependent. Our firmware (1.1.2.65)
supports 8+ concurrent contexts. XRT hw_context objects can coexist as long
as kernels are invoked sequentially.
