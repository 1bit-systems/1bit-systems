# Engine Status

## Production: FLM Proxy (94 tok/s) ✅

The NPU daemon proxies to FastFlowLM for production inference. Coherent output,
OpenAI-compatible API, used by `1bit chat` and all integration clients.

- **Port**: 9090
- **Models**: Qwen3-0.6B (turbo)
- **Status**: Verified, production stable

## C++ v12 Engine (97 tok/s) ⚠️ — Output Incoherent

The open-source C++ engine achieves 97 tok/s decode on Qwen3-0.6B, but has
**never produced coherent output**. All benchmarks measure throughput only —
correctness has not been validated.

### Known issues

1. **Q4NX weight format**: Dequantization produces incorrect values. The
   `dequant_q4nx.c` routine needs validation against FLM's reference output.
2. **Attention path**: CPU OpenMP attention works but numerical accuracy vs
   FLM's fused attention has not been verified.
3. **LM head**: The final projection layer may introduce errors that compound
   across tokens.

### Fix history

- 3 bugs found and fixed (details in [journey.md](journey.md#update-25))
- Output still incoherent after all fixes
- Root cause likely deeper — weight format or numerical path

### Next steps

- Lock-step comparison: run v12 and FLM side-by-side, dump activations per layer
- Compare dequantized weights against known-good reference
- Fused xclbin port (eliminates per-layer ioctl) may also fix correctness by
  matching FLM's dispatch path

## GPU ZINC Engine (22 tok/s) ✅

Vulkan compute shaders via ZINC. Verified coherent output on Bonsai-1.7B-F16.

---

*Last updated: July 4, 2026*
