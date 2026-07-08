# Hacker News Post Draft

## Title:
**Native ternary on AMD's NPU: 32 cores, 128/128 bit-exact, 118.9 µs — on a laptop**

## Body:

AMD shipped the Strix Halo with a 50 TOPS NPU and 32 AIE cores. Then they
locked sub-8-bit compute behind a proprietary runtime.

The silicon can do 2-bit packed ternary natively. The vector units are right
there. AMD just decided you shouldn't be allowed to use them without paying.

I bought one. I got angry. I fixed it.

**32-core native ternary kernel. 128/128 bit-exact. 118.9 microseconds.
314 KB xclbin. No FastFlowLM. No proprietary anything. MIT licensed.**

### What this is

- **Native ternary on the NPU**: 2-bit packed weights → BF16 MAC, decoded on-the-fly
- **128/128 bit-exact**: all-ones test = -256.0000 exactly. Not "within tolerance." EXACTLY.
- **118.9 µs per call**: 32,768 MACs (128 rows × 256 ternary) in one dispatch
- **314 KB xclbin**: built with open-source Chess C++ + MLIR. Zero proprietary bits.
- **Q2_0 decoder**: cos=1.000000 vs F16. We reverse-engineered the format bit-for-bit.

### The "wait, AMD locked that?" part

Ternary models store weights as {-1, 0, +1}. Bonsai-1.7B = 250 MB. Same model
at FP16 = 3.4 GB. That's 13.6× smaller. The entire thing fits in L3 cache.

AMD's NPU can process 4 ternary weights per byte — the AIE vector units have
native 2-bit decode paths. But the official toolchain only exposes INT8. So
everyone running 1-bit models on AMD NPUs has been upcasting to INT8 first,
throwing away the whole point of ternary.

We wrote the kernel they should have shipped: `mm_ternary_32x64x128`. Decodes
2-bit packed ternary on-the-fly. Multiplies against BF16 activations. Reduces
via dot product. Applies per-row scale. 4× less memory traffic than INT8.

### The routing bug that took out 31 of 32 cores

Built the 32-core xclbin. Chess compiled. MLIR generated. aiecc packaged.
Loaded it on the NPU. Ran the all-ones test.

1 core produced -256.0000. The other 31 produced zeros.

The AIE interconnect on XDNA2 does NOT support cross-column broadcast from
a mem tile to cores in other columns. The official examples rely on this
pattern. It silently fails on hardware. You get one column working and 31
cores sitting idle.

We discovered this the hard way — probing the raw output, finding exactly
4 correct values out of 128. The fix: per-column DMA routing. Each of 8
shim tiles feeds its column's 4 cores directly. Same-column routing works
fine. Cross-column is a lie.

First time this pattern has been proven on XDNA2.

### What's next

- Full Bonsai-1.7B on the NPU (tile the 2048-dim GEMV across kernel calls)
- Per-layer xclbins at production dimensions
- HIP kernel integration for the Radeon 8060S (13 GEMV kernels, 5 packing formats)
- End-to-end spec-decode: NPU draft + GPU verify

### Links

GitHub: https://github.com/bong-water-water-bong/1bit-systems  
Kernel: `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp`  
Docs: `docs/ternary-npu.md`

MIT. Open source. No NDAs. No inside access. Just a C++ compiler and spite.

---

*"The silicon was never the bottleneck. The business model was."*
