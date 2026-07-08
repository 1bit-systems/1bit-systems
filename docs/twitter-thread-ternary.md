# Twitter/X Thread — Native Ternary NPU

---

**Tweet 1** 🧵

AMD locked 2-bit compute behind a proprietary runtime on their own silicon.

I bought a Strix Halo. I got angry. I fixed it.

32-core native ternary kernel. 128/128 bit-exact. 118.9 µs.
314 KB xclbin. No FastFlowLM. MIT licensed.

The silicon was never the bottleneck. 🔥👇

---

**Tweet 2**

Ternary models: weights are {-1, 0, +1}. 1.58 bits.

Bonsai-1.7B = 250 MB. Same model FP16 = 3.4 GB.

AMD's NPU can process 4 ternary weights per byte. The vector units
are right there. AMD just decided you can't use them without paying.

So we wrote the kernel they should have shipped.

---

**Tweet 3**

`mm_ternary_32x64x128`:

Reads 2-bit packed ternary. Decodes on-the-fly.
Multiplies against BF16. Reduces via dot product.
Applies per-row scale. Outputs M bf16 scalars.

4× less memory traffic than INT8.
4× more weights per byte than the "official" path.

---

**Tweet 4**

The all-ones test:

Every single output = -256.0000 exactly.

Not "approximately." Not "within BF16 tolerance." EXACTLY.

128 outputs. 32 cores. All bit-exact.

On real silicon. Strix Halo XDNA2. Right now.

---

**Tweet 5**

The routing bug that took 31 of 32 cores:

Built 32-core xclbin. Loaded it. Ran it.

1 core: -256.0000 ✅
31 cores: 0.0000 ❌

AMD's AIE interconnect silently fails on cross-column broadcast.
The official examples rely on this pattern. It's a lie.

Fix: per-column DMA. Each column feeds its own cores.

---

**Tweet 6**

We discovered the NPU2 routing limitation the hard way —
probing raw bytes, finding 4 correct values out of 128.

Same-column routing works. Cross-column doesn't.
First time this has been proven on XDNA2.

"No NDAs. No inside access. Just a C++ compiler and spite."

---

**Tweet 7**

What's running:

| Config | Latency | Throughput | XCLBin |
|--------|---------|------------|--------|
| 1 core | 68.3 µs | 14,636/s | 16 KB |
| 32 cores | 118.9 µs | 8,410/s | 314 KB |

Q2_0 decoder: cos=1.000000 vs F16 (reverse-engineered bit-for-bit)

ZAYA1-8B: 5.6 GB, 40 layers, 2048d — benchmarked for reference

---

**Tweet 8**

Open source. MIT. Do whatever you want.

`github.com/bong-water-water-bong/1bit-systems`

No NDAs. No proprietary runtime.
Just a C++ compiler and the silicon AMD put in your laptop.

Your hardware. Not AMD's.

---

*"The silicon was never the bottleneck. The business model was."*

#NativeTernary #AMDNPU #StrixHalo #XDNA2 #32cores #BitExact
#YourHardware #MIT #OpenSource
