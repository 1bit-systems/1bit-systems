# Twitter/X Thread Draft

---

**Tweet 1** 🧵

Bought an AMD Strix Halo laptop for the NPU.
The official stack: one model format, no mixing NPU/GPU/CPU per request.

So I built a router. Drop in any GGUF model, it picks the backend automatically.

Zero Python at runtime. MIT licensed. 👇

---

**Tweet 2**

The router reads a model's own on-disk metadata — architecture, quantization,
tensor shapes — and routes it to NPU, GPU, or CPU. No config files, no
manifest, no model registry.

11 GGUF quant formats supported, each dequantizer checked bit-exact against
an independent Python reference. Not "looks right" — actually diffed.

---

**Tweet 3**

The honest part: our own in-process NPU kernel has a confirmed correctness
bug on real hardware. Numbers looked great, output was garbage.

So the default NPU path delegates to FastFlowLM (already correct) instead —
disclosed in the README, not discovered after you build it.

---

**Tweet 4**

Real, validated numbers on Radeon 8060S + XDNA 2:

GPU ROCm HIP: 64 tok/s (kernel-level)
NPU via FastFlowLM: 57 tok/s
zaya_server end-to-end, Qwen 27B: 30 tok/s (real prompt, not synthetic)

llama.cpp on the same box: 229 tok/s. We're behind it and say so.

---

**Tweet 5**

Also ships a video generation path (Wan2.2, LTX-Video, AnimateDiff,
CogVideoX) with LoRA support, vendored in as its own module.

One repo, one build, LLM inference + video gen, both model-agnostic where
the underlying pipeline allows it.

---

**Tweet 6**

MIT licensed. Not "community license." Not "source available." MIT.

https://github.com/bong-water-water-bong/1bit-systems

The bugs are in the issue tracker too — including the ones we found in our
own published benchmarks.

---

**Hashtags (in reply to last tweet):**
#OneBinary #ModelAgnostic #NoPython #ZeroDeps #AMDNPU #StrixHalo #Cpp23 #OpenSource
