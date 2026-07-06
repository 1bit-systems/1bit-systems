# Twitter/X Thread Draft

---

**Tweet 1** 🧵

AMD shipped Strix Halo with a 50 TOPS NPU.
Then locked INT8 behind proprietary software.

I bought one. I got angry. I fixed it.

4 days. 74KB. 94 tok/s. Open source.

The silicon was never the bottleneck. The business model was. 👇

---

**Tweet 2**

74KB binary. Think about that.

Your browser's favicon is bigger.
A 240p JPEG of a cat is bigger.

This binary runs 22 model architectures across video generation, photography, audio synthesis, and LLMs — all on your laptop.

Zero Python. Zero Docker. Zero pip. Just g++ and run.

---

**Tweet 3**

Model-agnostic isn't a buzzword here.

```
video-lora generate --model flux --prompt "cinematic portrait"
video-lora generate --model wan --prompt "cat walking"
video-lora generate --model stable-audio --prompt "rain on window"
```

Same CLI. Same engine. Auto-detected. LoRA on every backend.

22 models, 3 modalities, one 74KB binary.

---

**Tweet 4**

The NPU hits 94 tok/s on Qwen3-0.6B.
The GPU (Radeon 8060S) hits 22 tok/s on 1.7B.
The CPU scheduler fuses KV cache across all three.

On a consumer laptop. While it's on battery.

No data center. No A100. Just a ThinkPad with an NPU.

---

**Tweet 5**

"I reverse-engineered AMD's proprietary NPU stack in 4 days."

That sentence gets reactions. People think I'm exaggerating.

Day 1: downloaded their toolchain. Nothing worked.
Day 2: probed ioctl calls. Found the NPU interface.
Day 3: first inference. 244 ms/tok (terrible).
Day 4: 24× speedup. 16 ms/tok.

No NDAs. No inside access. Just a C++ compiler and spite.

---

**Tweet 6**

This is MIT licensed. Not "community license." Not "source available."
MIT. Do whatever you want.

https://github.com/1bit-systems/1bit

`curl -sL https://1bit.systems/npu-install.sh | bash`

Your hardware. Not AMD's.

---

**Hashtags (in reply to last tweet):**
#74kbBinary #OneBinaryToRuleThemAll #FusedEngine #ModelAgnostic
#NoPython #ZeroDeps #AMDNPU #StrixHalo #AntiVendorLock #Cpp23 #TheUnlock
