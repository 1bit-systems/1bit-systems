# 1bit.MONSTER

<sub>**One engine. Any model. Zero Python.**</sub>

A model-agnostic, hardware-agnostic, pure-C++23 inference engine — MIT licensed.
One binary that runs Hugging Face models on **NPU + GPU + CPU**. 16 model families,
the native **1BP** format, and **JARVIS** (the flagship voice assistant) all inside
`build/1bit`.

100% HF model coverage, any hardware. An open-source, pure-C++ inference engine.
NPU + GPU + CPU in one engine. Zero Python. MIT.

---

### The lines

- **One engine, any model** — detects the architecture and picks a kernel path. No config, no glue.
- **Zero Python** — pure C++23. No virtualenv, no pip, nothing to babysit. Source to model in three commands.
- **The NPU story** — AMD's closed XDNA 2 stack, 22 proprietary libraries, was reverse-engineered in four days. And now it beats it.

---

### Hardware targets

| Backend | Notes |
|---|---|
| **NPU** | XDNA 2 DPU kernels, Q4NX / 1BP, 64 MB SRAM |
| **GPU** | HIP (ROCm) + Vulkan, fused MoE shaders |
| **CPU** | scalar + OpenMP fallbacks |

One binary, every backend. `build/1bit` is the engine, the serving server, the CLI, and the voice assistant.

---

### Model coverage

16 documented families: `qwen`, `llama`, `deepseek`, `gemma`, `mistral`, `phi`,
`olmo`, `gpt-oss`, `falcon`, `granite`, `kimi`, `laguna`, `bitnet-bonsai`,
`smollm`, `whisper`, `zyphra`. The **Zyphra** family is the one the engine was
tuned against and powers JARVIS by default.

---

### Quick start

```bash
git clone https://github.com/1bit-MONSTER/1bit-MONSTER && cd 1bit-MONSTER
bash install.sh            # build
bash install.sh --with-jarvis   # build + JARVIS launcher + config

./build/1bit zaya               # serve on :8088
./build/1bit zaya --port 8080   # custom port
```

---

### The site

This repo hosts the **1bit.MONSTER marketing site** — a self-contained static
site (no build step, no dependencies), designed in a modern-minimal light system
with a 1-bit pixel identity.

| Page | File |
|---|---|
| Landing / index | [`index.html`](index.html) |
| Engine | [`1bit-monster-v2.html`](1bit-monster-v2.html) |
| Models | [`1bit-models.html`](1bit-models.html) |
| JARVIS | [`1bit-jarvis.html`](1bit-jarvis.html) |
| Blog | [`1bit-blog.html`](1bit-blog.html) |
| Store | [`1bit-store.html`](1bit-store.html) |
| Docs | [`1bit-docs.html`](1bit-docs.html) · `docs-*.html` |
| Benchmarks | [`1bit-benchmarks.html`](1bit-benchmarks.html) |

---

### Design system

- **Light modern-minimal ground** — near-white background, ink type, hairline structure.
- **1-bit pixel identity** — the relay mark, crisp pixel art, one blue accent + one status green.
- **Type** — system display sans + system body; mono for numerics, labels, micro-labels.

---

### License

MIT. See [`LICENSE`](LICENSE).
