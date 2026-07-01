# Packaging

1bit.systems is architecture-agnostic — any model with 1024 hidden dim, 28 layers,
16 Q heads, 8 KV heads, and 128 head dim works with the same INT8 xclbins.

## Supported Formats

| Format | Status | Command |
|--------|--------|---------|
| **GitHub Release** | ✅ [v2026.07.01](https://github.com/bong-water-water-bong/1bit-systems/releases/tag/v2026.07.01) | `gh release download v2026.07.01` |
| **One-liner install** | ✅ | `curl -sL https://1bit.systems/install.sh \| bash` |
| **Debian (.deb)** | 📋 | `dpkg -i 1bit-systems.deb` |
| **Arch (AUR)** | 📋 | `yay -S 1bit-systems-bin` |
| **Homebrew** | 📋 | `brew install 1bit-systems` |
| **Snap** | 📋 | `snap install 1bit-systems` |
| **Docker** | 📋 | `docker run 1bit-systems/npu qwen3:0.6b` |
| **Ollama backend** | 📋 | `ollama run qwen3:0.6b --backend 1bit-npu` |

## Model Compatibility

| Model | Family | Hidden | Layers | Q Heads | KV Heads | Compatible |
|-------|--------|--------|--------|---------|----------|-----------|
| Qwen3-0.6B | qwen3 | 1024 | 28 | 16 | 8 | ✅ Native |
| Qwen3-0.6B-Chat | qwen3 | 1024 | 28 | 16 | 8 | ✅ Weights work |
| Qwen3-0.6B-Base | qwen3 | 1024 | 28 | 16 | 8 | ✅ Weights work |
| Qwen3-Embedding-0.6B | qwen3 | 1024 | 28 | 16 | 8 | ✅ Weights work |
| Qwen3-0.6B-Q8_0 (GGUF) | qwen3 | 1024 | 28 | 16 | 8 | ⚠️ GGUF loader WIP |
| Qwen3.5-0.8B | qwen3 | 1024 | 28 | 16 | 8 | ⚠️ Needs verification |
| Bonsai-1.7B IQ1_S | bitnet | 2560 | 28 | 20 | 5 | ❌ Different arch |
| Qwen3-8B | qwen3 | 4096 | 36 | 32 | 8 | ❌ Needs new xclbins |
