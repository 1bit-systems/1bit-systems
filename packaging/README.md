# Packaging — 1bit.systems v2026.07.01

**Model-agnostic.** Any Qwen3-family model with 1024 hidden dim × 28 layers × 16 Q heads works with the same xclbins. **Client-agnostic.** The HTTP server speaks OpenAI-compatible JSON — Ollama, Open WebUI, LangChain, anything that hits `/v1/chat/completions` just works.

| Format | Status | Command |
|--------|--------|---------|
| **Binary tarball** | ✅ Built | `tar xzf 1bit-systems-2026.07.01-linux-amd64.tar.gz` |
| **Debian (.deb)** | ✅ Built | `sudo dpkg -i 1bit-systems_2026.07.01_amd64.deb` |
| **GitHub Release** | ✅ [v2026.07.01](https://github.com/bong-water-water-bong/1bit-systems/releases/tag/v2026.07.01) | `gh release download v2026.07.01` |
| **One-liner install** | ✅ | `curl -sL https://1bit.systems/install.sh \| bash` |
| **Ollama** | ✅ Modelfile | `ollama create qwen3-npu -f Modelfile` |
| **OpenAI SDK** | ✅ Drop-in | `client = OpenAI(base_url="http://localhost:8081/v1")` |
| **Open WebUI** | ✅ Compatible | Point `OPENAI_API_BASE` at the NPU server |
| **LangChain** | ✅ Compatible | `ChatOpenAI(openai_api_base="http://localhost:8081/v1")` |
| **Arch (AUR)** | 📋 PKGBUILD ready | `yay -S 1bit-systems-bin` (needs AUR submission) |
| **Homebrew** | 📋 Formula ready | `brew install 1bit-systems` (needs tap) |
| **Snap** | 📋 snapcraft.yaml ready | `snap install 1bit-systems` (needs snapcraft build) |
| **Docker** | 📋 Dockerfile ready | `docker run 1bit-systems/npu` (needs registry push) |

### Model Compatibility (same xclbins, no rebuild)

| Model | Hidden | Layers | Q Heads | KV Heads | Status |
|-------|--------|--------|---------|----------|--------|
| Qwen3-0.6B | 1024 | 28 | 16 | 8 | ✅ Native |
| Qwen3-0.6B-Chat | 1024 | 28 | 16 | 8 | ✅ Verified |
| Qwen3-0.6B-Base | 1024 | 28 | 16 | 8 | ✅ Compatible |
| Qwen3-Embedding-0.6B | 1024 | 28 | 16 | 8 | ✅ Compatible |
| Qwen3-VL-0.6B | 1024 | 28 | 16 | 8 | ✅ Weights compatible (vision head separate) |
| Qwen3.5-0.8B | 1024 | 28 | 16 | 8 | ✅ Compatible (verify) |

### Client Compatibility (same HTTP API, no SDK needed)

| Client | Integration | Effort |
|--------|-----------|--------|
| **Ollama** | `ollama create qwen3-npu -f Modelfile` | 1 command |
| **OpenAI Python** | `OpenAI(base_url="http://localhost:8081/v1")` | 1 line |
| **OpenAI JS** | `new OpenAI({baseURL: "http://localhost:8081/v1"})` | 1 line |
| **Open WebUI** | Set `OPENAI_API_BASE` env var | 1 env var |
| **LangChain** | `ChatOpenAI(openai_api_base=...)` | 1 param |
| **LlamaIndex** | `OpenAI(api_base=...)` | 1 param |
| **curl** | `curl -d '{"messages":[...]}' localhost:8081/v1/chat/completions` | 0 deps |
| **Anything with HTTP** | POST JSON → get JSON back | Universal |

## Included in every package

| Binary | Purpose | Size |
|--------|---------|------|
| `1bit-npu` | CLI inference engine | 56 KB |
| `1bit-server` | HTTP API server (OpenAI-compatible) | 43 KB |
| `dequant_q4nx.o` | Q4NX weight dequantizer | 2.4 KB |

## Build them yourself

```bash
# Binary tarball
tar -czf 1bit-systems-2026.07.01-linux-amd64.tar.gz \
  packaging/binary/1bit-npu packaging/binary/1bit-server \
  engine/npu/build/dequant_q4nx.o

# Debian package
mkdir -p packaging/deb/usr/bin packaging/deb/usr/lib/1bit
cp packaging/binary/1bit-npu packaging/deb/usr/bin/
cp packaging/binary/1bit-server packaging/deb/usr/bin/
cp engine/npu/build/dequant_q4nx.o packaging/deb/usr/lib/1bit/
dpkg-deb --build packaging/deb 1bit-systems_2026.07.01_amd64.deb

# Docker image
docker build -t 1bit-systems/npu:2026.07.01 -f packaging/docker/Dockerfile .
docker run --device /dev/accel/accel0 -p 8081:8081 1bit-systems/npu:2026.07.01

# Snap
snapcraft --destructive-mode

# AUR
cd packaging/aur && makepkg -si
```
