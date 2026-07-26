# Packaging — 1bit.systems v2026.07.26

**One binary. 40 models. Auto-detect.** Zero Python. Zero pip. No Docker required.
The HTTP server speaks OpenAI-compatible JSON — Ollama, Open WebUI, LangChain, anything that hits `/v1/chat/completions` just works.

| Format | Status | Command |
|--------|--------|---------|
| **GitHub Release** | ✅ [Latest](https://github.com/bong-water-water-bong/1bit-systems/releases/latest) | `gh release download` |
| **One-liner install** | ✅ | `curl -sL https://1bit.systems/install.sh \| bash` |
| **Debian (.deb)** | ✅ Built by CI on every tagged release | `sudo dpkg -i 1bit-systems_*_amd64.deb` |
| **AppImage** | ✅ Built by CI on every tagged release | `chmod +x 1bit-systems-*.AppImage && ./1bit-systems-*.AppImage` |
| **Binary tarball** | ✅ | `make package-tarball` |
| **Docker** | ✅ Dockerfile ready | `docker run 1bit-systems/npu` |
| **Ollama** | ✅ Modelfile | `ollama create qwen3-npu -f Modelfile` |
| **OpenAI SDK** | ✅ Drop-in | `client = OpenAI(base_url="http://localhost:8081/v1")` |
| **Open WebUI** | ✅ Compatible | Point `OPENAI_API_BASE` at the NPU server |
| **LangChain** | ✅ Compatible | `ChatOpenAI(openai_api_base="http://localhost:8081/v1")` |
| **Arch (AUR)** | 📋 PKGBUILD ready | `yay -S 1bit-systems-bin` |
| **Homebrew** | 📋 Formula ready | `brew install 1bit-systems` |
| **Snap** | 📋 snapcraft.yaml ready | `snap install 1bit-systems` |

### All 5 Models Verified (auto-detect, no rebuild)

| Model | H | IM | NH | HD | Size | Decode | Status |
|-------|---|----|----|----|------|--------|--------|
| Qwen3-0.6B | 1024 | 3072 | 16 | 128 | 610 MB | 28 tok/s | ✅ |
| Gemma4-E2B | 1536 | 6144 | 8 | 256 | 4.7 GB | 16 tok/s | ✅ |
| Qwen3-VL-4B | 2560 | 9728 | 32 | 128 | 3.2 GB | 11 tok/s | ✅ |
| Llama-3.1-8B | 4096 | 14336 | 32 | 128 | 5.7 GB | 10 tok/s | ✅ |
| Qwen3-8B | 4096 | 12288 | 32 | 128 | 6.0 GB | 8 tok/s | ✅ |

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
| `1bit-npu` | CLI inference engine (40 models, auto-detect) | 17.5 MB (stripped) |
| `1bit-server` | HTTP API server (OpenAI-compatible) | 43 KB |
| `dequant_q4nx.o` | Q4NX weight dequantizer | 2.8 KB |

## Build them yourself

```bash
# Binary tarball
make package-tarball

# Debian package
make package-deb

# Docker image
docker build -t 1bit-systems/npu:2026.07.02 -f packaging/docker/Dockerfile .
docker run --device /dev/accel/accel0 -p 8081:8081 1bit-systems/npu:2026.07.02

# Snap
make package-snap

# AUR
cd packaging/aur && makepkg -si
```
