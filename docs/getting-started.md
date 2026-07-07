# Getting Started — 1bit.systems NPU Engine

Run LLM inference on your AMD Strix Halo NPU. One binary, zero dependencies.

## What You Need

| Requirement | Details |
|-------------|---------|
| **Hardware** | AMD Ryzen AI Max+ 395 (Strix Halo) with XDNA 2 NPU |
| **OS** | Linux (Ubuntu 26.04 recommended) |
| **XRT** | 2.21+ (`sudo apt install libxrt2 libxrt-npu2 libxrt-dev`) |
| **A model** | Q4NX format (see below) |

## Install (30 seconds)

```bash
# Option A: One-liner install (recommended)
curl -sL https://1bit.systems/npu-install.sh | bash

# Option B: Download from GitHub Releases
# Go to https://github.com/bong-water-water-bong/1bit-systems/releases/latest
# Download 1bit-systems-2026.07.02-linux-amd64.tar.gz
tar xzf 1bit-systems-*.tar.gz
cd 1bit-systems-* && bash npu-install.sh
```

## Get a Model

Place a `.q4nx` model file in `~/.local/share/1bit/models/` or anywhere on disk.

Supported models (all auto-detected — no rebuild needed):

| Model | Size | Decode Speed | File |
|-------|------|-------------|------|
| Qwen3-0.6B | 610 MB | 28 tok/s | `qwen3-0.6b.q4nx` |
| Qwen3-VL-4B | 3.2 GB | 11 tok/s | `qwen3-vl-4b.q4nx` |
| Gemma4-E2B | 4.7 GB | 16 tok/s | `gemma4-e2b.q4nx` |
| Llama-3.1-8B | 5.7 GB | 10 tok/s | `llama-3.1-8b.q4nx` |
| Qwen3-8B | 6.0 GB | 8 tok/s | `qwen3-8b.q4nx` |

## Run

### CLI inference

```bash
# Auto-detect model in ~/.local/share/1bit/models/
1bit-npu --auto 16

# Or specify a model file directly
1bit-npu /path/to/model.q4nx 16
```

The second argument is the number of OpenMP threads. Use 16 on Ryzen AI Max+ 395.

### HTTP API server (OpenAI-compatible)

```bash
# Start the server on port 8081
1bit-server 8081

# In another terminal, chat with it:
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### From any OpenAI-compatible client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1")
response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Build from Source

> The standalone `engine/npu/` C++ engine referenced by earlier versions of
> this guide was retired (commit `cd232a091`) — superseded by the
> `spec-decode/` stack. The build command below for the HTTP server still
> applies; see [docs/wiki/performance.md](../docs/wiki/performance.md) for how
> to run current, production NPU/GPU inference (FLM proxy / ZINC Vulkan).

```bash
# Prerequisites
sudo apt install g++ libxrt2 libxrt-npu2 libxrt-dev

# Clone and build
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems

# Build the HTTP server
g++ -std=c++23 -O3 -o 1bit-server packaging/binary/server.cpp
```

See [building.md](building.md) for detailed prerequisites and [architecture.md](architecture.md) for engine design (historical — describes the now-retired NPU engine).

## Performance Tuning

For best performance:

```bash
# Use all cores
OMP_NUM_THREADS=16 1bit-npu model.q4nx 16

# Batch size matters — M=32 gives 28 tok/s on Qwen3-0.6B
# Single-token decode is slower (~4 tok/s)
```

Full benchmarks at [docs/wiki/performance.md](../docs/wiki/performance.md).

## Docker

```bash
docker run --device /dev/accel/accel0 -p 8081:8081 \
  1bit-systems/npu:latest
```

## Troubleshooting

| Symptom | Likely Fix |
|---------|-----------|
| `libxrt_coreutil not found` | Install XRT: `sudo apt install libxrt2 libxrt-npu2 libxrt-dev` |
| `No NPU device found` | Check `lspci \| grep -i npu`. Ensure NPU driver is loaded: `sudo modprobe amdxdna` |
| `Segfault on start` | XRT version mismatch. Run `xbutil examine` to check version |
| `Slow inference` | Ensure `OMP_NUM_THREADS` matches your core count |
| `Model not detected` | File must have `.q4nx` extension and be in supported format |

## Next Steps

- [Full documentation](https://1bit.systems)
- [Benchmarks](wiki/performance.md)
- [Architecture overview](architecture.md)
- [Build from source guide](docs/building.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)
- [Discord community](https://discord.gg/dSyV646eBs)
