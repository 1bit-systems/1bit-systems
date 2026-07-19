# Getting Started — zaya_server

**Zaya1‑8B** inference server. Pure C++/HIP, one binary (~282 KB). No Python,
no Rust, no virtualenvs, no containers. Runs on **AMD Strix Halo** (Ryzen AI
Max+ 395) with ROCm GPU acceleration.

---

## Prerequisites

| Component     | Requirement                                                       |
|---------------|-------------------------------------------------------------------|
| **Hardware**  | AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151)                       |
| **OS**        | Ubuntu 24.04 LTS or later                                         |
| **Kernel**    | **6.18.22-lts or 7.x** — avoid 6.19.x (see warning below)         |
| **ROCm**      | 7.2.4 (HIP runtime + device library)                              |

> ⚠️ **Kernel warning (issue #1).** On Strix Halo (gfx1151), Linux **6.19.x**
> kernels have a reproducible `amdgpu` OPTC CRTC hang under sustained NPU/GPU
> load — the display pipe locks up mid-inference. Confirmed-stable kernels are
> **6.18.22-lts** and the **7.x** series. `install.sh` detects a 6.19.x kernel
> and warns. Check yours with `uname -r` before running GPU inference.
| **CMake**     | ≥ 3.28                                                            |
| **Ninja**     | ≥ 1.12                                                            |
| **Compiler**  | GCC ≥ 13 (C++17) + ROCm's `amdclang++` for HIP                    |
| **Git**       | —                                                                 |

---

## 1. Clone

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
```

---

## 2. Install ROCm 7.2.4

<details>
<summary><b>Repository install (recommended)</b></summary>

```bash
# Add AMD ROCm repository
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/rocm/apt/7.2.4 noble main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-hip-libraries rocm-hip-dev rocm-device-libs
```

Add to `~/.bashrc`:

```bash
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```
</details>

<details>
<summary><b>Alternative — pip TheRock SDK</b></summary>

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  rocm[devel,libraries]
export THEROCK_PIP_ROOT="$HOME/.cache/pip/therock"
```
</details>

Verify the HIP compiler is reachable:

```bash
amdclang++ --version
```

---

## 3. Build

```bash
# Configure
cmake -B build -G Ninja

# Build the server (and its librocm_cpp dependency)
cmake --build build --target zaya_server -j$(nproc)
```

The build fetches three header-only dependencies automatically via CMake
`FetchContent` — no manual install needed:

- **cpp-httplib** — HTTP server
- **nlohmann_json** — JSON parsing
- **FTXUI** — terminal UI (used by other tools, not the server itself)

On success you'll have a single binary:

```bash
ls -lh build/zaya_server
# -rwxrwxr-x  ... 207K build/zaya_server
```

> **Build targets reference** — other useful targets in the same project:
>
> | Target | Description |
> |--------|-------------|
> | `zaya_server` | HTTP inference server (this guide) |
> | `zaya_full` | Full 40‑layer GPU inference loop (no HTTP) |
> | `zaya_gpu_decode` | Q4NX model decoder |
> | `bitnet_decode` | BitNet/tri‑bit decode CLI |
> | `test_zaya_moe_gemv` | MoE ternary GEMV functional test |
> | `test_cca_attn` | CCA attention kernel test |

---

## 4. Prepare Weights

The server expects **Zaya1‑8B** weights as flat float16 binary files under
`/tmp/zaya_weights/`. Each tensor is stored as a separate `.bin` file named
after its HuggingFace‑style key, for example:

```
/tmp/zaya_weights/
├── model_embed_tokens_weight.bin              # [262272 × 2048] fp16
├── model_norm_weight.bin                      # [2048] fp16
├── model_input_hidden_states_scale.bin        # [2048] fp32
├── model_input_hidden_states_bias.bin         # [2048] fp32
├── model_layers_0_input_layernorm_weight.bin
├── model_layers_0_self_attn_qkv_proj_q_proj_weight.bin
├── model_layers_0_self_attn_qkv_proj_k_proj_weight.bin
├── model_layers_0_self_attn_qkv_proj_v_proj_current_weight.bin
├── model_layers_0_self_attn_qkv_proj_v_proj_delayed_weight.bin
├── model_layers_0_self_attn_o_proj_weight.bin
├── model_layers_0_self_attn_qkv_proj_conv_qk_depthwise_weight.bin  # fp32
├── model_layers_0_self_attn_qkv_proj_conv_qk_grouped_weight.bin    # fp32
├── model_layers_0_self_attn_qk_norm_temp.bin                        # fp32
├── model_layers_0_post_attention_residual_scale_*.bin              # fp32
├── model_layers_0_mlp_gate_down_proj_weight.bin                    # fp32
├── model_layers_0_mlp_gate_router_mlp_*.bin                        # fp32
├── model_layers_0_mlp_experts_gate_up_proj.bin                     # fp16
├── model_layers_0_mlp_experts_down_proj.bin                        # fp16
├── model_layers_0_post_mlp_residual_scale_*.bin                    # fp32
├── model_layers_0_mlp_gate_router_states_scale.bin                 # fp32, optional
├── ...  (× 40 layers)
```

Weights are available from the [1bit.systems releases page](
https://github.com/bong-water-water-bong/1bit-systems/releases) or can be exported
from a HuggingFace Zaya1‑8B checkpoint using the included extraction script.

```bash
# Example: download and extract the weight bundle
# (URL placeholder — check releases for the current bundle)
curl -L https://github.com/bong-water-water-bong/1bit-systems/releases/download/v0.2.1/zaya-weights.tar.gz \
  | sudo tar xz -C /tmp/
```

---

## 5. Run the Server

```bash
# Source the environment (sets HSA_OVERRIDE_GFX_VERSION, etc.)
source env.sh

# Start on default port 8088
./build/zaya_server

# Or specify a custom port
./build/zaya_server 8080
```

You should see:

```
Zaya1-8B Server (pure C++, no Python/Rust)
Loading 40 layers...
Ready on port 8088
```

The server loads all 40 layers of weights into GPU memory on startup (~6 GB for
fp16 weights + MoE parameters). Allow a few seconds for HIP initialization and
weight transfer.

---

## 6. Send Requests

### Health check

```bash
curl http://localhost:8088/
```

```json
{"status":"ok","model":"Zaya1-8B","version":"pure-cpp"}
```

### Text generation (prompt)

```bash
curl -X POST http://localhost:8088/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The future of AI is","n_predict":32}'
```

Response:

```json
{
  "tokens": [2, 261, ...],
  "text": "The future of AI is bright and full of possibilities...",
  "gen_ms": 2850.12,
  "tok_s": 11.2
}
```

### Text generation (pre‑tokenized)

```bash
curl -X POST http://localhost:8088/completion \
  -H "Content-Type: application/json" \
  -d '{"tokens":[2,9259],"n_predict":64}'
```

### From Python (OpenAI‑compatible client)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8088",
    api_key="not-needed"   # zaya_server does not require an API key
)

response = client.completions.create(
    model="zaya-1-8b",
    prompt="The future of AI is",
    max_tokens=32
)

print(response.choices[0].text)
```

> **Note:** The server implements the legacy `/completion` endpoint (not the
> OpenAI chat completions format). Use `client.completions.create(...)` rather
> than `client.chat.completions.create(...)`.

---

## CLI Reference

```
zaya_server [port]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `port`   | `8088`  | TCP port to listen on |

### Request body (`POST /completion`)

| Field       | Type           | Default | Description                            |
|-------------|----------------|---------|----------------------------------------|
| `prompt`    | string         | —       | Input text (mutually exclusive with `tokens`) |
| `tokens`    | array[int]     | —       | Pre‑tokenized input IDs                |
| `n_predict` | int            | 16      | Number of tokens to generate            |

### Response

| Field    | Type         | Description                           |
|----------|--------------|---------------------------------------|
| `tokens` | array[int]   | All output token IDs (prompt + generated) |
| `text`   | string       | Decoded text output                   |
| `gen_ms` | float        | End‑to‑end generation time in ms      |
| `tok_s`  | float        | Throughput in tokens per second       |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   curl / OpenAI client               │
└──────────────────────┬──────────────────────────────┘
                       │  POST /completion
                       ▼
┌─────────────────────────────────────────────────────┐
│                  zaya_server (~282 KB)               │
│                                                      │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────┐│
│  │  HTTP    │  │  Tokenizer  │  │  Inference Engine ││
│  │  Server  │──┤ (built‑in)  │──┤  (ROCm HIP)      ││
│  │(httplib) │  │             │  │                   ││
│  └──────────┘  └─────────────┘  │  • CCA attention  ││
│                                  │  • MoE expere  s ││
│                                  │  • RMS norm       ││
│                                  │  • SiLU/MoE FFN   ││
│                                  │  • lm_head        ││
│                                  └──────────────────┘│
│                                   │                  │
│                            librocm_cpp (shared lib)   │
│                           ┌─────────────────────────┐│
│                           │  • ternar  GEMV/GEMM    ││
│                           │  • WMMA tiled kernels   ││
│                           │  • KV cach  attention   ││
│                           │  • Prefill dispatcher   ││
│                           └─────────────────────────┘│
└──────────────────────┬──────────────────────────────┘
                       │  HIP launches (gfx1151)
                       ▼
            ┌─────────────────────┐
            │  AMD Radeon 8060S   │
            │  (Strix Halo GPU)   │
            │  128 GB unified mem │
            └─────────────────────┘
```

The server is a single process with one binary. There is no separate Python
runtime, no Rust component, and no containerization layer. The tokenizer is
compiled in — no external vocabulary file is required at runtime (though
optional `.htok` files can be loaded for full BPE support).

---

## Performance

Measured on AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S):

| Configuration    | Tokens/s | Notes                            |
|------------------|:--------:|----------------------------------|
| Prefill (32 tok) | ~18      | First‑token latency              |
| Decode (32 tok)  | ~11      | Steady‑state generation          |
| Decode (64 tok)  | ~10      | Slightly lower due to KV cache   |

Performance depends on GPU clock speeds, system thermals, and whether the MoE
expert data fits in VRAM (16 experts × 2 layers = heavy memory pressure).

---

## Troubleshooting

| Symptom | Likely Fix |
|---------|-----------|
| `amdclang++: command not found` | Add ROCm to `PATH`: `export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH` |
| `hipErrorNoBinaryForGpu` | Set `HSA_OVERRIDE_GFX_VERSION=11.5.1` (done by `source env.sh`) |
| `Missing: /tmp/zaya_weights/...` | Download and extract weights to `/tmp/zaya_weights/` |
| `hipMalloc failed` | Not enough GPU memory. Zaya1‑8B needs ~6 GB free. Check `rocm-smi` |
| `bind: Address already in use` | Port taken. Use a different port: `./build/zaya_server 8080` |
| Slow generation | Ensure weights are on a fast NVMe SSD for first‑load time. Sequential decode speed is bound by GPU memory bandwidth |

---

## Next Steps

- [Complete build guide](building.md) — detailed ROCm setup, CMake options,
  alternate backend targets
- [Architecture overview](architecture.md) — kernel design, attention variants,
  MoE router internals
- [Roadmap](roadmap.md) — planned features (Vulkan backend, NPU fused path,
  OpenAI chat completions API)
- [Contributing](../../CONTRIBUTING.md)
- [Changelog](../../CHANGELOG.md)
