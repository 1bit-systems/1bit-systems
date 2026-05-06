# Control Plane Roadmap

`1bit-systems` should become a 1-bit inference engine with one control plane.
It is not there yet.

The current repository mixes several eras:

- native Arch/CachyOS installer for Lemonade + FastFlowLM
- `1bit-proxy` as a thin OpenAI-compatible routing layer
- GAIA/Open WebUI integration notes
- older Rust / halo / no-Python architecture notes
- static site copy that describes more coherence than the repo currently has

The repair plan is to make the control plane backend-agnostic.

## Target Contract

Apps should only need:

```text
base_url = http://127.0.0.1:13306/v1
api_key  = local-no-auth
```

Everything behind that is an implementation detail.

## Backend Types

| Backend | First role | Runtime |
|---|---|---|
| llama.cpp toolbox | primary local GGUF inference | RADV/Vulkan first, ROCm second |
| FastFlowLM | XDNA NPU side lane | native FLM or container if available |
| Lemonade | multimodal / OmniRouter | native when stable |
| vLLM toolbox | high-throughput serving | ROCm toolbox |
| ComfyUI toolbox | image/video generation | ROCm toolbox |
| fine-tuning toolbox | training/adapters | ROCm + Jupyter toolbox |
| remote worker | optional private Strix Halo worker | HTTPS or mesh |

## Efficiency Track

The control plane should treat efficient inference as a first-class routing
problem, not only a packaging problem. Useful research directions from the
Neural Noise publication list include KV-cache compression, attention
selection, activation sparsity, adaptive computation, and small/efficient LLM
workshops. Those ideas belong behind explicit route policy and benchmark
switches:

- prefer 1-bit / ternary GGUF models for the default local lane
- expose context and KV-cache budget as backend metadata
- keep prompt prefill, decode throughput, memory footprint, and quality checks
  in the same benchmark report
- make adaptive or sparse variants opt-in until measured on Strix Halo

## Milestone 1: Honest Bootstrap

- Keep native `install.sh` behavior for Arch/CachyOS.
- Make Ubuntu/Fedora enter a toolbox-backed installer path instead of failing
  on `pacman`.
- Add `1bit doctor` for device and backend readiness.
- Add `1bit toolbox bootstrap|up|down|status` for the first backend lifecycle.
- Let `1bit-proxy` point to a toolbox `llama-server` on `:13305`.

## Milestone 2: Backend Registry

Create a small config file:

```toml
[[backend]]
name = "llama-vulkan-radv"
kind = "openai"
base_url = "http://127.0.0.1:13305/v1"
start = "toolbox enter llama-vulkan-radv -- llama-server ..."
health = "/v1/models"

[[backend]]
name = "flm-npu"
kind = "openai"
base_url = "http://127.0.0.1:52625/v1"
start = "flm serve qwen3:1.7b --port 52625 --embed 1"
health = "/v1/models"
```

The proxy should read this registry instead of assuming Lemonade and FLM.

## Milestone 3: Lifecycle

`1bit up` should:

1. read configured backends
2. check device access
3. start missing backends
4. start proxy
5. report exact model and route state

`1bit down` should stop only what `1bit up` started.

## Milestone 4: Product Surface

Expose one local dashboard:

- backend status
- model list
- active route policy
- recent errors
- copyable app connection snippets
- guardrails for LAN exposure

## Non-Goals For Now

- no public hosted inference from the home Strix Halo box
- no custom kernel work on the critical path
- no single-vendor runtime bet
- no claims that NPU, ROCm, Vulkan, and Lemonade are already unified

The next useful engineering step is replacing the hard-coded Lemonade/FLM proxy
assumptions with a backend registry while keeping the toolbox lifecycle working.
