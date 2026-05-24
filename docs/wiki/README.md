# Project Wiki: 1bit-systems

## Mission
To provide a toolbox-first local inference workbench for AMD Strix Halo. It unifies multiple inference backends (Lemonade, FastFlowLM, llama.cpp) behind a single OpenAI-compatible union endpoint (`1bit-proxy`), enabling seamless integration for local AI applications.

## Architecture
- **1bit-proxy:** A Node.js service (port `:13306`) that acts as the primary union entry point for OpenAI-compatible clients.
- **Backends:**
  - **Lemonade Server:** Canonical multimodal and OmniRouter inference server (port `:13305`).
  - **FastFlowLM:** XDNA NPU runtime for FLM models and embeddings (port `:52625`).
  - **Toolbox llama.cpp:** A repair/bootstrap lane for non-Arch hosts using Vulkan or ROCm.
- **Control Plane:** The `1bit` CLI handles lifecycle (`up`, `down`, `status`), repair checks, and service wiring.
- **UI Layers:** GAIA Agent UI (primary control surface) and Open WebUI (secondary browser UI).
- **Lifecycle:** Managed via `systemd` units (`1bit-stack.target`).

## Agent Handoff
- **Installation:** Run `./install.sh`. It is idempotent and Arch/CachyOS-first. For Ubuntu/Fedora, use the `1bit toolbox` repair path.
- **Commands:**
  - `1bit up`: Start the entire stack.
  - `1bit status`: Check the health of all services.
  - `1bit doctor`: Inspect host readiness (GPU, NPU, toolbox).
- **Verification:** Connect to `http://127.0.0.1:13306/v1`. Test with a simple `curl` or the Python SDK example in the README.
- **Hot Paths:** `scripts/1bit` (CLI logic), `scripts/1bit-proxy.js` (request routing), and the `install.sh` unit generation.
- **Current Priorities:** Finalizing the single control plane for toolbox lifecycle management and stabilizing the NPU lane bootstrap.

## Decisions & Gotchas
- **Rule A: Core serving is Python-free.** The proxy, kernels, and native runtimes are built without Python to ensure performance and stability (see [[Why-No-Python]]).
- **Arch-First:** The native installer is optimized for CachyOS/Arch. Other distros require the `toolbox` container approach.
- **NPU Lanes:** Supports both FastFlowLM for serving and IRON/MLIR-AIE for custom kernel authoring.
- **Fork-Everything:** All runtime dependencies are maintained as forks under `bong-water-water-bong/` to ensure stability.
- **Lemond Segfault:** Uses standalone `lemon-asr` if the `lemond` whisper backend fails on certain hardware.

## Hard Rules
- **Rule A:** Core serving stays Python-free.
- **Rule B:** C++20 for kernels (archived branch).
- **Rule C:** hipBLAS is banned in the runtime path (archived branch).
- **Rule D:** Rust 1.88+, edition 2024.
- **Rule E:** NPU has two lanes (FastFlowLM serving + custom IRON kernels).
- **Compatibility:** OpenAI surface (`:13306/v1`) must never be broken.

## Article Index
(See original wiki for detailed article descriptions)
- [[Architecture-Deep]]
- [[Repo-Layout]]
- [[Why-Rust]]
- [[Why-No-Python]]
- [[Why-No-NPU-Yet]]
- [[Development]]
- [[Complete-Pack]]
- [[Installation]]
- [[Clients]]
- [[FAQ]]
- [[AMD-GAIA-Integration]]
- [[Fork-Everything]]
- [[Ternary-on-AIE-Pack-Plan]]
