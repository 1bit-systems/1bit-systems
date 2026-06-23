# Project Wiki: 1bit-systems

## Mission
To provide the fastest local 1-bit inference on AMD Strix Halo. The runtime is Lemonade SDK with a native `BitNetServer` backend wrapping rocm-cpp HIP kernels. One layer. Python-free inference hot path.

## Architecture
- **Runtime:** `lemond` (Lemonade Server, C++17) on port `:13305`
- **1-bit Backend:** `BitNetServer` — wraps `bitnet_decode --server` (rocm-cpp) as a subprocess
- **Fallback Backends:** `LlamaCppServer` (non-1bit models), `FastFlowLMServer` (NPU)
- **API Surface:** OpenAI-compatible `/v1/`, Ollama-compatible `/api/`, Anthropic-compatible `/api/messages`
- **UI:** Lemonade Tauri desktop app + web app at `/app`

## Agent Handoff
- **Installation:** Build rocm-cpp, then build 1bit-lemonade. See root README.
- **Commands:**
  - `lemonade run BitNet-b1.58-2B-4T`: Start 1-bit inference
  - `lemonade list`: Show available models
  - `lemonade pull <model>`: Download a model
- **Verification:** `curl http://127.0.0.1:13305/v1/models`
- **Hot Paths:** `lemond` binary, `bitnet_decode` subprocess, `librocm_cpp.so` kernels

## Decisions & Gotchas
- **Rule A: Core serving is Python-free.** The hot path is C++/HIP only.
- **Lemonade is the only runtime layer.** No proxy, no shell orchestration.
- **rocm-cpp kernels are the engine.** Custom HIP GEMV/GEMM for ternary weights.
- **Fork-everything.** Lemonade fork at `bong-water-water-bong/1bit-lemonade`.

## Hard Rules
- **Rule A:** Core serving stays Python-free (C++/HIP hot path)
- **Rule B:** C++20 for kernels (rocm-cpp repo)
- **Rule C:** hipBLAS is banned — all compute uses custom ternary-aware kernels
- **Rule D:** Lemonade SDK is the only runtime layer
- **Rule E:** NPU is an optional side lane (FastFlowLM via Lemonade FLM backend)
- **Compatibility:** OpenAI surface (`:13305/v1`) never breaks

## Article Index
- [[Architecture-Deep]]
- [[Why-Lemonade]]
- [[Why-No-Python]]
- [[BitNetServer-Backend]]
- [[Development]]
- [[Installation]]
- [[Clients]]
- [[FAQ]]
- [[Ternary-on-AIE-Pack-Plan]]
