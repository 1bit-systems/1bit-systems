# FAQ

## What is 1bit systems?

The 1-bit inference engine. A Rust orchestration stack running Microsoft's BitNet-b1.58-2B-4T on an AMD Strix Halo mini-PC with native HIP kernels. OpenAI-compatible API, MCP server, 17 specialist agents, live landing page. Bring any OpenAI-compat client, plug it in, the agents keep the stack running while you work.

## What does it run on?

AMD Ryzen AI MAX+ 395 (Strix Halo) mini-PC, $2-3k. 128 GB LPDDR5 unified memory, Radeon 8060S iGPU (gfx1151, RDNA 3.5), 256 GB/s bandwidth, <150 W under load. CachyOS for the kernel-7 NPU driver support.

## How fast?

- 83 tok/s at 64-token generation
- 68 tok/s at 1024-token generation
- Perplexity 9.18 on wikitext-103 (matches Microsoft paper's 9.1607 within ±0.05)
- 96.66% byte-identical output vs the reference C++ server after ~10 000 side-by-side rounds

## How do I install it?

```
git clone git@github.com:bong-water-water-bong/1bit systems-core.git
cd 1bit systems-core
./install-strixhalo.sh
```

One script, ~5 minutes on a fresh CachyOS box. After: `halo doctor`, `halo chat`, `halo say "hello"`.

## What clients work with it?

Anything that accepts an OpenAI-compatible `base_url`:
- **Open WebUI** — polished desktop-style chat, full RAG + MCP
- **LibreChat** — yaml-configured OpenAI alternative
- **DSPy** (Stanford) — compile declarative LM programs against 1bit-server
- **Claude Code** — 1bit-mcp registers as MCP server; 17 specialists appear as tools
- **1bit-helm** — our own native-Rust egui desktop client (renamed from halo-gaia 2026-04-20)
- **lemonade-python-sdk** via the Lemonade-compat shim on `:8200`

## Why 1.58-bit and not 1-bit?

See [`Why-Ternary.md`](./Why-Ternary.md). Summary: ternary has a load-bearing zero that maps to activation sparsity, hardware efficiency, and 10× memory reduction with near-zero accuracy loss.

## Why Rust and not Python?

See [`Why-No-Python.md`](./Why-No-Python.md). Summary: 200 ms cold-start vs 10 seconds, 15 MB binary vs 1.2 GB venv, `Result<T, E>` vs `KeyError` at line unknown.

## Is it ready?

For private-beta testers, yes. PPL parity proven, shadow-burnin at 96.66% byte-exact, CI green, tester-installable. Public launch (reddit post, Steam-community-style rollout) after the 72-hour burnin gate clears. See [`../../CUTOVER.md`](../../CUTOVER.md).

## Can I run it on Apple Silicon?

Yes. The workspace is feature-gated — `cmake --build cpp/build/strix --features mlx-apple` pulls in `bitnet-mlx-rs` and targets M-series via MLX. Same code, different kernel backend. AMD is the performance target; Apple is supported but not optimized.

## Can I run it on NVIDIA?

Not today. Our kernels are HIP. A future CUDA port is technically doable (HIP → CUDA is mostly mechanical) but not planned. A Windows/NVIDIA build would need a new maintainer.

## Can I use my own model?

Partially:
- Any BitNet GGUF with IQ2_S weights should load through our `1bit-core::gguf` path (commit `2d1ec89`). Currently parse-only; the IQ2_S → `.h1b` bit-unpack lands next sprint.
- Microsoft's `bitnet-b1.58-2B-4T` works today via our `.h1b` format + requantizer.
- Non-BitNet architectures (Llama, Mistral, Qwen) don't run on the ternary kernels. Those models use their own non-1.58-bit formats.

## What's 1bit-agents?

17 specialist agents running in the background — anvil rebuilds kernels on commit, librarian keeps the changelog, quartermaster triages issues, magistrate scans PRs for secrets + commit-msg compliance. See the `1bit-agents` crate. Exposed via MCP so Claude Code and DSPy can call them as tools.

## What's shadow-burnin?

See [`Why-Shadow-Burnin.md`](./Why-Shadow-Burnin.md). Continuous parity check between the gen-1 C++ server and gen-2 Rust server so we can cut over with evidence, not faith.

## Why not just use Ollama / LM Studio / vLLM?

- **Ollama** — CPU-first, runs BitNet via llama.cpp's ternary kernels, no native gfx1151 optimization. We're ~5× faster on the same box.
- **LM Studio** — GUI-only, closed-source. Can't drive it from automation.
- **vLLM** — NVIDIA-first, CUDA-only. No AMD iGPU path today.

Also: none of them ship an MCP server, self-maintaining agents, Lemonade-compat gateway, or a recording-ready landing page. 1bit systems is the batteries-included bundle, not just the kernel.

## Is it open source?

- Kernels (`bong-water-water-bong/rocm-cpp`) — MIT, public.
- Rust monorepo (`1bit-systems`) — private until launch. Collaborator invites available.
- Everything we fork or borrow is credited in [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Who's behind it?

One operator, one box, one Claude Code session that won't quit. See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) for acknowledgements — huge thanks to Light Heart Labs, Microsoft's BitNet team, the llama.cpp/ggml maintainers, and everyone whose shoulders we're standing on.
