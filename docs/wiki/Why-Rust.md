# Why Rust above, C++ below?

**One-line answer**: kernels that win on specific hardware live in C++/HIP because the GPU toolchains + intrinsics are there. Everything above the kernel — orchestration, HTTP, CLI, agents, file formats, clients — lives in Rust because we want memory safety, async runtimes, serde, and first-class binaries without a runtime dep.

## The split

```
 ┌────────────────────────────────────────────────────────┐
 │  Rust (halo-workspace, 11 crates, 121+ tests)          │
 │  1bit-cli · 1bit-server · lemond · 1bit-mcp       │
 │  1bit-agents · 1bit-lemonade · 1bit-helm · 1bit-landing│
 │  1bit-core (mmap + tokenizer + sampler) · halo-bitnet-*│
 └──────────────────────────┬─────────────────────────────┘
                            │ extern "C" FFI via 1bit-hip
 ┌──────────────────────────▼─────────────────────────────┐
 │  C++ / HIP (rocm-cpp, bong-water-water-bong canonical)             │
 │  ternary_gemv_halo · ternary_gemv_sherry · ternary_tq1 │
 │  kv_cache_attn_fd · kv_cache_attn_i8 · rotorquant_*    │
 │  h1b_loader · tokenizer                                │
 └────────────────────────────────────────────────────────┘
```

## Why kernels in C++ / HIP

- **hipcc + clang-HIP toolchain** — AMD's official path. Native `__global__` / `__device__` / `__shared__` / intrinsics like `__builtin_amdgcn_perm`, `__builtin_amdgcn_ballot_w32`, `v_dot4_i32_i8`. Rust-CUDA and Rust-HIP exist but lag the intrinsic surface.
- **WMMA / Tensile** — the matrix-accumulate units we target on gfx1151 have C-level APIs. Rust bindings exist but with layout caveats.
- **Iteration speed** — hipcc build cycle for one kernel is ~15 seconds. Rust-GPU is minutes.
- **Matching upstream** — Microsoft's reference BitNet kernels are C++. Porting them literally is cheaper than re-discovering their tricks in a different language.

## Why everything else in Rust

- **Memory safety** — the orchestration layer manages HTTP request lifecycles, SSE streams, model weight mmaps, tokenizer tables, KV cache ownership, systemd journal pipes. A Python `KeyError` or C++ use-after-free at that layer takes down the serving process. `Result<T, E>` + ownership prevents that class of bug entirely.
- **`tokio` + `axum`** — proven async HTTP/SSE stack. The 1bit-server implements OpenAI chat completions, SSE streaming, and per-request sharded metrics in ~1 500 LOC.
- **`serde` + `schemars`** — typed JSON-RPC, MCP tool schemas, config loading. Each specialist declares `#[derive(JsonSchema)]` on its input/output struct and the MCP `tools/list` output falls out for free.
- **`cargo` as build + package manager** — one command (`cargo build --workspace`) builds all 11 crates. CI runs `cargo test --workspace`. No `make`, no `cmake` at this layer.
- **Binary distribution** — Rust produces static-link-friendly binaries with no runtime dep. `1bit-server` is 2.4 MB and needs only `libamdhip64.so` + `librocm_cpp.so` from the system.

## Why not pure Rust end-to-end?

Tried considering it. Vetoed because:

- **Rust-HIP bindings aren't production** — `hip-sys`, `rocm-rs` exist but each is one person + a partial surface. Writing a HIP kernel in pure Rust today means giving up on intrinsics.
- **Rust-GPU (wgpu / Vulkan) doesn't map to ROCm's kernel model** — Metal/Vulkan compute shaders are a different abstraction than HIP's CUDA-style kernels. Porting rocm-cpp's ternary GEMV to wgpu would mean a complete rewrite that loses 30-50% perf.
- **The FFI boundary is stable** — `1bit-hip` exposes 30+ extern "C" functions. Over a year of development, this boundary has changed <5 times. FFI cost is low.

## Why not pure C++ end-to-end?

Rejected for the orchestration layer. Reasons:

- **Concurrency** — `tokio` gives us structured async for free. In C++ we'd pull `asio` + write equivalent plumbing for 3× the code.
- **Serde + typed JSON** — `nlohmann::json` works but loses type safety at every touchpoint. `serde_json::from_value` + `#[derive(Deserialize)]` catches shape errors at compile.
- **Ecosystem** — `clap` for CLIs, `tracing` for logs, `reqwest` for HTTP clients, `axum` for HTTP servers. Each equivalent in C++ is a one-person github project.
- **Distribution** — `1bit-server` binary is 2.4 MB static-ish. A C++ equivalent with openssl + asio + nlohmann + stdlib is 15-30 MB or dynamic-link hell.

## Not Python (ever, at runtime)

Rule A. See [`Why-No-Python.md`](./Why-No-Python.md).

Callers are free to use Python — DSPy, lemonade-python-sdk, jupyter. That's the user's box, not ours. The *service* side (anything under a systemd unit, anything serving an HTTP request) is Rust or C++.

## The inherited trade-off

We pay:
- One FFI boundary (1bit-hip) that needs manual memory + lifetime care across `extern "C"`.
- Two build systems (cargo + cmake).
- Two languages for contributors to read.

We get:
- Zero Python at runtime.
- Safe, fast orchestration on top of fast, hand-tuned kernels.
- Independent evolution: the kernel team can optimize HIP without touching Rust, and the platform team can iterate on the HTTP server without rebuilding kernels.

That trade has paid for itself four times in four months.
