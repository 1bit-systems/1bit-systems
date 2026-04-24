# 1bit.cpp — router-cpp

**Phase 2 scaffolding. Stub methods only; fill in sequentially as each
Rust crate gets ported.**

This subtree is the C++20 mirror of `crates/1bit-router/`. It exists to
kill the HTTP-overhead leak (sampler-pipe IPC + async-runtime overhead)
the bench identified as eating 50-80% of kernel tok/s in the Rust router.

The Rust router keeps compiling and shipping for one release cycle after
`1bit.cpp` reaches feature parity; retirement is scheduled for v0.2.
See the Roadmap wiki for the overall Phase-2 plan.

## Build

Opt-in at the rocm-cpp top level:

```
cd rocm-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_ROUTER_CPP=ON
cmake --build build -j$(nproc)
./build/router-cpp/test_router_smoke
```

The flag defaults OFF so the standard Rust-backed build doesn't pay extra
compile time on the iGPU host.

## Port status

| C++ header / source                      | Rust source of truth                             | Status this pass  |
| ---------------------------------------- | ------------------------------------------------ | ----------------- |
| `include/onebit_cpp/router.hpp`          | `crates/1bit-router/src/lib.rs`                  | Interface         |
| `include/onebit_cpp/backend.hpp`         | `crates/1bit-router/src/backend_impl.rs`         | Interface         |
| `include/onebit_cpp/sampler.hpp`         | `crates/1bit-router/src/sampler/mod.rs`          | Interface         |
| `include/onebit_cpp/chat_template.hpp`   | `crates/1bit-server/src/chat_template.rs`        | **Implemented**   |
| `include/onebit_cpp/model.hpp`           | `crates/1bit-core/src/h1b.rs` + loader wrapper   | Implemented (RAII over rcpp) |
| `src/router.cpp`                         | `crates/1bit-router/src/lib.rs`                  | Scaffold          |
| `src/backend_hip.cpp`                    | `crates/1bit-router/src/backend_impl.rs`         | Scaffold (throws `not yet wired`) |
| `src/sampler.cpp`                        | `crates/1bit-router/src/sampler/cpu.rs`          | Greedy impl; top-k / top-p throw |
| `src/chat_template.cpp`                  | `crates/1bit-server/src/chat_template.rs`        | **Byte-exact**    |
| `src/model.cpp`                          | Wraps `rocm-cpp/src/h1b_loader.cpp`              | Implemented       |

## Non-goals this pass

- No HTTP surface. The server wrapper (`1bit-server` analog) lands in a
  later commit.
- No async runtime. This is the reason the port exists — the router is
  called synchronously from whatever surface wraps it.
- No MCP, no tokenizer port. The MCP bridge lives in `halo-mcp` (Rust,
  Rule A exempt — caller-side) and the tokenizer stays in
  `rocm-cpp/src/tokenizer.cpp` which already has a C API this router
  will call.
- No kernel duplication (Rule B). The router depends on `librocm_cpp` for
  every compute step.

## Byte-exact parity gate

`test_router_smoke` runs 11 assertions, 4 of which compare byte-for-byte
against the Rust reference output for `ChatTemplate::{Llama3,Short,Raw}`
+ `sanitize()`. If any of these diverge, the shadow-burnin harness
(`benchmarks/shadow-burnin.sh`) will start flagging every request that
hits the C++ path — that's the regression gate.

## Next three commits

1. **HipBackend::forward_token layer sequence.** Port the loop body of
   `backend_impl.rs::HipBackend::forward_token` one sub-step at a time:
   embedding lookup → per-layer RMSNorm + quantize + QKV + RoPE + KV
   append + split-KV FD attn + O proj + FFN gate/up + fused ReLU²·GLU +
   FFN down → final RMSNorm → LM head. One commit per sub-block so the
   diff stays reviewable.
2. **Sampler CPU lane.** Port `sampler/cpu.rs` — persistent Zen5 worker
   + bounded-channel handoff equivalent (std::jthread + a minimal
   lock-free SPSC queue; no flume clone needed). Wire Top-K / Top-P
   sampling on top.
3. **`.h1b` RAII loader integration test.** Add a `BUILD_ROUTER_CPP_E2E`
   gated test that loads halo-1bit-2b from disk, runs one forward token,
   asserts the sampled id matches the Rust reference. Gated because
   the test wants a real GPU.

## Rule compliance

- **Rule A** (no Python runtime): C++ only. No Python deps, no
  subprocess dispatches.
- **Rule B** (kernels live in rocm-cpp): we link `librocm_cpp`, no
  kernels authored here.
- **Rule C** (no hipBLAS): every kernel entry point we bind to
  (`rcpp_*`) is native-Tensile-only by construction.
