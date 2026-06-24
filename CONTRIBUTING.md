# Contributing To 1bit-systems

`1bit-systems` is the docs, benchmarks, and website for a local 1-bit inference
stack on AMD Strix Halo. The runtime is [1bit-engine](https://github.com/bong-water-water-bong/1bit-engine)
(pure Rust) wrapping [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp)
(pure C++/HIP) kernels.

## Architecture

```
onebit :13305/v1  →  bitnet_decode --server  →  librocm_cpp.so  →  gfx1151
   (Rust)              (C++/HIP)                 (ternary GEMV)
```

## How To Help

- File issues with reproducible commands and hardware details.
- Test OpenAI-compatible clients against `http://127.0.0.1:13305/v1`.
- Run the benchmark sweep on real Strix Halo hardware:
  ```sh
  cd rocm-cpp && cmake -B build -G Ninja && ninja -C build bench_prefill_variants
  HSA_OVERRIDE_GFX_VERSION=11.5.1 HSA_ENABLE_SDMA=0 ./build/bench_prefill_variants
  ```
- Audit docs and website copy for stale architecture claims.
- Improve benchmarks, docs, and the website.

## Code Style

- Keep the repo focused on documentation and benchmarks.
- Do not reimplement inference kernels here — they live in rocm-cpp.
- Do not reimplement the HTTP server here — it lives in 1bit-engine.
- Preserve the OpenAI-compatible API shape (`:13305/v1`).

Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.).

## Tests And Checks

```sh
bash -n install.sh scripts/1bit benchmarks/bench-npu-ioctl-budget.sh 2>/dev/null
```

For public website changes, preview locally:

```sh
cd 1bit-site && python3 -m http.server 8765
```

Cloudflare Pages deployment is documented in `1bit-site/README.md`.

## Upstream Projects

- [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) — custom HIP kernels for 1-bit/ternary inference
- [1bit-engine](https://github.com/bong-water-water-bong/1bit-engine) — pure Rust HTTP server
- ROCm, XRT, and `amdxdna` for AMD GPU/NPU runtime support
