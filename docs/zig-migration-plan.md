# Zig → C++23 Migration Plan

Status: **Design proposal** · Tracking: ~8 weeks part-time

## Motivation

The codebase currently spans **4 languages** for the core inference engine:

| Language | Files | SLOC | Used in |
|----------|-------|------|---------|
| C++23 | ~60 | ~25K | NPU engine, server, HIP kernels, tools |
| Zig | ~100 | ~30K | GPU engine, NPU compat, fusion engine |
| C99 | ~5 | ~2K | Dequant, GGUF reader |
| Python | ~15 | ~3K | Converters, xclbin generators |

This language spread creates maintenance burden:
- Two build systems (CMake + `build.zig`)
- Duplicated type definitions across language boundaries
- New contributors must learn Zig to touch the GPU engine
- Zig's immature ecosystem (no stable ABI, frequent compiler breaks, sparse debugger support)

## Proposal: Port `engine/gpu/src/` to C++23 incrementally

### Phase 1: Dependency analysis (1 week)

Map the Zig → C++ dependency graph:

```
engine/gpu/src/
  main.zig              → entry point (can stay Zig or become thin C++ main)
  compute/*.zig         → GEMM, attention, elementwise kernels
  cuda/*.zig            → CUDA C interop → replace with direct CUDA C++
  vulkan/*.zig          → Vulkan C interop → replace with direct Vulkan C++
  scheduler/*.zig       → KV cache, request scheduling
  server/*.zig          → HTTP server, model manager
  model/*.zig           → GGUF loader, tokenizer, architecture dispatch
  zinc_rt/*.zig         → ZINC runtime (ISA, ring, KMD)
```

**Deliverable:** Dependency graph with port order.

### Phase 2: FFI boundary (1 week)

Add a C FFI layer at the Zig ↔ C++ boundary so both sides can coexist during the port:

```c
// engine/gpu/include/gpu_engine_c.h
typedef struct { /* ... */ } gpu_engine_t;
gpu_engine_t* gpu_engine_create(const char* model_path);
int gpu_engine_decode(gpu_engine_t* e, int* tokens, int n);
void gpu_engine_destroy(gpu_engine_t* e);
```

**Deliverable:** `gpu_engine_c.h` + C++ implementation, Zig calls through C FFI.

### Phase 3: Port compute layer (2 weeks)

Port `compute/*.zig` to C++23 HIP:
- `forward.zig` → `compute/forward.cpp` using `rocm_cpp/bonsai.h`
- `attention.zig` → `compute/attention.cpp` using HIP flash attention
- `dmmv.zig` → `compute/dmmv.cpp`

**Deliverable:** All matrix/attention kernels callable from C++, tested against original Zig output.

### Phase 4: Port server + scheduler (2 weeks)

Port `server/*.zig` and `scheduler/*.zig`:
- Leverage existing C++ HTTP server infrastructure (`tools/unified_server.cpp`)
- Port KV cache management to C++ with the same LRU + paging scheme
- Port model catalog and architecture dispatch into `src/model_router.cpp`

**Deliverable:** Feature-complete C++ inference server, Zig server can be deleted.

### Phase 5: Port ZINC runtime (2 weeks)

Port `zinc_rt/*.zig` (the GPU kernel dispatch runtime):
- KMD interface → C++ wrapper around `/dev/kfd`
- Ring buffer management → C++ lock-free queue
- IR graph → C++ with the same op definitions

This is the riskiest phase — ZINC has the most Zig-specific patterns (comptime dispatch, packed structs). Mitigation: keep ZINC in Zig behind a C FFI initially, port last.

**Deliverable:** Single `zaya_server` binary, no Zig build step.

## Risks

| Risk | Mitigation |
|------|------------|
| Zig `comptime` metaprogramming has no C++ equivalent | Use C++ templates + `constexpr` + codegen |
| GPU kernel correctness during port | Run `tests/download_and_run.sh` smoke test on every PR |
| Schedule slip | Phase 2 + 3 are minimum viable; 4 + 5 can follow |

## Success Criteria

- `cmake --build build --target zaya_server` produces a single binary
- All 9/11 CI tests pass
- No `.zig` files in `engine/gpu/src/` (can remain in `engine/fusion/`)
