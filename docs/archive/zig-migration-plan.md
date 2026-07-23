# Zig → C++23 Migration Plan (Revised)

Status: **Design proposal** · Effort: ~12-16 weeks part-time · **Not yet prioritized**

## Previous proposal was too aggressive

The original plan called for porting all ~30K SLOC of Zig to C++23 in 8 weeks. This was unrealistic because:

1. **ZINC runtime** (`engine/gpu/src/zinc_rt/`) uses Zig-specific features extensively:
   - `comptime` polymorphism for IR op dispatch — no direct C++ equivalent
   - Packed struct bitfields for GPU ring buffer packets — would need `#pragma pack` + manual bit twiddling
   - Error union propagation throughout the call chain — C++ exceptions or `std::expected` would restructure every caller
2. **FFI boundary overhead**: wrapping 100+ Zig functions in C extern calls adds serialization/deserialization at every call site, which is both a performance hit and a maintenance burden
3. **Opportunity cost**: the Zig GPU engine works reliably. Every week spent porting is a week not spent on the NPU multi-tile path (docs/mlir-air-integration.md) which has a higher performance ceiling.

## Revised strategy: isolate, don't eliminate

### Keep Zig for: ZINC runtime + GPU compute

The ZINC runtime (`engine/gpu/src/zinc_rt/`) and GPU compute layer (`engine/gpu/src/compute/`) are the parts that benefit most from Zig's low-level control and comptime metaprogramming. These ~20K SLOC are stable, tested, and produce correct output. Leave them in Zig.

### Port to C++: server + model loader + scheduler

The server (`engine/gpu/src/server/`), scheduler (`engine/gpu/src/scheduler/`), and model loader (`engine/gpu/src/model/`) layers benefit from C++ because:
- They interact with the C++ `zaya_server` infrastructure (HTTP, config, logging)
- They change frequently as new model architectures are added
- They don't need Zig-specific features — plain functions, structs, and vectors

### Plan

| Phase | What | Effort | Risk |
|-------|------|--------|------|
| 1 | Add C FFI around ZINC compute (a few extern "C" entry points) | 1 week | Low |
| 2 | Port server + HTTP routes to C++ using existing `tools/unified_server.cpp` | 3 weeks | Medium |
| 3 | Port model loader + tokenizer to C++ | 2 weeks | Medium |
| 4 | Port scheduler (KV cache, request batching) to C++ | 2 weeks | Medium |
| 5 | Delete ported Zig files, keep ZINC + compute in Zig | 1 week | Low |

### Result

- **One build system**: CMake for everything (ZINC still uses `build.zig` but called from CMake via `ExternalProject` or a custom target)
- **Eliminated language boundary** for the server path (no Zig↔C++ FFI on the hot path — the C FFI is only for ZINC compute calls)
- **ZINC stays in Zig** where it's most productive

### Success criteria

- `cmake --build build --target zaya_server` builds from a single CMake invocation
- Server, model loading, and scheduling run entirely in C++
- ZINC GPU kernels still compile and produce identical outputs
- All 9/11 CI tests pass
