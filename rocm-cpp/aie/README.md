# aie/ — Peano-compiled tile kernels for Strix Halo XDNA2 (AIE2P)

First-landing spot for our hand-rolled AIE2P tile code. Everything in this
directory is compiled by the Peano branch of LLVM (`/opt/peano/bin/clang`)
with the `aie2p-none-unknown-elf` triple. No IRON, no Python, no MLIR-AIE
here — this is the leaf kernel source only. The host-side xclbin, runtime
ND-DMA descriptors, and ObjectFifo wiring live elsewhere (future
`aie/host/` or equivalent) and are produced by `aiecc` once mlir-aie is
installed.

## Files

| File                              | Purpose                                                        |
|-----------------------------------|----------------------------------------------------------------|
| `halo_ternary_mm.h`               | Fused-path entry signature; parity oracle header.              |
| `halo_ternary_mm.cpp`             | Fused-path scalar reference + stubbed vectorized body.         |
| `kernels/bitnet_gemv_aie2p.cc`    | **Pipelined-path** tile entry stub (skeleton, signature only). |
| `scripts/build_aie.sh`            | IRON → Peano → hipcc link → xclbin wrapper (scaffold only).    |

Two BitNet-1.58 designs coexist while the lane bakes:

1. **Fused path** (`halo_ternary_mm.{h,cpp}`) — one tile kernel that unpacks
   ternary, multiplies by int8 activations, and accumulates into int32 in a
   single body. Simpler to reason about, used as the bit-exact parity oracle
   for bring-up.
2. **Pipelined path** (`kernels/bitnet_gemv_aie2p.cc` + stages in
   `../../npu-kernels/bitnet/`) — three ObjectFifo-connected stages
   (unpack → stock `matmul_vectorized_8x8x8_i8_i32` → scale). This reuses
   the stock upstream matmul tile, which already has a bit-exact M=K=N=512
   xclbin built for npu2 (see `project_npu_matmul_i8.md`). Default
   ship target.

## Build the tile object (fused parity-oracle path)

```sh
/opt/peano/bin/clang --target=aie2p-none-unknown-elf \
    -O2 -c halo_ternary_mm.cpp -o halo_ternary_mm.o
```

Expected output of `file halo_ternary_mm.o`:

```
halo_ternary_mm.o: ELF 32-bit LSB relocatable, *unknown arch 0x108* version 1 (SYSV), not stripped
```

(`0x108` is the ELF e_machine code the Peano backend emits for AIE2P; same
value observed for the `examples/aie_hello.cpp` smoke test.)

## Build the xclbin (pipelined path — the ship target)

```sh
./scripts/build_aie.sh            # scaffold today; kernel author fills in
# or override shape:
env M=1024 K=1024 N=16 ./scripts/build_aie.sh
```

This wraps the IRON + MLIR-AIE + Peano chain that produces
`build/bitnet_gemv/final_${M}x${K}x${N}_${m}x${k}x${n}.xclbin`. Load-time
dispatch is handled by `crates/1bit-aie` once the `AieBackend` trait's real
impl lands; see `docs/wiki/NPU-Kernel-Handoff.md` for the C++ ↔ Rust
boundary.

## What's stubbed vs live

- **Live, compiles today**: scalar reference matmul with ternary unpack.
  Same math as `rocm-cpp/kernels/ternary_gemv_halo.hip` — use it as the
  bit-exact parity oracle once vectorized lights up.
- **Stubbed, gated on `HAVE_AIE_API`**: `aie::mmul<8,8,8,int8,int8,accauto>`
  2x2 unroll cribbed from
  `/tmp/IRON-read/aie_kernels/aie2p/mm.cc:83-208`. Lands once
  `aie_api/aie.hpp` is on the include path. That header ships with
  mlir-aie, not Peano.

## Getting to a runnable xclbin

Peano alone gets you an AIE2P relocatable. To actually dispatch onto the
NPU you still need:

1. **Install mlir-aie** (headers + `aiecc` + `xchesscc` shim). This adds
   `aie_api/aie.hpp` for the vectorized path and the MLIR pass pipeline.
   Reference: `github.com/Xilinx/mlir-aie`.
2. **Write a host-side MLIR wrapper** that instantiates the ObjectFifo graph
   (4 A-lanes broadcast across rows, 8 B-lanes broadcast across cols, 8 C
   drains) and binds this tile kernel symbol into each of the 32 compute
   tiles. Template: `iron/operators/gemm/design.py:358-437`.
3. **Run `aiecc.py`** (or the equivalent `aie-opt` + `aie-translate` +
   `bootgen` chain) to fuse the MLIR + tile objects into an xclbin.
4. **Load via `halo-bitnet-xdna`** (new Rust crate, planned
   2026-04-20) using `xrt::kernel` + `xrt::bo` — no IRON Python.

## Cross-references

- `docs/wiki/NPU-Kernel-Design.md` — tile grid, DMA pattern, L1 budget,
  source map to IRON.
- `docs/wiki/Ternary-on-AIE-Pack-Plan.md` — 2-bit encoding (`-1=0b10`,
  `0=0b00`, `+1=0b01`, `0b11` reserved), unpack inlining, bandwidth model.

## Pointer hygiene note

`halo_ternary_mm_core` takes `__restrict`-able raw pointers by design. The
host xclbin guarantees no aliasing between `packed_A`, `B`, and `C`
(separate L1 ping-pong banks). When porting to the vectorized path, put
`__restrict` back on the args inside the `.cpp` — the C header drops it
because C++20 doesn't standardize `__restrict` and we want the header to
be safely includable from an MLIR-generated TU.
