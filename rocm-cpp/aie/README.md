# aie/ — Peano-compiled tile kernels for Strix Halo XDNA2 (AIE2P)

First-landing spot for our hand-rolled AIE2P tile code. Everything in this
directory is compiled by the Peano branch of LLVM (`/opt/peano/bin/clang`)
with the `aie2p-none-unknown-elf` triple. No IRON, no Python, no MLIR-AIE
here — this is the leaf kernel source only. The host-side xclbin, runtime
ND-DMA descriptors, and ObjectFifo wiring live elsewhere (future
`aie/host/` or equivalent) and are produced by `aiecc` once mlir-aie is
installed.

## Files

| File                   | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `halo_ternary_mm.h`    | Core entry signature; safe to include host-side for MLIR/FFI.  |
| `halo_ternary_mm.cpp`  | Scalar reference + stubbed vectorized path (AIE-API gated).    |

## Build the tile object

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
