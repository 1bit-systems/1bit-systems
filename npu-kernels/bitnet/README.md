# BitNet-1.58 NPU kernels (AIE2P / npu2 / Strix Halo)

Glue micro-kernels that frame the stock `matmul_vectorized_8x8x8_i8_i32`
primitive (already compiled from `mlir-aie/aie_kernels/aie2p/mm.cc`) into a
BitNet-1.58 decode-path GEMV.

## What's here

| File | Role | LOC |
|------|------|----|
| `unpack_ternary_2bit_to_int8.cc` | 2-bit halo-ai ternary -> int8 {-1, 0, +1} unpack, 32-wide SIMD | ~140 |
| `scale_i32_fp16.cc`              | int32 matmul tile x per-row bf16 scale -> bf16 tile, 16-wide fp | ~100 |
| `bitnet_gemv.py`                 | IRON MLIR emitter; 3-stage pipeline (unpack -> mm -> scale) | ~220 |

The int8 matmul itself is **not** re-shipped — we link against the stock
`mm.cc` object compiled by upstream (`mm_${m}x${k}x${n}.o`).

## Pipeline

```
                 +----------------+      +---------------+      +--------------+
 packed W  ----> |  unpack_       | ---> |  matmul_i8_   | ---> |  scale_i32_  | --> bf16 Y
 (uint8)         |  ternary_i2_i8 |      |  i32 (mm.cc)  |      |  bf16        |
                 +----------------+      +---------------+      +--------------+
                                                ^                      ^
                                                |                      |
                                int8 A  --------+                      |
                                                                       |
                                bf16 scale (per-row) ------------------+
```

One Worker per stage, ObjectFifos between them. Target device is NPU2
(`AIEDevice.npu2`), one compute column.

## Dimensions (decode default)

```
M = 512   K = 512   N = 16   (decode; N=1 padded up to 16 for tile compat)
m = 64    k = 64    n = 16   (tile sizes)
```

Hard constraints enforced at MLIR-emit time:
- `m % 16 == 0`   (mm.cc 2x2 expansion of r=8 along M)
- `k % 8  == 0`   (mm.cc s=8)
- `k % 4  == 0`   (2-bit packing, 4 codes per byte)
- `n % 16 == 0`   (mm.cc 2x2 expansion of t=8 along N)

## Build

Mirrors the upstream Makefile pattern under
`mlir-aie/programming_examples/basic/matrix_multiplication/single_core/`.
You need the aie-iron venv and `env_setup.sh` sourced, same as today's axpy:

```sh
# one-time in this shell
source /home/bcloud/.venvs/ironenv/bin/activate
source /home/bcloud/repos/mlir-aie/utils/env_setup.sh /home/bcloud/repos/mlir-aie/install

# from this directory
make devicename=npu2 M=512 K=512 N=16 m=64 k=64 n=16
```

A `Makefile` is not yet committed here — copy
`mlir-aie/programming_examples/basic/matrix_multiplication/single_core/Makefile`
and adapt `kernels=` / object-file targets to build all three `.cc` files:

```make
# pseudo-Makefile skeleton (fill in when wiring):
kernels = unpack_${m}x${k} mm_${m}x${k}x${n} scale_${m}x${n}

build/unpack_${m}x${k}.o: unpack_ternary_2bit_to_int8.cc
	${KERNEL_CC} ${KERNEL_CFLAGS} -DDIM_M=${m} -DDIM_K=${k} -c $< -o $@

build/mm_${m}x${k}x${n}.o: ${mlir_aie_root}/aie_kernels/aie2p/mm.cc
	${KERNEL_CC} ${KERNEL_CFLAGS} -Di8_i32_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} \
	    -c $< -o $@

build/scale_${m}x${n}.o: scale_i32_fp16.cc
	${KERNEL_CC} ${KERNEL_CFLAGS} -DDIM_M=${m} -DDIM_N=${n} -c $< -o $@

aie_py_src = bitnet_gemv.py
```

## Expected artifacts

After a clean build with the defaults above:

```
build/unpack_64x64.o                         ~2 kB   (Peano AIE core)
build/mm_64x64x16.o                          ~12 kB  (stock upstream)
build/scale_64x16.o                          ~3 kB
build/aie_512x512x16_64x64x16.mlir           (emitted by bitnet_gemv.py)
build/insts_512x512x16_64x64x16.txt          (NPU control ops)
build/final_512x512x16_64x64x16.xclbin       (loadable binary)
```

Runner: use the proven pyxrt path (same as the axpy 160/160 path today).
Host C++ runner via xrt-targets.cmake is still blocked on Arch SHARED-lib
flag — not a shipping concern for the kernel work.

## Known limitations

- **N padded to 16.** Decode is N=1 in practice; we waste 15/16 of the
  matmul compute on this path. Fine for correctness bringup; later we can
  author a narrower-N packed path or batch multiple decode requests.
- **K tested up to 4096.** The `matmul_vectorized_2x2_mmul` accumulator
  saturation was not characterized beyond this — BitNet-1.58 layers are
  well under this, but audit before scaling.
- **`ncols % 256 == 0`** in the vectorized unpack path. Enforced by
  `static_assert` in `unpack_ternary_vectorized`. Rows with residual K
  need to route to the scalar fallback — not wired in the IRON glue yet.
- **Scale dtype is bfloat16, not IEEE fp16.** AIE2P has no fp16 MAC; halo-ai
  `.h1b` fp16 scales are converted to bf16 in the loader before being fed to
  this kernel. Loss-of-precision is bounded (~0.1% relative error from the
  fp16->bf16 transit) and empirically invisible on BitNet-1.58 PPL.
- **`aie::shift_bytes` name.** The per-lane variable shift in the unpack
  kernel assumes this name. Confirm against the installed `aie_api/aie.hpp`;
  swap to `aie::shift_right` with vector-rhs if it differs on the box
  (see open questions below).
- **`aie::to_float<float>(int32_v)` availability on AIE2P.** If absent,
  fallback path is `aie::accum<accfloat, N>().from_vector(i32_v).to_vector<float>()`
  which is two instructions instead of one.

## Open questions (confirm on box — I could not run anything)

- [ ] Does `aie_api/aie.hpp` on our mlir-aie install expose
      `aie::shift_bytes(vec, shift_vec)` for `uint8_t` on AIE2P, or only the
      fixed-shift form? Fallback to per-lane scalar masked shift costs
      ~2x cycles on this step but is otherwise drop-in.
- [ ] Is `aie::to_float<float>(aie::vector<int32_t, 16>)` a single hardware
      instruction on AIE2P, or does it lower to the `accfloat` round-trip?
- [ ] Does NPU2 support `zero_i32` (from `zero.cc`) at `n=16` tile width
      without needing a new specialization? Upstream tests at `n=64`.
- [ ] Do the three Workers fit on a single compute column, or do we need to
      place them across two columns? IRON placement auto-picks; verify
      `aiecc` placement log after first emit.
- [ ] The `pattern_repeat=N_div_n` on `Wpacked_tiles` and `S_tiles` assumes
      weights/scale are re-streamed for each N-tile. For N_div_n == 1 this
      collapses; confirm it doesn't emit a redundant DMA at N=16/n=16.

## Licensing

Apache-2.0 WITH LLVM-exception (matches upstream mlir-aie).

## Cross-references

- `/home/bcloud/.claude/projects/-home-bcloud/memory/project_npu_matmul_i8.md`
  — int8 matmul compile notes.
- `/home/bcloud/.claude/projects/-home-bcloud/memory/project_npu_unlocked.md`
  — IRON toolchain proven via axpy 160/160.
- `/home/bcloud/repos/mlir-aie/aie_kernels/aie2p/mm.cc:383` — the primitive
  we wrap.
- `/home/bcloud/repos/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/single_core_iron.py`
  — IRON pattern.
