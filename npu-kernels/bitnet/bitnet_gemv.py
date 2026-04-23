#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 1bit.systems / bong-water-water-bong.
#
# BitNet-1.58 decode-path GEMV wiring for AIE2P (npu2 / Strix Halo).
#
#   Pipeline:
#     packed_w (uint8) --> [unpack_ternary_i2_i8]  --> int8 weights
#     activations (int8) -------------------------->
#                                                  \--> [matmul_i8_i32] --> int32 tile
#     per-row scale (bf16) --------------------------------------------->
#                                                                      \->
#                                                                         [scale_i32_bf16]
#                                                                         --> bf16 output
#
#   Dims (decode path, N padded to 16 for tile compat):
#     M = 512, K = 512, N = 16
#     Tile:  m = 64, k = 64, n = 16    (mm.cc int8_i32 requires m%16, k%8, n%16)
#
# This is a MLIR-emitter DSL file (one-shot build step — allowed under
# Rule A since nothing Python runs in the serving path; aiecc consumes
# its stdout as MLIR and walks away).
#
# Pattern-after: mlir-aie/programming_examples/basic/matrix_multiplication/
#                single_core/single_core_iron.py
# We keep the same ObjectFifo + per-stage Worker shape, swapping the single
# matmul kernel for our 3-kernel chain.

import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def ceildiv(a, b):
    return (a + b - 1) // b


def bitnet_gemv(M, K, N, m, k, n, trace_size=0):
    # -- Static shape checks ---------------------------------------------
    assert M % m == 0, f"M={M} not divisible by m={m}"
    assert K % k == 0, f"K={K} not divisible by k={k}"
    assert N % n == 0, f"N={N} not divisible by n={n}"
    # 2-bit ternary packing requires K tile divisible by 4 for byte alignment.
    assert k % 4 == 0, f"k={k} not divisible by 4 (2-bit packing)"
    # mm.cc int8_i32 static_asserts: m%(2*r=16), k%(s=8), n%(2*t=16).
    assert m % 16 == 0, f"m={m} not divisible by 16 (mm.cc r=8)"
    assert k % 8 == 0, f"k={k} not divisible by 8 (mm.cc s=8)"
    assert n % 16 == 0, f"n={n} not divisible by 16 (mm.cc t=8)"

    # -- Dtype aliases ----------------------------------------------------
    i8 = np.dtype[np.int8]
    u8 = np.dtype[np.uint8]
    i32 = np.dtype[np.int32]
    # bfloat16 in numpy: mlir-aie's str_to_dtype maps "bf16" -> a bf16 dtype.
    # IRON lowers `ml_dtypes.bfloat16` to bf16 at the MLIR level; we import
    # lazily so this file still imports if ml_dtypes isn't installed on the
    # aiecc-invoker's venv.
    try:
        from ml_dtypes import bfloat16 as _bf16
        bf16 = np.dtype(_bf16)
    except ImportError:
        # aie.iron ships its own bf16 hook via str_to_dtype on recent trees.
        from aie.iron import str_to_dtype
        bf16 = str_to_dtype("bf16")

    # -- Host-visible tensor types ---------------------------------------
    # Packed weights: one byte per 4 ternary values, K-contiguous, row-major.
    packed_bytes_per_row = K // 4
    W_packed_ty = np.ndarray[(M * packed_bytes_per_row,), u8]
    # Activations: int8, row-major [K, N].
    A_ty = np.ndarray[(K * N,), i8]
    # Per-row scale (bf16), length M.
    S_ty = np.ndarray[(M,), bf16]
    # Output: bf16 [M, N].
    Y_ty = np.ndarray[(M * N,), bf16]

    # -- Tile types (per-core L1 buffers) --------------------------------
    packed_tile_ty = np.ndarray[(m * (k // 4),), u8]      # packed weight tile
    w_tile_ty = np.ndarray[(m, k), i8]                    # unpacked weight tile
    a_tile_ty = np.ndarray[(k, n), i8]                    # activation tile
    c_tile_ty = np.ndarray[(m, n), i32]                   # matmul accumulator
    s_tile_ty = np.ndarray[(m,), bf16]                    # per-row scale slice
    y_tile_ty = np.ndarray[(m, n), bf16]                  # scaled output tile

    # -- Derived loop counts ---------------------------------------------
    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n  # output tiles

    # -- Kernel declarations ---------------------------------------------
    # Each Kernel(...) binds (C symbol, object file, arg types). All three
    # object files end up linked into the per-core program by aiecc; the
    # Makefile compiles each .cc separately with its own DIM_* defines.
    unpack_obj = f"unpack_{m}x{k}.o"
    mm_obj = f"mm_{m}x{k}x{n}.o"
    scale_obj = f"scale_{m}x{n}.o"

    unpack_kernel = Kernel(
        "unpack_ternary_i2_i8",
        unpack_obj,
        [packed_tile_ty, w_tile_ty],
    )

    # zero_i32 lives in mm.cc (same object file as the matmul) and is used to
    # reset the int32 accumulator at the start of each K-tile sweep.
    zero_kernel = Kernel("zero_i32", mm_obj, [c_tile_ty])
    matmul_kernel = Kernel(
        "matmul_i8_i32",
        mm_obj,
        [w_tile_ty, a_tile_ty, c_tile_ty],
    )

    scale_kernel = Kernel(
        "scale_i32_bf16",
        scale_obj,
        [c_tile_ty, s_tile_ty, y_tile_ty],
    )

    # -- ObjectFifos: shim -> memtile -> core, and core -> core ----------
    # Stage A: packed weights into the unpack worker.
    inWpacked = ObjectFifo(packed_tile_ty, name="inWpacked")
    memWpacked = inWpacked.cons().forward(name="memWpacked")

    # Stage B: activations into the matmul worker. (Already int8, no unpack.)
    inA = ObjectFifo(a_tile_ty, name="inA")
    # Stream-layout dims must match the mm.cc int8 A-operand layout:
    #   2x2 tile of (r=8, s=8) micro-mmuls across (m=2*r, k=2*s) -> use
    #   single_core_iron.py's a_dims pattern for int8 (r=8, s=8).
    r_mac, s_mac, t_mac = 8, 8, 8
    a_dims = [(k // s_mac, s_mac * n), (n // t_mac, t_mac), (s_mac, n), (t_mac, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # Stage C: per-row scale into the scale worker.
    inS = ObjectFifo(s_tile_ty, name="inS")
    memS = inS.cons().forward(name="memS")

    # Intermediate fifos between workers on the same compute tile cluster.
    # fifo_W_unpacked carries int8 weights from unpack -> matmul.
    fifo_W_unpacked = ObjectFifo(w_tile_ty, name="fifoWunp")
    # fifo_C carries int32 accumulator from matmul -> scale.
    fifo_C = ObjectFifo(c_tile_ty, name="fifoC")

    # Output fifo from scale back to shim.
    memY = ObjectFifo(y_tile_ty, name="memY")
    outY = memY.cons().forward(name="outY")

    # -- Worker bodies ---------------------------------------------------
    # Worker 1: unpack ternary.
    # Consumes one packed weight tile per (M-tile, K-tile) step; produces one
    # int8 weight tile to fifo_W_unpacked.
    def unpack_fn(of_wp, of_wu, unpack):
        for _ in range_(tiles) if tiles > 1 else range(1):
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                packed_in = of_wp.acquire(1)
                w_out = of_wu.acquire(1)
                unpack(packed_in, w_out)
                of_wp.release(1)
                of_wu.release(1)

    # Worker 2: matmul int8 x int8 -> int32, accumulating across K.
    def matmul_fn(of_w, of_a, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            acc = of_c.acquire(1)
            zero(acc)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                w_in = of_w.acquire(1)
                a_in = of_a.acquire(1)
                matmul(w_in, a_in, acc)
                of_w.release(1)
                of_a.release(1)
            of_c.release(1)

    # Worker 3: apply per-row bf16 scale to int32 tile -> bf16 tile.
    # Scale is acquired once per output tile (one m-row-slice per M-tile);
    # for N_div_n > 1 output tiles within the same M-tile, the scale is
    # re-used — we acquire/release per-tile for simplicity. The intermediate
    # memS forward fifo handles the re-fill from the shim side via a
    # pattern_repeat tap below.
    def scale_fn(of_c, of_s, of_y, scale):
        for _ in range_(tiles) if tiles > 1 else range(1):
            c_in = of_c.acquire(1)
            s_in = of_s.acquire(1)
            y_out = of_y.acquire(1)
            scale(c_in, s_in, y_out)
            of_c.release(1)
            of_s.release(1)
            of_y.release(1)

    worker_unpack = Worker(
        unpack_fn,
        [memWpacked.cons(), fifo_W_unpacked.prod(), unpack_kernel],
        stack_size=0x400,
    )
    worker_matmul = Worker(
        matmul_fn,
        [
            fifo_W_unpacked.cons(),
            memA.cons(),
            fifo_C.prod(),
            zero_kernel,
            matmul_kernel,
        ],
        stack_size=0xD00,
    )
    worker_scale = Worker(
        scale_fn,
        [fifo_C.cons(), memS.cons(), memY.prod(), scale_kernel],
        stack_size=0x400,
    )

    # -- Runtime DMA taps -------------------------------------------------
    # Packed weight taps: M/m tiles across rows, K/k tiles across cols.
    Wpacked_tiles = TensorTiler2D.group_tiler(
        (M, packed_bytes_per_row),
        (m, k // 4),
        (1, K_div_k),
        pattern_repeat=N_div_n,
        prune_step=False,
    )

    # Activations: one full K x N tile streamed per output-tile group.
    a_tap = TensorTiler2D.group_tiler(
        (K, N),
        (k, n),
        (K_div_k, N_div_n),
        tile_group_col_major=True,
        prune_step=False,
    )[0]

    # Scale: one m-length slice per M-tile; repeated for each N-tile column.
    S_tiles = TensorTiler2D.group_tiler(
        (1, M),
        (1, m),
        (1, M_div_m),
        pattern_repeat=N_div_n,
        prune_step=False,
    )

    # Output: (m, n) tiles, group-tiler scanning M then N.
    Y_tiles = TensorTiler2D.group_tiler(
        (M, N),
        (m, n),
        (M_div_m, N_div_n),
        prune_step=False,
    )

    rt = Runtime()
    with rt.sequence(W_packed_ty, A_ty, S_ty, Y_ty) as (Wp, A, S, Y):
        rt.enable_trace(trace_size, workers=[worker_unpack, worker_matmul, worker_scale])
        rt.start(worker_unpack, worker_matmul, worker_scale)

        # One fill of activations covers the whole output sweep (matmul_vec
        # pattern_repeat would be cleaner; we use a single group tap).
        rt.fill(inA.prod(), A, tap=a_tap)

        out_idx = 0
        for tile_row in range(M_div_m):
            rt.fill(inWpacked.prod(), Wp, tap=Wpacked_tiles[tile_row])
            rt.fill(inS.prod(), S, tap=S_tiles[tile_row])
            for _ in range(N_div_n):
                rt.drain(outY.cons(), Y, tap=Y_tiles[out_idx], wait=True)
                out_idx += 1

    my_program = Program(NPU2(), rt)
    return my_program.resolve_program()


def main():
    ap = argparse.ArgumentParser(
        prog="BitNet-1.58 AIE2P GEMV (decode path)",
        description="Emit MLIR for the 3-stage unpack->mm->scale pipeline.",
    )
    ap.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=512)
    ap.add_argument("-N", type=int, default=16)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--trace_size", type=int, default=0)
    # mm.cc tooling passes these; accept+ignore so makefile-common passes
    # through unchanged.
    ap.add_argument("--dtype_in", type=str, default="i8")
    ap.add_argument("--dtype_out", type=str, default="bf16")
    ap.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    ap.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    args, _unknown = ap.parse_known_args()

    if args.dev != "npu2":
        raise SystemExit("Only NPU2 (Strix Halo AIE2P) is supported by this kernel.")

    module = bitnet_gemv(
        args.M, args.K, args.N,
        args.m, args.k, args.n,
        trace_size=args.trace_size,
    )
    print(module)


if __name__ == "__main__":
    main()
