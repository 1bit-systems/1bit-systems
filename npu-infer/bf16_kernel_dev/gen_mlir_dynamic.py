#!/usr/bin/env python3
"""INT8 GEMM with scf.for loops + dynamic BD pool.

Two modes:

* static (default) — same output as gen_mlir_direct.py. Use with old aiecc.
* dynamic (--dynamic) — uses dma_configure_task + DMABdPoolPop for runtime
  bd_id allocation. Requires --aie-lower-dynamic-bd-pool then
  --aie-npu-to-cpp for C++ TXN builder generation.
"""
import sys

m, k, n = 32, 64, 128
r, s, t = 8, 8, 8


def gen_static(M, K, N):
    """Static path: compile-time constants, same binary as gen_mlir_direct.py."""
    tr = M // m
    tc = N // n
    k_steps = K // k
    a_total = M * K
    b_total = K * N
    c_total = M * N
    grp = 2

    loop_body = ""
    for gi in range(tr // grp):
        a_off = gi * grp * m * K
        c_off = gi * grp * m * N
        a_len = k_steps * grp * m * k
        b_len = k_steps * k * n
        c_len = tc * grp * m * n
        loop_body += f"""
      %a{gi} = aiex.dma_configure_task_for @inA {{
        aie.dma_bd(%arg0 : memref<{a_total}xi8> offset = {a_off} len = {a_len} sizes = [{tc}, {k_steps}, {grp * m}, {k}] strides = [0, {k}, {K}, 1])
        aie.end
      }} {{repeat_count = {tr * tc - 1} : i32}}
      aiex.dma_start_task(%a{gi})
      %b{gi} = aiex.dma_configure_task_for @inB {{
        aie.dma_bd(%arg1 : memref<{b_total}xi8> offset = 0 len = {b_len} sizes = [{tc}, {k_steps}, {k}, {n}] strides = [{n}, {k*N}, {N}, 1])
        aie.end
      }} {{repeat_count = {tr * tc - 1} : i32}}
      aiex.dma_start_task(%b{gi})
      %c_out{gi} = aiex.dma_configure_task_for @outC {{
        aie.dma_bd(%arg2 : memref<{c_total}xi32> offset = {c_off} len = {c_len} sizes = [1, {tc}, {grp * m}, {n}] strides = [{grp * m * N}, {n}, {N}, 1])
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%c_out{gi})
      aiex.dma_await_task(%c_out{gi})
      aiex.dma_free_task(%a{gi})
      aiex.dma_free_task(%b{gi})
      aiex.dma_free_task(%c_out{gi})"""

    return f"""// INT8 GEMM static M={M} K={K} N={N}
module {{
  aie.device(npu2_1col) {{
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_1 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_2 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_3 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @inA(%logical_shim_noc, {{%logical_mem}}, 2 : i32) : !aie.objectfifo<memref<{m}x{k}xi8>>
    aie.objectfifo @memA(%logical_mem dimensionsToStream
      [<size = {m//r}, stride = {r*k}>, <size = {k//s}, stride = {s}>, <size = {r}, stride = {k}>, <size = {s}, stride = 1>],
      {{%logical_core}}, 2 : i32) : !aie.objectfifo<memref<{m}x{k}xi8>>
    aie.objectfifo.link [@inA] -> [@memA]([] [0])
    aie.objectfifo @inB(%logical_shim_noc_0, {{%logical_mem_1}}, 2 : i32) : !aie.objectfifo<memref<{k}x{n}xi8>>
    aie.objectfifo @memB(%logical_mem_1 dimensionsToStream
      [<size = {k//s}, stride = {s*n}>, <size = {n//t}, stride = {t}>, <size = {s}, stride = {n}>, <size = {t}, stride = 1>],
      {{%logical_core}}, 2 : i32) : !aie.objectfifo<memref<{k}x{n}xi8>>
    aie.objectfifo.link [@inB] -> [@memB]([] [0])
    aie.objectfifo @memC(%logical_core, {{%logical_mem_2}}, 2 : i32) : !aie.objectfifo<memref<{m}x{n}xi32>>
    aie.objectfifo @outC(%logical_mem_2 dimensionsToStream
      [<size = {m//r}, stride = {r*n}>, <size = {r}, stride = {t}>, <size = {n//t}, stride = {r*t}>, <size = {t}, stride = 1>],
      {{%logical_shim_noc_3}}, 2 : i32) : !aie.objectfifo<memref<{m}x{n}xi32>>
    aie.objectfifo.link [@memC] -> [@outC]([] [0])
    func.func private @zero_i32(memref<{m*n}xi32>) attributes {{link_with = "mm_32x64x128.o"}}
    func.func private @matmul_i8_i32(memref<{m*k}xi8>, memref<{k*n}xi8>, memref<{m*n}xi32>) attributes {{link_with = "mm_32x64x128.o"}}
    %core = aie.core(%logical_core) {{
      %c0 = arith.constant 0 : index
      %c_big = arith.constant 1048576 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c_big step %c1 {{
        %c0_4 = arith.constant 0 : index
        %c_tiles = arith.constant {tr * tc} : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_4 to %c_tiles step %c1_5 {{
          %1 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<{m}x{n}xi32>>
          %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<{m}x{n}xi32>> -> memref<{m}x{n}xi32>
          %cs = memref.collapse_shape %2 [[0, 1]] : memref<{m}x{n}xi32> into memref<{m*n}xi32>
          func.call @zero_i32(%cs) : (memref<{m*n}xi32>) -> ()
          %c0_6 = arith.constant 0 : index
          %c_ks = arith.constant {k_steps} : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg2 = %c0_6 to %c_ks step %c1_7 {{
            %3 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<{m}x{k}xi8>>
            %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<{m}x{k}xi8>> -> memref<{m}x{k}xi8>
            %5 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<{k}x{n}xi8>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<{k}x{n}xi8>> -> memref<{k}x{n}xi8>
            %ca8 = memref.collapse_shape %4 [[0, 1]] : memref<{m}x{k}xi8> into memref<{m*k}xi8>
            %ca9 = memref.collapse_shape %6 [[0, 1]] : memref<{k}x{n}xi8> into memref<{k*n}xi8>
            %ca10 = memref.collapse_shape %2 [[0, 1]] : memref<{m}x{n}xi32> into memref<{m*n}xi32>
            func.call @matmul_i8_i32(%ca8, %ca9, %ca10) : (memref<{m*k}xi8>, memref<{k*n}xi8>, memref<{m*n}xi32>) -> ()
            aie.objectfifo.release @memA(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }}
          aie.objectfifo.release @memC(Produce, 1)
        }}
      }}
      aie.end
    }} {{stack_size = 3328 : i32}}
    aie.runtime_sequence @main(%arg0: memref<{a_total}xi8>, %arg1: memref<{b_total}xi8>, %arg2: memref<{c_total}xi32>) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c_tr = arith.constant {tr // grp} : index
      scf.for %gi = %c0 to %c_tr step %c1 {{
{loop_body}
      }}
    }}
  }}
}}
"""


def gen_dynamic(M, K, N):
    """Dynamic path: dma_configure_task + DMABdPoolPop ping-pong pattern.
    
    Each row-block group gets its own DMA task with bd_id from pool.
    scf.for iter_args carry the bd_id for recycling (ping-pong).
    Requires --aie-lower-dynamic-bd-pool + --aie-npu-to-cpp.
    """
    tr = M // m
    tc = N // n
    k_steps = K // k
    a_total = M * K
    b_total = K * N
    c_total = M * N
    grp = 2
    n_groups = tr // grp

    a_len = k_steps * grp * m * k
    b_len = k_steps * k * n
    c_len = tc * grp * m * n

    return f"""// INT8 GEMM dynamic M={M} K={K} N={N}
// Ping-pong bd_id pattern for dynamic pool pass.
module {{
  aie.device(npu2) {{
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @inA (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @inB (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @outC (%tile_0_0, S2MM, 0)
    aie.runtime_sequence @main(%inA: memref<{a_total}xi8>, %inB: memref<{b_total}xi8>, %outC: memref<{c_total}xi32>, %M: i32, %K: i32, %N: i32) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c_grp = arith.constant {grp * m} : i32
      %c_mK = arith.constant {grp * m * K} : i32
      %c_mN = arith.constant {grp * m * N} : i32
      %c_a_len = arith.constant {a_len} : i32
      %c_b_len = arith.constant {b_len} : i32
      %c_c_len = arith.constant {c_len} : i32
      %c_tc = arith.constant {tc} : i32
      %c_ks = arith.constant {k_steps} : i32
      %c_grp_m = arith.constant {grp * m} : i32
      %c_k = arith.constant {k} : i32
      %c_n = arith.constant {n} : i32
      %c_K = arith.constant {K} : i32
      %c_N = arith.constant {N} : i32
      %c_0 = arith.constant 0 : i32
      %c_stride_N = arith.constant {N} : i32
      %c_stride_KN = arith.constant {k * N} : i32
      %c_stride_mN = arith.constant {grp * m * N} : i32

      // i64 versions for dma_bd sizes/strides (DynamicIndexList expects i64)
      %c_tc_i64 = arith.extsi %c_tc : i32 to i64
      %c_ks_i64 = arith.extsi %c_ks : i32 to i64
      %c_grp_m_i64 = arith.extsi %c_grp_m : i32 to i64
      %c_k_i64 = arith.extsi %c_k : i32 to i64
      %c_n_i64 = arith.extsi %c_n : i32 to i64
      %c_K_i64 = arith.extsi %c_K : i32 to i64
      %c_N_i64 = arith.extsi %c_N : i32 to i64
      %c_stride_KN_i64 = arith.extsi %c_stride_KN : i32 to i64
      %c_stride_mN_i64 = arith.extsi %c_stride_mN : i32 to i64
      %c_stride_N_i64 = arith.extsi %c_stride_N : i32 to i64

      // tr = M / (grp*m)
      %tr_i32 = arith.divsi %M, %c_grp : i32
      %tr = arith.index_cast %tr_i32 : i32 to index

      // --- Init row block 0 ---
      %init_a = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {{
        aie.dma_bd(%inA : memref<{a_total}xi8> offset = 0 len = %c_a_len sizes = [%c_tc_i64, %c_ks_i64, %c_grp_m_i64, %c_k_i64] strides = [0, %c_k_i64, %c_K_i64, 1])
        aie.end
      }} {{repeat_count = {tc - 1} : i32}}
      aiex.dma_start_task(%init_a)

      %init_b = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {{
        aie.dma_bd(%inB : memref<{b_total}xi8> offset = 0 len = %c_b_len sizes = [%c_tc_i64, %c_ks_i64, %c_k_i64, %c_n_i64] strides = [%c_n_i64, %c_stride_KN_i64, %c_stride_N_i64, 1])
        aie.end
      }} {{repeat_count = {tc - 1} : i32}}
      aiex.dma_start_task(%init_b)

      %init_c = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {{
        aie.dma_bd(%outC : memref<{c_total}xi32> offset = 0 len = %c_c_len sizes = [1, %c_tc_i64, %c_grp_m_i64, %c_n_i64] strides = [%c_stride_mN_i64, %c_n_i64, %c_stride_N_i64, 1])
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%init_c)
      aiex.dma_await_task(%init_c)

      // --- Loop over remaining row blocks ---
      %last_a = scf.for %gi = %c1 to %tr step %c1 iter_args(%prev_a = %init_a) -> (index) {{
        %a = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {{
          aie.dma_bd(%inA : memref<{a_total}xi8> offset = 0 len = %c_a_len sizes = [%c_tc_i64, %c_ks_i64, %c_grp_m_i64, %c_k_i64] strides = [0, %c_k_i64, %c_K_i64, 1])
          aie.end
        }} {{repeat_count = {tc - 1} : i32}}
        aiex.dma_start_task(%a)
        aiex.dma_free_task(%prev_a)
        scf.yield %a : index
      }}

      %last_b = scf.for %gi = %c1 to %tr step %c1 iter_args(%prev_b = %init_b) -> (index) {{
        %b = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {{
          aie.dma_bd(%inB : memref<{b_total}xi8> offset = 0 len = %c_b_len sizes = [%c_tc_i64, %c_ks_i64, %c_k_i64, %c_n_i64] strides = [%c_n_i64, %c_stride_KN_i64, %c_stride_N_i64, 1])
          aie.end
        }} {{repeat_count = {tc - 1} : i32}}
        aiex.dma_start_task(%b)
        aiex.dma_free_task(%prev_b)
        scf.yield %b : index
      }}

      %last_c = scf.for %gi = %c1 to %tr step %c1 iter_args(%prev_c = %init_c) -> (index) {{
        %c = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {{
          aie.dma_bd(%outC : memref<{c_total}xi32> offset = 0 len = %c_c_len sizes = [1, %c_tc_i64, %c_grp_m_i64, %c_n_i64] strides = [%c_stride_mN_i64, %c_n_i64, %c_stride_N_i64, 1])
          aie.end
        }} {{issue_token = true}}
        aiex.dma_start_task(%c)
        aiex.dma_await_task(%c)
        aiex.dma_free_task(%prev_c)
        scf.yield %c : index
      }}

      aiex.dma_free_task(%last_a)
      aiex.dma_free_task(%last_b)
      aiex.dma_free_task(%last_c)
    }}
  }}
}}
"""


def main():
    import argparse
    p = argparse.ArgumentParser(description="INT8 GEMM MLIR generator")
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=1024)
    p.add_argument("-N", type=int, default=4096)
    p.add_argument("--dynamic", action="store_true",
                    help="Emit dynamic ping-pong pattern (needs pool pass + cpp)")
    args = p.parse_args()
    if args.dynamic:
        sys.stdout.write(gen_dynamic(args.M, args.K, args.N))
    else:
        sys.stdout.write(gen_static(args.M, args.K, args.N))


if __name__ == "__main__":
    main()
