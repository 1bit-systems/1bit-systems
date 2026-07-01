#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Qwen3.6 F32-router MoE boundary fusion.
//
// Fuses residual_rms_norm.metal with router_f32_topk_batched.metal for the
// single-token prefill path: hidden += residual, materialize ffn_norm for the
// expert DMMVs, then route the F32 router weights from the cached normalized
// vector in the same dispatch. Also emits the one-row F32 shared-gate dot so
// the following MoE finalizer can consume compact metadata instead of
// re-reading the normalized row.

struct Params {
    uint n;
    float eps;
    float scale;
    uint residual_offset;
    uint n_experts;
    uint K;
    uint k;
    uint a_offset;
    uint shared_gate_offset;
};

// Cycle-42: bump TG_SIZE 512→1024 to match the Q8 router sibling
// (ROUTER_TG_SIZE=1024 in residual_rms_norm_router_q8_0_topk_repacked_k2048).
// The kernel runs as a SINGLE threadgroup dispatch — `.{ 1, 1, 1 }` grid in
// dispatchQwenResidualRmsNormRouterF32TopkOnCmd — so adding simdgroups inside
// the TG is the only way to widen parallelism short of splitting the kernel.
// With TG_SIZE=1024, N_SIMDGROUPS=32 and ROWS_PER_TG=64, the 256-expert router
// GEMM completes in 4 sequential row_blocks instead of 8: per-simdgroup work
// halves on the hottest decode-token kernel (cycle-41 profile: hot #1 at
// 234.98us avg × 1436 calls = 337.43ms total, 23% of timed kernel time).
// Shared mem stays ~17.5 KB (16 KB x_cache4 + 1 KB values + small partials),
// well under Apple9's 32 KB/TG limit. Distinct from cycle-34 (reverted), which
// quadrupled ROWS_PER_SG and was likely register-bound; this change keeps
// ROWS_PER_SG=2 unchanged and only widens the simdgroup count.
#define TG_SIZE 1024
#define SIMD_WIDTH 32
#define N_SIMDGROUPS (TG_SIZE / SIMD_WIDTH)
#define ROWS_PER_TG (N_SIMDGROUPS * 2)
#define MAX_EXPERTS 256
#define MAX_K_USED 16
#define MAX_K_VEC4 1024

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* norm_out [[buffer(3)]],
    device const float* norm_weight [[buffer(4)]],
    device const float* W [[buffer(5)]],
    device uint* output_data [[buffer(6)]],
    device const float* W_shared_gate [[buffer(7)]],
    device float* shared_gate_out [[buffer(8)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (p.n == 0u || p.K != p.n || (p.K & 3u) != 0u ||
        (p.K >> 2) > MAX_K_VEC4 ||
        p.n_experts == 0u || p.n_experts > MAX_EXPERTS ||
        p.k == 0u || p.k > MAX_K_USED) {
        return;
    }

    threadgroup float4 x_cache4[MAX_K_VEC4];
    threadgroup float partial_sums[N_SIMDGROUPS];
    threadgroup float values[MAX_EXPERTS];
    threadgroup float selected_val[MAX_K_USED];
    threadgroup float shared_partials[N_SIMDGROUPS];

    // Cycle ~60: keep the per-thread `updated` float4 in REGISTERS across the
    // sum_sq → rms_inv reduction barrier, eliminating one TG mem write (Phase 1)
    // and one TG mem read (Phase 2) per active thread on the #1 hot decode-token
    // kernel (339 ms/req across 1436 calls, 23% of timed kernel time, single-TG
    // dispatch per `dispatchQwenResidualRmsNormRouterF32TopkOnCmd`). The
    // validation guards at lines 61-66 ensure `(p.K >> 2) <= MAX_K_VEC4 = 1024`
    // and TG_SIZE=1024, so the TG_SIZE-stride loop body executes AT MOST ONCE
    // per thread (each thread's `vi = local_id`; the next iter `vi += TG_SIZE`
    // is always ≥ k_vec4). Same-thread Phase-1→Phase-2 dataflow (each thread
    // reads x_cache4[local_id] that it itself wrote) goes through registers
    // instead of TG mem; cross-thread x_cache4 visibility for the SUBSEQUENT
    // router-GEMM and shared-gate phases is preserved by Phase 2's
    // `x_cache4[vi] = nval` write and the line-107 barrier. Adapts the reference implementation's
    // `kernel_mul_mv_q4_K_f32_impl` (ggml-metal.metal:7763-7811) pattern that
    // keeps `yl[16]/yh[16]` in registers across the per-row reduction loop —
    // here applied across the inter-phase RMS reduction barrier. The
    // `threadgroup_barrier` at line 90 still serializes partial_sums writes;
    // the line-107 barrier still serializes nval visibility for downstream.
    const uint k_vec4 = p.K >> 2;
    const uint vi = local_id;
    const bool active = vi < k_vec4;
    float sum_sq = 0.0f;
    float4 updated = float4(0.0f);
    if (active) {
        const uint idx = vi << 2;
        const float4 h = *(device const float4*)(hidden + idx);
        const float4 r = *(device const float4*)(residual + p.residual_offset + idx);
        updated = fma(float4(p.scale), r, h);
        *(device float4*)(hidden + idx) = updated;
        sum_sq = dot(updated, updated);
    }

    const float sg_sum = simd_sum(sum_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
    const float total_sq = simd_sum(partial);
    const float rms_inv = fast::rsqrt(fast::divide(total_sq, float(p.n)) + p.eps);

    // Cycle ~61: extend the cycle-60 inter-phase register-passing pattern to
    // Phase 4 (shared-gate dot, lines 173-180). Hoist `nval` out of the active
    // block into an outer-scope register so the shared-gate loop can consume
    // the SAME float4 the thread wrote to x_cache4 without re-reading TG mem.
    // Validation guards (lines 61-66) ensure k_vec4 = (p.K >> 2) ≤ MAX_K_VEC4
    // = TG_SIZE = 1024, so the Phase 4 stride loop body executes AT MOST ONCE
    // per thread (vi = local_id; next iter vi += TG_SIZE is always ≥ k_vec4).
    // Therefore each Phase 4 read of x_cache4[local_id] is intra-thread (same
    // thread that wrote it in Phase 2) and can route through the register
    // instead of TG mem. The Phase 2 `x_cache4[vi] = nval` write + line-127
    // barrier still preserve cross-thread x_cache4 visibility for Phase 3
    // (router GEMM uses lane-strided reads across simdgroups). Same reference implementation
    // `kernel_mul_mv_q4_K_f32_impl` register-resident-Y pattern adapted to
    // the SUBSEQUENT inter-phase boundary on hot kernel #1 (339 ms/req across
    // 1436 calls). Eliminates one TG mem read per active thread in Phase 4.
    float4 nval = float4(0.0f);
    if (active) {
        const uint idx = vi << 2;
        const float4 w = *(device const float4*)(norm_weight + idx);
        nval = w * (updated * rms_inv);
        x_cache4[vi] = nval;
        *(device float4*)(norm_out + idx) = nval;
    }

    if (local_id < MAX_EXPERTS) {
        values[local_id] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint weight_base = p.a_offset >> 2;
    for (uint row_block = 0u; row_block < p.n_experts; row_block += ROWS_PER_TG) {
        const uint base_row = row_block + sg_idx * 2u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        if (base_row < p.n_experts) {
            device const float* row0 = W + weight_base + base_row * p.K;
            device const float* row1 = row0 + p.K;

            for (uint vi = lane; vi < k_vec4; vi += SIMD_WIDTH) {
                const float4 x = x_cache4[vi];
                acc0 += dot(*(device const float4*)(row0 + (vi << 2)), x);
                if (base_row + 1u < p.n_experts) {
                    acc1 += dot(*(device const float4*)(row1 + (vi << 2)), x);
                }
            }

            // Cycle ~65: pack the two router-GEMM final-reduction `simd_sum`
            // calls into a single `simd_sum(float2)` — Apple's `simd_sum` on
            // vector types issues one fused butterfly per lane width (5
            // shuffle_xor pairs for SIMD32) instead of two sequential
            // reductions, halving cross-lane shuffle traffic on the
            // per-simdgroup row tail. Same pattern as cycle ~62/63 on
            // `dmmv_q5k_moe_k512_quad`/`dmmv_q4k_moe_gate_up_swiglu_k2048` and
            // cycle ~64 on `dmmv_q8_0_pair_swiglu`. Hot kernel #1 fires 4
            // row_blocks × 32 simdgroups = 128 reduction tails per kernel
            // call × 1436 calls/req — saving one `simd_sum` per tail amplifies
            // into ~184K shuffle-pair savings per request. acc1 stays semantic
            // even when `base_row + 1u >= p.n_experts` (always zero in that
            // case, would have summed zeros either way); the `store_row <
            // p.n_experts` predicate below still gates the write.
            const float2 sums = simd_sum(float2(acc0, acc1));
            // Cycle-52: lane-parallel 2-row writeback of router GEMM partials
            // into threadgroup `values[]`. After `simd_sum`, both `sums.x`
            // and `sums.y` are present on every lane; lanes 0 and 1 each
            // issue one store to consecutive `values[base_row..base_row+1]`
            // slots in a single SIMD scatter instead of lane 0 doing two
            // serial stores. The `store_row < p.n_experts` predicate
            // preserves the generic-validation tail (n_experts < ROWS_PER_TG
            // cases). Extends cycle-43/45/47/49/50 lane-parallel writeback
            // discipline from global memory to threadgroup memory; the
            // barrier at line 147 still serializes against the next-phase
            // top-k reader.
            const uint store_row = base_row + lane;
            if (lane < 2u && store_row < p.n_experts) {
                const float val = (lane == 0u) ? sums.x : sums.y;
                values[store_row] = val;
            }
        }
    }

    device const float* shared_row = W_shared_gate + (p.shared_gate_offset >> 2);
    float shared_acc = 0.0f;
    if (active) {
        shared_acc = dot(*(device const float4*)(shared_row + (local_id << 2)), nval);
    }
    const float shared_sum = simd_sum(shared_acc);
    if (lane == 0u) {
        shared_partials[sg_idx] = shared_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cycle-58: move `shared_partials` reduction + `shared_gate_out` store from
    // sg 0 to sg 1 so sg 0 can begin the 8-slot top-k loop IMMEDIATELY after the
    // line-161 barrier. Cycle-51 already collapsed the reduction to a single
    // `simd_sum` (~5-10 cycles); cycle-52 parallelized the router-GEMM writeback.
    // The remaining critical path through sg 0 is `simd_sum(shared_partials[])`
    // (5 shuffle_xor steps + 1 lane-0 global store) followed by the 8-slot
    // simd_max/simd_min top-k loop (8 slots × ~5 shuffles + accept/reject mask
    // work = ~400 cycles). Top-k is FAR longer than the shared reduction, and
    // both work blocks are data-independent — top-k reads `values[]` while the
    // reduction reads `shared_partials[]` and writes `shared_gate_out[0]`, a
    // distinct global output buffer. Running them on different simdgroups
    // (sg 0 ↔ top-k, sg 1 ↔ shared_total) lets the GPU dispatch both wave fronts
    // in parallel. sg 0's critical path drops by ~15 cycles per kernel call;
    // sg 1's critical path is only ~17 cycles (well under sg 0's top-k tail),
    // so the kernel-exit timing is still gated by sg 0 — meaning the savings
    // are real net gain rather than just shifted. Hot kernel #1
    // (337 ms/req across 1436 calls, 23% of timed kernel time, single-TG
    // `{1,1,1}` × `{1024,1,1}` dispatch per
    // `dispatchQwenResidualRmsNormRouterF32TopkOnCmd`). Bit-equivalent output —
    // same `simd_sum` math, just executed on a different sg. With TG_SIZE=1024,
    // sg 1 always exists (N_SIMDGROUPS=32), so this is safe under the same
    // validation guards (n_experts ≤ MAX_EXPERTS, k ≤ MAX_K_USED) that the
    // existing kernel relies on.
    if (sg_idx == 1u) {
        const float shared_total = simd_sum(shared_partials[lane]);
        if (lane == 0u) {
            shared_gate_out[0] = shared_total;
        }
    }

    if (p.n_experts == 256u && p.k == 8u) {
        if (sg_idx == 0u) {
            float selected_score[8];
            uint selected_mask = 0u;
            #pragma unroll
            for (uint slot = 0u; slot < 8u; ++slot) {
                selected_score[slot] = -INFINITY;
            }

            #pragma unroll
            for (uint slot = 0u; slot < 8u; ++slot) {
                float lane_best = -INFINITY;
                uint lane_best_idx = 0xffffffffu;
                #pragma unroll
                for (uint lane_row = 0u; lane_row < 8u; ++lane_row) {
                    const uint expert = lane + (lane_row << 5);
                    const float score = ((selected_mask & (1u << lane_row)) == 0u) ? values[expert] : -INFINITY;
                    if (score > lane_best) {
                        lane_best = score;
                        lane_best_idx = expert;
                    }
                }
                const float best_val = simd_max(lane_best);
                const uint best_idx = simd_min((lane_best == best_val) ? lane_best_idx : 0xffffffffu);
                selected_score[slot] = best_val;
                if ((best_idx & 31u) == lane) {
                    selected_mask |= 1u << (best_idx >> 5);
                }
                if (lane == 0u) {
                    output_data[slot] = best_idx;
                }
            }

            const bool weight_lane = lane < 8u;
            const float score = weight_lane ? selected_score[lane] : -INFINITY;
            const float max_sel = simd_max(score);
            const float exp_score = weight_lane ? fast::exp(score - max_sel) : 0.0f;
            const float sum = simd_sum(exp_score);
            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            if (weight_lane) {
                output_data[8u + lane] = as_type<uint>(exp_score * inv_sum);
            }
        }
        return;
    }

    if (local_id == 0u) {
        const uint k = min(p.k, uint(MAX_K_USED));
        for (uint slot = 0u; slot < k; ++slot) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < p.n_experts; ++expert) {
                const float score = values[expert];
                if (score > best_val) {
                    best_val = score;
                    best_idx = expert;
                }
            }
            output_data[slot] = best_idx;
            selected_val[slot] = best_val;
            values[best_idx] = -INFINITY;
        }

        float max_sel = -INFINITY;
        for (uint slot = 0u; slot < k; ++slot) {
            max_sel = max(max_sel, selected_val[slot]);
        }

        float sum = 0.0f;
        for (uint slot = 0u; slot < k; ++slot) {
            const float e = fast::exp(selected_val[slot] - max_sel);
            selected_val[slot] = e;
            sum += e;
        }

        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint slot = 0u; slot < k; ++slot) {
            output_data[p.k + slot] = as_type<uint>(selected_val[slot] * inv_sum);
        }
    }
}
