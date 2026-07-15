#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Qwen3.6 full-attention prompt fusion:
//   attn_delta = post_attn_norm(residual)
//   hidden = hidden + attn_delta
//   norm = ffn_norm(hidden)
//   routing = F32 router top-k(norm)
//   shared_gate = dot(F32 shared-gate, norm)
//
// This adapts the reference implementation's dependency-edge fusion discipline and vLLM's compact
// top-k metadata flow to the post-attention boundary. It keeps the same
// selected-expert row format as router_f32_topk_batched.

struct Params {
    uint n;
    float eps;
    uint n_experts;
    uint K;
    uint k;
    uint router_offset;
    uint shared_gate_offset;
    uint output_stride;
};

// Cycle ~92: bump TG_SIZE 512→1024 to match the SSM-boundary sibling
// `residual_rms_norm_router_f32_topk.metal` (cycle-42, TG_SIZE=1024).
// Dispatch is single-TG (`.{1,1,1}` × `.{TG_SIZE,1,1}` in
// `dispatchQwenPostNormResidualRouterF32SharedGateOnCmd`) so bumping does NOT
// reduce in-flight TG count — it only adds simdgroups within the same TG.
// With SG_PER_TG=32 and ROWS_PER_TG = SG_PER_TG * 2 = 64, the 256-expert
// router GEMM completes in 4 sequential row_blocks instead of 8: per-simdgroup
// work halves on the full-attn-boundary decode-token kernel (~360 calls/req
// across 10 attn layers). Shared mem stays well under Apple9's 32 KB/TG limit
// (~17 KB: x_cache4 16 KB + values 1 KB + small partials). Distinct from the
// failed cycle-42 attempt on `dmmv_q4k_moe_gate_up_swiglu_k2048` which was a
// many-TG DMMV shader where bumping reduced TG-level parallelism; here the
// single-TG geometry means strictly more parallelism with no tradeoff.
#define TG_SIZE 1024
#define SG_PER_TG (TG_SIZE / 32)
#define MAX_EXPERTS 256
#define MAX_K_USED 16
#define MAX_K_VEC4 1024

kernel void main0(
    device const float* W_router [[buffer(0)]],
    constant Params& p [[buffer(1)]],
    device float* hidden [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device const float* post_norm_weight [[buffer(4)]],
    device float* norm_out [[buffer(5)]],
    device const float* ffn_norm_weight [[buffer(6)]],
    device uint* output_data [[buffer(7)]],
    device const float* W_shared_gate [[buffer(8)]],
    device float* shared_gate_out [[buffer(9)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 x_cache4[MAX_K_VEC4];
    threadgroup float partial_sums[SG_PER_TG];
    threadgroup float values[MAX_EXPERTS];
    threadgroup float selected_val[MAX_K_USED];
    threadgroup float shared_partials[SG_PER_TG];

    if (p.n_experts == 0u || p.n_experts > MAX_EXPERTS ||
        p.k == 0u || p.k > MAX_K_USED ||
        p.n == 0u || p.n != p.K || (p.K & 3u) != 0u ||
        (p.K >> 2) > MAX_K_VEC4) {
        return;
    }

    const uint k_vec4 = p.K >> 2;

    float residual_sq = 0.0f;
    for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
        const float4 r = *(device const float4*)(residual + (vi << 2));
        residual_sq += dot(r, r);
    }

    float sg_sum = simd_sum(residual_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float part = (lane < SG_PER_TG) ? partial_sums[lane] : 0.0f;
    float total = simd_sum(part);
    const float residual_rms_inv = rsqrt((total / float(p.n)) + p.eps);

    float hidden_sq = 0.0f;
    for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
        const uint off = vi << 2;
        const float4 r = *(device const float4*)(residual + off);
        const float4 rn = r * *(device const float4*)(post_norm_weight + off) * residual_rms_inv;
        const float4 h = *(device const float4*)(hidden + off) + rn;
        *(device float4*)(hidden + off) = h;
        x_cache4[vi] = h;
        hidden_sq += dot(h, h);
    }

    sg_sum = simd_sum(hidden_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    part = (lane < SG_PER_TG) ? partial_sums[lane] : 0.0f;
    total = simd_sum(part);
    const float hidden_rms_inv = rsqrt((total / float(p.n)) + p.eps);

    for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
        const uint off = vi << 2;
        const float4 norm4 = x_cache4[vi] * *(device const float4*)(ffn_norm_weight + off) * hidden_rms_inv;
        x_cache4[vi] = norm4;
        *(device float4*)(norm_out + off) = norm4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint router_base = p.router_offset >> 2;
    // Cycle ~93 follow-up to cycle 92's TG_SIZE 512→1024 bump: with SG_PER_TG
    // now 32, promote ROWS_PER_SG 2→4 so each row_block covers SG_PER_TG*4=128
    // rows. For the Qwen3.6 256-expert router this collapses the outer loop
    // from 4 iterations to 2, AND packs the four per-row `simd_sum` tails into
    // a single `simd_sum(float4)` — Apple9's vec simd_sum lowers to one
    // log2(32)=5-level butterfly that shuffles 128-bit packed lanes per
    // `shuffle_xor` instead of four independent 32-bit trees, cutting
    // cross-lane shuffle traffic ~4× on each per-simdgroup tail (vs. the
    // 2-row cycle-91 baseline). Also halves outer-loop / barrier overhead and
    // doubles `x_cache4[vi]` LDS reuse (one load feeds 4 row dots instead of
    // 2). Hot user: every Qwen3.6 full-attention decode-token boundary (10
    // attn layers × per-step ≈ 360 calls/req); previous 32 sg × 4 iter × 2
    // tails ≈ 256 reduction tails / call drops to 32 sg × 2 iter × 1 tail =
    // 64 tails / call. The `base_row + N < p.n_experts` predicates stay
    // uniform within the simdgroup (depend only on row_block + sg_idx, not
    // lane) so no divergence. Register pressure: 4 accumulators per lane is
    // well within Apple9's per-thread register file (~256 32-bit regs). Same
    // proven pattern as cycles 83/85/91 across the dual/quad simd_sum family.
    for (uint row_block = 0u; row_block < p.n_experts; row_block += SG_PER_TG * 4u) {
        const uint base_row = row_block + sg_idx * 4u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        if (base_row < p.n_experts) {
            device const float* row0 = W_router + router_base + base_row * p.K;
            device const float* row1 = row0 + p.K;
            device const float* row2 = row1 + p.K;
            device const float* row3 = row2 + p.K;

            for (uint vi = lane; vi < k_vec4; vi += 32u) {
                const float4 x = x_cache4[vi];
                acc0 += dot(*(device const float4*)(row0 + (vi << 2)), x);
                if (base_row + 1u < p.n_experts) {
                    acc1 += dot(*(device const float4*)(row1 + (vi << 2)), x);
                }
                if (base_row + 2u < p.n_experts) {
                    acc2 += dot(*(device const float4*)(row2 + (vi << 2)), x);
                }
                if (base_row + 3u < p.n_experts) {
                    acc3 += dot(*(device const float4*)(row3 + (vi << 2)), x);
                }
            }

            const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
            const uint store_row = base_row + lane;
            if (lane < 4u && store_row < p.n_experts) {
                const float val = (lane == 0u) ? sums.x
                                : (lane == 1u) ? sums.y
                                : (lane == 2u) ? sums.z
                                               : sums.w;
                values[store_row] = val;
            }
        }
    }

    device const float* shared_row = W_shared_gate + (p.shared_gate_offset >> 2);
    float shared_acc = 0.0f;
    for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
        shared_acc += dot(*(device const float4*)(shared_row + (vi << 2)), x_cache4[vi]);
    }
    const float shared_sum = simd_sum(shared_acc);
    if (lane == 0u) {
        shared_partials[sg_idx] = shared_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cycle-58 sibling treatment: move `shared_partials` reduction +
    // `shared_gate_out` store from `local_id == 0` to sg 1 so sg 0 can begin
    // the 8-slot top-k loop IMMEDIATELY after the barrier. Mirrors the proven
    // pattern from the SSM-boundary sibling `residual_rms_norm_router_f32_topk.metal`
    // (lines 231-236, cycle 58). Two wins combined: (a) replaces the 16-iter
    // scalar `for` sum (~15 cycles) with one `simd_sum` (5 shuffle_xor steps),
    // and (b) parallelizes the shared_total store with sg 0's ~400-cycle top-k
    // loop — the two are data-independent (top-k reads `values[]`, this reads
    // `shared_partials[]` and writes `shared_gate_out[0]`, distinct buffers).
    // Post-cycle-92 TG bump: SG_PER_TG=32 now matches the SSM-boundary sibling
    // exactly, so the `(lane < SG_PER_TG)` mask becomes `(lane < 32)` which is
    // identically true across the 32-lane simdgroup; compiler will fold it.
    // sg 1 always exists (N_SIMDGROUPS=32).
    // Hot user: every Qwen3.6 full-attention decode-token boundary (10 attn
    // layers × per-step ≈ 360 calls/req).
    if (sg_idx == 1u) {
        const float part = (lane < SG_PER_TG) ? shared_partials[lane] : 0.0f;
        const float shared_total = simd_sum(part);
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
        const uint kk = min(p.k, uint(MAX_K_USED));
        for (uint slot = 0u; slot < kk; slot++) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < p.n_experts; expert++) {
                const float v = values[expert];
                if (v > best_val) {
                    best_val = v;
                    best_idx = expert;
                }
            }
            output_data[slot] = best_idx;
            selected_val[slot] = best_val;
            values[best_idx] = -INFINITY;
        }

        float max_sel = -INFINITY;
        for (uint slot = 0u; slot < kk; slot++) {
            max_sel = max(max_sel, selected_val[slot]);
        }

        float exp_sum = 0.0f;
        for (uint slot = 0u; slot < kk; slot++) {
            const float e = exp(selected_val[slot] - max_sel);
            selected_val[slot] = e;
            exp_sum += e;
        }

        const float inv_sum = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
        for (uint slot = 0u; slot < kk; slot++) {
            output_data[p.k + slot] = as_type<uint>(selected_val[slot] * inv_sum);
        }
    }
}
