#include <metal_stdlib>
using namespace metal;

struct ArgmaxPush {
    uint n;
};

kernel void main0(
    device const float* logits [[buffer(0)]],
    device uint* out [[buffer(1)]],
    constant ArgmaxPush& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n == 0) {
        if (tid == 0) out[0] = 0;
        return;
    }

    threadgroup float best_vals[32];
    threadgroup uint best_idxs[32];

    float best_val = -INFINITY;
    uint best_idx = 0;

    for (uint i = tid; i < p.n; i += 256) {
        const float v = logits[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    const float sg_best_val = simd_max(best_val);
    const uint sg_best_idx = simd_min(best_val == sg_best_val ? best_idx : 0xffffffffu);

    if (lane == 0u) {
        best_vals[sg_idx] = sg_best_val;
        best_idxs[sg_idx] = sg_best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0u) {
        const bool active = lane < simdgroups_per_tg;
        const float tg_val = active ? best_vals[lane] : -INFINITY;
        const uint tg_idx = active ? best_idxs[lane] : 0xffffffffu;
        const float final_val = simd_max(tg_val);
        const uint final_idx = simd_min(tg_val == final_val ? tg_idx : 0xffffffffu);
        if (lane == 0u) {
            out[0] = final_idx;
            out[1] = as_type<uint>(final_val);
        }
    }
}
