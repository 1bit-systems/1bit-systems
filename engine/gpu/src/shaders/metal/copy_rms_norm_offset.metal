#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    uint src_offset;
    float eps;
};

#define MAX_PER_THREAD 16

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device float* hidden [[buffer(2)]],
    device float* norm [[buffer(3)]],
    device const float* weights [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shmem[32];

    const uint src_base = p.src_offset;

    float vals[MAX_PER_THREAD];
    uint cached = 0;
    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = src[src_base + i];
        hidden[i] = v;
        sum_sq += v * v;
        if (cached < MAX_PER_THREAD) {
            vals[cached] = v;
            cached++;
        }
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        shmem[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_groups = (tg_size + subgroup_size - 1) / subgroup_size;
    float total = (simd_lane < n_groups) ? shmem[simd_lane] : 0.0f;
    total = simd_sum(total);
    const float rms_inv = fast::rsqrt(fast::divide(total, float(p.n)) + p.eps);

    uint c = 0;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = (c < MAX_PER_THREAD) ? vals[c] : src[src_base + i];
        norm[i] = weights[i] * (v * rms_inv);
        c++;
    }
}
