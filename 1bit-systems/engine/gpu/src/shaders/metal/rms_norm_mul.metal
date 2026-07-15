#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
};

// Multi-simdgroup RMS norm with threadgroup memory reduction.
// Uses up to 1024 threads (32 simdgroups) for better float32 precision
// when accumulating sum-of-squares over large vectors with wide value ranges.
//
// Pass 1 caches the input values it just read into a per-thread register
// array so pass 2 normalizes from registers instead of re-reading `input`
// from device memory. Mirrors the residual_rms_norm pattern. For Qwen3-8B
// the hot shapes are (n=4096,tg=1024) and (n=128,tg=32) — both yield 4
// iterations/thread, well within MAX_PER_THREAD=16. Models exceeding the
// cache size fall back to the device re-read for the overflow tail.
#define MAX_PER_THREAD 16

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shmem[32]; // one slot per simdgroup

    const uint base = group_id * p.n;

    float vals[MAX_PER_THREAD];
    uint cached = 0;
    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = input[base + i];
        sum_sq += v * v;
        if (cached < MAX_PER_THREAD) {
            vals[cached] = v;
            cached++;
        }
    }

    // Reduce within simdgroup
    sum_sq = simd_sum(sum_sq);

    // Write per-simdgroup result to shared memory
    if (simd_lane == 0) {
        shmem[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every simdgroup does the final reduction + rms_inv independently.
    // Avoids the rms_inv broadcast barrier (used to be: thread 0 computes
    // rsqrt, writes to threadgroup mem, barrier, all 1024 threads read).
    // Work is duplicated 32× but Apple7 issues the same fast::rsqrt in
    // parallel across simdgroups; the saved barrier — ~144 dispatches/token
    // on Qwen3-8B — is net positive.
    const uint n_groups = (tg_size + subgroup_size - 1) / subgroup_size;
    float total = (simd_lane < n_groups) ? shmem[simd_lane] : 0.0f;
    total = simd_sum(total);
    const float rms_inv = fast::rsqrt(fast::divide(total, float(p.n)) + p.eps);

    uint c = 0;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = (c < MAX_PER_THREAD) ? vals[c] : input[base + i];
        output[base + i] = weights[i] * (v * rms_inv);
        c++;
    }
}
