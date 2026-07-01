#include <metal_stdlib>

using namespace metal;

struct Params {
    uint n_tokens;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint ids_stride;
    uint profile_index;
    uint profile_slots;
};

#define NUM_COLS 8u
#define MAX_EXPERTS 256u
#define PROFILE_TAIL_BINS 7u
#define PROFILE_STATS_PER_LAYER (4u + PROFILE_TAIL_BINS)

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]],
    device uint* ids [[buffer(3)]],
    device atomic_uint* active_block_count [[buffer(4)]],
    device uint* active_blocks [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup uint route_counts[MAX_EXPERTS];
    threadgroup uint block_counts[MAX_EXPERTS];

    if (tid == 0u) {
        atomic_store_explicit(active_block_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    const uint total_routes = p.n_tokens * p.k;

    if (tid < p.n_experts) {
        atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
        route_counts[tid] = 0u;
        block_counts[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (tid < p.n_experts) {
        for (uint route = tid; route < total_routes; route += p.n_experts) {
            const uint token = route / p.k;
            const uint slot = route - token * p.k;
            const uint expert_id = routing[token * p.routing_stride + slot];
            if (expert_id >= p.n_experts) {
                continue;
            }

            const uint index = atomic_fetch_add_explicit(counts + expert_id, 1u, memory_order_relaxed);
            if (index < p.ids_stride) {
                ids[expert_id * p.ids_stride + index] = route;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (tid < p.n_experts) {
        const uint count = atomic_load_explicit(counts + tid, memory_order_relaxed);
        const uint stored_count = min(count, p.ids_stride);
        route_counts[tid] = stored_count;
        block_counts[tid] = (stored_count + NUM_COLS - 1u) / NUM_COLS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint total_blocks = 0u;
        uint full_blocks = 0u;
        uint tail_blocks = 0u;
        uint singleton_tail_blocks = 0u;
        uint padding_slots = 0u;
        uint tail_size_blocks[PROFILE_TAIL_BINS] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u };
        for (uint expert = 0u; expert < p.n_experts; expert++) {
            total_blocks += block_counts[expert];
            const uint stored_count = route_counts[expert];
            if (stored_count == 0u) {
                continue;
            }
            full_blocks += stored_count / NUM_COLS;
            const uint tail_routes = stored_count % NUM_COLS;
            if (tail_routes != 0u) {
                tail_blocks += 1u;
                tail_size_blocks[tail_routes - 1u] += 1u;
                padding_slots += NUM_COLS - tail_routes;
                if (tail_routes == 1u) {
                    singleton_tail_blocks += 1u;
                }
            }
        }
        if (p.profile_index < p.profile_slots) {
            atomic_store_explicit(active_block_count + 1u + p.profile_index, total_blocks, memory_order_relaxed);
            device atomic_uint* layer_stats = active_block_count + 1u + p.profile_slots + p.profile_index * PROFILE_STATS_PER_LAYER;
            atomic_store_explicit(layer_stats + 0u, full_blocks, memory_order_relaxed);
            atomic_store_explicit(layer_stats + 1u, tail_blocks, memory_order_relaxed);
            atomic_store_explicit(layer_stats + 2u, singleton_tail_blocks, memory_order_relaxed);
            atomic_store_explicit(layer_stats + 3u, padding_slots, memory_order_relaxed);
            for (uint i = 0u; i < PROFILE_TAIL_BINS; i++) {
                atomic_store_explicit(layer_stats + 4u + i, tail_size_blocks[i], memory_order_relaxed);
            }
        }
    }

    if (tid < p.n_experts) {
        const uint block_count = block_counts[tid];
        // Consumers read expert_id/block_idx from each entry, so stable
        // expert-major ordering is unnecessary. Reserve the output span
        // atomically instead of doing a per-expert prefix loop.
        const uint block_offset = atomic_fetch_add_explicit(active_block_count, block_count, memory_order_relaxed);
        for (uint block = 0u; block < block_count; block++) {
            active_blocks[block_offset + block] = tid | (block << 16u);
        }
    }
}
