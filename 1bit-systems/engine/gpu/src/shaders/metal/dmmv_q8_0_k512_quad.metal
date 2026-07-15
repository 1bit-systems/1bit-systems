#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Q8_0 DMMV specialization for Qwen3.6 shared-down (M=2048, K=512).
//
// The generic quad-row Q8 path assigns one Q8 block to one lane. At K=512 that
// leaves lanes 16..31 idle because there are only 16 blocks per row. Split each
// block across two lanes instead. This variant extends the reference implementation's adjacent-row
// reuse discipline to eight output rows per simdgroup; each lane loads the X
// half-block once and applies it to eight Q8 rows.
kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 8u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint blocks_per_row = 16u; // K=512, Q8_0 block size=32
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const bool has1 = base_row + 1u < p.M;
    const bool has2 = base_row + 2u < p.M;
    const bool has3 = base_row + 3u < p.M;
    const bool has4 = base_row + 4u < p.M;
    const bool has5 = base_row + 5u < p.M;
    const bool has6 = base_row + 6u < p.M;
    const bool has7 = base_row + 7u < p.M;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = has1 ? (row0 + row_bytes) : row0;
    device const uchar* row2 = has2 ? (row0 + 2ull * row_bytes) : row0;
    device const uchar* row3 = has3 ? (row0 + 3ull * row_bytes) : row0;
    device const uchar* row4 = has4 ? (row0 + 4ull * row_bytes) : row0;
    device const uchar* row5 = has5 ? (row0 + 5ull * row_bytes) : row0;
    device const uchar* row6 = has6 ? (row0 + 6ull * row_bytes) : row0;
    device const uchar* row7 = has7 ? (row0 + 7ull * row_bytes) : row0;

    const uint block_idx = lane >> 1;
    const uint half_sel = lane & 1u;
    const uint half_base = half_sel << 4; // 0 or 16 elements within the Q8_0 block
    const uint x_base = (block_idx << 5) + half_base;

    device const uchar* blk0 = row0 + block_idx * 34u;
    device const uchar* blk1 = row1 + block_idx * 34u;
    device const uchar* blk2 = row2 + block_idx * 34u;
    device const uchar* blk3 = row3 + block_idx * 34u;
    device const uchar* blk4 = row4 + block_idx * 34u;
    device const uchar* blk5 = row5 + block_idx * 34u;
    device const uchar* blk6 = row6 + block_idx * 34u;
    device const uchar* blk7 = row7 + block_idx * 34u;

    const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
    const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
    const float s2 = float(as_type<half>(*(device const ushort*)(blk2)));
    const float s3 = float(as_type<half>(*(device const ushort*)(blk3)));
    const float s4 = float(as_type<half>(*(device const ushort*)(blk4)));
    const float s5 = float(as_type<half>(*(device const ushort*)(blk5)));
    const float s6 = float(as_type<half>(*(device const ushort*)(blk6)));
    const float s7 = float(as_type<half>(*(device const ushort*)(blk7)));

    device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u + half_base);
    device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u + half_base);
    device const packed_char4* q2 = (device const packed_char4*)(blk2 + 2u + half_base);
    device const packed_char4* q3 = (device const packed_char4*)(blk3 + 2u + half_base);
    device const packed_char4* q4 = (device const packed_char4*)(blk4 + 2u + half_base);
    device const packed_char4* q5 = (device const packed_char4*)(blk5 + 2u + half_base);
    device const packed_char4* q6 = (device const packed_char4*)(blk6 + 2u + half_base);
    device const packed_char4* q7 = (device const packed_char4*)(blk7 + 2u + half_base);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    #pragma unroll
    for (uint vi = 0u; vi < 4u; ++vi) {
        const float4 x = *(device const float4*)(input + x_base + (vi << 2));
        acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
        acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
        acc2 = fma(s2, dot(float4(char4(q2[vi])), x), acc2);
        acc3 = fma(s3, dot(float4(char4(q3[vi])), x), acc3);
        acc4 = fma(s4, dot(float4(char4(q4[vi])), x), acc4);
        acc5 = fma(s5, dot(float4(char4(q5[vi])), x), acc5);
        acc6 = fma(s6, dot(float4(char4(q6[vi])), x), acc6);
        acc7 = fma(s7, dot(float4(char4(q7[vi])), x), acc7);
    }

    // Cycle ~70: pack the eight final-reduction `simd_sum` calls into two
    // `simd_sum(float4)` calls — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // shuffle_xor instead of eight independent 32-bit trees, cutting cross-lane
    // shuffle traffic ~4× on the per-simdgroup tail of hot kernel #5 (224 ms/req
    // across 1436 calls, 15% of timed kernel time — the Qwen3.6 shared-expert
    // down projection M=2048, K=512). The downstream lane<8 writeback consumes
    // all eight sums as simdgroup-uniform scalars, so picking float4 components
    // by lane is bit-equivalent. Same proven pattern as cycle ~62/64/68/69
    // (which packed four sums via float4) — this is the first 8-row packing,
    // bisecting into a low/high pair to keep each reduction at float4 width.
    const float4 sums_lo = simd_sum(float4(acc0, acc1, acc2, acc3));
    const float4 sums_hi = simd_sum(float4(acc4, acc5, acc6, acc7));
    // Parallelize the 8-row writeback across lanes 0..7 (lane 0 serial 8
    // stores → lanes 0..7 coalesced 32-byte store). After simd_sum all
    // eight sums are broadcast to every lane in registers; base_row+0..+7
    // are eight contiguous floats so the production Qwen3.6-35B shared-down
    // case (M=2048, M%8==0 ⇒ every simdgroup writes a full eight rows)
    // issues a single coalesced 32-byte store instead of eight serial
    // lane-0 stores. The `base_row + lane < p.M` predicate (previously
    // expressed as has1..has7) handles the generic-validation tail (M=19
    // in the in-tree shader test). Mirrors cycle-27/32/38/39/40/41/43/45
    // lane-parallel writeback discipline across the Q8/Q5_K family; this
    // is the widest (8-row) still-serial writeback hot kernel — profile
    // hot #5 (217.47ms/req, 1436 calls, avg 151us, ~40 calls/decode token
    // × 256 WGs ≈ 10K TGs/token) covering the Qwen3.6 shared-expert down
    // projection.
    if (lane < 8u && base_row + lane < p.M) {
        const float local_sum = (lane == 0u) ? sums_lo.x
                              : (lane == 1u) ? sums_lo.y
                              : (lane == 2u) ? sums_lo.z
                              : (lane == 3u) ? sums_lo.w
                              : (lane == 4u) ? sums_hi.x
                              : (lane == 5u) ? sums_hi.y
                              : (lane == 6u) ? sums_hi.z
                              : sums_hi.w;
        output[base_row + lane] = local_sum;
    }
}
