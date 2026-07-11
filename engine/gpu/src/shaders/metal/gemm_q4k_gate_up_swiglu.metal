#include <metal_stdlib>
using namespace metal;

// Q4_K gate/up GEMM fused with SwiGLU for dense Qwen3.5 9B queued prefill.
//
// This follows the reference implementation's Metal batched mul_mat shape: one 64 row x 32 token
// tile per threadgroup using simdgroup matrix multiply. The difference is that
// the gate and up matrices share the same f32 prompt tile, so the shader loads
// B once, computes both projections, and writes SiLU(gate) * up directly.

struct GemmGateUpPush {
    int32_t  ne00;      // K dimension
    int32_t  ne02;      // unused batch dim, kept for GemmPush alignment
    uint64_t nb01;      // row stride of each Q4_K weight matrix in bytes
    uint64_t nb02;      // unused batch stride
    int32_t  ne12;      // unused batch dim
    uint32_t pad0;
    uint64_t nb10;      // input element stride in bytes
    uint64_t nb11;      // input token stride in bytes
    uint64_t nb12;      // unused batch stride
    int32_t  ne0;       // M rows
    int32_t  ne1;       // N prompt tokens
    uint32_t src0_off;  // gate weight byte offset
    uint32_t src1_off;  // up weight byte offset
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

#define QK_K 256
#define QK_NL 16
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                           uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

static void dequantize_q4_K(device const block_q4_K * xb, short il, thread half4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float mn = float(xb->dmin);
    const float dl = d * sc[0];
    const float ml = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(dl * (q[i] & mask) - ml);
    }
}

static inline float swiglu(float gate, float up) {
    return up * gate * fast::divide(1.0f, 1.0f + fast::exp(-gate));
}

#ifndef ZINC_GEMM_GATE_UP_NR1
#define ZINC_GEMM_GATE_UP_NR1 32
#endif
#ifndef ZINC_GEMM_GATE_UP_THREADS
#define ZINC_GEMM_GATE_UP_THREADS 128
#endif

constant constexpr int NR0 = 64;
constant constexpr int NR1 = ZINC_GEMM_GATE_UP_NR1;
constant constexpr int NK = 32;
constant constexpr int NL0 = NK / 16;
constant constexpr int NL1 = NK / 8;
constant constexpr int NCB = NR1 / 8;
constant constexpr int NT = ZINC_GEMM_GATE_UP_THREADS;
constant constexpr int A_LOAD_THREADS = 128;

kernel void main0(
    constant GemmGateUpPush & args [[buffer(0)]],
    device const char * gate_w [[buffer(1)]],
    device const char * up_w [[buffer(2)]],
    device const char * src1 [[buffer(3)]],
    device       char * dst [[buffer(4)]],
    threadgroup  char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short nr0 = min(args.ne0 - r0, NR0);
    const short nr1 = min(args.ne1 - r1, NR1);

    const ushort a_tiitg = tiitg & (A_LOAD_THREADS - 1);
    const short lr0 = min((short)(a_tiitg / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / NL1), (short)(nr1 - 1));
    const short il0 = a_tiitg % NL0;
    const short il1 = tiitg % NL1;
    const short iy = 8 * il1;

    short il_gate = il0;
    short il_up = il0;
    const short offset1 = il0 / QK_NL;

    device const block_q4_K * gate_x =
        (device const block_q4_K *)(gate_w + args.src0_off + args.nb01 * (r0 + lr0)) + offset1;
    device const block_q4_K * up_x =
        (device const block_q4_K *)(up_w + args.src1_off + args.nb01 * (r0 + lr0)) + offset1;
    device const float * y = (device const float *)(src1 + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    FOR_UNROLL (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_gate;
        if (tiitg < A_LOAD_THREADS) {
            dequantize_q4_K(gate_x, il_gate, temp_gate);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg < A_LOAD_THREADS) {
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (a_tiitg / NL0) / 8;
                const short lx = (a_tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_gate[i / 4][i % 4];
            }
        }

        {
            const short sx = il1;
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = NCB * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) = (half2x4)(*((device float2x4 *) y));
        }

        il_gate = (il_gate + 2 < QK_NL) ? il_gate + 2 : il_gate % 2;
        gate_x = (il_gate < 2) ? gate_x + (2 + QK_NL - 1) / QK_NL : gate_x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_gate + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i / 4], ma[i % 4], mc_gate[i]);
            }
            lsma_gate += 8 * 64;
            lsmb += NCB * 64;
        }

        half4x4 temp_up;
        if (tiitg < A_LOAD_THREADS) {
            dequantize_q4_K(up_x, il_up, temp_up);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg < A_LOAD_THREADS) {
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (a_tiitg / NL0) / 8;
                const short lx = (a_tiitg / NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_up[i / 4][i % 4];
            }
        }

        il_up = (il_up + 2 < QK_NL) ? il_up + 2 : il_up % 2;
        up_x = (il_up < 2) ? up_x + (2 + QK_NL - 1) / QK_NL : up_x;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_up = sa + 4 * 64 * (sgitg % 2);
        lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_up + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_up[i], mb[i / 4], ma[i % 4], mc_up[i]);
            }
            lsma_up += 8 * 64;
            lsmb += NCB * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * gate_tmp = (threadgroup float *)shmem;
    threadgroup float * up_tmp = (threadgroup float *)(shmem + NR0 * NR1 * sizeof(float));

    threadgroup float * gate_store =
        gate_tmp + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    threadgroup float * up_store =
        up_tmp + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;

    FOR_UNROLL (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], gate_store + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        simdgroup_store(mc_up[i], up_store + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int total = int(nr0) * int(nr1);
    for (int idx = tiitg; idx < total; idx += NT) {
        const int col = idx / int(nr0);
        const int row = idx - col * int(nr0);
        const float gate = gate_tmp[col * NR0 + row];
        const float up = up_tmp[col * NR0 + row];
        device float * out = (device float *)dst + (r0 + row) + (r1 + col) * args.ne0;
        *out = swiglu(gate, up);
    }
}
