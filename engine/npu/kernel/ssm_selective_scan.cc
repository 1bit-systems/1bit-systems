// ssm_selective_scan.cc — AIE2 SSM selective scan kernel
//
// Maps Mamba2 selective scan onto AIE tiles:
//   Per head: state = expm1(A) * state + dt * B * x  (state update)
//             y    = state · C                        (output)
//
// Parallelism: 80 heads × 1 AIE tile each = 80 tiles (Strix Halo has 32,
// so 32 heads per batch, 3 batches)
//
// Each tile handles d_state=64 elements via 512-bit vectors.
//   state[64] × A[64] + dt × B[64] × x[64] = 128 MACs per step
//   1 token step = ~200 cycles per tile
//   32 tiles in parallel = ~6 cycles/token = ~6ns at 1GHz
//
// Licensed under Apache 2.0 with LLVM Exceptions.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

constexpr int D_STATE = 64;   // Mamba2 state dimension
constexpr int N_HEADS = 32;   // heads per tile batch (80 total → 3 batches)

extern "C" {

// ── One AIE tile processes N_HEADS × D_STATE selective scan ──
// Each head is independent — parallel across 32 AIE cores.
//
// Inputs (all bf16, contiguous per head):
//   state_in:  [N_HEADS × D_STATE] — initial state
//   A:         [N_HEADS × D_STATE] — log-state matrix (expm1 applied)
//   dt:        [N_HEADS]           — discretization step
//   B:         [N_HEADS × D_STATE] — input matrix
//   C:         [N_HEADS × D_STATE] — output matrix
//   x:         [N_HEADS × D_STATE] — input (from in_proj)
//
// Outputs:
//   state_out: [N_HEADS × D_STATE] — updated state
//   y:         [N_HEADS × D_STATE] — output (before gate)
void ssm_selective_scan(
    bfloat16 *state_in,
    bfloat16 *A,
    bfloat16 *dt,
    bfloat16 *B,
    bfloat16 *C,
    bfloat16 *x,
    bfloat16 *state_out,
    bfloat16 *y
) {
    event0();

    // Vectorized per-head: 64 elements = 8 × 8-element vectors
    for (int h = 0; h < N_HEADS; h++) {
        // Load state[64], A[64], B[64], C[64], x[64] as 8 vectors each
        alignas(32) bfloat16 s_buf[D_STATE], a_buf[D_STATE];
        alignas(32) bfloat16 b_buf[D_STATE], c_buf[D_STATE], x_buf[D_STATE];
        alignas(32) bfloat16 dt_buf[1];

        // Load from global memory via DMA
        for (int i = 0; i < D_STATE; i += 8) {
            auto vs = aie::load_v<8>(state_in + h * D_STATE + i);
            aie::store_v(s_buf + i, vs);

            auto va = aie::load_v<8>(A + h * D_STATE + i);
            aie::store_v(a_buf + i, va);

            auto vb = aie::load_v<8>(B + h * D_STATE + i);
            aie::store_v(b_buf + i, vb);

            auto vc = aie::load_v<8>(C + h * D_STATE + i);
            aie::store_v(c_buf + i, vc);

            auto vx = aie::load_v<8>(x + h * D_STATE + i);
            aie::store_v(x_buf + i, vx);
        }
        dt_buf[0] = dt[h];

        // ── State update ──
        // state_new = expm1(A) * state + dt * B * x
        //
        // Vectorized across 64 elements:
        //   s_new[i] = a[i] * s_old[i] + dt * b[i] * x[i]
        // where a[i] = expm1(A[i]) (precomputed on host)
        //
        // All operations are bf16 × bf16 → bf16
        alignas(32) bfloat16 s_new[D_STATE];
        for (int i = 0; i < D_STATE; i += 8) {
            auto vs = aie::load_v<8>(s_buf + i);
            auto va = aie::load_v<8>(a_buf + i);
            auto vb = aie::load_v<8>(b_buf + i);
            auto vx = aie::load_v<8>(x_buf + i);

            // state = A * state     (element-wise mul)
            auto vs_new = aie::mul(va, vs);
            // dt_B = dt * B         (scalar * vector)
            auto vdt_b = aie::mul(dt_buf[0], vb);
            // dt_B_x = dt_B * x
            auto vdt_b_x = aie::mul(vdt_b, vx);
            // state += dt * B * x
            vs_new = aie::add(vs_new, vdt_b_x);

            aie::store_v(s_new + i, vs_new);
        }

        // ── Output ──
        // y = C · state           (element-wise mul, not dot product)
        // y_output[h][d] = C[h][d] * state_new[h][d]
        alignas(32) bfloat16 y_buf[D_STATE];
        for (int i = 0; i < D_STATE; i += 8) {
            auto vs = aie::load_v<8>(s_new + i);
            auto vc = aie::load_v<8>(c_buf + i);
            auto vy = aie::mul(vc, vs);
            aie::store_v(y_buf + i, vy);

            aie::store_v(state_out + h * D_STATE + i, vs);
            aie::store_v(y + h * D_STATE + i, vy);
        }
    }

    event1();
}
}
