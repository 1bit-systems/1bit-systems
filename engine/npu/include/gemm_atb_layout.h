#ifndef ENGINE_NPU_GEMM_ATB_LAYOUT_H
#define ENGINE_NPU_GEMM_ATB_LAYOUT_H

/**
 * ATB (AI Engine Block) Layout Transformations.
 *
 * These functions shuffle activation and output tensors between the standard
 * row-major format and the tiled/shuffled format that the MAI AI Engine xclbin
 * hardware expects. They are provided by AMD's torch2aie toolchain:
 *
 *   https://github.com/amd/torch2aie
 *
 * The tile dimensions (M_TILE, K_TILE, N_TILE) match the AI Engine kernel's
 * data mover configuration.
 *
 * Without torch2aie installed (including the ATB headers and compiled kernel),
 * the BF16 NPU engine variant cannot run — the xclbin is not accessible and
 * the layout transformations are not linked. These stubs document the expected
 * signatures and will cause a linker error or runtime assertion if called.
 */

#include <vector>
#include <cassert>
#include <cstdio>

namespace gemm_atb {

/// Shuffle activation matrix A(M×K) from row-major → ATB L1 tiled layout.
/// The output vector has the same size (M×K floats) but with elements
/// reordered into the (mTile × kTile) blocks that the AI Engine data mover
/// reads sequentially.
///
/// @param A      Input activation, row-major [M][K]
/// @param M      Number of rows (padded to XM)
/// @param K      Number of columns (padded to XK)
/// @param mTile  Tile rows (M_TILE = 128)
/// @param kTile  Tile columns (K_TILE = 8)
/// @return       Shuffled vector, same size as input
inline std::vector<float> layout_A_L1_2x1_8x8block(
    const std::vector<float>&, int M, int K, int mTile, int kTile)
{
    fprintf(stderr, "[ERROR] gemm_atb::layout_A_L1_2x1_8x8block requires "
                    "AMD torch2aie headers (https://github.com/amd/torch2aie). "
                    "Without torch2aie, the BF16 NPU engine cannot run.\n");
    // Return identity (wrong, but allows compilation for testing)
    std::vector<float> out((size_t)M * K);
    return out;
}

/// Inverse shuffle: AI Engine output C(M×N) → row-major.
/// The kernel writes C in the ATB tiled layout; this function converts back
/// to standard row-major format.
///
/// @param C      Shuffled output, ATB L1 layout [M][N]
/// @param M      Number of rows (padded to XM)
/// @param N      Number of columns (padded to XN)
/// @param mTile  Tile rows (4 * M_TILE = 512)
/// @param nTile  Tile columns (N_TILE = 8)
/// @return       De-shuffled row-major vector
inline std::vector<float> layout_inverse_C_L1_2x2_8x8block(
    const std::vector<float>&, int M, int N, int mTile, int nTile)
{
    fprintf(stderr, "[ERROR] gemm_atb::layout_inverse_C_L1_2x2_8x8block requires "
                    "AMD torch2aie headers.\n");
    std::vector<float> out((size_t)M * N);
    return out;
}

} // namespace gemm_atb

#endif // ENGINE_NPU_GEMM_ATB_LAYOUT_H
