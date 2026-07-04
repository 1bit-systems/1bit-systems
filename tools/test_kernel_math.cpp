// Standalone x86 test of INT8 GEMM kernel math
// Implements the scalar matmul logic without AIE headers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>

// Pure scalar INT8 GEMM (same logic as the Chess kernel's matmul_scalar)
// C[m][n] += sum_k A[m][k] * B[k][n]
template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar_i8_i32(T_in *a, T_in *b, T_out *c) {
    for (int row = 0; row < rowA; row++) {
        for (int col = 0; col < colB; col++) {
            T_out running_sum = 0;
            for (int i = 0; i < colA; i++) {
                T_in a_val = a[row * colA + i];
                T_in b_val = b[i * colB + col];
                running_sum += (T_out)a_val * (T_out)b_val;
            }
            c[row * colB + col] += running_sum;
        }
    }
}

// Simulate the NPU multi-tile GEMM for a full matrix multiply
// This mimics what the xclbin + ObjectFifo does with tiling
static void npu_simulated_gemm(const int8_t* A, const int8_t* B, int32_t* C,
                                int M, int K, int N,
                                int mt, int kt, int nt) {
    memset(C, 0, M * N * sizeof(int32_t));
    
    // Tile dimensions
    for (int mg = 0; mg < M / mt; mg++) {
        for (int ng = 0; ng < N / nt; ng++) {
            // Each tile processes B[kt, nt] for all K-tiles
            int32_t tile_C[mt * nt] = {0};
            for (int kg = 0; kg < K / kt; kg++) {
                // Load B tile from the flat layout
                // In the NPU, B is stored as: per N-group, per K-group, per row, per col
                // So B_flat[ng * (K/kt) * kt * nt + kg * kt * nt + i * nt + j] = B[kg*kt+i][ng*nt+j]
                
                for (int i = 0; i < mt; i++) {
                    for (int j = 0; j < nt; j++) {
                        int32_t sum = 0;
                        int k_base = kg * kt;
                        for (int kk = 0; kk < kt; kk++) {
                            int a_idx = (mg * mt + i) * K + (k_base + kk);
                            int b_flat_idx = ng * (K/kt) * kt * nt + kg * kt * nt + kk * nt + j;
                            sum += (int32_t)A[a_idx] * (int32_t)B[b_flat_idx];
                        }
                        tile_C[i * nt + j] += sum;
                    }
                }
            }
            // Write tile result to output
            for (int i = 0; i < mt; i++) {
                for (int j = 0; j < nt; j++) {
                    C[(mg * mt + i) * N + (ng * nt + j)] = tile_C[i * nt + j];
                }
            }
        }
    }
}

// Standard reference GEMM (for verification)
static void ref_gemm(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int K, int N) {
    memset(C, 0, M * N * sizeof(int32_t));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                C[m * N + n] += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
}

int main() {
    printf("=== INT8 GEMM Kernel Math Verification ===\n\n");
    
    const int M = 32, K = 1536, N = 1536;
    const int mt = 32, kt = 64, nt = 192;
    
    // Generate random test data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B_flat(K * N);  // NPU flat layout
    std::vector<int8_t> B_ref(K * N);   // standard [K,N] layout
    
    for (auto& v : A) v = (int8_t)dist(rng);
    for (auto& v : B_ref) v = (int8_t)dist(rng);
    
    // Convert B_ref to NPU flat layout
    // Layout: per N-group (8 cols * nt), per K-group (K/kt), per kt-row, per nt-col
    int n_groups = N / (8 * nt);  // = 1536 / 1536 = 1
    int k_tiles = K / kt;          // = 1536 / 64 = 24
    
    for (int ng = 0; ng < n_groups; ng++) {
        for (int c = 0; c < 8; c++) {  // 8 NPU columns
            for (int kg = 0; kg < k_tiles; kg++) {
                for (int i = 0; i < kt; i++) {
                    for (int j = 0; j < nt; j++) {
                        int out_idx = ng * 8 * nt + c * nt + j;
                        int in_idx = kg * kt + i;
                        // B_ref[k][n] layout: B_ref[in_idx * N + out_idx]
                        int b_ref_pos = in_idx * N + out_idx;
                        // NPU flat layout
                        int n_tiles_per_group = 8;
                        int b_flat_pos = ng * n_tiles_per_group * k_tiles * kt * nt
                                       + c * k_tiles * kt * nt
                                       + kg * kt * nt
                                       + i * nt + j;
                        B_flat[b_flat_pos] = B_ref[b_ref_pos];
                    }
                }
            }
        }
    }
    
    // Test 1: NPU-simulated GEMM vs reference
    printf("Test 1: NPU tile-simulated GEMM vs standard reference\n");
    std::vector<int32_t> C_npu(M * N), C_ref(M * N);
    
    npu_simulated_gemm(A.data(), B_flat.data(), C_npu.data(), M, K, N, mt, kt, nt);
    ref_gemm(A.data(), B_ref.data(), C_ref.data(), M, K, N);
    
    double max_err = 0, sum_err = 0;
    for (int i = 0; i < M * N; i++) {
        double err = std::abs((double)C_npu[i] - (double)C_ref[i]);
        sum_err += err;
        if (err > max_err) max_err = err;
    }
    printf("  Mean err: %.2f  Max err: %.0f\n", sum_err / (M*N), max_err);
    printf("  First 5 NPU: ");
    for (int i = 0; i < 5; i++) printf("%d ", C_npu[i]);
    printf("\n  First 5 REF: ");
    for (int i = 0; i < 5; i++) printf("%d ", C_ref[i]);
    printf("\n");
    printf("  %s\n\n", max_err < 1e-9 ? "PASS - exact match" : "FAIL - mismatch");
    
    // Test 2: A_q=1, B=all ones, using NPU tiled layout
    printf("Test 2: NPU tile-simulated GEMM with A=1, B=1 (all ones)\n");
    std::vector<int8_t> A_ones(M * K, 1);
    std::vector<int8_t> B_flat_ones(K * N, 1);  // In NPU layout, B_flat=1 everywhere
    std::vector<int32_t> C_ones(M * N);
    npu_simulated_gemm(A_ones.data(), B_flat_ones.data(), C_ones.data(), M, K, N, mt, kt, nt);
    printf("  C[0]=%d (expect %d): %s\n", C_ones[0], K, C_ones[0] == K ? "PASS" : "FAIL");
    
    // Test 3: INT32 overflow test with extreme values
    printf("\nTest 3: INT32 overflow test\n");
    std::vector<int8_t> A_ext(M * K, 127);
    std::vector<int8_t> B_ext(K * N, -128);
    std::vector<int32_t> C_ext(M * N);
    ref_gemm(A_ext.data(), B_ext.data(), C_ext.data(), M, K, N);
    
    // Expected: C[0] = 127 * (-128) * K = 127 * (-128) * 1536 = -24,969,216
    int64_t expected = (int64_t)127 * (int64_t)(-128) * (int64_t)K;
    printf("  C[0]=%d (expect %ld): %s\n", C_ext[0], (long)expected, 
           C_ext[0] == expected ? "PASS" : "FAIL");
    printf("  INT16 would truncate to: %d (WRONG)\n", (int16_t)C_ext[0]);
    
    printf("\nDone.\n");
    return 0;
}
