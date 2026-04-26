// test_xdna_kernel.cpp — smoke test for rocm_cpp::XdnaKernel.
//
// Loads the stock mlir-aie 512×512×512 / 64×64×64 i8×i8→i32 matmul xclbin and
// verifies one run against a CPU reference. Bit-exact comparison: i8×i8→i32 is
// integer arithmetic, no FP roundoff.
//
// The xclbin path is hard-coded to where it lives in the user's tree right
// now. If it's missing we exit 0 with a "// SKIPPED" log so CI doesn't choke
// on a clean clone — this matches the spec's deliverable-A blocker policy.

#include "rocm_cpp/xdna_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kM = 512;
constexpr int kK = 512;
constexpr int kN = 512;

constexpr const char* kXclbinPath =
    "/home/bcloud/repos/mlir-aie/programming_examples/basic/"
    "matrix_multiplication/single_core/build/"
    "final_512x512x512_64x64x64.xclbin";

constexpr const char* kKernelName = "MLIR_AIE";

void cpu_matmul_i8_i32(const int8_t* A, const int8_t* B, int32_t* C,
                       int M, int K, int N) {
    // Plain row-major reference. M*K*N == 134M for 512^3 — under a second on
    // x86_64 Release.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[i * K + k]) *
                       static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = acc;
        }
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    if (!fs::exists(kXclbinPath)) {
        std::cout << "// SKIPPED test_xdna_kernel — xclbin not present at "
                  << kXclbinPath
                  << "\n// Build it via:\n"
                  << "//   cd /home/bcloud/repos/mlir-aie/programming_examples/basic/"
                     "matrix_multiplication/single_core\n"
                  << "//   env M=512 K=512 N=512 m=64 k=64 n=64 dtype_in=i8 "
                     "dtype_out=i32 make devicename=npu2\n";
        return 0;
    }

    try {
        rocm_cpp::XdnaKernel kernel(kXclbinPath, kKernelName);
        if (!kernel.is_loaded()) {
            std::cerr << "FAIL: kernel not loaded after construction\n";
            return 1;
        }
        std::cout << "Loaded xclbin. Kernel name: " << kernel.kernel_name() << "\n";

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-8, 7);

        std::vector<int8_t> A(kM * kK), B(kK * kN);
        for (auto& x : A) x = static_cast<int8_t>(dist(rng));
        for (auto& x : B) x = static_cast<int8_t>(dist(rng));

        std::vector<int32_t> C_npu(kM * kN, 0);
        std::vector<int32_t> C_ref(kM * kN, 0);

        kernel.matmul_i8_i32(A.data(), B.data(), C_npu.data(), kM, kK, kN);
        cpu_matmul_i8_i32(A.data(), B.data(), C_ref.data(), kM, kK, kN);

        int64_t max_abs = 0;
        long long mismatches = 0;
        for (size_t i = 0; i < C_npu.size(); ++i) {
            int64_t d = std::abs(static_cast<int64_t>(C_npu[i]) -
                                  static_cast<int64_t>(C_ref[i]));
            if (d != 0) ++mismatches;
            if (d > max_abs) max_abs = d;
        }

        std::cout << "max |C_npu - C_ref| = " << max_abs << "\n";
        std::cout << "mismatches          = " << mismatches << " / " << C_npu.size() << "\n";
        if (max_abs == 0) {
            std::cout << "PASS\n";
            return 0;
        }
        std::cout << "FAIL\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL with exception: " << e.what() << "\n";
        return 2;
    }
}
