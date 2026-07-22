// INT8 GEMM dynamic test harness — calls the generated TXN builder at runtime.
// One xclbin serves any M/N at tile granularity (m=32, n=128).
//
// Build:
//   source toolchain-env.sh
//   g++ -std=c++23 -I../build/int8 -I$(airenv)/include -o test_dynamic \
//     test_dynamic_int8.cpp -lxrt_core -lxrt_coreutil -lpthread
//
// Run:
//   ./test_dynamic <M> <K> <N>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include GEN_HDR

#ifndef XCLBIN
#define XCLBIN "dynamic.xclbin"
#endif

#ifndef KERNEL_NAME
#define KERNEL_NAME "MLIR_AIE"
#endif

using A_DTYPE = int8_t;
using B_DTYPE = int8_t;
using C_DTYPE = int32_t;

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " M K N\n";
    return 1;
  }
  int M = std::atoi(argv[1]);
  int K = std::atoi(argv[2]);
  int N = std::atoi(argv[3]);

  constexpr int m = 32, k = 64, n = 128;
  if (M % m != 0 || N % n != 0) {
    std::cerr << "M must be multiple of " << m << ", N multiple of " << n << "\n";
    return 1;
  }

  // Build instruction stream at runtime using generated TXN builder
  std::optional<std::vector<uint32_t>> instr_opt =
      generate_txn_main_main(M, K, N);
  if (!instr_opt) {
    std::cerr << "TXN builder failed (pool exhausted) for M=" << M
              << " K=" << K << " N=" << N << "\n";
    return 1;
  }
  std::vector<uint32_t> instr_v = std::move(*instr_opt);
  std::cout << "Generated " << instr_v.size() << " instructions for "
            << "M=" << M << " K=" << K << " N=" << N << "\n";

  // Open XRT device
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);
  std::string xclbin_path = XCLBIN;
  xrt::xclbin xclbin = xrt::xclbin(xclbin_path);

  // Find kernel
  auto xkernels = xclbin.get_kernels();
  auto xkernel = std::find_if(xkernels.begin(), xkernels.end(),
                              [](const xrt::xclbin::kernel &k) {
                                return k.get_name().rfind(KERNEL_NAME, 0) == 0;
                              });
  if (xkernel == xkernels.end()) {
    std::cerr << "No kernel matching '" << KERNEL_NAME << "'\n";
    return 1;
  }
  std::string kernel_name = xkernel->get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernel_name);

  // Allocate buffers
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, M * K * sizeof(A_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
  auto bo_b = xrt::bo(device, K * N * sizeof(B_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4));
  auto bo_c = xrt::bo(device, M * N * sizeof(C_DTYPE), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(5));

  // Fill inputs with random data
  std::vector<A_DTYPE> A_vec(M * K);
  std::vector<B_DTYPE> B_vec(K * N);
  srand(42);
  for (auto &v : A_vec) v = (rand() % 256) - 128;
  for (auto &v : B_vec) v = (rand() % 256) - 128;

  std::memcpy(bo_a.map<A_DTYPE *>(), A_vec.data(), M * K * sizeof(A_DTYPE));
  std::memcpy(bo_b.map<B_DTYPE *>(), B_vec.data(), K * N * sizeof(B_DTYPE));
  std::memset(bo_c.map<C_DTYPE *>(), 0, M * N * sizeof(C_DTYPE));

  std::memcpy(bo_instr.map<void *>(), instr_v.data(),
              instr_v.size() * sizeof(uint32_t));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Run kernel
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cerr << "Kernel failed: " << r << "\n";
    return 1;
  }

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  C_DTYPE *C_out = bo_c.map<C_DTYPE *>();

  // Verify against reference
  int64_t n_errors = 0;
  for (int i = 0; i < M && n_errors < 10; i++) {
    for (int j = 0; j < N && n_errors < 10; j++) {
      int64_t expected = 0;
      for (int kk = 0; kk < K; kk++)
        expected += (int64_t)A_vec[i * K + kk] * (int64_t)B_vec[kk * N + j];
      int64_t actual = C_out[i * N + j];
      if (expected != actual) {
        std::cerr << "ERR [" << i << "," << j << "] exp=" << expected
                  << " got=" << actual << "\n";
        n_errors++;
      }
    }
  }
  if (n_errors == 0)
    std::cout << "PASS!\n";
  else
    std::cout << "FAIL (" << n_errors << " errors)\n";

  return n_errors == 0 ? 0 : 1;
}
