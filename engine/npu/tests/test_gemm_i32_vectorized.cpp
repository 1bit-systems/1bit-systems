// test_gemm_i32_vectorized.cpp — Test vectorized matmul_i8_i32 with microtile pre-shuffle.
// Builds on test_gemm_i32.cpp but adds data layout transformation so the vectorized
// 8x8x8 microtile kernel receives correctly ordered data.
//
// The vectorized matmul_i8_i32 kernel expects each (m,k) tile of A and each (k,n) tile
// of B to be arranged in 8x8 microtile order: all (m/8 * k/8) microtiles of 8x8 = 64 bytes
// each, stored row-major by microtile-row then microtile-col.
//
// Usage: ./test_gemm_i32_vectorized <xclbin> <insts.txt> M K N [--no-shuffle]
//   --no-shuffle: run without microtile pre-shuffle (tests whether kernel actually needs it)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// Reference GEMM on flat row-major data
static void ref_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int32_t s = 0;
      for (int k = 0; k < K; k++) s += (int32_t)A[m*K+k] * (int32_t)B[k*N+n];
      C[m*N+n] = s;
    }
}

// Shuffle flat row-major (M,K) matrix into 8x8-microtile order.
// The vectorized matmul_i8_i32 kernel with r=s=t=8 and microtile counts
// rowA=M/8, colA=K/8 expects data: microtile[mt_row][mt_col][wr][wc] laid out as
//   offset = (mt_row * colA + mt_col) * 64 + wr * 8 + wc
// where mt_row=row/8, mt_col=col/8, wr=row%8, wc=col%8.
static void shuffle_to_microtile(const int8_t* flat, int8_t* micro, int rows, int cols) {
  int mt_cols = cols / 8;  // number of microtiles per row-group
  for (int r = 0; r < rows; r++) {
    int mt_r = r / 8;
    int wr = r % 8;
    for (int c = 0; c < cols; c++) {
      int mt_c = c / 8;
      int wc = c % 8;
      int flat_pos = r * cols + c;
      int micro_pos = (mt_r * mt_cols + mt_c) * 64 + wr * 8 + wc;
      micro[micro_pos] = flat[flat_pos];
    }
  }
}

// Reverse: unshuffle microtile-order C matrix back to flat row-major
static void unshuffle_from_microtile(const int32_t* micro, int32_t* flat, int rows, int cols) {
  int mt_cols = cols / 8;
  for (int r = 0; r < rows; r++) {
    int mt_r = r / 8;
    int wr = r % 8;
    for (int c = 0; c < cols; c++) {
      int mt_c = c / 8;
      int wc = c % 8;
      int micro_pos = (mt_r * mt_cols + mt_c) * 64 + wr * 8 + wc;
      int flat_pos = r * cols + c;
      flat[flat_pos] = micro[micro_pos];
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "Usage: %s <xclbin> <insts.txt> M K N [--no-shuffle]\n", argv[0]); return 1; }
  const char* xp = argv[1]; const char* ip = argv[2];
  int M = atoi(argv[3]), K = atoi(argv[4]), N = atoi(argv[5]);
  bool do_shuffle = true;
  if (argc > 6 && strcmp(argv[6], "--no-shuffle") == 0) do_shuffle = false;

  printf("INT8 GEMM vectorized test: M=%d K=%d N=%d shuffle=%s\n",
         M, K, N, do_shuffle ? "yes (microtile order)" : "no (flat row-major)");

  // Load instructions
  FILE* f = fopen(ip, "rb");
  if (!f) { fprintf(stderr, "no insts %s\n", ip); return 1; }
  fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
  std::vector<uint32_t> ins(sz/4); fread(ins.data(), 4, ins.size(), f); fclose(f);

  // Open device and load xclbin
  xrt::device d(0);
  FILE* xf = fopen(xp, "rb");
  if (!xf) { fprintf(stderr, "no xclbin %s\n", xp); return 1; }
  fseek(xf, 0, 2); long xsz = ftell(xf); fseek(xf, 0, 0);
  std::vector<char> xbuf(xsz); fread(xbuf.data(), 1, xsz, xf); fclose(xf);
  xrt::xclbin xc{xbuf}; d.register_xclbin(xc);
  xrt::hw_context h(d, xc.get_uuid());
  xrt::kernel k(h, "MLIR_AIE");

  // Allocate buffers
  auto bI = xrt::bo(d, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
  auto bA = xrt::bo(d, (size_t)M*K, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  auto bB = xrt::bo(d, (size_t)K*N, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bC = xrt::bo(d, (size_t)M*N*4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));

  // Copy instructions
  memcpy(bI.map(), ins.data(), ins.size()*4);
  bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Generate deterministic random-ish data
  int8_t* A_flat = (int8_t*)malloc(M*K);
  int8_t* B_flat = (int8_t*)malloc(K*N);
  for (int i=0;i<M*K;i++) A_flat[i] = (int8_t)((i*7+13)%63-31);
  for (int i=0;i<K*N;i++) B_flat[i] = (int8_t)((i*3+7)%63-31);

  // Compute reference on flat data
  std::vector<int32_t> ref(M*N);
  ref_gemm(A_flat, B_flat, ref.data(), M, K, N);

  // Prepare NPU data: either shuffle to microtile order or keep flat
  int8_t* A_npu = (int8_t*)bA.map();
  int8_t* B_npu = (int8_t*)bB.map();
  if (do_shuffle) {
    // Shuffle A (M×K) into microtile order
    shuffle_to_microtile(A_flat, A_npu, M, K);
    // Shuffle B (K×N) into microtile order
    shuffle_to_microtile(B_flat, B_npu, K, N);
  } else {
    memcpy(A_npu, A_flat, M*K);
    memcpy(B_npu, B_flat, K*N);
  }

  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bC.map(), 0, M*N*4);
  bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Launch NPU kernel (opcode 3 = GEMM)
  auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
  r.wait();
  bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Read NPU output
  std::vector<int32_t> C_npu_flat(M*N);
  int32_t* C_raw = (int32_t*)bC.map();
  if (do_shuffle) {
    unshuffle_from_microtile(C_raw, C_npu_flat.data(), M, N);
  } else {
    memcpy(C_npu_flat.data(), C_raw, M*N*4);
  }

  // Compare with reference
  long err = 0; int32_t mx = 0; long sum = 0;
  long cnt = std::min((long)10000, (long)M*N);
  printf("  [0..15] NPU: "); for (int i=0;i<16;i++) printf("%d ", C_npu_flat[i]); printf("\n");
  printf("  [0..15] REF: "); for (int i=0;i<16;i++) printf("%d ", ref[i]); printf("\n");
  for (long i=0;i<cnt;i++){ int32_t dd = std::abs((long)C_npu_flat[i]-(long)ref[i]); if (dd>1) err++; if (dd>mx) mx=dd; sum+=dd; }
  printf("  Errors(>1): %ld/%ld  maxdiff=%d  avgdiff=%.2f\n", err, cnt, mx, (double)sum/cnt);
  printf("%s\n", err==0 ? "PASS — xclbin GEMM correct (int32, vectorized layout)" : 
         (do_shuffle ? "FAIL" : "FAIL (try with --no-shuffle or with shuffle removed)"));

  free(A_flat); free(B_flat);
  return err ? 1 : 0;
}
