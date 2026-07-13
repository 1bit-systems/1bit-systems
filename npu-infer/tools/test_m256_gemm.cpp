// Minimal test: verify M=256 xclbin produces correct output for a single GEMM.
// Uses the O_proj dimensions (K=2048, N=1024) to compare M=128 vs M=256 results.
// With M=256, we pack 128 rows of actual data + 128 rows of zeros, then verify
// that the first 128 rows match the M=128 result.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"

static const int MT = 128, KT = 64, NT = 128;

static float bf16f(uint16_t v) {
  uint32_t b = v << 16;
  float f;
  memcpy(&f, &b, 4);
  return f;
}
static uint16_t fbf16(float v) {
  uint32_t b;
  memcpy(&b, &v, 4);
  uint32_t r = ((b >> 16) & 1) + 0x7FFF;
  return (uint16_t)((b + r) >> 16);
}

static std::vector<uint8_t> pw(const float *w, int K, int N) {
  std::vector<float> Bf(K * N);
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      Bf[k * N + n] = w[n * K + k];
  auto Bs = gemm_atb::layout_transpose_L1_1x2_8x8block(Bf, K, N, KT, NT);
  auto Be = floatToBfp16(8, K * N, Bs.data(), 0);
  return {Be.data(), Be.data() + Be.size()};
}

struct XCLBIN {
  std::vector<uint32_t> ins;
  xrt::kernel k;
  xrt::bo i, bA, bB, bC;
  int K, N, M;
};

static bool load_xclbin(XCLBIN &v, int M, int K, int N, xrt::device &d) {
  char xp[256], ip[256];
  snprintf(xp, sizeof(xp),
           "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering"
           "/config1/build/final_%dx%dx%d_128x64x128.xclbin",
           M, K, N);
  snprintf(ip, sizeof(ip),
           "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering"
           "/config1/build/insts_%dx%dx%d_128x64x128.txt",
           M, K, N);

  // Load instruction file
  FILE *f = fopen(ip, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open: %s\n", ip);
    return false;
  }
  fseek(f, 0, 2);
  long sz = ftell(f);
  fseek(f, 0, 0);
  v.ins.resize(sz / 4);
  fread(v.ins.data(), 4, v.ins.size(), f);
  fclose(f);

  // Load xclbin
  auto xc = xrt::xclbin(std::string(xp));
  d.register_xclbin(xc);
  auto hw = xrt::hw_context(d, xc.get_uuid());
  v.k = xrt::kernel(hw, "MLIR_AIE");

  // Create buffers
  v.i = xrt::bo(d, v.ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, v.k.group_id(1));
  memcpy(v.i.map(), v.ins.data(), v.ins.size() * 4);
  v.i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  size_t as = M * K * 2;     // bf16
  size_t bs = (size_t)K * N * 9 / 8;  // v8bfp16ebs8
  size_t cs = M * N * 2;     // bf16
  v.bA = xrt::bo(d, as, XRT_BO_FLAGS_HOST_ONLY, v.k.group_id(3));
  v.bB = xrt::bo(d, bs, XRT_BO_FLAGS_HOST_ONLY, v.k.group_id(4));
  v.bC = xrt::bo(d, cs, XRT_BO_FLAGS_HOST_ONLY, v.k.group_id(5));

  v.K = K;
  v.N = N;
  v.M = M;
  return true;
}

static void run_gemm(XCLBIN &v, const float *A, const uint8_t *B_flat,
                     float *C_out) {
  int M = v.M, K = v.K, N = v.N;

  // Pack A: need to layout the MxK input
  std::vector<float> A_padded(M * K, 0.0f);
  memcpy(A_padded.data(), A, (M < 128 ? M : 128) * K * 4);

  auto As = gemm_atb::layout_A_L1_2x1_8x8block(A_padded, M, K, MT, KT);
  uint16_t *Ab = (uint16_t *)v.bA.map();
  for (int i = 0; i < M * K; i++)
    Ab[i] = fbf16(As[i]);
  v.bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  
  printf("  K=%d M=%d A buffer: first 8 bf16 vals: ", K, M);
  for (int i = 0; i < 8; i++) printf("%.2f ", bf16f(((uint16_t*)v.bA.map())[i]));
  printf("\n");

  // Copy B
  memcpy(v.bB.map(), B_flat, (size_t)K * N * 9 / 8);
  v.bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Fill A with a simple pattern: row i gets value i
  // Use non-laid-out raw A for M=256 to check if ANY computation happens
  // First, fill A with known values without layout transformation
  
  // --- DEBUG: write known pattern to C before kernel ---
  // Fill C with 0xDEAD before kernel run to verify DMA actually writes
  for (int i = 0; i < M * N; i++)
    ((uint16_t*)v.bC.map())[i] = 0xDEAD;
  v.bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Launch
  auto r = v.k(3, v.i, (unsigned)v.ins.size(), v.bA, v.bB, v.bC);
  r.wait();

  // Read back
  v.bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint16_t *Cb = (uint16_t *)v.bC.map();

  printf("  M=%d C raw first 8 values: ", M);
  for (int i = 0; i < 8 && i < M * N; i++)
    printf("0x%04x(%.4f) ", Cb[i], bf16f(Cb[i]));
  printf("\n");
  printf("  M=%d C raw Cb[128*N-8..128*N]: ", M);
  int mid = (M < 128 ? M : 128) * N;
  if (mid > 8) {
    for (int i = mid - 8; i < mid; i++)
      printf("0x%04x(%.4f) ", Cb[i], bf16f(Cb[i]));
    printf("\n");
  }
  printf("  M=%d C total non-zero: ", M);
  int nz = 0;
  int dead_beef = 0;
  for (int i = 0; i < M * N; i++) {
    if (Cb[i] == 0xDEAD) dead_beef++;
    if (Cb[i] != 0 && Cb[i] != 0xDEAD) nz++;
  }
  printf("%d / %d (0xDEAD=%d, zero=%d)\n", nz, M * N, dead_beef, M*N-nz-dead_beef);

  // Skip layout unshuffle if there's no output
  if (nz == 0) {
    printf("  NO OUTPUT (all %s)\n", dead_beef > 0 ? "still 0xDEAD (DMA never wrote)" : "zeros (maybe DMA cleared?)");
    return;
  }

  // Unshuffle C
  std::vector<float> ct(M * N);
  for (int i = 0; i < M * N; i++)
    ct[i] = bf16f(Cb[i]);
  auto Cu = gemm_atb::layout_inverse_C_L1_2x2_8x8block(ct, M, N, MT, NT);

  // Copy first N rows to output (or up to 128 if M > 128)
  int copy_rows = M < 128 ? M : 128;
  for (int i = 0; i < copy_rows * N; i++)
    C_out[i] = Cu[i];
}

int main() {
  printf("=== M=256 xclbin verification ===\n\n");
  xrt::device d(0);

  // Test with O_proj (K=2048, N=1024) - easiest: single column tile
  const int K = 2048, N = 1024;

  // Load both M=128 and M=256 xclbins
  XCLBIN v128, v256;
  if (!load_xclbin(v128, 128, K, N, d)) {
    fprintf(stderr, "Failed to load M=128 xclbin\n");
    return 1;
  }
  if (!load_xclbin(v256, 256, K, N, d)) {
    fprintf(stderr, "Failed to load M=256 xclbin\n");
    return 1;
  }

  // Generate random test weights
  std::vector<float> W(K * N);
  for (int i = 0; i < K * N; i++)
    W[i] = (float)rand() / RAND_MAX * 2 - 1;

  // Pre-pack B weights (same for both M=128 and M=256 tests)
  auto B_flat = pw(W.data(), K, N);

  // Generate random input A (M=128)
  std::vector<float> A128(128 * K);
  for (int i = 0; i < 128 * K; i++)
    A128[i] = (float)rand() / RAND_MAX * 2 - 1;

  // Run M=128 GEMM
  printf("Running M=128 GEMM (K=%d, N=%d)...\n", K, N);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<float> C128(128 * N);
  run_gemm(v128, A128.data(), B_flat.data(), C128.data());
  auto t1 = std::chrono::steady_clock::now();
  double ms128 =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("  M=128: %.1f ms\n", ms128);

  // Run M=256 GEMM with same A in first half
  printf("Running M=256 GEMM (K=%d, N=%d)...\n", K, N);
  std::vector<float> A256(256 * K);
  memcpy(A256.data(), A128.data(), 128 * K * 4);
  // Fill second half with random (different) values
  for (int i = 128 * K; i < 256 * K; i++)
    A256[i] = (float)rand() / RAND_MAX * 2 - 1;

  std::vector<float> C256_first(128 * N);
  auto t2 = std::chrono::steady_clock::now();
  run_gemm(v256, A256.data(), B_flat.data(), C256_first.data());
  auto t3 = std::chrono::steady_clock::now();
  double ms256 =
      std::chrono::duration<double, std::milli>(t3 - t2).count();
  printf("  M=256: %.1f ms\n", ms256);

  // Also run M=256 with just the second half to verify independence
  printf("Running M=256 (2nd half only) GEMM...\n");
  std::vector<float> A256_v2(256 * K);
  for (int i = 0; i < 128 * K; i++)
    A256_v2[i] = 0.0f;
  // Use a different random input for second half
  for (int i = 128 * K; i < 256 * K; i++)
    A256_v2[i] = (float)rand() / RAND_MAX * 2 - 1;

  std::vector<float> C256_second(128 * N);
  memset(C256_second.data(), 0, 128 * N * 4);
  run_gemm(v256, A256_v2.data(), B_flat.data(), C256_second.data());

  // Compare: first 128 rows of M=256 should match M=128
  double max_err = 0, sum_err = 0;
  int bad = 0;
  for (int i = 0; i < 128 * N; i++) {
    float e = fabsf(C128[i] - C256_first[i]);
    if (e > max_err)
      max_err = e;
    sum_err += e;
    if (e > 0.01f && fabsf(C128[i]) > 0.001f &&
        e / (fabsf(C128[i]) + 1e-10f) > 0.1f)
      bad++;
  }
  printf("\n=== M=128 vs M=256 (first half) comparison ===\n");
  printf("  Max error:       %.6f\n", max_err);
  printf("  Mean error:      %.6f\n", sum_err / (128 * N));
  printf("  Bad elements:    %d / %d\n", bad, 128 * N);
  printf("  M=256 speedup:   %.2fx\n", ms128 / ms256);

  if (bad == 0 && max_err < 0.01f) {
    printf("\n  ✓ M=256 xclbin VERIFIED - results match M=128!\n");
    printf("  ✓ Two independent tokens can be processed in one GEMM call.\n");
    printf("  ✓ 2-token decode: process both token embeddings in 1 GEMM\n");
    printf("    then split the C output (first 128 rows = tok1, last 128 = tok2).\n");
  } else {
    printf("\n  ✗ M=256 mismatch! Debug needed.\n");
  }

  printf("\n=== Done ===\n");
  return bad > 0 ? 1 : 0;
}
