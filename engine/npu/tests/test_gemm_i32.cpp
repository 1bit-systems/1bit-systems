// test_gemm_i32.cpp — INT8 GEMM oracle, int32 output (matches engine Cm read).
// Usage: ./test_gemm_i32 <xclbin> <insts.txt> M K [N]
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

static void ref_gemm(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int32_t s = 0;
      for (int k = 0; k < K; k++) s += (int32_t)A[m*K+k] * (int32_t)B[k*N+n];
      C[m*N+n] = s;
    }
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "Usage: %s <xclbin> <insts.txt> M K [N]\n", argv[0]); return 1; }
  const char* xp = argv[1]; const char* ip = argv[2];
  int M = atoi(argv[3]), K = atoi(argv[4]), N = (argc>5)?atoi(argv[5]):K;
  FILE* f = fopen(ip, "rb");
  if (!f) { fprintf(stderr, "no insts %s\n", ip); return 1; }
  fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
  std::vector<uint32_t> ins(sz/4); fread(ins.data(), 4, ins.size(), f); fclose(f);

  xrt::device d(0);
  FILE* xf = fopen(xp, "rb");
  if (!xf) { fprintf(stderr, "no xclbin %s\n", xp); return 1; }
  fseek(xf, 0, 2); long xsz = ftell(xf); fseek(xf, 0, 0);
  std::vector<char> xbuf(xsz); fread(xbuf.data(), 1, xsz, xf); fclose(xf);
  xrt::xclbin xc{xbuf}; d.register_xclbin(xc);
  xrt::hw_context h(d, xc.get_uuid());
  xrt::kernel k(h, "MLIR_AIE");

  auto bI = xrt::bo(d, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
  auto bA = xrt::bo(d, (size_t)M*K, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
  auto bB = xrt::bo(d, (size_t)K*N, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
  auto bC = xrt::bo(d, (size_t)M*N*4, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
  memcpy(bI.map(), ins.data(), ins.size()*4); bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto* A = (int8_t*)bA.map(); for (int i=0;i<M*K;i++) A[i] = (int8_t)((i*7+13)%63-31);
  auto* B = (int8_t*)bB.map(); for (int i=0;i<K*N;i++) B[i] = (int8_t)((i*3+7)%63-31);
  bA.sync(XCL_BO_SYNC_BO_TO_DEVICE); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memset(bC.map(), 0, M*N*4); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto r = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC); r.wait();
  bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::vector<int32_t> ref(M*N);
  ref_gemm((int8_t*)bA.map(), (int8_t*)bB.map(), ref.data(), M, K, N);
  int32_t* C = (int32_t*)bC.map();
  long err = 0; int32_t mx = 0; long sum = 0;
  long cnt = std::min((long)10000, (long)M*N);
  printf("INT32 oracle: M=%d K=%d N=%d (checking %ld of %d)\n", M, K, N, cnt, M*N);
  printf("  [0..15] NPU: "); for (int i=0;i<16;i++) printf("%d ", C[i]); printf("\n");
  printf("  [0..15] REF: "); for (int i=0;i<16;i++) printf("%d ", ref[i]); printf("\n");
  for (long i=0;i<cnt;i++){ int32_t dd = std::abs((long)C[i]-(long)ref[i]); if (dd>1) err++; if (dd>mx) mx=dd; sum+=dd; }
  printf("  Errors(>1): %ld/%ld  maxdiff=%d  avgdiff=%.2f\n", err, cnt, mx, (double)sum/cnt);
  printf("%s\n", err==0 ? "PASS — xclbin GEMM correct (int32)" : "FAIL — xclbin GEMM wrong");
  return err ? 1 : 0;
}