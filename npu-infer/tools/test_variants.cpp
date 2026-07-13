/**
 * Test all 5 variants with simple deterministic data to detect any format issues.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "helper.h"
#include "gemm_atb_layout.h"
static float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static uint16_t fbf16(float v){uint32_t b;memcpy(&b,&v,4);uint32_t r=((b>>16)&1)+0x7FFF;return(uint16_t)((b+r)>>16);}

struct Variant {
    const char *path;
    const char *inspath;
    int XK, XN;
};
static const Variant VARIANTS[] = {
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x1024_128x64x128.xclbin",
     "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x1024_128x64x128.txt", 1024, 1024},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x2048_128x64x128.xclbin",
     "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x2048_128x64x128.txt", 1024, 2048},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x3072_128x64x128.xclbin",
     "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x3072_128x64x128.txt", 1024, 3072},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x2048x1024_128x64x128.xclbin",
     "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x2048x1024_128x64x128.txt", 2048, 1024},
    {"/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x3072x1024_128x64x128.xclbin",
     "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x3072x1024_128x64x128.txt", 3072, 1024},
};
static const int NV = 5;
static const int MT = 128, KT = 64, NT = 128, M_PAD = 128;

int test_variant(int vi, int M, int K, int N) {
    auto &va = VARIANTS[vi];
    printf("Testing variant %d (%dx%d) with M=%d K=%d N=%d ... ", vi, va.XK, va.XN, M, K, N);
    fflush(stdout);

    FILE *f = fopen(va.inspath, "rb"); if (!f) return 1;
    fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
    std::vector<uint32_t> ins(sz/4); fread(ins.data(), 4, ins.size(), f); fclose(f);

    auto d = xrt::device(0);
    auto xc = xrt::xclbin(std::string(va.path)); d.register_xclbin(xc);
    auto hw = xrt::hw_context(d, xc.get_uuid());
    auto k = xrt::kernel(hw, "MLIR_AIE");

    auto bi = xrt::bo(d, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
    auto bA = xrt::bo(d, (size_t)M_PAD*va.XK*2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
    auto bB = xrt::bo(d, (size_t)va.XK*va.XN*9/8, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
    auto bC = xrt::bo(d, (size_t)M_PAD*va.XN*2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));

    memcpy(bi.map(), ins.data(), ins.size()*4); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // A = 1.0 (uniform)
    std::vector<float> A(M * K, 1.0f);

    // Pad A
    std::vector<float> Af(M_PAD * va.XK);
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) Af[m * va.XK + k] = A[m * K + k];
    auto As = gemm_atb::layout_A_L1_2x1_8x8block(Af, M_PAD, va.XK, MT, KT);
    uint16_t *Ab = (uint16_t*)bA.map();
    for (int i = 0; i < M_PAD * va.XK; i++) Ab[i] = fbf16(As[i]);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // B = 3.0 (uniform)
    std::vector<float> Bf(va.XK * va.XN, 3.0f);
    auto Bs = gemm_atb::layout_transpose_L1_1x2_8x8block(Bf, va.XK, va.XN, KT, NT);
    auto Bp = floatToBfp16(8, va.XK * va.XN, Bs.data(), 0);
    memcpy(bB.map(), Bp.data(), Bp.size()); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    memset(bC.map(), 0, (size_t)M_PAD*va.XN*2); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = k((unsigned)3, bi, (unsigned)ins.size(), bA, bB, bC); run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    uint16_t *Cb = (uint16_t*)bC.map();
    float expected = 3.0f * K;
    int err = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float c = bf16f(Cb[m * va.XN + n]);
            if (fabsf(c - expected) > 2.0f) { err++; }
        }
    }
    if (err == 0) printf("PASS (%.0f)\n", expected);
    else printf("FAIL: %d errors (expected %.0f, first few: ", err, expected);
    if (err > 0) { for (int i = 0; i < 5 && i < M_PAD * va.XN; i++) printf("%.0f ", bf16f(Cb[i])); printf(")"); }
    if (err > 0) printf("\n");
    return err;
}

int main() {
    // Test all 5 variants with simple uniform data
    int total = 0;
    total += test_variant(0, 1, 1024, 1024);   // K,V
    total += test_variant(1, 1, 1024, 2048);   // Q
    total += test_variant(2, 1, 1024, 3072);   // gate,up
    total += test_variant(3, 1, 2048, 1024);   // O
    total += test_variant(4, 1, 3072, 1024);   // down
    printf("Total errors: %d\n", total);
    return total;
}
