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
static const int MT=128, KT=64, NT=128;

int main() {
    const char *xp = "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x1024_128x64x128.xclbin";
    const char *ip = "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x1024_128x64x128.txt";
    FILE *f = fopen(ip, "rb"); if (!f) return 1;
    fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
    std::vector<uint32_t> ins(sz/4); fread(ins.data(), 4, ins.size(), f); fclose(f);
    auto d = xrt::device(0); auto xc = xrt::xclbin(std::string(xp)); d.register_xclbin(xc);
    auto hw = xrt::hw_context(d, xc.get_uuid()); auto k = xrt::kernel(hw, "MLIR_AIE");
    const int MP=128, XK=1024, XN=1024;
    auto bi = xrt::bo(d, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
    auto bA = xrt::bo(d, (size_t)MP*XK*2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
    auto bB = xrt::bo(d, (size_t)XK*XN*9/8, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
    auto bC = xrt::bo(d, (size_t)MP*XN*2, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
    memcpy(bi.map(), ins.data(), ins.size()*4); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Test: use B where only K=0..63 has value=5.0, rest=0
    // If all K=1024 is consumed, result = 3*5*64 = 960
    // If only K=0..63 is consumed, result = 3*5*64 = 960 (same for uniform B)
    // Better: use B where different K ranges have different values
    // K=0..63: B=5
    // K=64..127: B=7
    // K=128..191: B=11
    // K=192..255: B=13
    // K=256..319: B=17
    // K=320..383: B=19
    // K=384..447: B=23
    // K=448..511: B=29
    // K=512..575: B=31
    // K=576..639: B=37
    // K=640..703: B=41
    // K=704..767: B=43
    // K=768..831: B=47
    // K=832..895: B=53
    // K=896..959: B=59
    // K=960..1023: B=61
    
    std::vector<float> Bf(XK * XN, 0.0f);
    int primes[] = {5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
    for (int k = 0; k < XK; k++) {
        int kt = k / 64;
        int v = (kt < 16) ? primes[kt] : 0;
        for (int n = 0; n < XN; n++) Bf[k * XN + n] = (float)v;
    }
    
    auto Bs = gemm_atb::layout_transpose_L1_1x2_8x8block(Bf, XK, XN, KT, NT);
    auto Bp = floatToBfp16(8, XK*XN, Bs.data(), 0);
    memcpy(bB.map(), Bp.data(), Bp.size()); bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // A = 3.0 for all rows
    std::vector<float> Af(MP * XK, 3.0f);
    auto As = gemm_atb::layout_A_L1_2x1_8x8block(Af, MP, XK, MT, KT);
    uint16_t *Ab = (uint16_t*)bA.map();
    for (int i = 0; i < MP * XK; i++) Ab[i] = fbf16(As[i]);
    bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    memset(bC.map(), 0, (size_t)MP*XN*2); bC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = k((unsigned)3, bi, (unsigned)ins.size(), bA, bB, bC); run.wait();
    bC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    uint16_t *Cb = (uint16_t*)bC.map();
    // Expected = 3.0 * sum(primes) * 64 = 3 * 476 * 64 = 91392
    int sum = 0; for (int p : primes) sum += p;
    float expected = 3.0f * sum * 64.0f;
    float first = bf16f(Cb[0]);
    printf("Expected: %.0f, Got: %.0f\n", expected, first);
    // Also check: if only first 8 K-tiles: 3*sum(primes[0..7])*64 = 3*103*64 = 19776
    int sum8 = 0; for (int i=0;i<8;i++) sum8+=primes[i];
    printf("If only K=0..511: %.0f\n", 3.0f*sum8*64);
    // If only first 4 K-tiles: 3*sum(primes[0..3])*64 = 3*36*64 = 6912
    int sum4 = 0; for (int i=0;i<4;i++) sum4+=primes[i];
    printf("If only K=0..255: %.0f\n", 3.0f*sum4*64);
    // If only first 1 K-tile: 3*5*64 = 960
    printf("If only K=0..63: %.0f\n", 3.0f*5*64);
    
    // Also check the BFP16 decode to ensure correctness
    printf("\nBFP16 decode (first 5 groups):\n");
    for (int g = 0; g < 5; g++) {
        int idx = g * 9;
        int exp = Bp[idx];
        int8_t m0 = (int8_t)Bp[idx+1];
        float val = (float)m0 * powf(2.0f, exp-127-7);
        printf("  group %d: exp=%d mant=%d val=%.1f\n", g, exp, m0, val);
    }
    
    return 0;
}
