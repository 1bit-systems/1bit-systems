#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "helper.h"
#include "gemm_atb_layout.h"
int main() {
    const int K=1024, N=1024, KT=64, NT=128;
    std::vector<float> Bf(K*N, 3.0f);
    auto Bs = gemm_atb::layout_transpose_L1_1x2_8x8block(Bf, K, N, KT, NT);
    auto Bp = floatToBfp16(8, K*N, Bs.data(), 0);
    printf("B raw: %d floats, %zu packed bytes\n", K*N, Bp.size());
    // The BFP16 format: first byte is padding, then groups of 8 mantissas + 1 exponent
    // Decode a few samples
    printf("BFP16 sample (first 20 bytes): ");
    for (int i = 0; i < 20; i++) printf("%02x ", Bp[i]);
    printf("\n");
    // BFP16 format: byte[0] = shared exponent, byte[1..8] = mantissa bytes (8 values)
    printf("Group 0: exp=%d (unbiased=%d) mantissa_bytes=", Bp[0], (int)Bp[0]-127);
    for (int i = 1; i <= 8; i++) printf("%d ", (int8_t)Bp[i]);
    printf("\n");
    int exp = Bp[0];
    float val = (float)(int8_t)Bp[1] * powf(2.0f, exp-127-7);
    printf("  decoded value: %f (expected 3.0)\n", val);
    size_t expected = ((size_t)K*N/8)*9;
    printf("Expected packed size: %zu, actual: %zu\n", expected, Bp.size());
    return 0;
}
