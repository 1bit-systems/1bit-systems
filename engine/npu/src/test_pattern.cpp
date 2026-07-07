#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static std::vector<uint32_t> load_insts(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> v(sz / 4);
    (void)fread(v.data(), 4, v.size(), f); fclose(f);
    return v;
}

int main() {
    auto device = xrt::device(0);
    
    // Test with pattern B data, not random
    int K=1536, N=1536, M=32;
    std::string xp = "/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_attn_1536.xclbin";
    auto xc = xrt::xclbin(xp); device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto krnl = xrt::kernel(hw, "MLIR_AIE");
    auto instr = load_insts("/home/bcloud/npu-sandbox/npu-infer/build/wan/insts_wan_attn_1536.txt");
    
    auto bi = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
    auto ba = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(3));
    auto bb = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(4));
    auto bc = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(5));
    
    memcpy(bi.map(), instr.data(), instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
    
    // A = 1 everywhere
    auto* a = ba.map<int8_t*>();
    for (int i = 0; i < M*K; i++) a[i] = 1;
    ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
    
    // B = simple ramp pattern in [K,N] layout, then convert to NPU flat layout
    auto* b = bb.map<int8_t*>();
    
    // Build in standard [K,N] layout
    std::vector<int8_t> b_ref(K * N);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            b_ref[k * N + n] = (int8_t)((k * 7 + n * 3) % 63 - 31);
    
    // Convert to NPU flat layout
    int kt = 64, nt = 192;
    int k_tiles = K / kt;
    int n_columns = 8;
    
    for (int ng = 0; ng < N / (8 * nt); ng++) {
        for (int c = 0; c < n_columns; c++) {
            for (int kg = 0; kg < k_tiles; kg++) {
                for (int i = 0; i < kt; i++) {
                    for (int j = 0; j < nt; j++) {
                        int out_idx = ng * 8 * nt + c * nt + j;
                        int in_idx = kg * kt + i;
                        int b_pos = in_idx * N + out_idx;
                        int flat_pos = ng * n_columns * k_tiles * kt * nt
                                     + c * k_tiles * kt * nt
                                     + kg * kt * nt
                                     + i * nt + j;
                        b[flat_pos] = b_ref[b_pos];
                    }
                }
            }
        }
    }
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
    
    auto* c = bc.map<int16_t*>(); memset(c, 0, M*N*2);
    bc.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
    
    printf("Running NPU with pattern B data (A=1)...\n");
    auto run = krnl((unsigned)3, bi, (unsigned)instr.size(), ba, bb, bc);
    run.wait();
    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
    
    // Compute expected: C[n] = sum_k 1 * b_ref[k][n]
    printf("Results (first 16):\n");
    int match = 0;
    for (int n = 0; n < 16; n++) {
        int64_t expected = 0;
        for (int k = 0; k < K; k++) expected += b_ref[k * N + n];
        printf("  [%d] NPU=%6d ref=%6ld %s\n", n, c[n], (long)expected, c[n]==expected ? "OK" : "FAIL");
        if (c[n] == expected) match++;
    }
    printf("Match: %d/16\n", match);
    
    // Extra check: what does position 0 and 8 give?
    printf("\nPeriod-8 check:\n");
    for (int n = 0; n < 192; n += 8) {
        printf("  [%d]=%d [%d]=%d\n", n, c[n], n+8, c[n+8]);
    }
    
    return 0;
}
