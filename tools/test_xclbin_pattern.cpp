#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    auto bench = [&](const char* xp, const char* ip, int M, int K, int N, int n, const char* label) {
        printf("\n=== %s ===\n", label);
        auto xc = xrt::xclbin(std::string(xp)); device.register_xclbin(xc);
        auto hw = xrt::hw_context(device, xc.get_uuid());
        auto kernel = xrt::kernel(hw, "MLIR_AIE");
        auto instr = load_insts(ip);
        
        auto bo_instr = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        
        memcpy(bo_instr.map(), instr.data(), instr.size()*4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
        
        auto* a = bo_a.map<int8_t*>(); for (int i = 0; i < M*K; i++) a[i] = 1;
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
        auto* b = bo_b.map<int8_t*>(); for (int i = 0; i < K*N; i++) b[i] = 1;
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
        auto* c = bo_c.map<int16_t*>(); memset(c, 0, M*N*2);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
        
        auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
        
        printf("  First 16 results (should all = %d):\n", K);
        int ok = 0;
        for (int i = 0; i < 16; i++) {
            printf("  [%d]=%d", i, c[i]);
            if (c[i] == K) { printf("✓"); ok++; }
            printf("\n");
        }
        printf("  OK: %d / %d\n", ok, M*N);
        
        // Now test: does the n tile size match?
        // For n=192, N=1536: we should have 8 columns × 192 = 1536
        // Check that position n*8 = 192 has value K (start of 2nd column)
        printf("  C[192]=%d (start of core1 result)\n", c[192]);
        printf("  C[384]=%d (start of core2 result)\n", c[384]);
    };
    
    // INT16 xclbin (known working from earlier test)
    bench(
        "/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_attn_1536.xclbin",
        "/home/bcloud/npu-sandbox/npu-infer/build/wan/insts_wan_attn_1536.txt",
        32, 1536, 1536, 192, "INT16 attn (all-ones A,B)"
    );
    
    // Also test INT32
    {
        printf("\n=== INT32 attn (all-ones A,B) ===\n");
        auto xc = xrt::xclbin(std::string("/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_attn_1536_i32.xclbin"));
        device.register_xclbin(xc);
        auto hw = xrt::hw_context(device, xc.get_uuid());
        auto kernel = xrt::kernel(hw, "MLIR_AIE");
        auto instr = load_insts("/home/bcloud/npu-sandbox/npu-infer/build/wan/insts_wan_attn_1536_i32.txt");
        int M=32, K=1536, N=1536;
        auto bo_instr = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        memcpy(bo_instr.map(), instr.data(), instr.size()*4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
        auto* a = bo_a.map<int8_t*>(); for (int i = 0; i < M*K; i++) a[i] = 1;
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
        auto* b = bo_b.map<int8_t*>(); for (int i = 0; i < K*N; i++) b[i] = 1;
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
        auto* cc = bo_c.map<int32_t*>(); memset(cc, 0, M*N*4);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*4, 0);
        auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*4, 0);
        printf("  First 16 (all = %d):\n", K);
        int ok = 0;
        for (int i = 0; i < 16; i++) { printf("  [%d]=%d", i, cc[i]); if (cc[i] == K) ok++; printf("\n"); }
        printf("  OK: %d / %d\n", ok, M*N);
        printf("  C[192]=%d C[384]=%d\n", cc[192], cc[384]);
    }
    return 0;
}
