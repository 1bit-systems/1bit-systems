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
    // Load real weights
    FILE* f = fopen("/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_weights.bin", "rb");
    fseek(f, 0, SEEK_END); long wsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* wdata = (uint8_t*)malloc(wsz);
    (void)fread(wdata, 1, wsz, f); fclose(f);
    
    size_t q_offset = 25657368;
    const int8_t* q_i8 = (const int8_t*)(wdata + q_offset);
    float scale = 0.004174705594778061f;
    
    auto device = xrt::device(0);
    
    // Test 1: A_quantized=1, B=real_weights → should match sum_k q_i8[k][n]
    printf("=== Test A=1, B=real weights (INT16 xclbin) ===\n");
    {
        int M=32, K=1536, N=1536;
        std::string _xp = "/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_attn_1536.xclbin"; auto xc = xrt::xclbin(_xp);
        device.register_xclbin(xc);
        auto hw = xrt::hw_context(device, xc.get_uuid());
        auto kernel = xrt::kernel(hw, "MLIR_AIE");
        auto instr = load_insts("/home/bcloud/npu-sandbox/npu-infer/build/wan/insts_wan_attn_1536.txt");
        
        auto bo_instr = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        
        memcpy(bo_instr.map(), instr.data(), instr.size()*4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
        
        auto* a = bo_a.map<int8_t*>(); for (int i = 0; i < M*K; i++) a[i] = 1;
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
        
        auto* b = bo_b.map<int8_t*>(); memcpy(b, q_i8, (size_t)K*N);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
        
        auto* c = bo_c.map<int16_t*>(); memset(c, 0, M*N*2);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
        
        auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
        
        // Check first 5 elements: C[n] = sum_k A_quant[0][k] * B[k][n] = sum_k 1 * q_i8[k][n]
        printf("NPU raw first 10 vs expected (sum_k q_i8[k][n]):\n");
        int ok = 0;
        for (int n = 0; n < 10; n++) {
            int64_t expected = 0;
            for (int k = 0; k < K; k++) expected += q_i8[(size_t)k*N+n];
            printf("  [%d] NPU=%6d expected=%6ld %s\n", n, c[n], (long)expected, c[n]==expected ? "MATCH" : "MISMATCH");
            if (c[n] == expected) ok++;
        }
        printf("  OK: %d/10\n", ok);
    }
    
    // Test 2: A_quantized=127, B=real_weights (same as engine does for input=1.0)
    printf("\n=== Test A=127, B=real weights (INT32 xclbin) ===\n");
    {
        int M=32, K=1536, N=1536;
        std::string _xp = "/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_attn_1536_i32.xclbin"; auto xc = xrt::xclbin(_xp);
        device.register_xclbin(xc);
        auto hw = xrt::hw_context(device, xc.get_uuid());
        auto kernel = xrt::kernel(hw, "MLIR_AIE");
        auto instr = load_insts("/home/bcloud/npu-sandbox/npu-infer/build/wan/insts_wan_attn_1536_i32.txt");
        
        auto bo_instr = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        
        memcpy(bo_instr.map(), instr.data(), instr.size()*4);
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
        
        auto* a = bo_a.map<int8_t*>(); for (int i = 0; i < M*K; i++) a[i] = 127;
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
        
        auto* b = bo_b.map<int8_t*>(); memcpy(b, q_i8, (size_t)K*N);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
        
        auto* c = bo_c.map<int32_t*>(); memset(c, 0, M*N*4);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*4, 0);
        
        auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*4, 0);
        
        // With A=127: C[n] = sum_k 127 * q_i8[k][n] = 127 * sum_k q_i8[k][n]
        printf("NPU raw first 10 vs expected (127*sum):\n");
        int ok = 0;
        for (int n = 0; n < 10; n++) {
            int64_t expected = 0;
            for (int k = 0; k < K; k++) expected += 127 * (int64_t)q_i8[(size_t)k*N+n];
            printf("  [%d] NPU=%6d expected=%6ld diff=%.1f%%\n", n, c[n], (long)expected, (double)abs(c[n]-expected)/expected*100);
            if (abs(c[n]-expected) < 1000) ok++;
        }
        printf("  OK: %d/10\n", ok);
    }
    
    free(wdata);
    return 0;
}
