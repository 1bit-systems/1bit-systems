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
    FILE* f = fopen("/home/bcloud/npu-sandbox/npu-infer/build/wan/wan_weights.bin", "rb");
    fseek(f, 0, SEEK_END); long wsz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* wdata = (uint8_t*)malloc(wsz);
    (void)fread(wdata, 1, wsz, f); fclose(f);
    size_t q_off = 25657368;
    const int8_t* q_i8 = (const int8_t*)(wdata + q_off);
    
    auto device = xrt::device(0);
    
    // Test A=1, B=real (INT16)
    printf("=== A=1, B=real (INT16) ===\n");
    {
        int M=32, K=1536, N=1536;
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
        auto* a = ba.map<int8_t*>(); for (int i = 0; i < M*K; i++) a[i] = 1;
        ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
        auto* b = bb.map<int8_t*>(); memcpy(b, q_i8, (size_t)K*N);
        // Debug: verify first few weight values in BO
        printf("Weight BO first 10: ");
        for (int i = 0; i < 10; i++) printf("%d ", b[i]);
        printf("\nExpected first 10:  ");
        for (int i = 0; i < 10; i++) printf("%d ", q_i8[i]);
        printf("\n");
        bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
        auto* c = bc.map<int16_t*>(); memset(c, 0, M*N*2);
        bc.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
        
        auto run = krnl((unsigned)3, bi, (unsigned)instr.size(), ba, bb, bc);
        run.wait();
        bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
        
        printf("C[n] = sum_k 1*q_i8[k][n]:\n");
        int ok = 0;
        for (int n = 0; n < 10; n++) {
            int64_t expected = 0;
            for (int k = 0; k < K; k++) expected += q_i8[(size_t)k*N+n];
            printf("  [%d] NPU=%d exp=%ld %s\n", n, c[n], (long)expected, c[n]==expected ? "OK" : "FAIL");
            if (c[n]==expected) ok++;
        }
        printf("  %d/10 OK\n", ok);
    }
    free(wdata);
    return 0;
}
