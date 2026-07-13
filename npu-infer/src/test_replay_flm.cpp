// test_replay_flm.cpp — Replay FLM's captured BOs to verify NPU pipeline
// Build: g++ -std=c++17 -I/opt/xilinx/xrt/include -o /tmp/test_replay test_replay_flm.cpp \
//          -L/opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -Wl,-rpath,/opt/xilinx/xrt/lib -ldl -lpthread

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <unistd.h>

namespace fs = std::filesystem;

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"

static const char* XCLBIN_DIR = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
static const char* BO_DUMP_DIR = "/home/bcloud/npu-sandbox/xrt-direct/captured_bo_dump";

struct CapturedBO {
    std::string label;
    size_t size;
    std::vector<uint8_t> data;
};

static std::vector<CapturedBO> load_captured_bos() {
    std::vector<CapturedBO> bos;
    struct Entry { const char* label; size_t size; const char* file; };
    
    Entry entries[] = {
        {"act_layer0",  10*1024*1024, "h6_10MB_act_layer0.bin"},
        {"weight_prepack_layer0", 1*1024*1024, "h7_1MB_weight_prepack_layer0.bin"},
        {"cmd_layer0",  128*1024*1024, "h8_128MB_cmds_layer0.bin"},
        {"scale_prepack_layer0", 1*1024*1024, "h9_1MB_scale_prepack_layer0.bin"},
    };
    
    for (auto& e : entries) {
        CapturedBO bo;
        bo.label = e.label;
        bo.size = e.size;
        
        std::string path = std::string(BO_DUMP_DIR) + "/" + e.file;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { printf("  [WARN] %s not found\n", path.c_str()); continue; }
        fseek(f, 0, SEEK_END);
        size_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        bo.data.resize(file_size);
        fread(bo.data.data(), 1, file_size, f);
        fclose(f);
        printf("  Loaded %s: %zu bytes\n", e.label, file_size);
        bos.push_back(std::move(bo));
    }
    return bos;
}

int main() {
    printf("=== FLM BO Replay Test ===\n\n");
    
    // Load captured BOs
    printf("Loading captured BOs...\n");
    auto captured = load_captured_bos();
    
    auto find_bo = [&](const char* label) -> CapturedBO* {
        for (auto& b : captured) if (b.label == label) return &b;
        return nullptr;
    };
    auto* c_act = find_bo("act_layer0");
    auto* c_weight = find_bo("weight_prepack_layer0");
    if (!c_act || !c_weight) {
        printf("FAIL: missing essential BOs\n"); return 1;
    }
    
    // Try ALL 4 xclbins
    const char* xclbin_names[] = {"mm.xclbin", "attn.xclbin", "layer.xclbin", "dequant.xclbin"};
    
    for (int xi = 0; xi < 4; xi++) {
        printf("\n=== XPU: %s ===\n", xclbin_names[xi]);
        
        xrt::device device(0);
        std::string xclbin_path = std::string(XCLBIN_DIR) + "/" + xclbin_names[xi];
        auto uuid = device.load_xclbin(xclbin_path);
        auto hwctx = xrt::hw_context(device, uuid, xrt::hw_context::access_mode::shared);
        auto kern = xrt::kernel(hwctx, "MLIR_AIE");
        
        // Create BOs with captured data
        xrt::bo bo_act(device, c_act->data.size(), xrt::bo::flags::host_only, 0);
        xrt::bo bo_weight(device, c_weight->data.size(), xrt::bo::flags::host_only, 0);
        
        memcpy(bo_act.map(), c_act->data.data(), c_act->data.size());
        memcpy(bo_weight.map(), c_weight->data.data(), c_weight->data.size());
        bo_act.sync(XCL_BO_SYNC_BO_TO_DEVICE, c_act->data.size(), 0);
        bo_weight.sync(XCL_BO_SYNC_BO_TO_DEVICE, c_weight->data.size(), 0);
        
        // Save pre-exec copies for comparison
        std::vector<uint8_t> act_before(c_act->data.begin(), c_act->data.end());
        std::vector<uint8_t> weight_before(c_weight->data.begin(), c_weight->data.end());
        
        // Try opcode=3 with captured data
        printf("  opcode=3...\n");
        auto t0 = std::chrono::steady_clock::now();
        try {
            auto run = kern((uint64_t)3, (uint64_t)0, (uint32_t)0,
                           bo_act, bo_weight, bo_act, bo_weight, bo_act);
            run.wait();
        } catch (const std::exception& e) {
            printf("  FAIL: %s\n", e.what());
            continue;
        }
        auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        
        bo_act.sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_act->data.size(), 0);
        
        // Check changes
        uint8_t* act_out = (uint8_t*)bo_act.map();
        int changed = 0;
        for (size_t i = 0; i < act_before.size() && i < 4096; i++)
            if (act_out[i] != act_before[i]) changed++;
        printf("  %.2f ms, %d/%d bytes changed\n", ms, changed, 4096);
        
        if (changed > 0) {
            printf("  Output (first 32 bytes):\n  ");
            for (int i = 0; i < 32; i++) printf("%02x ", act_out[i]);
            printf("\n");
            
            // Try to decode as BF16
            printf("  As BF16: ");
            for (int i = 0; i < 8; i++) {
                uint16_t v; memcpy(&v, &act_out[i*2], 2);
                float f; uint32_t bits = (uint32_t)v << 16; memcpy(&f, &bits, 4);
                printf("%.4f ", f);
            }
            printf("\n");
        }
    }
    
    printf("\n=== DONE ===\n");
    return 0;
}
