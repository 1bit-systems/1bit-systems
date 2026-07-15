// backend_npu.cpp — NPU XRT backend for Zaya1-8B
// Uses pre-compiled ternary TQ1 xclbins on AMD XDNA NPU

#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>

// ── NPU XRT headers ──
// These are only included when XRT is available at build time
#ifdef HAS_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#endif

struct NPUBackend : Backend {
    bool available = false;

    // XRT state (only valid when HAS_XRT)
#ifdef HAS_XRT
    xrt::device xrt_dev;
    std::vector<xrt::kernel> kernels;
#endif

    NPUBackend() { type = BackendType::NPU_XRT; name = "NPU XDNA (XRT) [STUB]"; }

    ~NPUBackend() override { destroy(); }

    // NPU inference (forward/lm_head/generate) is not implemented — this backend
    // detects the XRT device only. can_infer()=false so BackendManager reports it as
    // available but never selects it for inference (fixes #82). Real inference needs
    // XRT/xclbin dispatch.
    bool can_infer() const override { return false; }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        printf("NPU[stub]: Initializing (detection only -- no inference)...\n");

        // Try to open NPU device via XRT
#ifdef HAS_XRT
        try {
            xrt_dev = xrt::device(0);
            printf("NPU[stub]: XRT device found: %s\n",
                   xrt_dev.get_info<xrt::info::device::name>().c_str());
            printf("NPU[stub]: >>> Inference NOT available via this backend. <<<\n");
            printf("NPU[stub]: Real NPU inference uses engine/npu/npu_engine_universal via subprocess.\n");
            available = true;
        } catch (const std::exception& e) {
            printf("NPU[stub]: No XRT device: %s\n", e.what());
            available = false;
        }
#else
        printf("NPU[stub]: Built without XRT support -- NPU detection unavailable\n");
        available = false;
#endif
        initialized = available;
        // NPU[stub]: inference not available - can_infer()=false
        if (available) {
            printf("NPU[stub]: device detected, but inference NOT available via this backend — reporting as available but not inference-capable (see #82)\n");
        }
        return available;
    }

    bool reset() override { return true; }

    bool forward(int token_id, float* hidden_out) override {
        (void)token_id; (void)hidden_out;
        return false; // TODO: implement NPU forward (needs xclbins)
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        return false; // TODO: implement NPU lm_head
    }

    int generate(int token_id) override {
        (void)token_id;
        return -1; // TODO: implement NPU generate
    }

    float benchmark(int tokens = 10) override {
        (void)tokens;
        return 0;
    }

    void destroy() override {
        initialized = false;
    }
};

Backend* create_npu_backend() { return new NPUBackend(); }
