// backend_fused_npu.cpp — NPU FFN caller (pure C++, not HIP).
// Compiled as CXX to avoid HIP compiler context conflicts with XRT.
#include "backend_fused_npu.h"
#include "../engine/fusion/zero_copy/npu_gemm_kernel.h"
#include <cmath>
#include <vector>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

struct NpuState {
    xrt::device dev{0};
    std::unique_ptr<fusion::NpuGemmKernel> gu, d;
    std::vector<float> gu_scale, d_scale;  // per-layer scales
    int H = 0, IM = 0, NC = 0;
    bool ok = false;
};

NpuState* npu_state_create(const char* xclbin_dir, int H, int IM, int NC) {
    int npu_fd = open("/dev/accel/accel0", O_RDONLY);
    if (npu_fd < 0) return nullptr;
    close(npu_fd);

    auto* s = new NpuState();
    s->H = H; s->IM = IM; s->NC = NC;
    try { s->dev = xrt::device(0); } catch (...) { delete s; return nullptr; }

    auto find_xclbin = [&](const char* tag) -> std::pair<std::string,std::string> {
        std::string xd(xclbin_dir ? xclbin_dir : "engine/npu/xclbins");
        std::string b = xd + "/final_i8_" + tag;
        std::string i = xd + "/insts_i8_" + tag;
        std::string ms = b + "_qwen3_0_6b.xclbin", mi = i + "_qwen3_0_6b.txt";
        if (access(ms.c_str(), F_OK) == 0 && access(mi.c_str(), F_OK) == 0) return {ms, mi};
        std::string ts = b + "_v.xclbin", ti = i + "_v.txt";
        if (access(ts.c_str(), F_OK) == 0 && access(ti.c_str(), F_OK) == 0) return {ts, ti};
        return {"", ""};
    };

    auto [xgu, igu] = find_xclbin("GU");
    auto [xdd, idd] = find_xclbin("D");
    if (xgu.empty()) { delete s; return nullptr; }

    s->gu = std::make_unique<fusion::NpuGemmKernel>();
    s->d  = std::make_unique<fusion::NpuGemmKernel>();
    if (!s->gu->init(s->dev, xgu.c_str(), igu.c_str(), 128, H, 2*IM) ||
        !s->d->init(s->dev, xdd.c_str(), idd.c_str(), 128, IM, H)) {
        delete s; return nullptr;
    }

    s->ok = true;
    return s;
}

void npu_state_destroy(NpuState* s) {
    delete s;
}

void npu_state_pack_layer(NpuState* s, int layer,
                           const float* w1, const float* w2, const float* w3) {
    if (!s || !s->ok) return;
    if ((int)s->gu_scale.size() <= layer) {
        s->gu_scale.resize(layer + 1);
        s->d_scale.resize(layer + 1);
    }
    int H = s->H, IM = s->IM;

    // Transpose GGUF [out,in] → packB's [in,out]
    std::vector<float> gu((size_t)H * 2 * IM);
    for (int k = 0; k < H; k++) for (int n = 0; n < IM; n++) {
        gu[(size_t)k*(2*IM)+n]       = w1[(size_t)n*H + k];
        gu[(size_t)k*(2*IM)+IM+n]    = w2[(size_t)n*H + k];
    }
    std::vector<float> dw((size_t)IM * H);
    for (int k = 0; k < IM; k++) for (int n = 0; n < H; n++)
        dw[(size_t)k*H + n] = w3[(size_t)n*IM + k];

    s->gu->packB(gu.data(), H, 2*IM, s->gu_scale[layer]);
    s->d->packB(dw.data(), IM, H, s->d_scale[layer]);
}

bool npu_state_ffn(NpuState* s, int layer, float* h, int H) {
    if (!s || !s->ok || layer >= (int)s->gu_scale.size()) return false;
    int IM = s->IM;

    float ascale = 0;
    for (int i = 0; i < H; i++) { float a = fabsf(h[i]); if (a > ascale) ascale = a; }
    ascale = (ascale < 1e-12f) ? 1.0f : ascale / 127.0f;

    try {
        std::vector<float> gu(2*IM);
        s->gu->go(h, 1, H, ascale, s->gu_scale[layer], gu.data(), 2*IM);
        for (int i = 0; i < IM; i++)
            gu[i] = (gu[i] / (1.0f + expf(-gu[i]))) * gu[IM+i];

        float dscale = 0;
        for (int i = 0; i < IM; i++) { float a = fabsf(gu[i]); if (a > dscale) dscale = a; }
        dscale = (dscale < 1e-12f) ? 1.0f : dscale / 127.0f;

        std::vector<float> ffn_out(H);
        s->d->go(gu.data(), 1, IM, dscale, s->d_scale[layer], ffn_out.data(), H);
        for (int i = 0; i < H; i++) h[i] += ffn_out[i];
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[npu_ffn] l=%d exception: %s\n", layer, e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[npu_ffn] l=%d unknown exception\n", layer);
        return false;
    }
}
