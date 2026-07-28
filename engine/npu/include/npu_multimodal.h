#pragma once
// npu_multimodal.h — I8Ctx with autonomous hw_context and CPU fallback

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include "bfp16_pack.h"

struct I8Ctx {
    int MD, KD, ND, NL;
    bool use_bf16 = false, use_cpu = false;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI, bA, bC;
    std::vector<std::unique_ptr<xrt::bo>> layerB;
    int8_t* Am = nullptr;
    int16_t* Cm = nullptr;
    bool initialized = false, layerB_cached = true;
    int gid_B = 4;
    std::vector<float> cpu_w;

    ~I8Ctx() {}
    bool isReady() { return initialized && (use_cpu || (k && bA && bC)); }

    size_t a_size() const { return use_bf16 ? (size_t)MD * KD * 2 : (size_t)MD * KD; }
    size_t b_size() const { return use_bf16 ? ((size_t)KD * ND * 6 + 7) / 8 : (size_t)KD * ND; }
    size_t c_size() const { return (size_t)MD * ND * 2; }

    bool init(xrt::device& d, const char* xp, const char* ip, int gid, int nlayers, bool bf16 = false) {
        use_bf16 = bf16; NL = nlayers; gid_B = gid;
        FILE* f = fopen(ip, "rb");
        if (!f) { fprintf(stderr, "  I8Ctx: fopen(%s) failed\n", ip); return false; }
        fseek(f, 0, 2); long sz = ftell(f); fseek(f, 0, 0);
        ins.resize(sz / 4); fread(ins.data(), 4, ins.size(), f); fclose(f);
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
            bI = std::make_unique<xrt::bo>(d, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
            memcpy(bI->map(), ins.data(), ins.size() * 4); bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bA = std::make_unique<xrt::bo>(d, a_size(), XCL_BO_FLAGS_CACHEABLE, k->group_id(3));
            bC = std::make_unique<xrt::bo>(d, c_size(), XCL_BO_FLAGS_CACHEABLE, k->group_id(5));
            Am = (int8_t*)bA->map(); Cm = (int16_t*)bC->map();
            layerB_cached = true;
            try { layerB.emplace_back(std::make_unique<xrt::bo>(d, b_size(), XCL_BO_FLAGS_CACHEABLE, k->group_id(gid_B))); }
            catch (...) { layerB_cached = false; layerB.emplace_back(std::make_unique<xrt::bo>(d, b_size(), XRT_BO_FLAGS_HOST_ONLY, k->group_id(gid_B))); }
        } catch (std::exception& e) {
            fprintf(stderr, "  I8Ctx::init: %s — CPU fallback\n", e.what());
            use_cpu = true;
        }
        initialized = true;
        return true;
    }

    void packB(const float* w, int K, int N, float& sout) {
        if (use_cpu) { cpu_w.assign(w, w + (size_t)K * N); sout = 1.0f; return; }
        if (use_bf16) {
            // BFP16 packing (matches universal engine)
            pack_bfp16_weights(w, K, N, (uint8_t*)layerB[0]->map(), b_size());
            sout = 1.0f;
        }
        else {
            float amax = 0;
            for (int i = 0; i < K * N; i++) { float a = fabsf(w[i]); if (std::isfinite(a) && a > amax) amax = a; }
            if (amax < 1e-12f) amax = 1.0f; sout = amax / 127.0f; float is = 127.0f / amax;
            auto* Bm = (int8_t*)layerB[0]->map();
            for (int i = 0; i < K * N; i++) {
                float v = w[i]; if (!std::isfinite(v)) v = 0;
                int x = (int)roundf(v * is); if (x > 127) x = 127; else if (x < -127) x = -127;
                Bm[i] = (int8_t)x;
            }
        }
        layerB[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    bool go(const float* A, int am, int ak, float ascale, float Bscale, float* C, int an) {
        if (use_cpu) {
            const float* W = cpu_w.empty() ? nullptr : cpu_w.data();
            if (!W) { memset(C, 0, (size_t)am * an * 4); return true; }
            for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
                double s = 0; for (int k = 0; k < ak; k++) s += (double)A[m * ak + k] * (double)W[(size_t)k * an + n];
                C[(size_t)m * an + n] = (float)s;
            }
            return true;
        }
        if (!isReady()) { memset(C, 0, (size_t)am * an * 4); return true; }
        // NPU path: quantize → sync → launch → wait → dequantize
        if (use_bf16) {
            auto* ABuf = (uint16_t*)Am;
            for (int i = 0; i < am * ak; i++) { uint32_t bits; memcpy(&bits, &A[i], 4); ABuf[i] = (uint16_t)(bits >> 16); }
        } else {
            float ais = 1.0f / ascale; memset(Am, 0, (size_t)am * KD);
            for (int m = 0; m < am; m++) for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k]; if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais); if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        layerB[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto r = (*k)((unsigned)3, *bI, (unsigned)ins.size(), *bA, *layerB[0], *bC);
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (use_bf16) {
            auto* COut = (uint16_t*)Cm;
            for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
                uint32_t bits = (uint32_t)COut[m * ND + n] << 16;
                float val; memcpy(&val, &bits, 4);
                C[m * an + n] = std::isfinite(val) ? val : 0;
            }
        } else {
            float cs = ascale * Bscale;
            for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                C[m * an + n] = std::isfinite(val) ? val : 0;
            }
        }
        return true;
    }

    // Async launch — returns run that caller waits on
    xrt::run launch_async(const float* A, int am, int ak, float ascale) {
        if (use_bf16) {
            auto* ABuf = (uint16_t*)Am;
            for (int i = 0; i < am * ak; i++) { uint32_t bits; memcpy(&bits, &A[i], 4); ABuf[i] = (uint16_t)(bits >> 16); }
        } else {
            float ais = 1.0f / ascale; memset(Am, 0, (size_t)am * KD);
            for (int m = 0; m < am; m++) for (int k = 0; k < ak; k++) {
                float v = A[m * ak + k]; if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * ais); if (q > 127) q = 127; else if (q < -127) q = -127;
                Am[m * KD + k] = (int8_t)q;
            }
        }
        bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bC->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        layerB[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)((unsigned)3, *bI, (unsigned)ins.size(), *bA, *layerB[0], *bC);
    }

    // Finish async — wait and dequantize
    void finish_async(xrt::run& r, float* C, int am, int an, float ascale, float Bscale) {
        r.wait();
        bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (use_bf16) {
            auto* COut = (uint16_t*)Cm;
            for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
                uint32_t bits = (uint32_t)COut[m * ND + n] << 16;
                float val; memcpy(&val, &bits, 4);
                C[m * an + n] = std::isfinite(val) ? val : 0;
            }
        } else {
            float cs = ascale * Bscale;
            for (int m = 0; m < am; m++) for (int n = 0; n < an; n++) {
                float val = (float)Cm[m * ND + n] * cs;
                C[m * an + n] = std::isfinite(val) ? val : 0;
            }
        }
    }
};

struct AttnCtx {
    int max_seq, NH, NKV, HD, XM;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::kernel> k;
    std::vector<uint32_t> ins;
    std::unique_ptr<xrt::bo> bI, bQ, bK, bV, bOut, bDummy;
    bool initialized = false;

    ~AttnCtx() {}
    bool isReady() { return initialized && k && bI && bQ && bK && bV && bOut; }

    bool init(xrt::device& d, const char* xp, const std::vector<uint32_t>& instrs,
              int max_seq_len, int nh, int nkv, int hd, int xm) {
        max_seq = max_seq_len; NH = nh; NKV = nkv; HD = hd; XM = xm; ins = instrs;
        try {
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");
            bI = std::make_unique<xrt::bo>(d, ins.size()*4, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
            memcpy(bI->map(), ins.data(), ins.size()*4);
            bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            size_t q_bytes = (size_t)XM * NH * HD;
            size_t kv_bytes = (size_t)max_seq * NKV * HD;
            size_t out_bytes = (size_t)XM * NH * HD * 2;
            int gid = k->group_id(5);
            bQ = std::make_unique<xrt::bo>(d, q_bytes, XRT_BO_FLAGS_HOST_ONLY, gid);
            bK = std::make_unique<xrt::bo>(d, kv_bytes, XRT_BO_FLAGS_HOST_ONLY, gid);
            bV = std::make_unique<xrt::bo>(d, kv_bytes, XRT_BO_FLAGS_HOST_ONLY, gid);
            bOut = std::make_unique<xrt::bo>(d, out_bytes, XRT_BO_FLAGS_HOST_ONLY, gid);
            bDummy = std::make_unique<xrt::bo>(d, 64, XRT_BO_FLAGS_HOST_ONLY, gid);
        } catch (std::exception& ex) {
            fprintf(stderr, "  AttnCtx init failed: %s\n", ex.what());
            return false;
        }
        initialized = true;
        return true;
    }

    xrt::run launch(const float* Q_f32, const float* K_cache, const float* V_cache,
                    int seq_len, int batch, float q_scale, float kv_scale) {
        auto* q_i8 = (int8_t*)bQ->map();
        float q_is = 1.0f / q_scale;
        for (int i = 0; i < batch * NH * HD; i++) {
            float v = Q_f32[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * q_is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            q_i8[i] = (int8_t)q;
        }
        bQ->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        size_t kv_len = (size_t)seq_len * NKV * HD;
        float kv_is = 1.0f / kv_scale;
        auto* k_i8 = (int8_t*)bK->map();
        for (size_t i = 0; i < kv_len; i++) {
            float v = K_cache[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * kv_is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            k_i8[i] = (int8_t)q;
        }
        bK->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto* v_i8 = (int8_t*)bV->map();
        for (size_t i = 0; i < kv_len; i++) {
            float v = V_cache[i]; if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * kv_is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            v_i8[i] = (int8_t)q;
        }
        bV->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return (*k)(3ULL, *bI, (unsigned)ins.size(), *bQ, *bK, *bV, *bOut, *bDummy);
    }

    void finish(xrt::run& r, float* out, int batch, float q_scale, float kv_scale) {
        r.wait();
        bOut->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto* out_i16 = (int16_t*)bOut->map();
        float cs = q_scale * kv_scale;
        for (int i = 0; i < batch * NH * HD; i++) {
            float val = (float)out_i16[i] * cs;
            out[i] = std::isfinite(val) ? val : 0;
        }
    }
};

struct MMConfig {
    int H = 0, NC = 0, NH = 0, NKV = 0, HD = 0, IM = 0, NV = 0;
    int GQA = 0, XM = 128;
    int qkv_k_offset = 0, qkv_v_offset = 0, qkv_total = 0;
    int xclbin_qkv_k = 0, xclbin_qkv_n = 0;
    int xclbin_o_k = 0, xclbin_o_n = 0;
    int xclbin_gu_k = 0, xclbin_gu_n = 0;
    int xclbin_d_k = 0, xclbin_d_n = 0;
    bool gu_split = false, has_lm_head = false, has_q_norm = false, has_k_norm = false;
    float rope_theta = 1000000.0f;
    std::string model_tag;
};
