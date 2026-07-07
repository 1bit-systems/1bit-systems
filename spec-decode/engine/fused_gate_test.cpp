// Standalone gate test: reproduce the validated single-layer full-layer xclbin gate
// (layer L @ token T) from the reference fixture and compare to expected output.
// Isolates C++ driver plumbing from real-generation semantics.
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/include fused_gate_test.cpp -o fused_gate_test \
//       -L/opt/xilinx/xrt/lib64 -lxrt_coreutil -luuid -lm
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static std::vector<char> load(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "MISSING %s\n", p.c_str()); exit(2); }
    size_t n = f.tellg(); f.seekg(0);
    std::vector<char> d(n); f.read(d.data(), n); return d;
}
static float bf16f(uint16_t v) { uint32_t b = (uint32_t)v << 16; float f; memcpy(&f, &b, 4); return f; }

int main(int argc, char** argv) {
    const char* xclbin_dir = argc > 1 ? argv[1]
        : "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127";
    const char* gate_dir = argc > 2 ? argv[2]
        : "/home/bcloud/npu-sandbox/npu-infer/build/int8/capref/gate";
    int token = argc > 3 ? atoi(argv[3]) : 31;
    unsigned opcode = getenv("FUSED_OPCODE") ? atoi(getenv("FUSED_OPCODE")) : 3;

    auto hid = load(std::string(gate_dir) + "/gate_hidden.bin");
    auto kc  = load(std::string(gate_dir) + "/gate_kcache.bin");
    auto vc  = load(std::string(gate_dir) + "/gate_vcache.bin");
    auto wt  = load(std::string(gate_dir) + "/gate_weights.bin");
    auto exp = load(std::string(gate_dir) + "/gate_expected.bin");
    char instr_path[512];
    snprintf(instr_path, sizeof instr_path, "%s/design-token127-to-token%d.bin", xclbin_dir, token);
    auto ins = load(instr_path);
    printf("hidden=%zuB kcache=%zuB vcache=%zuB weights=%zuB expected=%zuB instr=%zuB(words=%zu)\n",
           hid.size(), kc.size(), vc.size(), wt.size(), exp.size(), ins.size(), ins.size()/4);

    xrt::device dev(0);
    auto xclbin = xrt::xclbin(std::string(xclbin_dir) + "/design.xclbin");
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx(dev, xclbin.get_uuid());
    xrt::kernel k(ctx, "MLIR_AIE");

    auto mk = [&](size_t bytes, int arg, xrt::bo::flags fl) {
        return xrt::bo(dev, bytes, fl, k.group_id(arg));
    };
    auto bI = mk(ins.size(), 1, xrt::bo::flags::cacheable);
    auto bK = mk(kc.size(), 3, xrt::bo::flags::host_only);
    auto bV = mk(vc.size(), 4, xrt::bo::flags::host_only);
    auto bW = mk(wt.size(), 5, xrt::bo::flags::host_only);
    auto bO = mk(exp.size(), 6, xrt::bo::flags::host_only);
    auto bH = mk(hid.size(), 7, xrt::bo::flags::host_only);
    memcpy(bI.map(), ins.data(), ins.size());
    memcpy(bK.map(), kc.data(), kc.size());
    memcpy(bV.map(), vc.data(), vc.size());
    memcpy(bW.map(), wt.data(), wt.size());
    memcpy(bH.map(), hid.data(), hid.size());
    memset(bO.map(), 0, exp.size());
    for (auto* b : {&bI,&bK,&bV,&bW,&bH,&bO}) b->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    int repeat = getenv("REPEAT") ? atoi(getenv("REPEAT")) : 1;
    int reuse  = getenv("REUSE") ? atoi(getenv("REUSE")) : 0;
    int resync = getenv("RESYNC") ? atoi(getenv("RESYNC")) : 0;
    if (getenv("RELOAD")) {
        // Measure per-call cost of a fresh hw_context+kernel+run (avoids the intra-process hang).
        int reps = atoi(getenv("RELOAD"));
        for (int rep = 0; rep < reps; rep++) {
            auto t0 = std::chrono::steady_clock::now();
            xrt::hw_context c2(dev, xclbin.get_uuid());
            xrt::kernel k2(c2, "MLIR_AIE");
            auto b2I = xrt::bo(dev, ins.size(), xrt::bo::flags::cacheable, k2.group_id(1));
            auto b2K = xrt::bo(dev, kc.size(), xrt::bo::flags::host_only, k2.group_id(3));
            auto b2V = xrt::bo(dev, vc.size(), xrt::bo::flags::host_only, k2.group_id(4));
            auto b2W = xrt::bo(dev, wt.size(), xrt::bo::flags::host_only, k2.group_id(5));
            auto b2O = xrt::bo(dev, exp.size(), xrt::bo::flags::host_only, k2.group_id(6));
            auto b2H = xrt::bo(dev, hid.size(), xrt::bo::flags::host_only, k2.group_id(7));
            memcpy(b2I.map(), ins.data(), ins.size()); memcpy(b2K.map(), kc.data(), kc.size());
            memcpy(b2V.map(), vc.data(), vc.size()); memcpy(b2W.map(), wt.data(), wt.size());
            memcpy(b2H.map(), hid.data(), hid.size()); memset(b2O.map(), 0, exp.size());
            for (auto* b : {&b2I,&b2K,&b2V,&b2W,&b2H}) b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto rr = k2(opcode, b2I, (uint32_t)(ins.size()/4), b2K, b2V, b2W, b2O, b2H);
            rr.wait(); b2O.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            fprintf(stderr, "[reload rep %d] %.1f ms  out[0]=%.4f\n", rep, ms, bf16f(((uint16_t*)b2O.map())[0]));
        }
        return 0;
    }
    if (reuse) {
        xrt::run r(k);
        r.set_arg(0, opcode); r.set_arg(1, bI); r.set_arg(2, (uint32_t)(ins.size()/4));
        r.set_arg(3, bK); r.set_arg(4, bV); r.set_arg(5, bW); r.set_arg(6, bO); r.set_arg(7, bH);
        for (int rep = 0; rep < repeat; rep++) {
            if (resync) bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            r.start(); r.wait();
            bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            fprintf(stderr, "[reuse rep %d] out[0]=%.4f\n", rep, bf16f(((uint16_t*)bO.map())[0]));
        }
    } else {
        for (int rep = 0; rep < repeat; rep++) {
            if (resync) bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto r = k(opcode, bI, (uint32_t)(ins.size()/4), bK, bV, bW, bO, bH);
            r.wait();
            bO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            fprintf(stderr, "[rep %d] done, out[0]=%.4f\n", rep, bf16f(((uint16_t*)bO.map())[0]));
        }
    }

    const uint16_t* got = (const uint16_t*)bO.map();
    const uint16_t* want = (const uint16_t*)exp.data();
    int n = exp.size() / 2;   // bf16 count
    int nan_ct = 0, close = 0; double maxad = 0, sumad = 0;
    for (int i = 0; i < n; i++) {
        float g = bf16f(got[i]), w = bf16f(want[i]);
        if (!std::isfinite(g)) nan_ct++;
        double ad = std::fabs((double)g - w);
        if (std::isfinite(ad)) { sumad += ad; if (ad > maxad) maxad = ad; if (ad < 0.05) close++; }
    }
    printf("bf16 elems=%d  close(<0.05)=%d (%.1f%%)  nan=%d  max_abs_diff=%.4f  mean_abs_diff=%.5f\n",
           n, close, 100.0*close/n, nan_ct, maxad, sumad/n);
    printf("got[:6] : "); for (int i=0;i<6;i++) printf("%.4f ", bf16f(got[i]));  printf("\n");
    printf("want[:6]: "); for (int i=0;i<6;i++) printf("%.4f ", bf16f(want[i])); printf("\n");
    printf(close > n*0.9 ? "GATE: PASS\n" : "GATE: FAIL\n");
    return 0;
}
