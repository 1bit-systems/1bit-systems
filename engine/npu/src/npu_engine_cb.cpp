/** NPU Engine v6 — FLM mm.xclbin, opcode=3, no instruction BO
 *  FLM's xclbin expects opcode=3 with instr=0, ninstr=0.
 *  Weights must be in tile SRAM (pre-loaded via DMA).
 *  This version passes weights as kernel args to see if FLM's 
 *  xclbin also supports traditional BO-based weight passing. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <cassert>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
extern "C" float* dequant_i8_to_float(const uint8_t*,int,int*,int*);
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static void transpose_pack(const float* s, int o, int i, float* d, int ds, int dof) {
    for (int a = 0; a < o; a++) for (int b = 0; b < i; b++) d[(size_t)b*ds+dof+a] = s[(size_t)a*i+b];
}
static inline float dynamic_ascale(const float* x, int n) {
    float am = 0; for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > am) am = a; }
    if (am < 1e-12f) am = 1.0f; return am / 127.0f;
}
static void matmul_ref(const float* A, const float* B, int M, int N, int K, float* C) {
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { float s = 0; for (int k = 0; k < K; k++) s += A[(size_t)m*K+k] * B[(size_t)k*N+n]; C[(size_t)m*N+n] = s; }
}
static void load_bin(const std::string& path, float* buf, size_t n) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { fprintf(stderr, "  FAIL: can't open %s\n", path.c_str()); exit(1); }
    size_t r = fread(buf, 4, n, f); fclose(f);
    if (r != n) fprintf(stderr, "  WARN: %s: read %zu/%zu floats\n", path.c_str(), r, n);
}
static void save_bin(const std::string& path, const float* buf, size_t n) {
    FILE* f = fopen(path.c_str(), "wb"); fwrite(buf, 4, n, f); fclose(f);
}
static float compare(const float* a, const float* b, int n) {
    double se = 0; float ae = 0; for (int i = 0; i < n; i++) { float d = a[i]-b[i]; float ad = fabsf(d); se += d*d; if (ad > ae) ae = ad; }
    return ae;
}
static int load_txt_i32(const std::string& path, int* buf, int n) {
    FILE* f = fopen(path.c_str(), "r"); if (!f) return -1;
    int k = 0; char line[256]; while (k < n && fgets(line, sizeof(line), f)) { buf[k++] = atoi(line); } fclose(f); return k;
}
struct GEMMSpec {
    int M, N, K, T, MP, NP, KP;
    int A_ch, A_cw, A_cd, B_ch, B_cw, B_cd, C_ch, C_cw, C_cd;
    bool is_a4, is_b4, use_mt, use_veco;
    uint8_t shift_a, shift_b;
    int mt_k, mt_n;
};
static GEMMSpec parse_instr(const std::string& path) {
    GEMMSpec s = {};
    std::ifstream f(path); std::string l;
    int ln = 0; while (std::getline(f, l)) {
        ln++;
        auto p = l.find('#'); if (p != std::string::npos) l = l.substr(0, p);
        while (!l.empty() && (l.back()=='\r'||l.back()==' '||l.back()=='\t')) l.pop_back();
        if (l.empty()) continue;
        char k[64], v[256];
        if (sscanf(l.c_str(), " %63[^=] = %255[^#\r\n] ", k, v) == 2) {
            std::string sk(k); while (!sk.empty() && sk.back()==' ') sk.pop_back();
            std::string sv(v); while (!sv.empty() && sv.back()==' ') sv.pop_back();
            if (sk == "M") s.M = atoi(sv.c_str());
            else if (sk == "N") s.N = atoi(sv.c_str());
            else if (sk == "K") s.K = atoi(sv.c_str());
            else if (sk == "T") s.T = atoi(sv.c_str());
            else if (sk == "MP") s.MP = atoi(sv.c_str());
            else if (sk == "NP") s.NP = atoi(sv.c_str());
            else if (sk == "KP") s.KP = atoi(sv.c_str());
            else if (sk == "A_ch") s.A_ch = atoi(sv.c_str());
            else if (sk == "A_cw") s.A_cw = atoi(sv.c_str());
            else if (sk == "A_cd") s.A_cd = atoi(sv.c_str());
            else if (sk == "B_ch") s.B_ch = atoi(sv.c_str());
            else if (sk == "B_cw") s.B_cw = atoi(sv.c_str());
            else if (sk == "B_cd") s.B_cd = atoi(sv.c_str());
            else if (sk == "C_ch") s.C_ch = atoi(sv.c_str());
            else if (sk == "C_cw") s.C_cw = atoi(sv.c_str());
            else if (sk == "C_cd") s.C_cd = atoi(sv.c_str());
            else if (sk == "is_a4") s.is_a4 = atoi(sv.c_str());
            else if (sk == "is_b4") s.is_b4 = atoi(sv.c_str());
            else if (sk == "use_mt") s.use_mt = atoi(sv.c_str());
            else if (sk == "use_veco") s.use_veco = atoi(sv.c_str());
            else if (sk == "shift_a") s.shift_a = (uint8_t)atoi(sv.c_str());
            else if (sk == "shift_b") s.shift_b = (uint8_t)atoi(sv.c_str());
            else if (sk == "mt_k") s.mt_k = atoi(sv.c_str());
            else if (sk == "mt_n") s.mt_n = atoi(sv.c_str());
        }
    }
    return s;
}
int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <xclbin> <instr.txt> <weights.bin> <out.bin>\n", argv[0]);
        fprintf(stderr, "  or:  %s <xclbin> <instr.txt> <weights.bin> <out.bin> <A.bin>\n");
        return 1;
    }
    std::string xclbin_path = argv[1];
    std::string instr_path  = argv[2];
    std::string weights_path= argv[3];
    std::string out_path    = argv[4];
    std::string A_path      = (argc > 5) ? argv[5] : "a_f32.bin";
    fprintf(stderr, "=== NPU Engine v6 (cb) ===\n");
    fprintf(stderr, "  xclbin:  %s\n", xclbin_path.c_str());
    fprintf(stderr, "  instr:   %s\n", instr_path.c_str());
    fprintf(stderr, "  weights: %s\n", weights_path.c_str());
    fprintf(stderr, "  A:       %s\n", A_path.c_str());
    fprintf(stderr, "  out:     %s\n", out_path.c_str());
    GEMMSpec spec = parse_instr(instr_path);
    fprintf(stderr, "\n  GEMM Spec: M=%d N=%d K=%d T=%d  MP=%d NP=%d KP=%d\n", spec.M, spec.N, spec.K, spec.T, spec.MP, spec.NP, spec.KP);
    fprintf(stderr, "  A: ch=%d cw=%d cd=%d is_a4=%d\n", spec.A_ch, spec.A_cw, spec.A_cd, spec.is_a4);
    fprintf(stderr, "  B: ch=%d cw=%d cd=%d is_b4=%d  shift=%u\n", spec.B_ch, spec.B_cw, spec.B_cd, spec.is_b4, spec.shift_b);
    fprintf(stderr, "  C: ch=%d cw=%d cd=%d\n", spec.C_ch, spec.C_cw, spec.C_cd);
    fprintf(stderr, "  use_mt=%d use_veco=%d mt_k=%d mt_n=%d\n", spec.use_mt, spec.use_veco, spec.mt_k, spec.mt_n);
    size_t A_sz = (size_t)spec.M * spec.K;
    size_t B_sz = (size_t)spec.K * spec.N;
    size_t C_sz = (size_t)spec.M * spec.N;
    std::vector<float> A(A_sz), B(B_sz), C(C_sz, 0);
    load_bin(A_path, A.data(), A_sz);
    std::vector<float> B_flat(B_sz);
    load_bin(weights_path, B_flat.data(), B_sz);
    std::vector<float> C_ref(C_sz);
    matmul_ref(A.data(), B_flat.data(), spec.M, spec.N, spec.K, C_ref.data());
    fprintf(stderr, "\n  CPU ref computed.\n");
    auto t0 = std::chrono::steady_clock::now();
    unsigned int dev_idx = 0;
    xrt::device device = xrt::device(dev_idx);
    xrt::uuid xclbin_uuid;
    try {
        auto xclbin = xrt::xclbin(xclbin_path);
        xclbin_uuid = device.register_xclbin(xclbin);
    } catch (const std::exception& e) {
        fprintf(stderr, "  FAIL: xclbin load: %s\n", e.what());
        try { xclbin_uuid = device.get_xclbin_uuid(); } catch (...) {}
    }
    fprintf(stderr, "  xclbin UUID: %s\n", xclbin_uuid.to_string().c_str());
    xrt::kernel krnl = xrt::kernel(device, xclbin_uuid, "MLIR_AIE");
    int M = spec.M, N = spec.N, K = spec.K;
    size_t A_bytes = (size_t)M * K * 4;
    size_t B_bytes = (size_t)K * N * 4;
    size_t C_bytes = (size_t)M * N * 4;
    xrt::bo A_bo = xrt::bo(device, A_bytes, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(1));
    xrt::bo B_bo = xrt::bo(device, B_bytes, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(2));
    xrt::bo C_bo = xrt::bo(device, C_bytes, XRT_BO_FLAGS_HOST_ONLY, krnl.group_id(3));
    A_bo.write(A.data());
    B_bo.write(B_flat.data());
    A_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    B_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fprintf(stderr, "  BOs written and synced.\n");
    auto run = xrt::run(krnl);
    run.set_arg(0, 0);          // instr_bo (null — using default FLM instruction)
    run.set_arg(1, A_bo);       // A
    run.set_arg(2, B_bo);       // B
    run.set_arg(3, C_bo);       // C
    run.set_arg(4, M);          // M
    run.set_arg(5, N);          // N
    run.set_arg(6, K);          // K
    run.start();
    run.wait();
    auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    C_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    C_bo.read(C.data());
    save_bin(out_path, C.data(), C_sz);
    float ae = compare(C.data(), C_ref.data(), (int)C_sz);
    fprintf(stderr, "  NPU result vs CPU ref: max_abs=%.4f  (%.1f ms)\n", ae, ms);
    fprintf(stderr, "  Output saved to %s\n", out_path.c_str());
    return 0;
}
