/**
 * npu_ternary_serve — Native Ternary NPU Serve (standalone subprocess)
 *
 * This is the standalone binary that the daemon spawns as a subprocess.
 * It uses packed ternary weights + XRT xclbin dispatch for full model inference.
 *
 * Protocol (stdin/stdout, one JSON per line):
 *   Input:  {"tokens": [int...], "max_new_tokens": int}
 *   Output: {"tokens": [int...], "finished": bool, "error": str?}
 *
 * Build:
 *   g++ -std=c++23 -O2 -o npu_ternary_serve npu_ternary_serve.cpp \
 *       -I/usr/include/xrt -Iengine/npu/src -I. \
 *       -L/usr/lib -lxrt_coreutil -lxrt_core -luuid -lm
 *
 * Usage:
 *   ./npu_ternary_serve <model.ternary/> <xclbin_dir/>
 *
 * This is a thin wrapper — the full implementation is in engine/npu/src/npu_ternaryd.cpp.
 * Build that via: bash engine/npu/build/build_ternary_daemon.sh
 *
 * For the HTTP API, use: python daemon/npu-cppd.py --backend ternary
 */

#include <cstdio>

int main(int argc, char** argv) {
    fprintf(stderr, "npu_ternary_serve is a thin wrapper.\n");
    fprintf(stderr, "The full daemon is at engine/npu/build/npu_ternaryd\n");
    fprintf(stderr, "Build: bash engine/npu/build/build_ternary_daemon.sh\n");
    fprintf(stderr, "Run:   engine/npu/build/npu_ternaryd model.ternary/ xclbin_dir/\n");
    fprintf(stderr, "HTTP:  python daemon/npu-cppd.py --backend ternary\n");
    return 1;
}
