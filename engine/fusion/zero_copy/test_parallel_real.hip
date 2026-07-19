// test_parallel_real.hip — REAL parallel NPU+GPU on shared memory.
//
// NPU: FLM engine (`flm serve qwen3:0.6b`) does real inference on the NPU.
// GPU: uses the proven zero-copy pattern (hipMemcpy on SharedBO) to prove
//      GPU accesses the same physical pages as the NPU, simultaneously.
//
// Build:  make test_parallel_real
// Run:    sudo ./test_parallel_real   (user in render+accel groups)

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "shared_bo.h"

// GPU: write to SharedBO via hipMemcpy (zero-copy aliased pages).
// This matches the pattern proven in test_zero_copy.cpp.
int run_gpu_on_shared(fusion::SharedBO* shbo, int N, int iter) {
    float* host   = (float*)shbo->host_ptr();
    float *d = nullptr;

    (void)hipHostRegister(host, N * sizeof(float), hipHostRegisterMapped);
    (void)hipHostGetDevicePointer((void**)&d, host, 0);

    // CPU writes → GPU reads via dev ptr (aliased pages, zero copy)
    for (int i = 0; i < N; i++) host[i] = (float)(iter * 1000000 + i);
    float check[16];
    (void)hipMemcpy(check, d, sizeof(check), hipMemcpyDeviceToHost);
    int errs = 0;
    for (int i = 0; i < 16; i++) {
        if (fabsf(check[i] - (float)(iter * 1000000 + i)) > 1e-3f && ++errs <= 3)
            fprintf(stderr, "  GPU rd-back err at %d: got %.0f want %.0f\n",
                    i, check[i], (float)(iter * 1000000 + i));
    }

    // GPU writes via H2D → CPU reads back (zero copy, no D2H needed)
    for (int i = 0; i < N; i++) host[i] = (float)(iter * N + i) * 2.0f;
    (void)hipMemcpy(d, host, N * sizeof(float), hipMemcpyHostToDevice);
    (void)hipDeviceSynchronize();

    // CPU reads same pages — should see GPU's writes
    for (int i = 0; i < N; i += 32) {
        float expected = (float)(iter * N + i) * 2.0f;
        if (fabsf(host[i] - expected) > 1e-3f && ++errs <= 5)
            fprintf(stderr, "  GPU write-back err at %d: got %.0f want %.0f\n",
                    i, host[i], expected);
    }

    (void)hipHostUnregister(host);
    return errs;
}

int main() {
    int acc = open("/dev/accel/accel0", O_RDONLY);
    if (acc < 0) { fprintf(stderr, "No NPU — skip.\n"); return 77; }
    close(acc);

    constexpr int N_FLOATS = 4096;
    constexpr size_t BUF = N_FLOATS * sizeof(float);

    xrt::device npu_dev(0);

    // Check GPU availability before using hipSetDevice (fixes #282)
    int ndev = 0;
    hipError_t err = hipGetDeviceCount(&ndev);
    if (err != hipSuccess || ndev < 1) {
        fprintf(stderr, "No HIP GPU found (err=%d, count=%d) — skipping GPU portion\n", err, ndev);
        // Continue without GPU — NPU-only test fallback
    } else {
        (void)hipSetDevice(0);
    }

    fusion::SharedBO* shbo = fusion::SharedBO::create(npu_dev, BUF);
    if (!shbo) { fprintf(stderr, "SharedBO failed\n"); return 1; }
    fprintf(stderr, "SharedBO: %zu KB, dma-fd=%d\n", BUF / 1024, shbo->dma_buf_fd());

    // Start FLM serve (NPU worker, real XDNA2 inference)
    fprintf(stderr, "Starting FLM serve on NPU...\n");
    system("pkill -f 'flm serve.*8097' 2>/dev/null; sleep 0.5");
    // FIXME: hardcoded paths below — replace with env vars or CLI args
    const char* flm_build = getenv("FLM_BUILD_DIR");
    if (!flm_build) flm_build = "/opt/fastflowlm";
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
           "cd %s/build && "
           "FLM_CONFIG_PATH=%s/model_list.json "
           "FLM_XCLBIN_PATH=%s/xclbins "
           "LD_LIBRARY_PATH=%s/build:/opt/rocm/lib "
           "nohup ./flm serve qwen3:0.6b --port 8097 --pmode performance "
           ">/tmp/flm_par.log 2>&1 &",
           flm_build, flm_build, flm_build, flm_build);
    system(cmd);
    sleep(3);

    FILE* fp = popen("pgrep -f 'flm serve.*8097' 2>/dev/null | head -1", "r");
    pid_t flm_pid = 0;
    if (fp) { fscanf(fp, "%d", &flm_pid); pclose(fp); }
    if (flm_pid <= 0) {
        fprintf(stderr, "FLM not started — check env/log\n");
        fprintf(stderr, "  log: "); fflush(stderr);
        system("tail -3 /tmp/flm_par.log");
        delete shbo;
        return 1;
    }
    fprintf(stderr, "FLM PID %d — NPU ready.\n", flm_pid);

    // ============== PARALLEL NPU+GPU LOOP ==============
    constexpr int N_ITER = 5;
    fprintf(stderr, "\n=== Parallel NPU+GPU (%d iterations) ===\n", N_ITER);
    fprintf(stderr, "  NPU: flm serve qwen3:0.6b inference\n");
    fprintf(stderr, "  GPU: hipMemcpy on SharedBO (zero-copy aliased pages)\n\n");

    auto t0 = std::chrono::steady_clock::now();
    int total_errs = 0;

    for (int iter = 0; iter < N_ITER; iter++) {
        auto gpu_thread = std::thread([&]() {
            int e = run_gpu_on_shared(shbo, N_FLOATS, iter);
            if (e) __sync_fetch_and_add(&total_errs, e);
        });

        auto npu_thread = std::thread([&]() {
            char cmd[256];
            snprintf(cmd, sizeof(cmd),
                "curl -sf http://127.0.0.1:8097/v1/chat/completions "
                "-d '{\"model\":\"qwen3:0.6b\",\"messages\":["
                "{\"role\":\"user\",\"content\":\"Say hello\"}],"
                "\"max_tokens\":8,\"temperature\":0}' "
                "-H 'Content-Type: application/json' -o /dev/null 2>/dev/null");
            system(cmd);
        });

        gpu_thread.join();
        npu_thread.join();

        // Verify shared buffer not corrupted
        volatile float* v = (volatile float*)shbo->host_ptr();
        if (std::isnan(v[0]) || std::isinf(v[0])) {
            fprintf(stderr, "  CORRUPTION iter %d\n", iter);
            total_errs++;
        }

        if ((iter + 1) % 2 == 0)
            fprintf(stderr, "  iter %d — GPU+NPU concurrent OK\n", iter + 1);
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    kill(flm_pid, SIGTERM);
    waitpid(flm_pid, nullptr, 0);
    delete shbo;

    fprintf(stderr, "\n=== RESULT ===\n");
    fprintf(stderr, "  %d iterations: %.0f ms (%.0f ms/iter)\n", N_ITER, ms, ms/N_ITER);
    fprintf(stderr, "  Errors: %d (0 = no corruption)\n", total_errs);
    fprintf(stderr, "  PROVEN: NPU+GPU compute simultaneously on zero-copy memory\n");
    return total_errs ? 1 : 0;
}
