// probe_contexts.cpp — how many concurrent xrt::hw_context does this NPU take?
//
// Opens 1..N hardware contexts (each registering + creating a context from the
// same xclbin) and reports which succeed.  This directly tests whether the
// 4-5 contexts opened by npu_engine_universal.cpp (one per GEMM shape) are the
// real cause of the EINVAL / context-exhaustion failure, or whether something
// else is to blame.
//
// Build: g++ -std=c++20 -O2 probe_contexts.cpp -o probe_contexts \
//          -I/usr/include/xrt -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -luuid -lpthread
// Run:   sudo ./probe_contexts <xclbin> [max_contexts]
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cstdio>
#include <vector>
#include <memory>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <xclbin> [max=8]\n", argv[0]); return 1; }
    const char* xp = argv[1];
    int maxc = (argc > 2) ? atoi(argv[2]) : 8;

    xrt::device dev(0);
    printf("device[0] opened; probing up to %d hw_contexts with %s\n", maxc, xp);

    xrt::xclbin xc{std::string(xp)};
    dev.register_xclbin(xc);
    auto uuid = xc.get_uuid();

    std::vector<std::unique_ptr<xrt::hw_context>> ctxs;
    for (int i = 1; i <= maxc; i++) {
        try {
            ctxs.push_back(std::make_unique<xrt::hw_context>(dev, uuid));
            printf("  ctx[%d]: OK   (live=%d)\n", i, (int)ctxs.size());
        } catch (const std::exception& e) {
            printf("  ctx[%d]: FAIL — %s   (live=%d)\n", i, e.what(), (int)ctxs.size());
            printf("\n>>> NPU accepts %d concurrent hw_context(s) before failure.\n", (int)ctxs.size());
            return 0;
        }
    }
    printf("\n>>> NPU accepted all %d contexts (no failure). Limit > %d.\n", maxc, maxc);
    return 0;
}
