// probe_multi_xclbin.cpp — open one hw_context per DISTINCT xclbin, mimicking
// npu_engine_universal.cpp (which loads QKV/O/GU/D xclbins separately).
// Tests whether mixing different xclbins — not raw context count — triggers the
// CREATE_HWCTX EINVAL the 1bit engine hits.
//
// Build: g++ -std=c++20 -O2 probe_multi_xclbin.cpp -o probe_multi_xclbin \
//          -I/usr/include/xrt -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -luuid -lpthread
// Run:   sudo ./probe_multi_xclbin <xc1> <xc2> [<xc3> ...]
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cstdio>
#include <vector>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <xc1> [<xc2> ...]\n", argv[0]); return 1; }
    xrt::device dev(0);
    printf("device[0] opened; opening one hw_context per distinct xclbin (%d given)\n", argc - 1);

    std::vector<std::unique_ptr<xrt::hw_context>> ctxs;
    for (int i = 1; i < argc; i++) {
        try {
            xrt::xclbin xc{std::string(argv[i])};
            auto uuid = dev.register_xclbin(xc);   // returns the xclbin uuid
            ctxs.push_back(std::make_unique<xrt::hw_context>(dev, uuid));
            printf("  %-60s ctx OK (live=%d)\n", argv[i], (int)ctxs.size());
        } catch (const std::exception& e) {
            printf("  %-60s ctx FAIL — %s\n", argv[i], e.what());
        }
    }
    printf("\n>>> %d/%d distinct-xclbin contexts live.\n", (int)ctxs.size(), argc - 1);
    return 0;
}
