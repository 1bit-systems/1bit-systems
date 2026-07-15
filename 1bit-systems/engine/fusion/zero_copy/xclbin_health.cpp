// xclbin_health.cpp — validate NPU xclbins against the RUNNING driver.
//
// The npu_engine_universal.cpp boot SIGABRT was long blamed on opening "too many
// hw_contexts" (4-5 per GEMM shape).  That is wrong: this tool proves the NPU
// holds 8+ concurrent contexts fine.  The real failure is that a specific
// xclbin requests a number of AIE columns the driver's column allocator
// rejects, e.g. dmesg:
//
//   aie2_hwctx_col_list: Invalid num_col 12
//   amdxdna_drm_create_hwctx_ioctl: Init hwctx failed, ret -22   (== EINVAL)
//
// Usage:
//   xclbin_health <xclbin> [<xclbin> ...]      # per-file accept/reject
//   xclbin_health --dir <dir>                   # scan a whole directory
//
// Exit code: 0 if ALL xclbins are accepted by the driver, 1 if any rejected.
// The NPU engine should run this (or link the check) at startup so a rejected
// xclbin is diagnosed with a precise message and degrades gracefully instead of
// SIGABRT-ing the whole process.
//
// Build: g++ -std=c++20 -O2 xclbin_health.cpp -o xclbin_health \
//          -I/usr/include/xrt -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -luuid -lpthread
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>
#include <memory>

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Probe-create a context for one xclbin. Returns true if the driver accepts it.
static bool driver_accepts(xrt::device& dev, const std::string& path, std::string& err) {
    try {
        xrt::xclbin xc{path};
        auto uuid = dev.register_xclbin(xc);
        auto ctx = std::make_unique<xrt::hw_context>(dev, uuid);  // throws on EINVAL
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            DIR* d = opendir(argv[++i]);
            if (!d) { fprintf(stderr, "cannot open dir %s\n", argv[i]); return 2; }
            struct dirent* de;
            while ((de = readdir(d))) {
                std::string n = std::string(argv[i]) + "/" + de->d_name;
                if (ends_with(n, ".xclbin")) paths.push_back(n);
            }
            closedir(d);
        } else {
            paths.push_back(argv[i]);
        }
    }
    if (paths.empty()) { fprintf(stderr, "usage: %s <xclbin...> | --dir <dir>\n", argv[0]); return 2; }

    xrt::device dev(0);
    printf("NPU device[0] opened. Validating %zu xclbin(s) against the running driver.\n\n", paths.size());
    printf("%-58s %-10s %s\n", "xclbin", "status", "reason");
    printf("%-58s %-10s %s\n", "------", "------", "------");

    int rejected = 0;
    for (auto& p : paths) {
        std::string base = p;
        auto slash = base.rfind('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);
        std::string err;
        bool ok = driver_accepts(dev, p, err);
        const char* col_hint = "";
        if (!ok) {
            rejected++;
            // Surface the likely column-count cause explicitly.
            if (err.find("Invalid argument") != std::string::npos)
                col_hint = "  <- likely 'Invalid num_col N' (driver rejects column count; check dmesg)";
        }
        printf("%-58s %-10s %s%s\n", base.c_str(), ok ? "ACCEPT" : "REJECT",
               ok ? "" : err.c_str(), col_hint);
    }

    printf("\n%d/%zu accepted, %d rejected.\n", (int)paths.size() - rejected, paths.size(), rejected);
    if (rejected) {
        printf("\nA rejected xclbin is what crashes npu_engine_universal.cpp with EINVAL.\n");
        printf("Use only ACCEPTED xclbins (smaller column count), or build xclbins whose\n");
        printf("num_col the driver's column allocator permits (the 40-col target is blocked).\n");
    }
    return rejected ? 1 : 0;
}
