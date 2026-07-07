/**
 * FLM BO Capture v3 — works by intercepting xrt::bo::sync (always called).
 *
 * Build: g++ -shared -fPIC -std=c++23 -O2 -o flm_bo_capture.so flm_bo_capture.cpp -ldl
 * Run:   LD_PRELOAD=./flm_bo_capture.so flm serve qwen3:0.6b
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <dlfcn.h>
#include <atomic>
#include <unistd.h>

struct CapBO { void* self; void* impl; size_t sz; int dir; };
static std::vector<CapBO> g_bos;
static std::mutex g_mtx;
static void* g_dev = nullptr;
static std::atomic<int> g_syncs{0};
static std::atomic<bool> g_ready{false};

extern "C" {
    void*  flm_dev()          { return g_dev; }
    int    flm_bo_cnt()       { std::lock_guard<std::mutex> l(mtx); return (int)g_bos.size(); }
    int    flm_ok()           { return g_ready ? 1 : 0; }
}

// ─── npu_app_manager ctor → capture device ───
typedef void (*app_mgr_t)(void*,int,void*,const std::string&,bool);
static app_mgr_t _app_mgr;
extern "C" void _ZN15npu_app_managerC1E10npu_devicePN3xrt6deviceENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb(
    void* s, int n, void* d, const std::string& x, bool f)
{
    g_dev = d;
    fprintf(stderr, "[capture] app_mgr dev=%p xclbin=%s\n", d, x.c_str());
    if (!_app_mgr) {
        void* lib = dlopen("/opt/fastflowlm/lib/libqwen3_npu.so", RTLD_LAZY|RTLD_NOLOAD);
        if (lib) { _app_mgr = (app_mgr_t)dlsym(lib, __func__); dlclose(lib); }
    }
    if (_app_mgr) _app_mgr(s, n, d, x, f);
}

// ─── xrt::bo::sync → capture BO info ───
typedef void (*sync_t)(void*,int,size_t,size_t);
static sync_t _sync;
extern "C" void _ZN3xrt2bo4syncE18xclBOSyncDirectionmm(
    void* s, int dir, size_t off, size_t sz)
{
    if (!_sync) {
        void* lib = dlopen("libxrt_coreutil.so.2", RTLD_LAZY|RTLD_NOLOAD);
        if (lib) { _sync = (sync_t)dlsym(lib, __func__); dlclose(lib); }
    }
    if (_sync) _sync(s, dir, off, sz);

    if (sz > 4096) {
        g_syncs++;
        void* impl = *(void**)s;
        std::lock_guard<std::mutex> l(g_mtx);
        bool found = false;
        for (auto& b : g_bos) { if (b.self == s) { b.sz = sz; b.dir = dir; found = true; break; } }
        if (!found) g_bos.push_back({s, impl, sz, dir});
        if (g_syncs == 8) {
            g_ready = true;
            fprintf(stderr, "[capture] READY: %d syncs, %zu BOs, dev=%p\n",
                g_syncs.load(), g_bos.size(), g_dev);
        }
    }
}

__attribute__((constructor)) static void _init() {
    fprintf(stderr, "\n[capture-v3] pid=%d\n", getpid());
}
__attribute__((destructor)) static void _fini() {
    fprintf(stderr, "[capture-v3] done: dev=%p BOs=%zu syncs=%d\n",
        g_dev, g_bos.size(), g_syncs.load());
}
