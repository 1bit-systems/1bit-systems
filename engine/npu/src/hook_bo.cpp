/**
 * LD_PRELOAD — capture FLM's xrt::bo allocations for fused engine submission.
 *
 * Intercepts xrt::bo::bo(device, size, flags, group_id) to capture BO handles,
 * then runs fused layer instructions using captured BOs + FLM's instruction generators.
 *
 * Build:
 *   g++ -shared -fPIC -O2 -o hook_bo.so hook_bo.cpp -ldl
 *
 * Run:
 *   LD_PRELOAD=./hook_bo.so flm serve qwen3:0.6b --port 52628
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <dlfcn.h>

static FILE* logf = nullptr;
static std::mutex mtx;
static bool capturing = true;

// Captured BOs: (handle, size, group_id)
struct BOInfo {
    void* impl;      // xrt::bo internals
    size_t size;
    int group_id;
};
static std::vector<BOInfo> captured_bos;
static std::set<void*> seen_impls; // dedup

// Capture a newly constructed xrt::bo
static void capture_bo(void* self, size_t size, int flags, unsigned int group_id) {
    if (!capturing) return;
    std::lock_guard<std::mutex> lk(mtx);
    if (!logf) { logf = fopen("/tmp/hook_bo.log", "w"); setbuf(logf, NULL); }
    void* impl = *(void**)self;
    if (seen_impls.find(impl) != seen_impls.end()) return;
    seen_impls.insert(impl);
    captured_bos.push_back({impl, size, (int)group_id});
    fprintf(logf, "[hook] bo ctor: self=%p impl=%p size=%zu(%.1fMB) flags=%d gid=%d\n",
            self, impl, size, size/1048576.0, flags, group_id);
    fflush(logf);
}

// Intercept xrt::bo::bo(device, size, group_id) — no flags version
// _ZN3xrt2boC1ERKNS_6deviceEmj
typedef void (*bo_ctor1_t)(void*, const void*, size_t, unsigned int);
static bo_ctor1_t real_bo1 = nullptr;
extern "C" void _ZN3xrt2boC1ERKNS_6deviceEmj(
    void* self, const void* dev, size_t sz, unsigned int gid)
{
    if (!real_bo1) real_bo1 = (bo_ctor1_t)dlsym(RTLD_NEXT, "_ZN3xrt2boC1ERKNS_6deviceEmj");
    real_bo1(self, dev, sz, gid);
    capture_bo(self, sz, 0, gid);
}

// Intercept xrt::bo::bo(device, size, flags, group_id) — with flags
// _ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj
typedef void (*bo_ctor2_t)(void*, const void*, size_t, int, unsigned int);
static bo_ctor2_t real_bo2 = nullptr;
extern "C" void _ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj(
    void* self, const void* dev, size_t sz, int flags, unsigned int gid)
{
    if (!real_bo2) real_bo2 = (bo_ctor2_t)dlsym(RTLD_NEXT, "_ZN3xrt2boC1ERKNS_6deviceEmNS0_5flagsEj");
    real_bo2(self, dev, sz, flags, gid);
    capture_bo(self, sz, flags, gid);
}

// Intercept xrt::bo::bo(hw_context, size, group_id) — FLM uses this for weight BOs
// _ZN3xrt2boC1ERKNS_10hw_contextEmj
typedef void (*bo_ctor3_t)(void*, const void*, size_t, unsigned int);
static bo_ctor3_t real_bo3 = nullptr;
extern "C" void _ZN3xrt2boC1ERKNS_10hw_contextEmj(
    void* self, const void* hwctx, size_t sz, unsigned int gid)
{
    if (!real_bo3) real_bo3 = (bo_ctor3_t)dlsym(RTLD_NEXT, "_ZN3xrt2boC1ERKNS_10hw_contextEmj");
    real_bo3(self, hwctx, sz, gid);
    capture_bo(self, sz, 0, gid);
}

// Intercept xrt::bo::bo(hw_context, size, flags, group_id)
// _ZN3xrt2boC1ERKNS_10hw_contextEmNS0_5flagsEj
typedef void (*bo_ctor4_t)(void*, const void*, size_t, int, unsigned int);
static bo_ctor4_t real_bo4 = nullptr;
extern "C" void _ZN3xrt2boC1ERKNS_10hw_contextEmNS0_5flagsEj(
    void* self, const void* hwctx, size_t sz, int flags, unsigned int gid)
{
    if (!real_bo4) real_bo4 = (bo_ctor4_t)dlsym(RTLD_NEXT, "_ZN3xrt2boC1ERKNS_10hw_contextEmNS0_5flagsEj");
    real_bo4(self, hwctx, sz, flags, gid);
    capture_bo(self, sz, flags, gid);
}

// ─── FLM uses xrt::ext::bo — Different class! ───
// _ZN3xrt3ext2boC1ERKNS_6deviceEm
typedef void (*ext_bo1_t)(void*, const void*, size_t);
static ext_bo1_t real_ext_bo1 = nullptr;
extern "C" void _ZN3xrt3ext2boC1ERKNS_6deviceEm(
    void* self, const void* dev, size_t sz)
{
    if (!real_ext_bo1) real_ext_bo1 = (ext_bo1_t)dlsym(RTLD_NEXT, "_ZN3xrt3ext2boC1ERKNS_6deviceEm");
    real_ext_bo1(self, dev, sz);
    capture_bo(self, sz, 0, 0);
}

// _ZN3xrt3ext2boC1ERKNS_10hw_contextEm
typedef void (*ext_bo2_t)(void*, const void*, size_t);
static ext_bo2_t real_ext_bo2 = nullptr;
extern "C" void _ZN3xrt3ext2boC1ERKNS_10hw_contextEm(
    void* self, const void* hwctx, size_t sz)
{
    if (!real_ext_bo2) real_ext_bo2 = (ext_bo2_t)dlsym(RTLD_NEXT, "_ZN3xrt3ext2boC1ERKNS_10hw_contextEm");
    real_ext_bo2(self, hwctx, sz);
    capture_bo(self, sz, 0, 0);
}

// ─── Also intercept xrt::bo move constructor (used when storing BOs in containers) ───
// _ZN3xrt2boC1EOS_  = xrt::bo::bo(xrt::bo&&)
typedef void (*bo_move_t)(void* self, void* other);
static bo_move_t real_bo_move = nullptr;

extern "C" void _ZN3xrt2boC1EOS_(void* self, void* other) {
    if (!real_bo_move)
        real_bo_move = (bo_move_t)dlsym(RTLD_NEXT, "_ZN3xrt2boC1EOS_");
    real_bo_move(self, other);
    
    // Track moved BOs too
    if (capturing) {
        void* impl = *(void**)self;
        std::lock_guard<std::mutex> lk(mtx);
        if (seen_impls.find(impl) == seen_impls.end()) {
            seen_impls.insert(impl);
            if (logf) fprintf(logf, "[hook] bo move: self=%p impl=%p (from=%p)\n", self, impl, other);
        }
    }
}

// ─── Intercept xrt::run::wait to measure kernel timing ───
// _ZN3xrt3run4waitEv
typedef void (*run_wait_t)(void*);
static run_wait_t real_run_wait = nullptr;

extern "C" void _ZN3xrt3run4waitEv(void* self) {
    if (!real_run_wait)
        real_run_wait = (run_wait_t)dlsym(RTLD_NEXT, "_ZN3xrt3run4waitEv");
    auto t0 = std::chrono::steady_clock::now();
    real_run_wait(self);
    auto ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    if (ms > 0.1) { // Only log meaningful waits
        std::lock_guard<std::mutex> lk(mtx);
        if (logf) fprintf(logf, "[hook] run::wait: %.3f ms\n", ms);
    }
}

// ─── Dump all captured BOs periodically ───
__attribute__((destructor))
void fini() {
    if (logf) {
        fprintf(logf, "\n=== Captured BOs (%zu total) ===\n", captured_bos.size());
        for (size_t i = 0; i < captured_bos.size(); i++) {
            fprintf(logf, "  BO[%zu]: impl=%p size=%zu(%.1fMB) gid=%d\n",
                    i, captured_bos[i].impl, captured_bos[i].size,
                    captured_bos[i].size/1048576.0, captured_bos[i].group_id);
        }
        fclose(logf);
    }
    printf("\n[Hook] Captured %zu BOs, logged to /tmp/hook_bo.log\n", captured_bos.size());
}

__attribute__((constructor))
void init() {
    printf("\n=== XRT BO Hook — capturing weight BOs ===\n");
    fflush(stdout);
}
