// unified_pool.cpp — Unified model memory pool (no GPU deps).
// GPU upload lives in backend_hip_1bp.cpp via upload_to_gpu().

#include "unified_pool.h"
#include "../engine/npu/src/onebp_loader.cpp"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

UnifiedModelPool::~UnifiedModelPool() {
    for (int i = (int)slots_.size() - 1; i >= 0; i--)
        unload(i);
}

int UnifiedModelPool::load(const std::string& path) {
    for (auto& s : slots_)
        if (s.path == path) { s.refcount++; return (int)(&s - &slots_[0]); }

    printf("[pool] Loading: %s\n", path.c_str());

    OnebpModel model;
    if (!model.open(path.c_str())) return -1;

    auto& hdr = model.header();
    ModelSlot slot;
    slot.name = path.substr(path.find_last_of('/') + 1);
    auto dot = slot.name.find_last_of('.');
    if (dot != std::string::npos) slot.name = slot.name.substr(0, dot);
    slot.path = path;
    slot.H = hdr.hidden_size; slot.NC = hdr.num_layers;
    slot.NH = hdr.num_attention_heads; slot.NKV = hdr.num_kv_heads;
    slot.HD = hdr.head_dim; slot.IM = hdr.intermediate_size;
    slot.V = hdr.vocab_size; slot.quant = hdr.quant;
    slot.refcount = 1;

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct stat st; fstat(fd, &st);
    slot.mmap_size = st.st_size;
    slot.mmap_data = mmap(NULL, slot.mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (slot.mmap_data == MAP_FAILED) { slot.mmap_data = nullptr; return -1; }

    slots_.push_back(std::move(slot));
    auto& s = slots_.back();
    printf("[pool]   %s: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d quant=%u %.0fMB\n",
           s.name.c_str(), s.H, s.NC, s.NH, s.NKV, s.HD, s.IM, s.V, s.quant,
           (double)s.mmap_size / 1024 / 1024);
    return (int)slots_.size() - 1;
}

ModelSlot* UnifiedModelPool::find(const std::string& name) {
    for (auto& s : slots_)
        if (s.name == name) return &s;
    return nullptr;
}

bool UnifiedModelPool::unload(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return false;
    auto& s = slots_[slot];
    if (--s.refcount > 0) return true;
    printf("[pool] Unloading: %s\n", s.name.c_str());
    s.gpu.reset();
    if (s.mmap_data) munmap(s.mmap_data, s.mmap_size);
    s.mmap_data = nullptr;
    return true;
}

int UnifiedModelPool::count() const { return (int)slots_.size(); }

void UnifiedModelPool::report() const {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║     Unified Model Pool                   ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  %d model(s) loaded\n\n", (int)slots_.size());
    for (auto& s : slots_) {
        printf("  %-25s %s\n", s.name.c_str(), s.gpu ? "✅ GPU" : "  mmap");
        printf("  %-25s H=%d NC=%d NH=%d IM=%d V=%d\n", "",
               s.H, s.NC, s.NH, s.IM, s.V);
    }
    printf("\n");
}
