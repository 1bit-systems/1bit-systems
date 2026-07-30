// unified_pool.h — Unified model memory pool for GPU+NPU.

#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

struct ModelSlot {
    std::string name;
    std::string path;
    void* mmap_data = nullptr;
    size_t mmap_size = 0;

    struct GpuWeights {
        float *embed = nullptr, *final_norm = nullptr, *output = nullptr;
        struct Layer { float *wq, *wk, *wv, *wo, *w1, *w2, *w3, *pre_norm, *post_norm; };
        std::vector<Layer> layers;
    };
    std::unique_ptr<GpuWeights> gpu;

    int H = 0, NC = 0, NH = 0, NKV = 0, HD = 0, IM = 0, V = 0;
    int quant = 0;
    int refcount = 0;
};

class UnifiedModelPool {
public:
    ~UnifiedModelPool();
    int load(const std::string& path);
    ModelSlot* get(int slot);
    ModelSlot* find(const std::string& name);
    bool unload(int slot);
    int count() const;
    void report() const;
private:
    std::vector<ModelSlot> slots_;
};
