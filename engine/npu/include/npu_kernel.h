#pragma once
/// NPU XRT xclbin kernel manager.
/// Manages one xclbin kernel (e.g., QKV, O, GU, D) and its associated buffer objects.
/// Ported from engine/npu/src/npu_kernels.zig and engine/npu/src/npu_engine.zig
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include "ternary_npu_bridge.h"
#include "xrt_wrapper.h"

// ─── NPU Engine Config (from npu_engine.zig) ─────────────────────

struct NpuConfig {
    uint32_t NC = 28;       // number of layers
    uint32_t H = 2048;      // hidden dim
    uint32_t NH = 16;       // num query heads
    uint32_t NKV = 8;       // num KV heads
    uint32_t HD = 64;       // head dim
    uint32_t vocab_size = 151936;
    uint32_t max_seq_len = 4096;
};

// ─── XclbinKernel ────────────────────────────────────────────────

/// Manages one NPU xclbin kernel instance with its instruction, activation,
/// weight, and output buffer objects.
class XclbinKernel {
public:
    XclbinKernel() = default;

    /// Initialize the kernel: load xclbin, create hardware context, open kernel,
    /// allocate buffer objects, and load instructions.
    void load(XrtDevice& device,
              const std::string& xclbin_path,
              const std::string& insts_path,
              uint32_t md, uint32_t kd, uint32_t nd,
              uint32_t n_layers, uint32_t weight_group,
              const std::string& kernel_name = "MLIR_AIE")
    {
        name_ = kernel_name;
        md_ = md;
        kd_ = kd;
        nd_ = nd;
        n_layers_ = n_layers;

        // Load xclbin and get UUID
        uuid_ = device.loadXclbin(xclbin_path);
        printf("Loaded xclbin %s\n", name_.c_str());

        // Create HW context
        hw_context_ = XrtHwContext(device.createHwContext(uuid_));

        // Open kernel
        kernel_ = XrtKernel(device.createKernel(uuid_, kernel_name.c_str()));

        // Read instructions file
        auto instrs = readInstructionsFile(insts_path);
        instr_count_ = (int)instrs.size();

        // Allocate instruction BO (cacheable, group 1)
        bo_instr_ = XrtBuffer(device.allocBO(instrs.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, 1));

        // Map and copy instructions
        uint8_t* instr_mem = bo_instr_.map(instrs.size() * sizeof(uint32_t));
        instr_mapped_ = reinterpret_cast<uint32_t*>(instr_mem);
        std::memcpy(instr_mapped_, instrs.data(), instrs.size() * sizeof(uint32_t));
        bo_instr_.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, instrs.size() * sizeof(uint32_t));

        // Allocate activation BO (host_only, group 3)
        // md * kd bytes (int8_t)
        size_t act_size = (size_t)md * kd;
        bo_act_ = XrtBuffer(device.allocBO(act_size, XRT_BO_FLAGS_HOST_ONLY, 3));
        uint8_t* act_mem = bo_act_.map(act_size);
        act_mapped_ = reinterpret_cast<int8_t*>(act_mem);

        // Allocate output BO (host_only, group 5)
        // md * nd * 2 bytes (int16_t)
        size_t out_size = (size_t)md * nd * 2;
        bo_out_ = XrtBuffer(device.allocBO(out_size, XRT_BO_FLAGS_HOST_ONLY, 5));
        uint8_t* out_mem = bo_out_.map(out_size);
        out_mapped_ = reinterpret_cast<int16_t*>(out_mem);

        // Allocate per-layer weight BOs (host_only, weight_group)
        layer_bos_.reserve(n_layers);
        size_t weight_size = (size_t)kd * nd;
        for (uint32_t i = 0; i < n_layers; ++i) {
            layer_bos_.emplace_back(device.allocBO(weight_size, XRT_BO_FLAGS_HOST_ONLY, weight_group));
        }

        initialized_ = true;
        printf("Kernel %s initialized: md=%u kd=%u nd=%u layers=%u instrs=%zu\n",
               name_.c_str(), md, kd, nd, n_layers, instrs.size());
    }

    /// INT8 quantize and pack a weight matrix into the layer's weight BO.
    /// weights is [k * n] row-major floats.
    /// Returns the quantization scale (for dequant on output).
    float packWeight(uint32_t layer, const float* weights, uint32_t k, uint32_t n) {
        if (layer >= n_layers_) throw std::runtime_error("Layer index out of bounds");

        // Find max absolute value
        float amax = 0.0f;
        size_t total = (size_t)k * n;
        for (size_t i = 0; i < total; ++i) {
            if (std::isfinite(weights[i])) {
                float a = std::abs(weights[i]);
                if (a > amax) amax = a;
            }
        }
        if (amax < 1e-12f) amax = 1.0f;

        float scale = amax / 127.0f;
        float inv_scale = 127.0f / amax;

        // Map the weight BO
        uint8_t* mem = layer_bos_[layer].map(total);
        int8_t* dst = reinterpret_cast<int8_t*>(mem);

        for (size_t i = 0; i < total; ++i) {
            float v = weights[i];
            if (!std::isfinite(v)) v = 0.0f;
            int32_t q = (int32_t)(v * inv_scale + 0.5f);
            q = std::max(-127, std::min(127, q));
            dst[i] = (int8_t)q;
        }

        // Sync to device
        layer_bos_[layer].sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, total);

        return scale;
    }

    /// Run the kernel for a given layer:
    /// 1. INT8-quantize activations, upload to act_bo
    /// 2. Launch kernel on the NPU
    /// 3. Download and dequantize output
    /// activations: [am * ak] f32
    /// output: [am * an] f32
    void run(uint32_t layer, const float* activations, uint32_t am, uint32_t ak,
             float ascale, float bscale, float* output, uint32_t an) {
        if (!initialized_) throw std::runtime_error("Kernel not initialized");
        if (layer >= n_layers_) throw std::runtime_error("Layer index out of bounds");

        // Quantize activations to INT8
        float ais = 1.0f / ascale;
        size_t act_size = (size_t)am * kd_;
        std::memset(act_mapped_, 0, act_size);

        for (uint32_t m = 0; m < am; ++m) {
            for (uint32_t k = 0; k < ak; ++k) {
                float v = activations[m * ak + k];
                if (!std::isfinite(v)) v = 0.0f;
                int32_t q = (int32_t)(v * ais + 0.5f);
                q = std::max(-127, std::min(127, q));
                act_mapped_[m * kd_ + k] = (int8_t)q;
            }
        }

        // Sync activation BO to device
        bo_act_.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, act_size);

        // Ensure weight BO is synced
        size_t weight_size = (size_t)kd_ * nd_;
        layer_bos_[layer].sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, weight_size);

        // Launch kernel: run(3, instr_bo, instr_count, act_bo, weight_bo, out_bo)
        XrtRun run_handle(kernel_.run(bo_instr_.handle(), instr_count_,
                                       bo_act_.handle(), layer_bos_[layer].handle(),
                                       bo_out_.handle()));
        run_handle.wait();

        // Sync output BO from device
        size_t out_size = (size_t)am * nd_ * 2;
        bo_out_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, out_size);

        // Dequantize: out[i] = Cm[i] * ascale * bscale
        float cs = ascale * bscale;
        for (uint32_t m = 0; m < am; ++m) {
            for (uint32_t n = 0; n < an; ++n) {
                float val = (float)out_mapped_[m * nd_ + n] * cs;
                output[m * an + n] = std::isfinite(val) ? val : 0.0f;
            }
        }
    }

    /// Free all resources.
    void deinit() {
        // Free layer weight BOs
        layer_bos_.clear();

        // Free output BO
        bo_out_.free();
        // Free activation BO
        bo_act_.free();
        // Free instruction BO
        bo_instr_.free();

        // Close kernel
        kernel_.close();

        // Destroy HW context
        hw_context_.destroy();

        initialized_ = false;
    }

    ~XclbinKernel() { deinit(); }

    // No copy
    XclbinKernel(const XclbinKernel&) = delete;
    XclbinKernel& operator=(const XclbinKernel&) = delete;

    // Move
    XclbinKernel(XclbinKernel&& other) noexcept
        : name_(std::move(other.name_))
        , md_(other.md_), kd_(other.kd_), nd_(other.nd_), n_layers_(other.n_layers_)
        , uuid_(other.uuid_)
        , device_(other.device_)
        , hw_context_(std::move(other.hw_context_))
        , kernel_(std::move(other.kernel_))
        , bo_instr_(std::move(other.bo_instr_))
        , bo_act_(std::move(other.bo_act_))
        , bo_out_(std::move(other.bo_out_))
        , layer_bos_(std::move(other.layer_bos_))
        , instr_mapped_(other.instr_mapped_)
        , out_mapped_(other.out_mapped_)
        , act_mapped_(other.act_mapped_)
        , instr_count_(other.instr_count_)
        , initialized_(other.initialized_)
    {
        other.initialized_ = false;
        other.instr_mapped_ = nullptr;
        other.out_mapped_ = nullptr;
        other.act_mapped_ = nullptr;
    }

    XclbinKernel& operator=(XclbinKernel&& other) noexcept {
        if (this != &other) {
            deinit();
            name_ = std::move(other.name_);
            md_ = other.md_; kd_ = other.kd_; nd_ = other.nd_; n_layers_ = other.n_layers_;
            uuid_ = other.uuid_;
            device_ = other.device_;
            hw_context_ = std::move(other.hw_context_);
            kernel_ = std::move(other.kernel_);
            bo_instr_ = std::move(other.bo_instr_);
            bo_act_ = std::move(other.bo_act_);
            bo_out_ = std::move(other.bo_out_);
            layer_bos_ = std::move(other.layer_bos_);
            instr_mapped_ = other.instr_mapped_;
            out_mapped_ = other.out_mapped_;
            act_mapped_ = other.act_mapped_;
            instr_count_ = other.instr_count_;
            initialized_ = other.initialized_;
            other.initialized_ = false;
            other.instr_mapped_ = nullptr;
            other.out_mapped_ = nullptr;
            other.act_mapped_ = nullptr;
        }
        return *this;
    }

    /// TQ2 ternary → INT8: pack TQ2 ternary weights into the layer's weight BO.
    /// tq2_data: raw TQ2 tile data in 1BP format (32x256 tiles, 2-bit codes)
    /// k, n: logical matrix dimensions
    /// tile_rows, tile_cols, group_size: tile geometry (default 32, 256, 32)
    /// Returns the quantization scale (for dequant on output).
    float packTQ2Weight(uint32_t layer, const uint8_t* tq2_data, uint32_t k, uint32_t n,
                         uint32_t tile_rows = 32, uint32_t tile_cols = 256, uint32_t group_size = 32) {
        if (layer >= n_layers_) throw std::runtime_error("Layer index out of bounds");

        // Use the ternary NPU bridge to convert TQ2 → INT8
        auto result = pack_tq2_to_npu_int8(
            tq2_data, (int)k, (int)n, (int)tile_rows, (int)tile_cols, (int)group_size);

        if (!result.weights) {
            fprintf(stderr, "[npu_kernel] TQ2 pack failed for layer %u\n", layer);
            return 1.0f;
        }

        size_t total = (size_t)k * n;
        uint8_t* mem = layer_bos_[layer].map(total);
        std::memcpy(mem, result.weights, total);
        layer_bos_[layer].sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, total);

        float scale = result.dequant_scale;
        free_ternary_npu_pack(&result);
        return scale;
    }

    /// TQ1 (1.58-bit) → INT8: pack TQ1 base-3 ternary weights into the layer's weight BO.
    float packTQ1Weight(uint32_t layer, const uint8_t* tq1_data, uint32_t k, uint32_t n,
                         uint32_t tile_rows = 32, uint32_t tile_cols = 256) {
        if (layer >= n_layers_) throw std::runtime_error("Layer index out of bounds");

        auto result = pack_tq1_to_npu_int8(
            tq1_data, (int)k, (int)n, (int)tile_rows, (int)tile_cols);

        if (!result.weights) {
            fprintf(stderr, "[npu_kernel] TQ1 pack failed for layer %u\n", layer);
            return 1.0f;
        }

        size_t total = (size_t)k * n;
        uint8_t* mem = layer_bos_[layer].map(total);
        std::memcpy(mem, result.weights, total);
        layer_bos_[layer].sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, total);

        float scale = result.dequant_scale;
        free_ternary_npu_pack(&result);
        return scale;
    }

    bool initialized() const { return initialized_; }

private:
    std::string name_;
    uint32_t md_ = 0;
    uint32_t kd_ = 0;
    uint32_t nd_ = 0;
    uint32_t n_layers_ = 0;

    XrtUuid uuid_{};
    XrtDevice* device_ = nullptr;
    XrtHwContext hw_context_;
    XrtKernel kernel_;

    XrtBuffer bo_instr_;
    XrtBuffer bo_act_;
    XrtBuffer bo_out_;
    std::vector<XrtBuffer> layer_bos_;

    int8_t* act_mapped_ = nullptr;
    int16_t* out_mapped_ = nullptr;
    uint32_t* instr_mapped_ = nullptr;
    int instr_count_ = 0;

    bool initialized_ = false;
};
