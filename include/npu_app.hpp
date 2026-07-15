/*!
 *  Copyright (c) 2025 (adapted from FastFlowLM MIT source)
 * \file npu_app.hpp
 * \brief NPU application manager — wraps npu_sequence with xrt::ext::kernel
 *        for dynamic instruction execution on AMD XDNA NPUs.
 *
 * Architecture:
 *   npu_xclbin_manager (singleton) owns xrt::device and manages xclbin registration.
 *   npu_app_manager      wraps one xclbin + hw_context, creates npu_app instances.
 *   npu_app              holds an npu_sequence + dynamically generated xrt::ext::kernel.
 *
 * Usage:
 *   xrt::device dev(0);
 *   npu_xclbin_manager mgr(npu_device::npu2, &dev);
 *   npu_app_manager* gemm_mgr = mgr.register_xclbin("mm.xclbin");
 *   npu_app app = gemm_mgr->create_app();
 *   app.seq()->cmds2seq();  // compile instructions
 *   app.run(act_bo, weight_bo, out_bo);  // execute on NPU
 *
 * This replaces the older xrt::kernel + pre-compiled .insts approach.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>

#include "npu_instr.hpp"

// ---------------------------------------------------------------------------
// ELF generation from instruction stream
// ---------------------------------------------------------------------------

/// Generate an ELF binary from an npu_sequence instruction stream.
/// Returns a malloc'd buffer and size. Caller must free the buffer.
inline std::pair<char*, size_t> generate_elf_from_sequence(npu_sequence& seq) {
    auto [instr_data, instr_words] = seq.dump();
    if (!instr_data || instr_words == 0) {
        throw std::runtime_error("Empty instruction sequence — call cmds2seq() first");
    }

    // Use aiebu assembler to generate ELF from raw instructions
    // This is the same API FLM uses internally
    char* elf_buf = nullptr;
    size_t elf_size = 0;

    // First call: get required buffer size
    elf_size = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        reinterpret_cast<char*>(instr_data),
        instr_words * sizeof(uint32_t),
        nullptr,            // no col mapping
        0,
        &elf_buf,           // nullptr → query size
        0,                  // zero buffer size → returns required size
        "",                 // no custom device
        "",
        nullptr, 0);

    if (elf_size == 0) {
        throw std::runtime_error("aiebu_assembler_get_elf query failed");
    }

    elf_buf = static_cast<char*>(std::malloc(elf_size));
    if (!elf_buf) {
        throw std::bad_alloc();
    }

    // Second call: actually generate ELF
    elf_size = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        reinterpret_cast<char*>(instr_data),
        instr_words * sizeof(uint32_t),
        nullptr, 0,
        &elf_buf, elf_size,
        "", "",
        nullptr, 0);

    if (elf_size == 0) {
        std::free(elf_buf);
        throw std::runtime_error("aiebu_assembler_get_elf generation failed");
    }

    return {elf_buf, elf_size};
}

// ---------------------------------------------------------------------------
// npu_app — single NPU application with dynamically generated kernel
// ---------------------------------------------------------------------------

class npu_app {
public:
    /// Default constructor — creates empty/invalid app
    npu_app() = default;

    /// Construct from device, context, and kernel name
    npu_app(npu_device dev_gen, xrt::device* dev, xrt::hw_context* ctx,
            const std::string& kernel_name, bool enable_preemption = false)
        : dev_gen_(dev_gen)
        , device_(dev)
        , context_(ctx)
        , kernel_name_(kernel_name)
        , enable_preemption_(enable_preemption)
        , seq_(std::make_unique<npu_sequence>(dev_gen, enable_preemption))
        , module_valid_(false)
        , module_version_(0xFF)
    {}

    /// Move constructor
    npu_app(npu_app&& other) noexcept
        : dev_gen_(other.dev_gen_)
        , device_(other.device_)
        , context_(other.context_)
        , kernel_name_(std::move(other.kernel_name_))
        , enable_preemption_(other.enable_preemption_)
        , seq_(std::move(other.seq_))
        , module_(std::move(other.module_))
        , elf_(std::move(other.elf_))
        , kernel_(std::move(other.kernel_))
        , module_valid_(other.module_valid_)
        , module_version_(other.module_version_)
    {
        other.device_ = nullptr;
        other.context_ = nullptr;
        other.module_valid_ = false;
    }

    /// Access the instruction sequence
    npu_sequence* seq() { return seq_.get(); }
    const npu_sequence* seq() const { return seq_.get(); }

    /// Rebuild the kernel from the current sequence (call after cmds2seq())
    void update_kernel() {
        if (!seq_ || !device_ || !context_) {
            throw std::runtime_error("npu_app not properly initialized");
        }
        if (!seq_->valid()) {
            throw std::runtime_error("Sequence not compiled — call cmds2seq() first");
        }

        // Generate ELF from sequence
        auto [elf_buf, elf_size] = generate_elf_from_sequence(*seq_);

        // Create xrt::elf from the binary
        elf_ = std::make_unique<xrt::elf>(elf_buf, elf_size);
        std::free(elf_buf);

        // Create module from ELF
        module_ = std::make_unique<xrt::module>(*elf_);

        // Create kernel from module
        kernel_ = std::make_unique<xrt::ext::kernel>(*context_, *module_, kernel_name_);

        module_valid_ = true;
        module_version_ = seq_->version();
    }

    /// Load a pre-compiled ELF directly (skip instruction generation)
    void load_elf(const std::string& elf_path) {
        if (!seq_) {
            throw std::runtime_error("npu_app not initialized");
        }
        seq_->clear();
        seq_->cmds2seq();  // empty valid sequence

        elf_ = std::make_unique<xrt::elf>(elf_path);
        module_ = std::make_unique<xrt::module>(*elf_);
        kernel_ = std::make_unique<xrt::ext::kernel>(*context_, *module_, kernel_name_);
        module_valid_ = true;
        module_version_ = seq_->version();
    }

    /// Save the current sequence as an ELF file
    void save_elf(const std::string& path) {
        if (!module_valid_) {
            update_kernel();
        }
        // Write the raw instruction data as an ELF
        auto [data, words] = seq_->dump();
        auto [elf_buf, elf_size] = generate_elf_from_sequence(*seq_);
        std::ofstream f(path, std::ios::binary);
        f.write(elf_buf, elf_size);
        std::free(elf_buf);
    }

    /// Run the kernel with BO arguments
    /// Uses opcode=3 (run pre-loaded program) with no inline instructions
    template<typename... BoArgs>
    ert_cmd_state run(BoArgs&&... args) {
        if (!module_valid_ || !kernel_ ||
            module_version_ != seq_->version()) {
            update_kernel();
        }

        // FLM calls: kernel->operator()(3, 0, 0, args.bo()...)
        // Arg 0 = opcode (3 = execute)
        // Arg 1 = 0 (no inline instructions)
        // Arg 2 = 0 (no inline instruction count)
        // Arg 3+ = BO handles
        auto run_obj = (*kernel_)(3, 0, 0, args.bo()...);
        return run_obj.wait();
    }

    /// Create a run object (for async execution with runlist)
    template<typename... BoArgs>
    xrt::run create_run(BoArgs&&... args) {
        if (!module_valid_ || !kernel_ ||
            module_version_ != seq_->version()) {
            update_kernel();
        }

        xrt::run run_obj(*kernel_);
        run_obj.set_arg(0, 3);
        run_obj.set_arg(1, 0);
        run_obj.set_arg(2, 0);

        // Set BO args starting at index 3
        set_run_args(run_obj, 3, std::forward<BoArgs>(args)...);

        return run_obj;
    }

    /// Safe run: sync BOs to device, execute, sync BOs back
    template<typename... BoArgs>
    ert_cmd_state safe_run(BoArgs&&... args) {
        // Sync all BOs to device
        (args.sync(XCL_BO_SYNC_BO_TO_DEVICE), ...);

        ert_cmd_state state = run(std::forward<BoArgs>(args)...);

        // Sync all BOs back from device
        (args.sync(XCL_BO_SYNC_BO_FROM_DEVICE), ...);

        return state;
    }

    /// Create a buffer BO managed by this app's device
    template<typename T>
    xrt::bo create_bo(size_t count, int group_id = 0) {
        if (!device_) throw std::runtime_error("No device");
        size_t bytes = count * sizeof(T);
        return xrt::bo(*device_, bytes, XRT_BO_FLAGS_HOST_ONLY, group_id);
    }

    bool is_valid() const { return device_ != nullptr; }

private:
    template<typename... BoArgs>
    void set_run_args(xrt::run& run_obj, int idx, xrt::bo& bo, BoArgs&&... rest) {
        run_obj.set_arg(idx, bo);
        set_run_args(run_obj, idx + 1, std::forward<BoArgs>(rest)...);
    }
    void set_run_args(xrt::run&, int) {}  // base case

    npu_device dev_gen_ = npu_device::npu2;
    xrt::device* device_ = nullptr;
    xrt::hw_context* context_ = nullptr;
    std::string kernel_name_ = "MLIR_AIE";
    bool enable_preemption_ = false;

    std::unique_ptr<npu_sequence> seq_;
    std::unique_ptr<xrt::module> module_;
    std::unique_ptr<xrt::elf> elf_;
    std::unique_ptr<xrt::ext::kernel> kernel_;

    bool module_valid_ = false;
    uint32_t module_version_ = 0xFF;
};

// ---------------------------------------------------------------------------
// npu_app_manager — manages one xclbin and creates npu_app instances
// ---------------------------------------------------------------------------

class npu_app_manager {
public:
    npu_app_manager() = default;

    /// Construct from device and xclbin path
    npu_app_manager(npu_device dev_gen, xrt::device* dev,
                    const std::string& xclbin_path, bool enable_preemption = false)
        : dev_gen_(dev_gen)
        , device_(dev)
        , xclbin_path_(xclbin_path)
        , enable_preemption_(enable_preemption)
    {
        if (!dev) throw std::runtime_error("Null device pointer");

        // Load xclbin
        xrt::xclbin xclbin(xclbin_path);

        // Find the MLIR_AIE kernel in the xclbin
        auto kernels = xclbin.get_kernels();
        auto it = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel& k) {
                return k.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (it == kernels.end()) {
            throw std::runtime_error("No MLIR_AIE kernel found in " + xclbin_path);
        }
        kernel_name_ = it->get_name();

        // Register xclbin with device
        device_->register_xclbin(xclbin);

        // Create hardware context
        context_ = std::make_unique<xrt::hw_context>(*device_, xclbin.get_uuid());

        xclbin_valid_ = true;
    }

    /// Copy constructor (shares context)
    npu_app_manager(const npu_app_manager& other)
        : dev_gen_(other.dev_gen_)
        , device_(other.device_)
        , context_(other.context_ ? std::make_unique<xrt::hw_context>(*other.context_) : nullptr)
        , kernel_name_(other.kernel_name_)
        , xclbin_path_(other.xclbin_path_)
        , xclbin_valid_(other.xclbin_valid_)
        , enable_preemption_(other.enable_preemption_)
    {}

    /// Create a new npu_app from this manager
    npu_app create_app() {
        if (!xclbin_valid_) throw std::runtime_error("Invalid xclbin manager");
        return npu_app(dev_gen_, device_, context_.get(), kernel_name_, enable_preemption_);
    }

    /// Create a buffer BO
    template<typename T>
    xrt::bo create_bo(size_t count, int group_id = 0) {
        if (!device_) throw std::runtime_error("No device");
        size_t bytes = count * sizeof(T);
        return xrt::bo(*device_, bytes, XRT_BO_FLAGS_HOST_ONLY, group_id);
    }

    /// Create an xrt::runlist for this context
    xrt::runlist create_runlist() {
        if (!context_) throw std::runtime_error("No context");
        return xrt::runlist(*context_);
    }

    const std::string& xclbin_path() const { return xclbin_path_; }
    bool xclbin_valid() const { return xclbin_valid_; }

private:
    npu_device dev_gen_ = npu_device::npu2;
    xrt::device* device_ = nullptr;
    std::unique_ptr<xrt::hw_context> context_;
    std::string kernel_name_ = "MLIR_AIE";
    std::string xclbin_path_;
    bool xclbin_valid_ = false;
    bool enable_preemption_ = false;
};

// ---------------------------------------------------------------------------
// npu_xclbin_manager — top-level manager, owns device, manages xclbins
// ---------------------------------------------------------------------------

class npu_xclbin_manager {
public:
    static constexpr int MAX_XCLBINS = 16;

    npu_xclbin_manager(npu_device dev = npu_device::npu2,
                       xrt::device* device_inst = nullptr,
                       bool enable_preemption = false)
        : device_(device_inst)
        , npu_gen_(dev)
        , enable_preemption_(enable_preemption)
    {
        managers_.reserve(MAX_XCLBINS);
    }

    /// Register an xclbin (deduplicates by path)
    npu_app_manager* register_xclbin(const std::string& xclbin_path) {
        if (xclbin_path.empty()) throw std::runtime_error("Empty xclbin path");

        // Check if already registered
        for (auto& mgr : managers_) {
            if (mgr && mgr->xclbin_path() == xclbin_path) {
                return mgr.get();
            }
        }

        if (managers_.size() >= MAX_XCLBINS) {
            throw std::runtime_error("Max xclbins (" + std::to_string(MAX_XCLBINS) + ") reached");
        }

        auto mgr = std::make_unique<npu_app_manager>(
            npu_gen_, device_, xclbin_path, enable_preemption_);
        auto* ptr = mgr.get();
        managers_.push_back(std::move(mgr));
        return ptr;
    }

    /// Create a buffer BO
    template<typename T>
    xrt::bo create_bo(size_t count, int group_id = 0) {
        if (!device_) throw std::runtime_error("No device");
        size_t bytes = count * sizeof(T);
        return xrt::bo(*device_, bytes, XRT_BO_FLAGS_HOST_ONLY, group_id);
    }

    /// List registered xclbins
    void list_xclbins() const {
        for (size_t i = 0; i < managers_.size(); i++) {
            std::cout << "  [" << i << "] " << managers_[i]->xclbin_path() << std::endl;
        }
    }

    xrt::device* device() { return device_; }

private:
    xrt::device* device_ = nullptr;
    npu_device npu_gen_ = npu_device::npu2;
    bool enable_preemption_ = false;
    std::vector<std::unique_ptr<npu_app_manager>> managers_;
};

#endif // __NPU_APP_HPP__
