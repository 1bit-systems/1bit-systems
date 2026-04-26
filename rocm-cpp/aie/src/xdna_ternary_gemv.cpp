// xdna_ternary_gemv.cpp — libxrt-backed ternary GEMV wrapper.
//
// Mirrors the BO/kernel layout used by xdna_kernel.cpp (matmul). The
// kernel arity is the same: opcode=3, instr-bo, instr-count, A-bo, B-bo,
// C-bo. The mlir-aie matrix-vector design uses identical group ids:
//   group_id(1) -> instr-stream BO
//   group_id(3) -> A buffer
//   group_id(4) -> B buffer
//   group_id(5) -> C buffer
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2026 Daniel <d1r7yman@gmail.com>

#include "rocm_cpp/xdna_ternary_gemv.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_uuid.h"

namespace rocm_cpp {

namespace {

// V0 baked-in shape. Must match the .xclbin in aie/build/. If you rebuild
// for different M/K, bump these and re-compile.
constexpr int kFixedM = 64;
constexpr int kFixedK = 64;
constexpr int kFixedKpack = kFixedK / 4;  // 4 trits per packed byte

// Same opcode used by the matmul xclbin — "configure-then-run instr seq".
constexpr unsigned int kOpcodeRunInstr = 3;

constexpr xrt::bo::flags kFlagsCacheable = xrt::bo::flags::cacheable;
constexpr xrt::bo::flags kFlagsHostOnly  = xrt::bo::flags::host_only;

std::string derive_insts_path(const std::string& xclbin_path) {
    const std::string final_prefix = "final_";
    const std::string xclbin_ext   = ".xclbin";
    auto slash = xclbin_path.find_last_of('/');
    std::string dir  = (slash == std::string::npos) ? std::string()
                                                    : xclbin_path.substr(0, slash + 1);
    std::string base = (slash == std::string::npos) ? xclbin_path
                                                    : xclbin_path.substr(slash + 1);
    if (base.size() < final_prefix.size() + xclbin_ext.size()) return "";
    if (base.compare(0, final_prefix.size(), final_prefix) != 0) return "";
    if (base.compare(base.size() - xclbin_ext.size(),
                     xclbin_ext.size(), xclbin_ext) != 0) return "";
    std::string stem = base.substr(final_prefix.size(),
                                   base.size() - final_prefix.size() - xclbin_ext.size());
    return dir + "insts_" + stem + ".txt";
}

// Load the instruction stream — same probe (hex-text vs raw u32) used by
// xdna_kernel.cpp. Kept private here to avoid leaking xrt headers from the
// xdna_kernel.cpp TU.
std::vector<uint32_t> load_instr_sequence(const std::string& path) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "XdnaTernaryGemv: cannot open instruction stream: " + path);
    }
    bool is_txt = false;
    {
        char buf[16] = {};
        f.read(buf, sizeof(buf));
        std::streamsize n = f.gcount();
        is_txt = (n > 0);
        for (std::streamsize i = 0; i < n; ++i) {
            unsigned char c = static_cast<unsigned char>(buf[i]);
            const bool printable = (c >= 0x20 && c < 0x7f) || c == '\n' ||
                                   c == '\r' || c == '\t' || c == ' ';
            if (!printable) { is_txt = false; break; }
        }
        f.clear();
        f.seekg(0, std::ios::beg);
    }

    std::vector<uint32_t> out;
    if (is_txt) {
        std::string tok;
        while (f >> tok) {
            unsigned long long v = std::stoull(tok, nullptr, 16);
            out.push_back(static_cast<uint32_t>(v));
        }
    } else {
        f.seekg(0, std::ios::end);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        if (sz <= 0 || (sz % sizeof(uint32_t)) != 0) {
            throw std::runtime_error(
                "XdnaTernaryGemv: instruction stream size not multiple of u32: " + path);
        }
        out.resize(static_cast<size_t>(sz) / sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(out.data()), sz);
    }
    if (out.empty()) {
        throw std::runtime_error(
            "XdnaTernaryGemv: empty instruction stream: " + path);
    }
    return out;
}

} // namespace

struct XdnaTernaryGemv::Impl {
    xrt::device       device;
    xrt::xclbin       xclbin;
    xrt::hw_context   ctx;
    xrt::kernel       kernel;
    xrt::bo           bo_instr;
    xrt::bo           bo_a;       // M * Kpack bytes
    xrt::bo           bo_b;       // K i8 elements
    xrt::bo           bo_c;       // M i32 elements
    std::vector<uint32_t> instrs;
    std::string       resolved_kernel_name;
    int               M = kFixedM;
    int               K = kFixedK;
    int               Kpack = kFixedKpack;
    bool              ready = false;
};

XdnaTernaryGemv::XdnaTernaryGemv(const std::string& xclbin_path)
    : XdnaTernaryGemv(xclbin_path, derive_insts_path(xclbin_path)) {}

XdnaTernaryGemv::XdnaTernaryGemv(const std::string& xclbin_path,
                                 const std::string& insts_path)
    : impl_(std::make_unique<Impl>()) {
    try {
        impl_->instrs = load_instr_sequence(insts_path);

        impl_->device = xrt::device(0u);
        impl_->xclbin = xrt::xclbin(xclbin_path);

        const auto kernels = impl_->xclbin.get_kernels();
        if (kernels.empty()) {
            throw std::runtime_error(
                "XdnaTernaryGemv: xclbin contains no kernels");
        }
        // mlir-aie names the entry "MLIR_AIE_<hash>" — first one wins.
        const auto& k0 = kernels.front();
        impl_->resolved_kernel_name = k0.get_name();

        impl_->device.register_xclbin(impl_->xclbin);
        impl_->ctx = xrt::hw_context(impl_->device, impl_->xclbin.get_uuid());
        impl_->kernel = xrt::kernel(impl_->ctx, impl_->resolved_kernel_name);

        const size_t a_bytes = static_cast<size_t>(impl_->M) * impl_->Kpack
                               * sizeof(uint8_t);
        const size_t b_bytes = static_cast<size_t>(impl_->K) * sizeof(int8_t);
        const size_t c_bytes = static_cast<size_t>(impl_->M) * sizeof(int32_t);
        const size_t instr_bytes = impl_->instrs.size() * sizeof(uint32_t);

        impl_->bo_instr = xrt::bo(impl_->device, instr_bytes,
                                  kFlagsCacheable, impl_->kernel.group_id(1));
        impl_->bo_a = xrt::bo(impl_->device, a_bytes,
                              kFlagsHostOnly, impl_->kernel.group_id(3));
        impl_->bo_b = xrt::bo(impl_->device, b_bytes,
                              kFlagsHostOnly, impl_->kernel.group_id(4));
        impl_->bo_c = xrt::bo(impl_->device, c_bytes,
                              kFlagsHostOnly, impl_->kernel.group_id(5));

        std::memcpy(impl_->bo_instr.map<void*>(),
                    impl_->instrs.data(),
                    instr_bytes);
        impl_->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        impl_->ready = true;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("XdnaTernaryGemv construction failed: ") + e.what());
    }
}

XdnaTernaryGemv::~XdnaTernaryGemv() = default;

bool XdnaTernaryGemv::is_loaded() const noexcept {
    return impl_ && impl_->ready;
}

const std::string& XdnaTernaryGemv::kernel_name() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->resolved_kernel_name : empty;
}

int XdnaTernaryGemv::M() const noexcept { return impl_ ? impl_->M : 0; }
int XdnaTernaryGemv::K() const noexcept { return impl_ ? impl_->K : 0; }

void XdnaTernaryGemv::run(const uint8_t* A_packed,
                          const int8_t* x_i8,
                          const float* scales,
                          float x_scale,
                          _Float16* y_out,
                          int M_in, int K_in) {
    if (!is_loaded()) {
        throw std::runtime_error("XdnaTernaryGemv: not loaded");
    }
    if (!A_packed || !x_i8 || !scales || !y_out) {
        throw std::runtime_error("XdnaTernaryGemv::run: null buffer");
    }
    if (M_in != impl_->M || K_in != impl_->K) {
        std::ostringstream oss;
        oss << "XdnaTernaryGemv: shape (M=" << M_in << ", K=" << K_in
            << ") not supported by loaded xclbin (compiled for M="
            << impl_->M << ", K=" << impl_->K << ")";
        throw std::runtime_error(oss.str());
    }

    const size_t a_bytes = static_cast<size_t>(impl_->M) * impl_->Kpack
                           * sizeof(uint8_t);
    const size_t b_bytes = static_cast<size_t>(impl_->K) * sizeof(int8_t);
    const size_t c_bytes = static_cast<size_t>(impl_->M) * sizeof(int32_t);

    std::memcpy(impl_->bo_a.map<void*>(), A_packed, a_bytes);
    std::memcpy(impl_->bo_b.map<void*>(), x_i8, b_bytes);
    impl_->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    try {
        auto run_handle = impl_->kernel(kOpcodeRunInstr,
                                        impl_->bo_instr,
                                        static_cast<uint32_t>(impl_->instrs.size()),
                                        impl_->bo_a,
                                        impl_->bo_b,
                                        impl_->bo_c);
        const auto state = run_handle.wait();
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::ostringstream oss;
            oss << "XdnaTernaryGemv: kernel did not complete (state="
                << int(state) << ")";
            throw std::runtime_error(oss.str());
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("XdnaTernaryGemv run failed: ") + e.what());
    }

    impl_->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Apply scales + uniform x_scale, cast to fp16. Done host-side because
    // the v0 NPU kernel is integer-only. Compiler vectorizes this trivially.
    const int32_t* acc = impl_->bo_c.map<int32_t*>();
    for (int m = 0; m < impl_->M; ++m) {
        const float v = static_cast<float>(acc[m]) * scales[m] * x_scale;
        y_out[m] = static_cast<_Float16>(v);
    }
}

} // namespace rocm_cpp
