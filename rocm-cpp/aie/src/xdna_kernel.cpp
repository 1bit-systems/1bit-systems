// xdna_kernel.cpp — libxrt-backed implementation. See header for v0 scope.
//
// Calling convention (MLIR-AIE single-core matmul, npu2/AIE2P):
//   kernel(opcode=3, bo_instr, instr_count_u32, bo_a, bo_b, bo_c)
// BO group_ids: instr=1, a=3, b=4, c=5. Buffer flags: instr is CACHEABLE,
// a/b/c are HOST_ONLY. Verified against
// mlir-aie/programming_examples/basic/matrix_multiplication/single_core/run_pyxrt_i8.py
// and the upstream test.cpp.
//
// We baked the M=K=N=512 / 64×64×64 shape in here because that's the stock
// xclbin shipped at npu_matmul_i8 unlock (2026-04-23). Generalizing means
// either compiling more xclbins or shipping our own authored kernel; both are
// out of scope for v0.

#include "rocm_cpp/xdna_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_uuid.h"

namespace rocm_cpp {

namespace {

constexpr int kFixedM = 512;
constexpr int kFixedK = 512;
constexpr int kFixedN = 512;

// Opcode 3 == NPU "configure-then-run instruction sequence" — the only opcode
// the mlir-aie matmul kernel honors. Hard-coded in upstream test.cpp:225.
constexpr unsigned int kOpcodeRunInstr = 3;

// Buffer-flag literals — we use the integer values directly because we don't
// want to drag XRT_BO_FLAGS_HOST_ONLY's macro into our public header.
// Mapping is the same as upstream's pyxrt enum:
//   XCL_BO_FLAGS_CACHEABLE  == 0x01000000  (XRT_BO_FLAGS_CACHEABLE)
//   XRT_BO_FLAGS_HOST_ONLY  == 0x04000000  (host-only allocator)
// We pull both from <xrt/xrt_bo.h> at TU scope so any future libxrt rev that
// renames these still produces a clean compile error here (instead of a
// silent ABI mismatch).
constexpr xrt::bo::flags kFlagsCacheable = xrt::bo::flags::cacheable;
constexpr xrt::bo::flags kFlagsHostOnly  = xrt::bo::flags::host_only;

// Derive `insts_*.txt` path from `final_*.xclbin`. The two files always live
// next to each other in the mlir-aie build dir; only the prefix and suffix
// differ. Returns "" if we don't recognize the layout.
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

// Load instruction stream. The file is either raw u32 little-endian binary
// (the MLIR-AIE `.bin` flavor) or a whitespace/newline-separated list of
// hexadecimal u32s (the `.txt` flavor). We probe by extension.
std::vector<uint32_t> load_instr_sequence(const std::string& path) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        throw std::runtime_error("XdnaKernel: cannot open instruction stream: " + path);
    }

    // Don't trust the extension — mlir-aie ships some `.txt` files that are
    // actually raw u32 binary streams (no ASCII at all). Probe by sampling
    // the first few bytes: if any byte isn't printable ASCII or whitespace,
    // treat the file as binary.
    bool is_txt = false;
    {
        char buf[16] = {};
        f.read(buf, sizeof(buf));
        std::streamsize n = f.gcount();
        is_txt = (n > 0);
        for (std::streamsize i = 0; i < n; ++i) {
            unsigned char c = static_cast<unsigned char>(buf[i]);
            const bool printable = (c >= 0x20 && c < 0x7f) || c == '\n' || c == '\r' || c == '\t' || c == ' ';
            if (!printable) { is_txt = false; break; }
        }
        f.clear();
        f.seekg(0, std::ios::beg);
    }

    std::vector<uint32_t> out;
    if (is_txt) {
        std::string tok;
        while (f >> tok) {
            // Tokens may be plain hex (e.g. "06030100") or 0x-prefixed.
            unsigned long long v = std::stoull(tok, nullptr, 16);
            out.push_back(static_cast<uint32_t>(v));
        }
    } else {
        f.seekg(0, std::ios::end);
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        if (sz <= 0 || (sz % sizeof(uint32_t)) != 0) {
            throw std::runtime_error(
                "XdnaKernel: instruction stream size not multiple of u32: " + path);
        }
        out.resize(static_cast<size_t>(sz) / sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(out.data()), sz);
    }

    if (out.empty()) {
        throw std::runtime_error("XdnaKernel: empty instruction stream: " + path);
    }
    return out;
}

} // namespace

struct XdnaKernel::Impl {
    xrt::device       device;
    xrt::xclbin       xclbin;
    xrt::hw_context   ctx;
    xrt::kernel       kernel;
    xrt::bo           bo_instr;
    xrt::bo           bo_a;
    xrt::bo           bo_b;
    xrt::bo           bo_c;
    std::vector<uint32_t> instrs;
    std::string       resolved_kernel_name;
    int               M = kFixedM, K = kFixedK, N = kFixedN;
    bool              ready = false;
};

XdnaKernel::XdnaKernel(const std::string& xclbin_path,
                       const std::string& kernel_name)
    : XdnaKernel(xclbin_path, kernel_name, derive_insts_path(xclbin_path)) {}

XdnaKernel::XdnaKernel(const std::string& xclbin_path,
                       const std::string& kernel_name,
                       const std::string& insts_path)
    : impl_(std::make_unique<Impl>()) {
    try {
        impl_->instrs = load_instr_sequence(insts_path);

        impl_->device = xrt::device(0u);

        impl_->xclbin = xrt::xclbin(xclbin_path);

        // Resolve kernel by prefix match (mlir-aie pads the registered name
        // with target metadata, so "MLIR_AIE" matches "MLIR_AIE_<hash>").
        const auto kernels = impl_->xclbin.get_kernels();
        if (kernels.empty()) {
            throw std::runtime_error("XdnaKernel: xclbin contains no kernels");
        }
        std::string resolved;
        if (kernel_name.empty()) {
            resolved = kernels.front().get_name();
        } else {
            for (const auto& k : kernels) {
                const auto name = k.get_name();
                if (name.rfind(kernel_name, 0) == 0) { resolved = name; break; }
            }
            if (resolved.empty()) {
                std::ostringstream oss;
                oss << "XdnaKernel: kernel '" << kernel_name
                    << "' not found in xclbin. Available: ";
                for (size_t i = 0; i < kernels.size(); ++i) {
                    if (i) oss << ", ";
                    oss << kernels[i].get_name();
                }
                throw std::runtime_error(oss.str());
            }
        }
        impl_->resolved_kernel_name = resolved;

        impl_->device.register_xclbin(impl_->xclbin);
        impl_->ctx = xrt::hw_context(impl_->device, impl_->xclbin.get_uuid());
        impl_->kernel = xrt::kernel(impl_->ctx, resolved);

        // Persistent BOs sized for the fixed compile-time shape.
        const size_t a_bytes = static_cast<size_t>(impl_->M) * impl_->K * sizeof(int8_t);
        const size_t b_bytes = static_cast<size_t>(impl_->K) * impl_->N * sizeof(int8_t);
        const size_t c_bytes = static_cast<size_t>(impl_->M) * impl_->N * sizeof(int32_t);
        const size_t instr_bytes = impl_->instrs.size() * sizeof(uint32_t);

        impl_->bo_instr = xrt::bo(impl_->device, instr_bytes,
                                  kFlagsCacheable, impl_->kernel.group_id(1));
        impl_->bo_a = xrt::bo(impl_->device, a_bytes,
                              kFlagsHostOnly,  impl_->kernel.group_id(3));
        impl_->bo_b = xrt::bo(impl_->device, b_bytes,
                              kFlagsHostOnly,  impl_->kernel.group_id(4));
        impl_->bo_c = xrt::bo(impl_->device, c_bytes,
                              kFlagsHostOnly,  impl_->kernel.group_id(5));

        // Stage the instruction stream once. Inputs/output sync per call.
        std::memcpy(impl_->bo_instr.map<void*>(),
                    impl_->instrs.data(),
                    instr_bytes);
        impl_->bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        impl_->ready = true;
    } catch (const std::exception& e) {
        // Re-wrap so callers see a single error type. The libxrt exception
        // hierarchy isn't part of our public surface.
        throw std::runtime_error(std::string("XdnaKernel construction failed: ") + e.what());
    }
}

XdnaKernel::~XdnaKernel() = default;

bool XdnaKernel::is_loaded() const noexcept {
    return impl_ && impl_->ready;
}

const std::string& XdnaKernel::kernel_name() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->resolved_kernel_name : empty;
}

void XdnaKernel::matmul_512_i8_i32(const int8_t* A_host,
                                    const int8_t* B_host,
                                    int32_t* C_host) {
    if (!is_loaded()) {
        throw std::runtime_error("XdnaKernel: not loaded");
    }
    if (!A_host || !B_host || !C_host) {
        throw std::runtime_error("XdnaKernel::matmul_512_i8_i32: null buffer");
    }

    const size_t a_bytes = static_cast<size_t>(impl_->M) * impl_->K * sizeof(int8_t);
    const size_t b_bytes = static_cast<size_t>(impl_->K) * impl_->N * sizeof(int8_t);
    const size_t c_bytes = static_cast<size_t>(impl_->M) * impl_->N * sizeof(int32_t);

    std::memcpy(impl_->bo_a.map<void*>(), A_host, a_bytes);
    std::memcpy(impl_->bo_b.map<void*>(), B_host, b_bytes);
    impl_->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    try {
        auto run = impl_->kernel(kOpcodeRunInstr,
                                 impl_->bo_instr,
                                 static_cast<uint32_t>(impl_->instrs.size()),
                                 impl_->bo_a,
                                 impl_->bo_b,
                                 impl_->bo_c);
        const auto state = run.wait();
        if (state != ERT_CMD_STATE_COMPLETED) {
            std::ostringstream oss;
            oss << "XdnaKernel: kernel did not complete (state=" << int(state) << ")";
            throw std::runtime_error(oss.str());
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("XdnaKernel run failed: ") + e.what());
    }

    impl_->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C_host, impl_->bo_c.map<void*>(), c_bytes);
}

void XdnaKernel::matmul_i8_i32(const int8_t* A_host,
                                const int8_t* B_host,
                                int32_t* C_host,
                                int M, int K, int N) {
    if (M != impl_->M || K != impl_->K || N != impl_->N) {
        std::ostringstream oss;
        oss << "XdnaKernel: shape (" << M << "," << K << "," << N
            << ") not supported by loaded xclbin (compiled for "
            << impl_->M << "," << impl_->K << "," << impl_->N << ")";
        throw std::runtime_error(oss.str());
    }
    matmul_512_i8_i32(A_host, B_host, C_host);
}

} // namespace rocm_cpp
