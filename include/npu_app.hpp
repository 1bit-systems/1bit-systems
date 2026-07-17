/*!
 *  Copyright (c) 2025 (adapted from FastFlowLM MIT source)
 * \file npu_app.hpp
 * \brief NPU application manager for dynamic instruction generation.
 *
 * Uses npu_sequence to dynamically generate instruction streams, then submits
 * them via the proven xrt::kernel API (instruction BO + args). This is simpler
 * and more reliable than xrt::ext::kernel with ELF generation.
 *
 * Architecture:
 *   npu_xclbin_manager (singleton) owns xrt::device and manages xclbin registration.
 *   Each I8Ctx creates its own app from the manager.
 *   npu_sequence generates instruction stream -> compiled to BO -> submitted to kernel.
 */
#ifndef __NPU_APP_HPP__
#define __NPU_APP_HPP__

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

#include "npu_instr.hpp"

// ---------------------------------------------------------------------------
// npu_xclbin_manager -- top-level manager, owns device and xclbins
// ---------------------------------------------------------------------------

class npu_xclbin_manager {
public:
    static constexpr int MAX_XCLBINS = 16;

    npu_xclbin_manager(npu_device dev, xrt::device* dev_inst)
        : device_(dev_inst), npu_gen_(dev)
    {}

    /// Register an xclbin and return a kernel object. Deduplicates by path.
    /// Returns the instruction group_id and an initialized xrt::kernel.
    struct XclbinInfo {
        xrt::kernel kernel;
        int group_id_instr;  // group_id for instruction BO
        int group_id_act;    // group_id for activation BO  (arg 3)
        int group_id_wgt;    // group_id for weight BO     (arg 4)
        int group_id_out;    // group_id for output BO     (arg 5)
    };

    XclbinInfo register_xclbin(const std::string& xclbin_path) {
        // Load xclbin
        xrt::xclbin xclbin(xclbin_path);
        device_->register_xclbin(xclbin);

        // Create hardware context
        xrt::hw_context hc(*device_, xclbin.get_uuid());

        // Find first kernel
        auto kernels = xclbin.get_kernels();
        auto it = std::find_if(kernels.begin(), kernels.end(),
            [](const xrt::xclbin::kernel& k) {
                return k.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        std::string kname = (it != kernels.end()) ? it->get_name() : "MLIR_AIE";

        xrt::kernel kernel(hc, kname);

        // Query group IDs from kernel arguments
        int gid_instr = kernel.group_id(1);
        int gid_act   = kernel.group_id(3);
        int gid_wgt   = kernel.group_id(4);
        int gid_out   = kernel.group_id(5);

        return { std::move(kernel), gid_instr, gid_act, gid_wgt, gid_out };
    }

    xrt::device* device() { return device_; }

private:
    xrt::device* device_ = nullptr;
    npu_device npu_gen_ = npu_device::npu2;
};

#endif // __NPU_APP_HPP__
