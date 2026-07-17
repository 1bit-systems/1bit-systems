// shared_bo.h — Zero-copy shared buffer for NPU <-> GPU fusion.
//
// ONE physical allocation, owned by the NPU, viewed zero-copy by everyone:
//   OWNER = NPU (amdxdna).  Allocated as an XRT HOST_ONLY BO.  The NPU's own
//          IOMMU domain owns the pages, so NPU DMA always reaches them — this
//          is the fix for the AMD-Vi IO_PAGE_FAULT seen when the *GPU* owned
//          the buffer and the NPU tried to import it (gpu_npu_bridge.cpp).
//   HOST  = xrt::bo::map() — coherent system RAM on Strix Halo (unified mem).
//   GPU   = two equivalent routes, both consume the SAME dma-buf fd exported
//          below:
//            (a) Vulkan import:  VkImportMemoryFdInfoKHR with
//                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT  (the
//                production path — matches engine/fusion/gpu_attn.zig; this is
//                the ONLY route that works under ROCm 7.2.4, whose HIP lacks a
//                DmaBuf external-memory handle type).
//            (b) hipHostRegister()+hipHostGetDevicePointer() on the host ptr:
//                the integrated-GPU zero-copy idiom, used in the test to PROVE
//                the NPU pages are GPU-reachable without any copy.
//
// Direction matters on Strix Halo: NPU and GPU are separate PCI devices
// (c6:00.1 vs c5:00.0) behind one AMD-Vi IOMMU but in distinct domains.  The
// reliable owner is the NPU; amdgpu and the CPU import from it.  Reversing
// that (the old code) is what faulted.
//
// Deliberately HIP-free: this module links only XRT + libc.  GPU integration
// lives in the caller so the production Vulkan path and the test's HIP-idiom
// path can share the same SharedBO without forcing a HIP dependency here.

#pragma once
#include <cstddef>
#include <cstdint>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>

namespace fusion {

class SharedBO {
public:
    // Allocate `bytes` of zero-copy shared memory owned by the NPU device.
    // `npu_dev` must outlive this object.  Returns nullptr on failure.
    static SharedBO* create(xrt::device& npu_dev, size_t bytes);

    ~SharedBO();

    // ---- zero-copy views (all alias the same physical pages) ----
    void*     host_ptr();        // CPU coherent view (XRT HOST_ONLY map)
    xrt::bo&  npu_bo();          // NPU view (the owning BO)
    int       dma_buf_fd() const { return dup_fd_; }  // shared dma-buf fd (for Vulkan/HIP import)
    size_t    size() const { return bytes_; }

    // NPU cache sync.  Cheap on coherent HOST_ONLY memory, but kept so callers
    // can express ordering intent on the NPU side.
    void sync_to_npu(size_t off = 0, size_t n = 0);
    void sync_from_npu(size_t off = 0, size_t n = 0);

    SharedBO(const SharedBO&) = delete;
    SharedBO& operator=(const SharedBO&) = delete;

private:
    SharedBO() = default;
    xrt::device* dev_    = nullptr;
    xrt::bo      bo_;
    void*        host_   = nullptr;
    int          dup_fd_ = -1;   // our dup of the export fd (callers may dup again)
    size_t       bytes_  = 0;
};

} // namespace fusion
