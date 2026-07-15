// shared_bo.cpp — see shared_bo.h for the architecture rationale.
#include "shared_bo.h"
#include <unistd.h>
#include <cstdio>

namespace fusion {

SharedBO* SharedBO::create(xrt::device& npu_dev, size_t bytes) {
    auto* self = new SharedBO();
    self->dev_   = &npu_dev;
    self->bytes_ = bytes;

    // 1) NPU owns the allocation.  HOST_ONLY = coherent host RAM the NPU DMAs
    //    into via its own (correctly mapped) IOMMU domain.
    try {
        self->bo_ = xrt::bo(npu_dev, bytes, xrt::bo::flags::host_only,
                            static_cast<xrt::memory_group>(0));
    } catch (const std::exception& e) {
        fprintf(stderr, "SharedBO: NPU BO alloc failed: %s\n", e.what());
        delete self;
        return nullptr;
    }

    // 2) Host view — direct coherent CPU pointer into the same pages.
    self->host_ = self->bo_.map();

    // 3) Export as dma-buf.  export_handle IS the fd (int32_t on Linux).
    xrt::bo::export_handle raw_fd = self->bo_.export_buffer();
    if (raw_fd < 0) {
        fprintf(stderr, "SharedBO: NPU export_buffer failed (fd=%d)\n", (int)raw_fd);
        delete self;
        return nullptr;
    }
    // dup it so our fd survives independent of the XRT export-handle lifetime
    // and so callers can import without closing our copy.
    self->dup_fd_ = dup(raw_fd);

    return self;
}

SharedBO::~SharedBO() {
    if (dup_fd_ >= 0) close(dup_fd_);
    // bo_ destroys itself (its pages return to the NPU domain).
}

void*    SharedBO::host_ptr() { return host_; }
xrt::bo& SharedBO::npu_bo()   { return bo_; }

void SharedBO::sync_to_npu(size_t off, size_t n) {
    size_t len = n ? n : (bytes_ - off);
    bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE, len, off);
}
void SharedBO::sync_from_npu(size_t off, size_t n) {
    size_t len = n ? n : (bytes_ - off);
    bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, len, off);
}

} // namespace fusion
