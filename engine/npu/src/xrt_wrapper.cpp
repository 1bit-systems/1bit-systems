/// C++ wrappers around the XRT C API — implementation.
/// Ported from engine/npu/src/xrt.zig
#include "xrt_wrapper.h"
#include <stdexcept>
#include <cstring>

// ─── XrtDevice ───────────────────────────────────────────────────

XrtDevice XrtDevice::open(unsigned int index) {
    XrtDevice dev;
    dev.handle_ = xrtDeviceOpen(index);
    if (!dev.handle_) {
        throw std::runtime_error("XrtDevice::open: failed to open device " + std::to_string(index));
    }
    return dev;
}

XrtUuid XrtDevice::loadXclbin(const std::string& path) {
    // Read the xclbin file into memory
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open xclbin file: " + path);
    }
    size_t file_size = file.tellg();
    file.seekg(0);
    std::vector<char> buf(file_size);
    file.read(buf.data(), file_size);

    // Create xclbin handle
    xrtXclbinHandle* alloc = xrtXclbinAlloc();
    if (!alloc) {
        throw std::runtime_error("xrtXclbinAlloc failed");
    }

    // Load xclbin from raw bytes
    int rc = xrtDeviceLoadXclbinHandle(handle_, buf.data(), buf.size());
    if (rc != 0) {
        xrtXclbinFree(alloc);
        throw std::runtime_error("xrtDeviceLoadXclbinHandle failed for: " + path);
    }

    // Get UUID
    XrtUuid uuid;
    int rc2 = xrtXclbinGetUUID(alloc, &uuid);
    xrtXclbinFree(alloc);
    if (rc2 != 0) {
        throw std::runtime_error("xrtXclbinGetUUID failed");
    }

    return uuid;
}

xrtBufferObjectHandle* XrtDevice::allocBO(size_t size, uint64_t flags, unsigned int group) {
    xrtBufferObjectHandle* bo = xrtBOAlloc(handle_, size, flags, group);
    if (!bo) {
        throw std::runtime_error("xrtBOAlloc failed: size=" + std::to_string(size) +
                                 " flags=" + std::to_string(flags) + " group=" + std::to_string(group));
    }
    return bo;
}

xrtHwContextHandle* XrtDevice::createHwContext(const XrtUuid& uuid) {
    xrtHwContextHandle* ctx = xrtHwContextCreate(handle_, &uuid);
    if (!ctx) {
        throw std::runtime_error("xrtHwContextCreate failed");
    }
    return ctx;
}

xrtKernelHandle* XrtDevice::createKernel(const XrtUuid& uuid, const char* name) {
    xrtKernelHandle* k = xrtKernelOpen(handle_, &uuid, name);
    if (!k) {
        throw std::runtime_error("xrtKernelOpen failed for: " + std::string(name));
    }
    return k;
}

void XrtDevice::close() {
    if (handle_) {
        xrtDeviceClose(handle_);
        handle_ = nullptr;
    }
}

// ─── XrtBuffer ───────────────────────────────────────────────────

uint8_t* XrtBuffer::map(size_t size) {
    void* ptr = xrtBOMap(handle_);
    if (!ptr) {
        throw std::runtime_error("xrtBOMap failed");
    }
    return static_cast<uint8_t*>(ptr);
}

void XrtBuffer::sync(unsigned int dir, uint64_t offset, uint64_t size) {
    int rc = xrtBOSync(handle_, dir, offset, size);
    if (rc != 0) {
        throw std::runtime_error("xrtBOSync failed");
    }
}

void XrtBuffer::free() {
    if (handle_) {
        xrtBOFree(handle_);
        handle_ = nullptr;
    }
}

// ─── XrtKernel ───────────────────────────────────────────────────

xrtRunHandle* XrtKernel::run(xrtBufferObjectHandle* instr, int instr_count,
                              xrtBufferObjectHandle* act, xrtBufferObjectHandle* weight,
                              xrtBufferObjectHandle* out) {
    xrtRunHandle* r = xrtKernelRun(handle_, 3, instr, instr_count, act, weight, out);
    if (!r) {
        throw std::runtime_error("xrtKernelRun failed");
    }
    return r;
}

void XrtKernel::close() {
    if (handle_) {
        xrtKernelClose(handle_);
        handle_ = nullptr;
    }
}

// ─── XrtRun ──────────────────────────────────────────────────────

void XrtRun::wait() {
    int rc = xrtRunWait(handle_);
    if (rc != 0) {
        throw std::runtime_error("xrtRunWait failed");
    }
}

void XrtRun::close() {
    if (handle_) {
        xrtRunClose(handle_);
        handle_ = nullptr;
    }
}

// ─── XrtHwContext ────────────────────────────────────────────────

void XrtHwContext::destroy() {
    if (handle_) {
        xrtHwContextDestroy(handle_);
        handle_ = nullptr;
    }
}
