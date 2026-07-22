#pragma once
/// C++ wrappers around the XRT C API (libxrt_coreutil).
/// Ported from engine/npu/src/xrt.zig
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

// XRT C API declarations (from xrt.h)
extern "C" {
    struct xrtDeviceHandle;
    struct xrtBufferObjectHandle;
    struct xrtKernelHandle;
    struct xrtRunHandle;
    struct xrtHwContextHandle;
    struct xrtXclbinHandle;

    xrtDeviceHandle* xrtDeviceOpen(unsigned int index);
    void xrtDeviceClose(xrtDeviceHandle* handle);
    int xrtDeviceLoadXclbinHandle(xrtDeviceHandle* handle, const void* xclbin, size_t size);

    xrtXclbinHandle* xrtXclbinAlloc();
    void xrtXclbinFree(xrtXclbinHandle* xclbin);
    int xrtXclbinGetUUID(xrtXclbinHandle* xclbin, void* uuid);

    xrtHwContextHandle* xrtHwContextCreate(xrtDeviceHandle* device, const void* uuid);
    void xrtHwContextDestroy(xrtHwContextHandle* hwctx);

    xrtKernelHandle* xrtKernelOpen(xrtDeviceHandle* device, const void* uuid, const char* name);
    void xrtKernelClose(xrtKernelHandle* kernel);

    xrtRunHandle* xrtKernelRun(xrtKernelHandle* kernel, int nargs,
                                xrtBufferObjectHandle* arg1, int arg2,
                                xrtBufferObjectHandle* arg3,
                                xrtBufferObjectHandle* arg4,
                                xrtBufferObjectHandle* arg5);
    void xrtRunClose(xrtRunHandle* run);
    int xrtRunWait(xrtRunHandle* run);

    xrtBufferObjectHandle* xrtBOAlloc(xrtDeviceHandle* device, size_t size, uint64_t flags, unsigned int group);
    void xrtBOFree(xrtBufferObjectHandle* bo);
    void* xrtBOMap(xrtBufferObjectHandle* bo);
    int xrtBOSync(xrtBufferObjectHandle* bo, unsigned int dir, uint64_t offset, uint64_t size);
}

// XRT constants
constexpr uint64_t XCL_BO_FLAGS_CACHEABLE = 0;
constexpr uint64_t XRT_BO_FLAGS_HOST_ONLY = 2;
constexpr unsigned int XCL_BO_SYNC_BO_TO_DEVICE = 0;
constexpr unsigned int XCL_BO_SYNC_BO_FROM_DEVICE = 1;

// XRT UUID — 16 bytes
struct XrtUuid {
    uint8_t data[16] = {0};
};

// ─── Safe C++ wrappers ───────────────────────────────────────────

class XrtDevice {
public:
    XrtDevice() : handle_(nullptr) {}
    ~XrtDevice() { close(); }

    XrtDevice(const XrtDevice&) = delete;
    XrtDevice& operator=(const XrtDevice&) = delete;
    XrtDevice(XrtDevice&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    XrtDevice& operator=(XrtDevice&& other) noexcept {
        if (this != &other) { close(); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }

    static XrtDevice open(unsigned int index);
    XrtUuid loadXclbin(const std::string& path);
    xrtBufferObjectHandle* allocBO(size_t size, uint64_t flags, unsigned int group);
    xrtHwContextHandle* createHwContext(const XrtUuid& uuid);
    xrtKernelHandle* createKernel(const XrtUuid& uuid, const char* name);
    void close();

    xrtDeviceHandle* handle() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }

private:
    xrtDeviceHandle* handle_;
};

class XrtBuffer {
public:
    XrtBuffer() : handle_(nullptr) {}
    explicit XrtBuffer(xrtBufferObjectHandle* h) : handle_(h) {}
    ~XrtBuffer() { free(); }

    XrtBuffer(const XrtBuffer&) = delete;
    XrtBuffer& operator=(const XrtBuffer&) = delete;
    XrtBuffer(XrtBuffer&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    XrtBuffer& operator=(XrtBuffer&& other) noexcept {
        if (this != &other) { free(); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }

    uint8_t* map(size_t size);
    void sync(unsigned int dir, uint64_t offset, uint64_t size);
    void free();

    xrtBufferObjectHandle* handle() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }

private:
    xrtBufferObjectHandle* handle_;
};

class XrtKernel {
public:
    XrtKernel() : handle_(nullptr) {}
    explicit XrtKernel(xrtKernelHandle* h) : handle_(h) {}
    ~XrtKernel() { close(); }

    XrtKernel(const XrtKernel&) = delete;
    XrtKernel& operator=(const XrtKernel&) = delete;
    XrtKernel(XrtKernel&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    XrtKernel& operator=(XrtKernel&& other) noexcept {
        if (this != &other) { close(); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }

    xrtRunHandle* run(xrtBufferObjectHandle* instr, int instr_count,
                      xrtBufferObjectHandle* act, xrtBufferObjectHandle* weight,
                      xrtBufferObjectHandle* out);
    void close();

    xrtKernelHandle* handle() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }

private:
    xrtKernelHandle* handle_;
};

class XrtRun {
public:
    XrtRun() : handle_(nullptr) {}
    explicit XrtRun(xrtRunHandle* h) : handle_(h) {}
    ~XrtRun() { close(); }

    XrtRun(const XrtRun&) = delete;
    XrtRun& operator=(const XrtRun&) = delete;
    XrtRun(XrtRun&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    XrtRun& operator=(XrtRun&& other) noexcept {
        if (this != &other) { close(); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }

    void wait();
    void close();

private:
    xrtRunHandle* handle_;
};

class XrtHwContext {
public:
    XrtHwContext() : handle_(nullptr) {}
    explicit XrtHwContext(xrtHwContextHandle* h) : handle_(h) {}
    ~XrtHwContext() { destroy(); }

    XrtHwContext(const XrtHwContext&) = delete;
    XrtHwContext& operator=(const XrtHwContext&) = delete;
    XrtHwContext(XrtHwContext&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    XrtHwContext& operator=(XrtHwContext&& other) noexcept {
        if (this != &other) { destroy(); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }

    void destroy();

private:
    xrtHwContextHandle* handle_;
};

/// Read a text file of instruction words (hex or decimal uint32_t values).
/// Returns the parsed instructions.
inline std::vector<uint32_t> readInstructionsFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open instruction file: " + path);
    }

    std::vector<uint32_t> result;
    std::string token;
    while (file >> token) {
        if (token.empty()) continue;
        uint32_t val;
        if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
            val = std::stoul(token.substr(2), nullptr, 16);
        } else {
            val = std::stoul(token, nullptr, 10);
        }
        result.push_back(val);
    }
    return result;
}
