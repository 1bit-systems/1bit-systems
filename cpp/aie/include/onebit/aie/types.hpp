// onebit::aie types — buffer / kernel-handle / dtype / device-info.
//
// Match AieBuffer / AieKernelHandle / AieDtype / AieDeviceInfo from
// the Rust crate.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace onebit::aie {

// Element types the BitNet NPU pipeline understands. Deliberately
// narrow — AIE2P does int8 / int16 / int32 / bf16 natively; IEEE fp16
// is emulated and not competitive.
enum class Dtype : std::uint8_t {
    PackedT2 = 0,   // 2-bit packed ternary (4 codes / byte)
    I8       = 1,
    I32      = 2,
    Bf16     = 3,
};

[[nodiscard]] constexpr std::string_view label(Dtype d) noexcept {
    switch (d) {
        case Dtype::PackedT2: return "packed_t2";
        case Dtype::I8:       return "i8";
        case Dtype::I32:      return "i32";
        case Dtype::Bf16:     return "bf16";
    }
    return "unknown";
}

// Opaque handle for a loaded .xclbin kernel. The u32 is a backend-local
// index, NOT a pointer — it is stable across calls but does not alias
// any host or device memory.
struct KernelHandle {
    std::uint32_t id{0};
    constexpr bool operator==(const KernelHandle&) const noexcept = default;
};

// Buffer object metadata. The actual `xrt::bo` (when libxrt is loaded)
// lives behind the backend's pImpl indexed by `id`.
struct Buffer {
    std::uint32_t id{0};
    std::size_t   len_bytes{0};
    Dtype         dtype{Dtype::I8};
    constexpr bool operator==(const Buffer&) const noexcept = default;
};

// Reported by Backend::device_info. Stable shape; if it grows to live
// telemetry, wrap with a method instead of breaking this.
struct DeviceInfo {
    std::string device_name;          // e.g. "RyzenAI-npu5"
    std::string firmware_version;     // e.g. "1.0.0.166"
    std::uint8_t columns{0};          // compute columns reported by xrt-smi
    std::string_view tile_class{"AIE2P"}; // AIE2 (Phoenix) / AIE2P (Strix Halo)
};

} // namespace onebit::aie
