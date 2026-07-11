//! Shared C import for the CUDA shim — all CUDA backend modules import from here
//! to ensure type identity across compilation units (mirrors src/metal/c.zig).
//! @section CUDA Runtime
//!
//! Keeping the `@cImport` in one place avoids duplicate opaque C types across
//! Zig compilation units, which is critical for safely passing shim handles
//! between the CUDA device, buffer, pipeline, and command helpers.
/// Raw CUDA shim C bindings (Driver API + NVRTC) imported from cuda_shim.h.
pub const shim = @cImport(@cInclude("cuda_shim.h"));
