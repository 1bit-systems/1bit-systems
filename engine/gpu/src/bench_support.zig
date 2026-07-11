//! Shared helpers for benchmark and standalone runner entrypoints.
//!
//! This module re-exports the Metal runtime pieces that the benchmark tools
//! need and centralizes the GPU-process-lock error path so the small bench
//! binaries do not duplicate server/runtime boilerplate.
//! @section Inference Runtime
const std = @import("std");

/// Metal device initialization and capability queries (MTLDevice wrapper).
pub const metal_device = @import("metal/device.zig");
/// Model loader that maps GGUF weights onto Metal buffers.
pub const metal_loader = @import("model/loader_metal.zig");
/// Metal buffer allocation and management utilities.
pub const metal_buffer = @import("metal/buffer.zig");
/// Metal command queue and command buffer submission helpers.
pub const metal_command = @import("metal/command.zig");
/// Per-kernel Metal dispatch timing probe for profiling individual GPU kernels.
pub const kernel_timing = @import("metal/kernel_timing.zig");
/// Metal compute pipeline state cache and compilation helpers.
pub const metal_pipeline = @import("metal/pipeline.zig");
/// Raw Objective-C/Metal C shim types and bindings.
pub const metal_c = @import("metal/c.zig");
/// GGUF file parser for reading quantized model weights and metadata.
pub const gguf = @import("model/gguf.zig");
/// BPE tokenizer encode and decode for text pre/post-processing.
pub const tokenizer_mod = @import("model/tokenizer.zig");
/// Metal forward-pass runtime that runs the full model inference graph.
pub const forward_metal = @import("compute/forward_metal.zig");
/// Cross-process GPU ownership lock preventing two zinc processes from sharing a GPU.
pub const process_lock = @import("gpu/process_lock.zig");

/// Log a user-facing GPU-process-lock error and terminate the benchmark binary.
///
/// Prints a human-readable message to stderr explaining why the lock could not
/// be acquired, then calls `std.process.exit(1)`.
///
/// @param err         The lock-acquisition error; `error.GpuAlreadyReserved` gets a
///                    dedicated "stop the other instance" message; all other errors
///                    fall back to a generic failure message.
/// @param backend     The GPU backend whose lock failed (used in the log message).
/// @param device_index Index of the GPU device that could not be locked (used in the log message).
pub fn reportGpuProcessLockError(err: anyerror, backend: process_lock.Backend, device_index: u32) noreturn {
    switch (err) {
        error.GpuAlreadyReserved => std.log.err(
            "GPU {s}:{d} is already reserved by another zinc process. Stop the other instance before loading a second model on the same GPU.",
            .{ @tagName(backend), device_index },
        ),
        else => std.log.err("Failed to acquire GPU process lock for {s}:{d}: {s}", .{
            @tagName(backend),
            device_index,
            @errorName(err),
        }),
    }
    std.process.exit(1);
}
