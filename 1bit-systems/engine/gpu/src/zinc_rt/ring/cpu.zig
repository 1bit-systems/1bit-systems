//! T-CPU ring backend.
//! @section Inference Runtime
//! Walks packet batches and executes pure Zig kernels as the validation oracle
//! used by every other ring tier (T1 PM4-direct, UMQ, Metal) to cross-check
//! their outputs bit-for-bit against a reference run.
const ring = @import("mod.zig");
const kernels = @import("../isa/cpu_zig/mod.zig");

/// T-CPU ring backend that executes packet batches synchronously on the CPU.
/// @note Acts as the validation oracle for the GPU ring implementations; every
/// dispatch is run by the pure Zig kernels in `isa/cpu_zig/mod.zig`.
pub const CpuRing = struct {
    /// Construct a fresh CPU ring. The backend is stateless, so this is a
    /// trivial value initializer that exists to mirror the GPU ring API.
    pub fn init() CpuRing {
        return .{};
    }

    /// Release any resources held by the ring. No-op for the CPU backend.
    pub fn deinit(_: *CpuRing) void {}

    /// Execute every packet in `batch` in order on the CPU and return when the
    /// last kernel has retired. Dispatches each opcode to its pure-Zig kernel
    /// implementation and treats `.barrier` as a no-op since execution is
    /// already synchronous.
    /// @param batch Sequence of typed packets produced by the planner.
    /// @returns Propagates any kernel error verbatim; success means every
    /// packet completed.
    pub fn submitAndWait(_: *CpuRing, batch: ring.PacketBatch) !void {
        for (batch.packets) |packet| {
            switch (packet) {
                .embed => |params| try kernels.embed.run(params),
                .rms_norm => |params| try kernels.rms_norm.run(params),
                .lm_head => |params| try kernels.lm_head.run(params),
                .swiglu => |params| try kernels.swiglu.run(params),
                .argmax => |params| try kernels.argmax.run(params),
                .barrier => {},
            }
        }
    }
};
