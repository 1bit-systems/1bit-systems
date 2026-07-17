//! ZINC_RT reference-runtime module.
//! Exposes the M0 engine, IR, CPU ring, and scalar CPU kernels through one
//! importable package so `forward_zinc_rt` can exercise the runtime without
//! pulling the same files into multiple Zig modules.
//! @section Inference Runtime

/// Top-level runtime handle and tier-selection helpers.
/// Other zinc_rt submodules go through this surface to obtain the active
/// execution tier (CPU reference, T1 PM4, T2 UMQ, Metal, etc.).
pub const engine = @import("engine.zig");

/// IR opcode definitions used by the M0/M1 decode graph.
/// Op kinds are stable identifiers shared by the emitter and the validator.
pub const ir_op = @import("ir/op.zig");

/// IR graph container — nodes plus their op/argument metadata.
/// `forward_zinc_rt` emits one of these per token before lowering to the tier
/// the engine selected.
pub const ir_graph = @import("ir/graph.zig");

/// Kernel-mode-driver glue. Houses helpers and constants shared by the ring
/// backends (UMQ user queues, KFD compute queues) so they describe submissions
/// in one place.
pub const kmd = @import("kmd.zig");

/// Common ring-submission surface — what every concrete ring backend
/// (CPU, UMQ, KFD, CS) implements so the engine can speak to them uniformly.
pub const ring = @import("ring/mod.zig");

/// Reference CPU ring used by the scalar M1 forward path. Executes ops
/// synchronously on the host so we have a ground-truth correctness oracle for
/// the GPU rings.
pub const cpu_ring = @import("ring/cpu.zig");

/// T2 user-mode queue ring backed by `DRM_IOCTL_AMDGPU_USERQ`. The "blessed"
/// direct path when amdgpu firmware admits compute user queues.
pub const umq = @import("ring/umq.zig");

/// T1 PM4 ring backed by `/dev/kfd` (the ROCm/tinygrad ABI). Fallback when
/// UMQ is rejected; talks PM4 packets to the compute scheduler directly.
pub const kfd = @import("ring/kfd.zig");

/// Generic libdrm compute-stream submission used by the PM4 backends.
/// Wraps context create/submit so the ring backends share one BO/syncobj path.
pub const cs = @import("ring/cs.zig");

/// PM4 packet builders (NOPs, indirect-buffer dispatch, COPY_DATA, fences).
/// Consumed by both the UMQ and KFD rings to produce wire-compatible
/// command-buffer bytes.
pub const pm4_packet = @import("ring/packet.zig");

/// Scalar CPU kernels (dequantization, matvec, softmax, etc.) used by the
/// reference forward path and by GPU-tier correctness checks.
pub const kernels = @import("isa/cpu_zig/mod.zig");

/// Tiny fixed-size worker pool used to fan out per-layer / per-expert work in
/// the scalar M1 decode path without paying allocator/scheduler overhead.
pub const fast_pool = @import("fast_pool.zig");

/// Tenant-aware admission and batch-selection primitives for the M3
/// continuous-batching executor.
pub const batching = @import("batching.zig");

comptime {
    _ = @import("tests/ir_smoke.zig");
    _ = @import("fast_pool.zig");
    _ = @import("batching.zig");
}
