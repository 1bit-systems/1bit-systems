//! Backend-neutral packet batch types for ZINC_RT rings.
//! These packet structs are the handoff point between lowered IR and concrete
//! ring implementations such as T-CPU, T2 UMQ, and future direct tiers.
//! @section Inference Runtime
const rms_norm = @import("../isa/cpu_zig/rms_norm.zig");
const swiglu = @import("../isa/cpu_zig/swiglu.zig");
const argmax = @import("../isa/cpu_zig/argmax.zig");
const embed = @import("../isa/cpu_zig/embed.zig");
const lm_head = @import("../isa/cpu_zig/lm_head.zig");

/// Tagged union describing one unit of work submitted to a ZINC_RT ring.
/// Each variant carries the CPU ISA parameter struct that fully describes a
/// single decode-step kernel; the `barrier` variant marks an in-stream
/// ordering point with no shader payload.
pub const Packet = union(enum) {
    embed: embed.Params,
    rms_norm: rms_norm.Params,
    lm_head: lm_head.Params,
    swiglu: swiglu.Params,
    argmax: argmax.Params,
    barrier: void,
};

/// Borrowed slice of packets that form one submission to a ring.
/// Ring implementations consume a batch in order, treating `.barrier` entries
/// as completion fences between adjacent dispatches.
pub const PacketBatch = struct {
    packets: []const Packet,
};
