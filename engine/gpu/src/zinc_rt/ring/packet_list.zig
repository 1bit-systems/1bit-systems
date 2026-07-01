//! Dynamic packet list for building per-token decode sequences.
//! T-CPU forward passes use this to accumulate packets before submitting
//! them to the CPU ring for execution.
//! @section Inference Runtime
const std = @import("std");
const ring = @import("mod.zig");

/// Growable buffer used to assemble a `PacketBatch` for one decode step.
/// Lowering code pushes packets in execution order, interleaves explicit
/// `barrier` entries between dependent dispatches, and then publishes the
/// resulting slice via `slice()` to whichever ring will run the batch.
pub const PacketList = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(ring.Packet),

    /// Create an empty list bound to the given allocator.
    /// @param allocator Owner of the underlying ArrayList storage; must
    ///     outlive the list.
    /// @returns A `PacketList` with zero packets queued.
    pub fn init(allocator: std.mem.Allocator) PacketList {
        return .{
            .allocator = allocator,
            .items = std.ArrayList(ring.Packet).init(allocator),
        };
    }

    /// Release the backing storage and poison `self` for use-after-free debug.
    /// @param self List to tear down; must not be reused afterwards.
    pub fn deinit(self: *PacketList) void {
        self.items.deinit(self.allocator);
        self.* = undefined;
    }

    /// Push a payload-bearing packet (embed, rms_norm, lm_head, etc.) onto the
    /// end of the list.
    /// @param self List receiving the packet.
    /// @param packet Fully populated packet to enqueue; the value is copied.
    pub fn append(self: *PacketList, packet: ring.Packet) !void {
        try self.items.append(self.allocator, packet);
    }

    /// Append a `.barrier` marker that forces the ring to drain prior packets
    /// before continuing.
    /// @param self List receiving the barrier.
    /// @note Used between producer/consumer kernels that share a tensor
    ///     buffer (for example, between attention output and the next RMSNorm).
    pub fn appendBarrier(self: *PacketList) !void {
        try self.items.append(self.allocator, .barrier);
    }

    /// Borrowed view of the current packet sequence.
    /// @param self List to inspect.
    /// @returns A slice that stays valid until the next mutation of the list.
    pub fn slice(self: *const PacketList) []const ring.Packet {
        return self.items.items;
    }

    /// Number of packets currently queued, including any barriers.
    /// @param self List to inspect.
    /// @returns Packet count.
    pub fn len(self: *const PacketList) usize {
        return self.items.items.len;
    }

    /// Drop all queued packets while keeping the allocated capacity, so the
    /// list can be reused for the next decode step without re-allocating.
    /// @param self List to reset.
    pub fn clear(self: *PacketList) void {
        self.items.clearRetainingCapacity();
    }
};
