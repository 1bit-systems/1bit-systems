//! ZINC_RT — the ZINC Runtime.
//! Owns tier selection and the top-level runtime handle used by future IR
//! emitters and ring backends.
//! @section Inference Runtime
// SPDX-FileCopyrightText: ZINC Authors
const std = @import("std");
const builtin = @import("builtin");
const kmd = @import("kmd.zig");
const umq = @import("ring/umq.zig");
const kfd = @import("ring/kfd.zig");

/// Execution tier the engine will dispatch through.
/// `t1_pm4` and `t2_umq` are the two direct AMDGPU paths; `t_cpu` is the
/// reference scalar fallback; `t_metal`, `t_intel`, and `t_cuda` are reserved
/// for the corresponding native backends.
pub const Tier = enum {
    t1_pm4,
    t2_umq,
    t_cpu,
    t_metal,
    t_intel,
    t_cuda,
};

/// Caller-supplied configuration for `Engine.init`. Currently just the desired
/// tier; future fields (worker counts, ring depths, telemetry hooks) live here.
pub const Options = struct {
    tier: Tier = .t_cpu,
};

/// Runtime capability bits surfaced to harnesses and server integration code.
/// The batch planner is present, but continuous batched model execution is not
/// wired into `forward_zinc_rt` yet.
pub const Capabilities = struct {
    /// Tenant-aware admission and prefill/decode batch selection is available.
    multitenant_batch_planning: bool = true,
    /// Requests must name a registered tenant before admission.
    requires_explicit_tenant_registration: bool = true,
    /// End-to-end batched inference execution is available.
    multitenant_batched_execution: bool = false,
    /// The `-Dbackend=zinc_rt` binary exposes an HTTP server entrypoint.
    server_entrypoint: bool = false,
};

/// True when ZINC_RT can plan multitenant batches.
pub const supports_multitenant_batch_planning = true;
/// True only when ZINC_RT can execute multiple tenants in one continuous
/// inference loop. This remains false until M3 wires the planner into
/// `forward_zinc_rt` and the server runtime.
pub const supports_multitenant_batched_execution = false;

/// Top-level runtime handle. Owns the allocator the engine was built with and
/// the selected tier; future revisions will also own the ring backend.
pub const Engine = struct {
    allocator: std.mem.Allocator,
    tier: Tier,

    /// Construct an engine pinned to the requested tier.
    /// @param allocator Allocator used for any engine-owned state.
    /// @param options Configuration block; `options.tier` selects the backend.
    /// @returns A ready-to-use `Engine`.
    pub fn init(allocator: std.mem.Allocator, options: Options) !Engine {
        return .{
            .allocator = allocator,
            .tier = options.tier,
        };
    }

    /// Release any engine-owned state and poison the handle. Safe to call
    /// once per successful `init`.
    pub fn deinit(self: *Engine) void {
        self.* = undefined;
    }

    /// Return static runtime capability bits for the selected build.
    pub fn capabilities(self: *const Engine) Capabilities {
        _ = self;
        return .{};
    }
};

/// Parse a textual tier identifier (e.g. from `ZINC_RT_TIER` or a CLI flag)
/// into a `Tier`. Accepts both short (`t1`, `t2`) and canonical (`t1_pm4`,
/// `t2_umq`) names; `auto` defers to `autoTier`.
/// @param value String to parse.
/// @returns The selected `Tier`, or `error.UnknownZincRtTier` if `value` is
/// not a recognised name.
pub fn parseTier(value: []const u8) !Tier {
    if (std.mem.eql(u8, value, "auto")) return autoTier();
    if (std.mem.eql(u8, value, "t1") or std.mem.eql(u8, value, "t1_pm4")) return .t1_pm4;
    if (std.mem.eql(u8, value, "t2") or std.mem.eql(u8, value, "t2_umq")) return .t2_umq;
    if (std.mem.eql(u8, value, "t_cpu")) return .t_cpu;
    if (std.mem.eql(u8, value, "t_metal")) return .t_metal;
    if (std.mem.eql(u8, value, "t_intel")) return .t_intel;
    if (std.mem.eql(u8, value, "t_cuda")) return .t_cuda;
    return error.UnknownZincRtTier;
}

/// Read `ZINC_RT_TIER` and parse it, falling back to `autoTier` when unset.
/// @returns The selected `Tier`, or a parse error from `parseTier`.
pub fn tierFromEnv() !Tier {
    const value = std.posix.getenv("ZINC_RT_TIER") orelse return autoTier();
    return parseTier(value);
}

/// Probe the host for direct-execution paths and return the best available
/// tier. Tries T2 UMQ first (the blessed AMDGPU user-queue path), falls back
/// to T1 PM4 over `/dev/kfd` when UMQ admission is refused, and finally to
/// the scalar CPU reference.
/// @returns The best tier the current host can run today.
/// @note On the bench node the amdgpu firmware rejects compute user queues,
/// so T2 admission usually fails and we end up on T1 PM4.
pub fn autoTier() Tier {
    // T2 UMQ (DRM_IOCTL_AMDGPU_USERQ) is the blessed direct path, but the
    // amdgpu firmware on the bench node rejects compute user queues, so we
    // fall through to the T1 PM4 path on `/dev/kfd` (the ROCm/tinygrad ABI)
    // whenever it is reachable, and only then to the T-CPU reference.
    if (!directModelKernelsAvailableDefault()) return .t_cpu;
    if (umq.admissionProbeDefault()) return .t2_umq;
    if (kfd.reachable()) return .t1_pm4;
    return .t_cpu;
}

fn directModelKernelsAvailableDefault() bool {
    if (builtin.os.tag != .linux) return false;

    var file = std.fs.openFileAbsolute(umq.default_render_node, .{ .mode = .read_write }) catch return false;
    defer file.close();

    const hw_ip = kmd.queryHwIp(file, kmd.AMDGPU_HW_IP_COMPUTE) catch return false;
    return supportsEmbeddedGfx12Kernels(hw_ip);
}

fn supportsEmbeddedGfx12Kernels(hw_ip: kmd.DrmAmdgpuInfoHwIp) bool {
    return hw_ip.hw_ip_version_major == 12;
}

test "parseTier maps explicit CPU tier" {
    try std.testing.expectEqual(Tier.t_cpu, try parseTier("t_cpu"));
}

test "capabilities distinguish batch planning from batched execution" {
    var eng = try Engine.init(std.testing.allocator, .{});
    defer eng.deinit();

    const caps = eng.capabilities();
    try std.testing.expect(caps.multitenant_batch_planning);
    try std.testing.expect(caps.requires_explicit_tenant_registration);
    try std.testing.expect(!caps.multitenant_batched_execution);
    try std.testing.expect(!caps.server_entrypoint);
    try std.testing.expect(supports_multitenant_batch_planning);
    try std.testing.expect(!supports_multitenant_batched_execution);
}

test "auto tier embedded-kernel gate rejects pre-gfx12 compute IPs" {
    var hw_ip: kmd.DrmAmdgpuInfoHwIp = std.mem.zeroes(kmd.DrmAmdgpuInfoHwIp);
    hw_ip.hw_ip_version_major = 10;
    try std.testing.expect(!supportsEmbeddedGfx12Kernels(hw_ip));

    hw_ip.hw_ip_version_major = 11;
    try std.testing.expect(!supportsEmbeddedGfx12Kernels(hw_ip));

    hw_ip.hw_ip_version_major = 12;
    try std.testing.expect(supportsEmbeddedGfx12Kernels(hw_ip));

    hw_ip.hw_ip_version_major = 13;
    try std.testing.expect(!supportsEmbeddedGfx12Kernels(hw_ip));
}

test "auto tier remains CPU on non-Linux hosts" {
    if (@import("builtin").os.tag == .linux) return error.SkipZigTest;
    try std.testing.expectEqual(Tier.t_cpu, try parseTier("auto"));
}
