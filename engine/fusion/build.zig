//! Build configuration for the fused NPU+GPU inference engine.
//! Links the NPU backend (XRT xclbin kernels), FLM proxy, unified scheduler,
//! and HTTP server into a single binary.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const target = b.standardTargetOptions(.{ .default_target = .{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
    } });

    // ── Unified scheduler module (shared KV cache) ──
    const sched_module = b.createModule(.{
        .root_source_file = b.path("sched/scheduler.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── NPU engine module (from engine/npu/src) ──
    const npu_engine_module = b.createModule(.{
        .root_source_file = b.path("../npu/src/npu_engine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    npu_engine_module.addImport("sched", sched_module);

    // ── Fusion engine main module ──
    const root_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    root_mod.addImport("sched", sched_module);
    root_mod.addImport("npu_engine", npu_engine_module);

    // ── Executable ──
    const exe = b.addExecutable(.{
        .name = "fused-engine",
        .root_module = root_mod,
    });

    // ── Install ──
    b.installArtifact(exe);

    // ── Run step ──
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the fused NPU+GPU engine");
    run_step.dependOn(&run_cmd.step);

    // ── Test step ──
    const test_step = b.step("test", "Run unit tests");

    const test_sources = [_][]const u8{
        "interop.zig",
        "dispatcher.zig",
        "memory.zig",
        "flm_proxy.zig",
    };

    for (test_sources) |src| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
        });

        const test_exe = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }
}
