//! Build configuration for the fused NPU+GPU inference engine.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const target = b.standardTargetOptions(.{ .default_target = .{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
        .abi = .gnu,
    } });

    // ── Unified scheduler module ──
    const sched_module = b.createModule(.{
        .root_source_file = b.path("sched/scheduler.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── NPU engine module ──
    const npu_engine_module = b.createModule(.{
        .root_source_file = b.path("../npu/src/npu_engine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    npu_engine_module.addImport("sched", sched_module);

    // ── NPU page table module ──
    const npu_page_table_module = b.createModule(.{
        .root_source_file = b.path("../npu/src/npu_page_table.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── Vulkan module ──
    const vk_wrapper_module = b.createModule(.{
        .root_source_file = b.path("vk_wrapper.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    vk_wrapper_module.linkSystemLibrary("vulkan", .{});
    vk_wrapper_module.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
    vk_wrapper_module.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // ── Fusion engine main module ──
    const root_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root_mod.addImport("sched", sched_module);
    root_mod.addImport("npu_engine", npu_engine_module);
    root_mod.addImport("npu_page_table", npu_page_table_module);
    root_mod.addImport("vk_wrapper", vk_wrapper_module);

    // ── Executable ──
    const exe = b.addExecutable(.{ .name = "fused-engine", .root_module = root_mod });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the fused NPU+GPU engine");
    run_step.dependOn(&run_cmd.step);

    // ── Tests ──
    const test_step = b.step("test", "Run unit tests");

    for ([_] []const u8{ "dispatcher.zig", "memory.zig", "flm_proxy.zig" }) |src| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
        });
        const test_exe = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }

    // Interop test (needs Vulkan deps)
    {
        const test_mod = b.createModule(.{
            .root_source_file = b.path("interop.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        test_mod.addImport("sched", sched_module);
        test_mod.addImport("npu_page_table", npu_page_table_module);
        test_mod.addImport("vk_wrapper", vk_wrapper_module);
        test_mod.linkSystemLibrary("vulkan", .{});
        test_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

        const test_exe = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }
}
