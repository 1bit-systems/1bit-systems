//! Build configuration for the fused NPU+GPU inference engine.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSmall });
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu } });

    // Modules
    const sched_mod = b.createModule(.{ .root_source_file = b.path("sched/scheduler.zig"), .target = target, .optimize = optimize });
    const npu_mod = b.createModule(.{ .root_source_file = b.path("../npu/src/npu_engine.zig"), .target = target, .optimize = optimize, .link_libc = true });
    npu_mod.addImport("sched", sched_mod);

    const vk_mod = b.createModule(.{ .root_source_file = b.path("vk_wrapper.zig"), .target = target, .optimize = optimize, .link_libc = true });
    vk_mod.linkSystemLibrary("vulkan", .{});
    vk_mod.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
    vk_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // Main module
    const root_mod = b.createModule(.{ .root_source_file = b.path("main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    root_mod.addImport("sched", sched_mod);
    root_mod.addImport("npu_engine", npu_mod);
    root_mod.addImport("vk_wrapper", vk_mod);

    const exe = b.addExecutable(.{ .name = "fused-engine", .root_module = root_mod });
    exe.linker_allow_shlib_undefined = true;
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run fused engine").dependOn(&run_cmd.step);

    // Tests
    const test_step = b.step("test", "Run unit tests");
    for ([_] []const u8{ "dispatcher.zig", "memory.zig", "flm_proxy.zig" }) |src| {
        const m = b.createModule(.{ .root_source_file = b.path(src), .target = target, .optimize = optimize });
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Fused execute test
    {
        const m = b.createModule(.{ .root_source_file = b.path("fused_execute.zig"), .target = target, .optimize = optimize, .link_libc = true });
        m.addImport("sched", sched_mod);
        m.addImport("npu_engine", npu_mod);
        m.addImport("vk_wrapper", vk_mod);
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Interop test
    {
        const m = b.createModule(.{ .root_source_file = b.path("interop.zig"), .target = target, .optimize = optimize, .link_libc = true });
        m.addImport("sched", sched_mod);
        m.addImport("npu_engine", npu_mod);
        m.addImport("vk_wrapper", vk_mod);
        m.linkSystemLibrary("vulkan", .{});
        m.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
