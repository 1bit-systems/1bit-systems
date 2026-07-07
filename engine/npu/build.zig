const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const target = b.standardTargetOptions(.{ .default_target = .{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
    } });

    // XRT paths from env or defaults
    const xrt_include = b.option([]const u8, "xrt-include", "XRT include dir") orelse
        "/home/bcloud/torch2aie/toolchain/xrt/include";
    const xrt_lib = b.option([]const u8, "xrt-lib", "XRT lib dir") orelse
        "/home/bcloud/torch2aie/toolchain/xrt/lib";

    // Create a single module for the GPU scheduler (unified KV cache layer)
    // scheduler.zig is the entry point; it re-exports KvPagePool, Request, etc.
    const sched_module = b.createModule(.{
        .root_source_file = b.path("../fusion/sched/scheduler.zig"),
    });

    // ── Reusable NPU engine module (for fusion engine import) ──
    _ = b.createModule(.{
        .root_source_file = b.path("src/npu_engine.zig"),
    });
    // Note: external builds (e.g., engine/fusion/) can set up the npu_engine module
    // themselves with: module.addImport("sched", sched_module)

    // Create the executable module
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Register scheduler module
    root_mod.addImport("sched", sched_module);

    const exe = b.addExecutable(.{
        .name = "npu-engine",
        .root_module = root_mod,
    });

    // C source for dequant
    root_mod.addCSourceFile(.{
        .file = b.path("src/dequant_q4nx.c"),
        .flags = &.{"-std=c11"},
    });

    // System libraries
    inline for (.{ "xrt_coreutil", "uuid", "m", "dl", "pthread" }) |lib| {
        root_mod.linkSystemLibrary(lib, .{});
    }

    // XRT paths
    root_mod.addIncludePath(.{ .cwd_relative = xrt_include });
    root_mod.addLibraryPath(.{ .cwd_relative = xrt_lib });
    root_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // Install
    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the NPU engine");
    run_step.dependOn(&run_cmd.step);

    // Test step
    const test_step = b.step("test", "Run unit tests");

    const test_sources = [_][]const u8{
        "src/model_reader.zig",
        "src/cpu_ops.zig",
        "src/xrt.zig",
        "src/npu_page_table.zig",
        "src/cdequant.zig",
    };

    for (test_sources) |src| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        test_mod.addIncludePath(.{ .cwd_relative = xrt_include });
        test_mod.addLibraryPath(.{ .cwd_relative = xrt_lib });
        inline for (.{ "xrt_coreutil", "uuid", "m" }) |lib| {
            test_mod.linkSystemLibrary(lib, .{});
        }

        const test_exe = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }
}
