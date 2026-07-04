const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const compile_shaders = b.option(bool, "shaders", "Compile GLSL shaders to SPIR-V (requires glslc)") orelse true;

    // ---- Shaders -------------------------------------------------------
    if (compile_shaders) {
        const shader_names: []const []const u8 = &.{
            "conv2d", "group_norm", "silu", "elementwise", "attention", "lora_merge",
        };
        for (shader_names) |name| {
            const comp = b.pathJoin(&.{ "shaders", b.fmt("{s}.comp", .{name}) });
            const spv = b.pathJoin(&.{ "shaders", b.fmt("{s}.spv", .{name}) });
            const cmd = b.addSystemCommand(&.{ "glslc", "-o", b.pathJoin(&.{ b.install_path, spv }), comp });
            b.getInstallStep().dependOn(&cmd.step);
        }
    }

    // ---- Executable ----------------------------------------------------
    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.linkSystemLibrary("vulkan", .{});
    mod.linkLibC();

    const exe = b.addExecutable(.{
        .name = "video-lora-vk",
        .root_module = mod,
    });
    b.installArtifact(exe);

    // ---- Tests ---------------------------------------------------------
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.linkSystemLibrary("vulkan", .{});
    test_mod.linkLibC();

    const test_exe = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
