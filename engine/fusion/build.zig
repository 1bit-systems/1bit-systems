//! Build configuration for the fused NPU+GPU inference engine.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSmall });
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu } });

    // Modules
    const sched_mod = b.createModule(.{ .root_source_file = b.path("sched/scheduler.zig"), .target = target, .optimize = optimize });
    const npu_mod = b.createModule(.{ .root_source_file = b.path("../npu/src/npu_engine.zig"), .target = target, .optimize = optimize, .link_libc = true });
    npu_mod.addImport("sched", sched_mod);

    const cpu_mod = b.createModule(.{ .root_source_file = b.path("cpu_backend.zig"), .target = target, .optimize = optimize, .link_libc = true });

    const vk_mod = b.createModule(.{ .root_source_file = b.path("vk_wrapper.zig"), .target = target, .optimize = optimize, .link_libc = true });
    vk_mod.linkSystemLibrary("vulkan", .{});
    vk_mod.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
    vk_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    const vulkan_c_mod = b.createModule(.{ .root_source_file = b.path("vulkan_c.zig"), .target = target, .optimize = optimize, .link_libc = true });
    vulkan_c_mod.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
    vulkan_c_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // Main module
    const root_mod = b.createModule(.{ .root_source_file = b.path("main.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true });
    root_mod.addImport("sched", sched_mod);
    root_mod.addImport("npu_engine", npu_mod);
    root_mod.addImport("cpu_backend", cpu_mod);
    root_mod.addImport("vk_wrapper", vk_mod);
    root_mod.addImport("vulkan_c", vulkan_c_mod);
    root_mod.addCSourceFile(.{ .file = b.path("cpu_layer.cpp"), .flags = &.{ "-O3", "-march=native", "-std=c++17" } });
    root_mod.addCSourceFile(.{ .file = b.path("../npu/src/dequant_q4nx.c"), .flags = &.{ "-O3", "-march=native", "-std=c11" } });
    root_mod.addCSourceFile(.{ .file = b.path("tokenize.cpp"), .flags = &.{ "-O3", "-march=native", "-std=c++17", "-DTOKENIZER_NO_MAIN" } });

    const exe = b.addExecutable(.{ .name = "fused-engine", .root_module = root_mod });
    exe.linker_allow_shlib_undefined = true;
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run fused engine").dependOn(&run_cmd.step);

    // Sci-Fi submodules
    const gguf_mod = b.createModule(.{ .root_source_file = b.path("gguf/gguf_reader.zig"), .target = target, .optimize = optimize });
    const gpu_attn_mod = b.createModule(.{ .root_source_file = b.path("gpu_attn.zig"), .target = target, .optimize = optimize, .link_libc = true });
    gpu_attn_mod.addImport("vk_wrapper", vk_mod);
    gpu_attn_mod.addImport("vulkan_c", vulkan_c_mod);

    const server_mod = b.createModule(.{ .root_source_file = b.path("server.zig"), .target = target, .optimize = optimize });
    server_mod.addImport("flm_proxy", b.createModule(.{ .root_source_file = b.path("flm_proxy.zig"), .target = target, .optimize = optimize }));
    server_mod.addImport("dispatcher", b.createModule(.{ .root_source_file = b.path("dispatcher.zig"), .target = target, .optimize = optimize }));
    server_mod.addImport("engine", b.createModule(.{ .root_source_file = b.path("engine.zig"), .target = target, .optimize = optimize }));

    const interop_mod = b.createModule(.{ .root_source_file = b.path("interop.zig"), .target = target, .optimize = optimize, .link_libc = true });
    _ = gguf_mod;
    _ = interop_mod;

    // Tests
    const test_step = b.step("test", "Run unit tests");
    for ([_] []const u8{ "dispatcher.zig", "memory.zig", "flm_proxy.zig", "cpu_backend.zig", "model_data.zig", "sched/request.zig", "sched/kv_cache.zig", "sched/scheduler.zig", "gguf/gguf_reader.zig" }) |src| {
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
        m.addImport("vulkan_c", vulkan_c_mod);
        m.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Interop test
    {
        const m = b.createModule(.{ .root_source_file = b.path("interop.zig"), .target = target, .optimize = optimize, .link_libc = true });
        m.addImport("sched", sched_mod);
        m.addImport("npu_engine", npu_mod);
        m.addImport("vk_wrapper", vk_mod);
        m.addImport("vulkan_c", vulkan_c_mod);
        m.linkSystemLibrary("vulkan", .{});
        m.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // GpuAttn test
    {
        const m = b.createModule(.{ .root_source_file = b.path("gpu_attn.zig"), .target = target, .optimize = optimize, .link_libc = true });
        m.addImport("vk_wrapper", vk_mod);
        m.addImport("vulkan_c", vulkan_c_mod);
        m.linkSystemLibrary("vulkan", .{});
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Shader build step
    const shader_step = b.step("shaders", "Build Vulkan/CUDA/Metal shaders from source");
    const shader_build = b.addSystemCommand(&.{ "bash", "shaders/build_shaders.sh" });
    shader_step.dependOn(&shader_build.step);

    // Convenience: build all
    b.step("all", "Build engine + shaders + tests").dependOn(shader_step);
}
