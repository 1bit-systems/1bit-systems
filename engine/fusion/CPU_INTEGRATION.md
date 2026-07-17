// engine/fusion/build.zig patch — add cpu_backend module
//
// Add to the exe import in build.zig:
//
//   const cpu_backend = b.createModule(.{
//       .source_file = .{ .path = "engine/fusion/cpu_backend.zig" },
//       .dependencies = &.{},
//   });
//   exe.addModule("cpu_backend", cpu_backend);
//
// And add cpu_layer.cpp to the exe link:
//
//   exe.addCSourceFile("engine/fusion/cpu_layer.cpp", &.{ "-O3", "-march=native", "-std=c++17" });
//   exe.linkLibC();
