class OnebitSystems < Formula
  desc "Model-agnostic NPU+GPU+CPU inference engine for AMD Strix Halo"
  homepage "https://1bit.systems"
  head "https://github.com/bong-water-water-bong/1bit-systems.git"
  url "https://github.com/bong-water-water-bong/1bit-systems/archive/refs/tags/v2026.07.20"
  sha256 "SKIP"
  license "MIT"
  version "2026.07.20"

  depends_on "gcc" => :build
  depends_on "xrt"

  def install
    system "mkdir", "-p", "engine/npu/build"
    system "gcc", "-c", "-std=c11", "-O3", "-o", "engine/npu/build/dequant_q4nx.o",
           "engine/npu/src/dequant_q4nx.c"
    # engine/npu/src/npu_engine_all.cpp was removed in the orphaned-variant
    # cleanup (PR #312); npu_engine_universal.cpp is the current engine source
    # (see README.md's Backends section and .github/workflows/bench.yml, which
    # builds this exact target on real Strix Halo hardware).
    system "g++", "-std=c++23", "-O3", "-march=native", "-fopenmp", "-ffast-math",
           "-o", "1bit-npu",
           "engine/npu/src/npu_engine_universal.cpp",
           "engine/npu/build/dequant_q4nx.o",
           "npu-infer/src/flm_bridge.cpp",
           "-Iengine/npu/include", "-Inpu-infer/include", "-Iinclude", "-Iengine/npu", "-I.",
           "-lxrt_coreutil", "-luuid", "-lm", "-ldl", "-lpthread", "-laiebu"
    system "g++", "-std=c++23", "-O3", "-o", "1bit-server",
           "packaging/binary/server.cpp"
    bin.install "1bit-npu"
    bin.install "1bit-server"
  end

  test do
    system "#{bin}/1bit-server", "--help"
  end
end
