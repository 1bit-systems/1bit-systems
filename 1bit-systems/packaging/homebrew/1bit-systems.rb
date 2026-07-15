class OnebitSystems < Formula
  desc "One binary to rule them all — 5 models, 120KB, 28 tok/s NPU"
  homepage "https://1bit.systems"
  head "https://github.com/bong-water-water-bong/1bit-systems.git"
  url "https://github.com/bong-water-water-bong/1bit-systems/archive/refs/tags/v2026.07.02-all5models.tar.gz"
  sha256 "SKIP"
  license "MIT"
  version "2026.07.02"

  depends_on "gcc" => :build
  depends_on "xrt"

  def install
    system "gcc", "-c", "-std=c11", "-O3", "-o", "engine/npu/build/dequant_q4nx.o",
           "engine/npu/src/dequant_q4nx.c"
    system "g++", "-std=c++23", "-O3", "-march=native", "-fopenmp", "-ffast-math",
           "-o", "1bit-npu",
           "engine/npu/src/npu_engine_all.cpp",
           "engine/npu/build/dequant_q4nx.o",
           "-Iengine/npu/src",
           "-lxrt_coreutil", "-luuid", "-lm", "-ldl"
    system "g++", "-std=c++23", "-O3", "-o", "1bit-server",
           "packaging/binary/server.cpp"
    bin.install "1bit-npu"
    bin.install "1bit-server"
  end

  test do
    system "#{bin}/1bit-server", "--help"
  end
end
