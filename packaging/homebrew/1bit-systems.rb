class OnebitSystems < Formula
  desc "50 TOPS INT8 NPU inference engine for AMD Strix Halo"
  homepage "https://1bit.systems"
  url "https://github.com/bong-water-water-bong/1bit-systems/archive/refs/tags/v2026.07.01.tar.gz"
  sha256 "SKIP"
  license "MIT"
  version "2026.07.01"

  depends_on "gcc" => :build
  depends_on "xrt"

  def install
    system "gcc", "-c", "-O3", "-o", "engine/npu/build/dequant_q4nx.o",
           "engine/npu/src/dequant_q4nx.c"
    system "g++", "-std=c++23", "-O3", "-o", "1bit-npu",
           "engine/npu/src/npu_engine_cb.cpp",
           "engine/npu/build/dequant_q4nx.o",
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
