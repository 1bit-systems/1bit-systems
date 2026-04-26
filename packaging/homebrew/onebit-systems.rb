# Homebrew formula for 1bit-systems.
#
# Tap install:
#   brew tap bong-water-water-bong/1bit
#   brew install onebit-systems
#
# Or one-liner:
#   brew install bong-water-water-bong/1bit/onebit-systems
#
# Note on naming: the user-facing product and the binary on disk are both
# `1bit`. Homebrew formula filenames and class names, however, are Ruby
# constants and cannot begin with a digit — so the formula file is
# `onebit-systems.rb` and the class is `OnebitSystems`. Once installed,
# everything the operator types is `1bit ...` as usual.
#
# Rule A note: this formula installs only the user-facing C++23 binaries
# from the `cpp/` tree. On macOS, the LLM / image / NPU lanes are no-ops
# (no gfx1151 silicon). The CLI + watchdog + helm tray + install manager
# still work for operating a remote 1bit server over HTTP. For full local
# inference you need a Linux + gfx1151 host and the AppImage / source
# install path.
#
# Rust gut 2026-04-26: this formula used to drive cargo. We use cmake
# now via `cmake --preset release-strix`.

class OnebitSystems < Formula
  desc "Local AI inference on Strix Halo — LLM, TTS, STT, image, video, NPU"
  homepage "https://1bit.systems"
  license "MIT"
  head "https://github.com/bong-water-water-bong/1bit-systems.git", branch: "main"

  # Tagged release artifact. Source tarball regenerates when GitHub serves
  # the v0.1.8 tag; SHA will fill in on first bottle build via CI.
  stable do
    url "https://github.com/bong-water-water-bong/1bit-systems/archive/refs/tags/v0.1.8.tar.gz"
    sha256 "SHA256_PLACEHOLDER_FILLED_BY_CI"
    version "0.1.8"
  end

  depends_on "cmake"   => :build
  depends_on "ninja"   => :build
  depends_on "gcc"     => :build  # macOS clang lacks std::print/expected at parity

  # We build only the portable subset of components. The GPU-linked
  # ones (server, lemonade, whisper, kokoro) need ROCm/HIP at link time
  # and are skipped on Homebrew (macOS lacks gfx1151 anyway).
  def install
    cd "cpp" do
      system "cmake", "--preset", "release-strix"
      system "cmake", "--build", "--preset", "release-strix",
             "--target", "1bit", "1bit-power", "1bit-watchdog",
             "1bit-helm", "1bit-voice", "1bit-echo", "1bit-mcp"
    end

    bin.install "cpp/build/strix/cli/1bit"
    bin.install "cpp/build/strix/power/1bit-power"
    bin.install "cpp/build/strix/watchdog/1bit-watchdog"
    bin.install "cpp/build/strix/helm/1bit-helm"
    bin.install "cpp/build/strix/voice/1bit-voice"
    bin.install "cpp/build/strix/echo/1bit-echo"
    bin.install "cpp/build/strix/mcp/1bit-mcp"
  end

  def caveats
    <<~EOS
      This formula installs only the orchestration binaries. For full local
      inference you need:
        - Linux host with AMD Strix Halo (gfx1151) or compatible RDNA3/3.5/4 GPU
        - ROCm 7+ (install via vendor packages, not Homebrew)
        - Models downloaded via `1bit install <model-id>`

      On macOS, this install lets you operate a remote 1bit server via HTTP
      and run the helm desktop UI pointed at it. See `1bit doctor` for a
      health summary.

      Join the community: https://github.com/bong-water-water-bong/1bit-systems
    EOS
  end

  test do
    # Version smoke — confirms the CLI binary links + runs.
    assert_match "1bit", shell_output("#{bin}/1bit --version 2>&1", 0)
    # Doctor smoke — must exit cleanly even without a gfx1151 host (it just
    # reports "no GPU detected" and returns 0).
    system "#{bin}/1bit", "doctor"
  end
end
