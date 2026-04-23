# Homebrew formula for 1bit-systems.
#
# Tap install:
#   brew tap bong-water-water-bong/1bit
#   brew install 1bit-systems
#
# Or one-liner:
#   brew install bong-water-water-bong/1bit/1bit-systems
#
# Rule A note: this formula installs only the user-facing Rust binaries. On
# macOS, the LLM / image / NPU lanes are no-ops (no gfx1151 silicon). The
# CLI + watchdog + helm tray + install manager still work for operating a
# remote 1bit server over HTTP. For full local inference you need a
# Linux + gfx1151 host and the AppImage / source install path.

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

  depends_on "rust" => :build

  # Installed per-binary rather than `cargo install --workspace` because the
  # workspace contains GPU-feature-gated crates (1bit-hip, 1bit-mlx) that
  # won't build on a generic Homebrew box. We ship only the portable Rust
  # orchestration binaries.
  def install
    portable_bins = %w[
      1bit-cli
      1bit-power
      1bit-watchdog
      1bit-helm
      1bit-voice
      1bit-echo
      1bit-mcp
    ]

    portable_bins.each do |c|
      system "cargo", "install", *std_cargo_args(path: "crates/#{c}")
    end

    # The core CLI's actual bin name is `1bit`, not `1bit-cli`. Symlink so
    # both shell histories work.
    bin.install_symlink bin/"1bit-cli" => "1bit" if (bin/"1bit-cli").exist?
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
