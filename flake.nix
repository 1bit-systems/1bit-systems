# Nix flake for 1bit-systems.
#
# Install via:
#   nix profile install github:bong-water-water-bong/1bit-systems
#   # or one-shot run:
#   nix run github:bong-water-water-bong/1bit-systems
#
# Development shell with rust 1.86 + cmake + cargo-nextest:
#   nix develop github:bong-water-water-bong/1bit-systems
#
# Rule A note: this flake only builds the portable Rust orchestration
# binaries. GPU-feature crates (1bit-hip, 1bit-mlx) are excluded because
# they depend on ROCm / MLX system libs outside the flake closure. End
# users on gfx1151 should install ROCm via their distro + run
# `1bit install all` to wire the HIP / NPU lanes.

{
  description = "1bit-systems — local AI inference on Strix Halo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };
        portableCrates = [
          "onebit-cli"
          "onebit-power"
          "onebit-watchdog"
          "onebit-helm"
          "onebit-voice"
          "onebit-echo"
          "onebit-mcp"
        ];
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "1bit-systems";
          version = "0.1.8";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [ cmake pkg-config ];
          buildInputs = with pkgs; [ openssl ];

          # Only build the portable crates; the GPU-gated ones need ROCm.
          cargoBuildFlags =
            builtins.concatMap (c: [ "-p" c ]) portableCrates;
          cargoTestFlags =
            builtins.concatMap (c: [ "-p" c ]) [
              "onebit-cli"
              "onebit-power"
              "onebit-watchdog"
              "onebit-voice"
            ];

          meta = with pkgs.lib; {
            description = "Local AI inference on Strix Halo — LLM, TTS, STT, image, video, NPU";
            homepage = "https://1bit.systems";
            license = licenses.mit;
            maintainers = [ ];
            platforms = platforms.unix;
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (rust-bin.stable."1.86.0".default.override {
              extensions = [ "rust-src" "rustfmt" "clippy" ];
            }) or rustc
            cargo
            cmake
            pkg-config
            openssl
            cargo-nextest
            cargo-audit
          ];
          shellHook = ''
            echo "1bit-systems dev shell — rust $(rustc --version | cut -d' ' -f2)"
          '';
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/1bit";
        };
      });
}
