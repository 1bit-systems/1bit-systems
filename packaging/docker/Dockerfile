# 1bit-systems — container image (8th distribution channel)
#
# Rule A (hard): no Python in this image. No `pip install`, no `apt install
# python3-*` runtime deps, no torch. The serving path is pure Rust+HIP.
# If a future contributor proposes a Python layer, reject the PR and point
# them at `CLAUDE.md` → Rule A.
#
# ROCm / HIP is NOT bundled. Kernels are loaded from the HOST via device
# passthrough. At `docker run` time, the operator must pass:
#
#   docker run --rm -it \
#     --device=/dev/kfd \
#     --device=/dev/dri \
#     --group-add video \
#     --group-add render \
#     -v 1bit-models:/root/.local/share/1bit \
#     -p 8180:8180 -p 8190:8190 -p 8200:8200 -p 8085:8085 -p 8095:8095 -p 1234:1234 \
#     1bit-systems:0.1.8-ci doctor
#
# Without those device flags the runtime will fall back to CPU-only paths
# and all GPU engines (HIP/Vulkan) will fail `1bit doctor` preflight.
#
# Build:
#   docker build -f packaging/docker/Dockerfile -t 1bit-systems:0.1.8-ci .
#
# Multi-stage layout:
#   Stage 1 (builder): rust:1.88-slim-bookworm, cargo build --release for the
#                      portable (non-GPU-linked) crate subset.
#   Stage 2 (runtime): debian:bookworm-slim, libc + libvulkan1, binaries
#                      copied from builder stage, non-root-safe layout.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM rust:1.88-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Minimal build deps. NO python3, NO pip, NO torch. Rule A.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy workspace manifests + source. Cache-friendly: Cargo.toml/lock first.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY tools  ./tools
COPY rocm-cpp ./rocm-cpp

# Portable subset only — GPU-linked crates (1bit-server, 1bit-router, 1bit-hip,
# 1bit-lemonade, 1bit-whisper, 1bit-kokoro, 1bit-landing) pull rocm-cpp/HIP at
# link time and are NOT built inside the container. They come from the host
# via bind-mount or live on the host's install path. The image ships only the
# orchestration/CLI/observability binaries.
RUN cargo build --release \
        -p onebit-cli \
        -p onebit-power \
        -p onebit-watchdog \
        -p onebit-helm \
        -p onebit-voice \
        -p onebit-echo \
        -p onebit-mcp

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Runtime deps only. libvulkan1 because some engines probe Vulkan before
# falling back; libc is implicit in the base image but called out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libc6 \
        libvulkan1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Binaries. Names mirror the Cargo `[[bin]]` entries discovered via
# `cargo metadata`; if a crate renames a bin, update here.
COPY --from=builder /src/target/release/1bit            /usr/local/bin/1bit
COPY --from=builder /src/target/release/1bit-halo-power /usr/local/bin/1bit-halo-power
COPY --from=builder /src/target/release/1bit-watchdog   /usr/local/bin/1bit-watchdog
COPY --from=builder /src/target/release/1bit-helm       /usr/local/bin/1bit-helm
COPY --from=builder /src/target/release/1bit-voice      /usr/local/bin/1bit-voice
COPY --from=builder /src/target/release/1bit-echo       /usr/local/bin/1bit-echo
COPY --from=builder /src/target/release/1bit-mcp        /usr/local/bin/1bit-mcp

# Model cache. First-run `1bit install <model>` writes here.
VOLUME ["/root/.local/share/1bit"]

# Port map (matches strixhalo systemd units):
#   8085 — echo (voice-duplex)
#   8095 — tts  (kokoro)
#   1234 — sd   (stable-diffusion.cpp)
#   8180 — server (OpenAI-compat /v1)
#   8190 — whisper / landing (dual-bound on host; container picks one)
#   8200 — lemonade (/v1/models gateway)
EXPOSE 8085 8095 1234 8180 8190 8200

# OCI labels for registry/ghcr surfacing.
LABEL org.opencontainers.image.source="https://github.com/bong-water-water-bong/1bit-systems" \
      org.opencontainers.image.version="0.1.8-ci" \
      org.opencontainers.image.title="1bit-systems" \
      org.opencontainers.image.description="Local AI inference on Strix Halo — LLM / TTS / STT / image / video / NPU. Rust orchestration, HIP kernels, no Python." \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/usr/local/bin/1bit"]
CMD ["doctor"]
