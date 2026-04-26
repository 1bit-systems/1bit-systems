# 1bit-systems — container image (8th distribution channel)
#
# Rule A (hard): no Python in this image. No `pip install`, no `apt install
# python3-*` runtime deps, no torch. The serving path is pure C++23+HIP.
# If a future contributor proposes a Python layer, reject the PR and point
# them at `CLAUDE.md` → Rule A.
#
# Rust gut 2026-04-26: this Dockerfile used to drive cargo. The
# orchestration tower is C++23 in `cpp/` now. Builder stage uses
# debian:bookworm-slim + g++-14 + cmake + ninja.
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
#   Stage 1 (builder): debian:bookworm-slim + g++-14 + cmake + ninja, drives
#                      `cmake --preset release-strix` against cpp/.
#   Stage 2 (runtime): debian:bookworm-slim, libc + libvulkan1, binaries
#                      copied from builder stage, non-root-safe layout.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Minimal build deps. NO python3, NO pip, NO torch. Rule A. Bookworm's
# default g++ is 12; we pull g++-14 from bookworm-backports for C++23
# (`std::expected`, `std::print`).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        ninja-build \
        g++-14 \
        gcc-14 \
        pkg-config \
        libdbus-1-dev \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

WORKDIR /src

# Copy the cpp tree + ancillary inputs. Cache-friendly: CMakePresets first.
COPY cpp ./cpp
COPY rocm-cpp ./rocm-cpp

# Portable subset only — GPU-linked components (server, lemonade,
# whisper, kokoro, landing) pull rocm-cpp/HIP at link time and are NOT
# built inside the container. They come from the host via bind-mount or
# live on the host's install path. The image ships only the
# orchestration/CLI/observability binaries.
RUN cd cpp \
    && cmake --preset release-strix \
    && cmake --build --preset release-strix \
        --target 1bit 1bit-power 1bit-watchdog 1bit-helm 1bit-voice 1bit-echo 1bit-mcp

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
        libstdc++6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Binaries from cpp/build/strix/<component>/<bin>.
COPY --from=builder /src/cpp/build/strix/cli/1bit                 /usr/local/bin/1bit
COPY --from=builder /src/cpp/build/strix/power/1bit-power         /usr/local/bin/1bit-power
COPY --from=builder /src/cpp/build/strix/watchdog/1bit-watchdog   /usr/local/bin/1bit-watchdog
COPY --from=builder /src/cpp/build/strix/helm/1bit-helm           /usr/local/bin/1bit-helm
COPY --from=builder /src/cpp/build/strix/voice/1bit-voice         /usr/local/bin/1bit-voice
COPY --from=builder /src/cpp/build/strix/echo/1bit-echo           /usr/local/bin/1bit-echo
COPY --from=builder /src/cpp/build/strix/mcp/1bit-mcp             /usr/local/bin/1bit-mcp

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
      org.opencontainers.image.description="Local AI inference on Strix Halo — LLM / TTS / STT / image / video / NPU. C++23 orchestration, HIP kernels, no Python." \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/usr/local/bin/1bit"]
CMD ["doctor"]
