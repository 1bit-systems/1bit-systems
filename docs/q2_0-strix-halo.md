# Q2_0 Strix Halo Plan

This note tracks the practical contribution path for Q2_0 support on AMD Strix
Halo (`gfx1151`) through llama.cpp HIP/ROCm.

## Current State

Prism's `llama.cpp` fork currently carries Q2_0 support on the `prism` branch.
The public Ternary Bonsai GGUF models are documented as Q2_0 with group size
128 (`g128`):

```text
block_q2_0 = fp16 scale + 32 packed bytes
QK2_0      = 128
codes      = 2 bits each, q in {0, 1, 2, 3}
value      = (q - 1) * scale
bpw        = (16 + 128 * 2) / 128 = 2.125
```

The upstream llama.cpp preference is expected to be official Q2_0 with group
size 64 (`g64`). If that holds, g128 should stay fork-specific or move under a
separate explicit format name. Do not bake g128 assumptions into the Strix Halo
HIP path intended for upstream.

## Upstream Split

| Track | Format | Purpose |
|---|---|---|
| Upstream llama.cpp | Q2_0 g64 | Official portable format and first HIP target |
| 1bit/Prism fork | Q2_0 g128 or renamed variant | Existing Bonsai model compatibility |
| Repack tooling | g128 -> g64 | Converts public Bonsai GGUFs to upstream-compatible layout |

Expected repack cost: roughly 7-10% larger model files because the same tensor
gets twice as many scale values.

## Prism Files To Watch

Format and GGUF:

- `ggml/include/ggml.h`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-quants.c`
- `ggml/src/ggml-quants.h`
- `gguf-py/gguf/constants.py`
- `gguf-py/gguf/quants.py`

CPU validation:

- `ggml/src/ggml-cpu/quants.c`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/src/ggml-cpu/arch/arm/quants.c`

CUDA/HIP path:

- `ggml/src/ggml-cuda/dequantize.cuh`
- `ggml/src/ggml-cuda/vecdotq.cuh`
- `ggml/src/ggml-cuda/mmq.cuh`
- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/mmvq.cu`
- `ggml/src/ggml-cuda/convert.cu`
- `ggml/src/ggml-cuda/getrows.cu`
- `ggml/src/ggml-cuda/common.cuh`
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`
- `ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu`

llama.cpp's HIP backend compiles through the CUDA backend source with
`GGML_USE_HIP`, so the Prism CUDA Q2_0 implementation is the starting point for
Strix Halo. The important distinction is not file location; it is whether the
kernel compiles and runs correctly under HIP for `gfx1151`.

## Strix Halo Build Target

Start from upstream llama.cpp after the official Q2_0 g64 PR exists.

```bash
cmake -B build-q2-hip \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-q2-hip -j"$(nproc)"
```

If building inside a toolbox, the container must see the host GPU devices:

```bash
ls -ld /dev/dri /dev/kfd
groups
```

The user should have active `video` and `render` groups. A configured group that
is not active means log out and back in or reboot before blaming HIP.

## Validation Checklist

The repo includes a harness for the build, benchmark, and endpoint smoke test:

```bash
Q2_MODEL=/path/to/q2_0-g64.gguf \
Q2_FORMAT=g64 \
LLAMA_CPP_DIR=/home/bcloud/prism-llama.cpp \
  benchmarks/q2_0-strix-halo-validate.sh --build --bench --server-smoke
```

Preview the exact commands without mutating the build tree:

```bash
Q2_MODEL=/path/to/q2_0-g64.gguf \
Q2_FORMAT=g64 \
  benchmarks/q2_0-strix-halo-validate.sh --dry-run
```

Correctness:

- `test-quantize-fns` covers Q2_0 quantize/dequantize error bounds.
- Compare CPU output against HIP output on a small prompt.
- Confirm `/v1/models` and `/v1/chat/completions` through `llama-server`.

### Host Validation Log

2026-05-06 on Strix Halo:

- ROCm 7.2.1 packages are installed under `/opt/rocm-7.2.1`.
- `rocminfo` sees `gfx1151` as `AMD Radeon Graphics`.
- `llama-bench --list-devices` sees `ROCm0: AMD Radeon Graphics`.
- Prism `llama.cpp` branch `prism` at `d104cf1b6` configures and builds with:

```bash
PATH=/opt/rocm/bin:$PATH cmake -S /home/bcloud/prism-llama.cpp \
  -B /home/bcloud/prism-llama.cpp/build-q2-hip \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release

PATH=/opt/rocm/bin:$PATH cmake --build /home/bcloud/prism-llama.cpp/build-q2-hip -j"$(nproc)"
```

Two install blockers were found and fixed on the host:

- Missing `hip-lang-config.cmake`: install `hip-dev` and `rocm-cmake`.
- Missing `hipblas-config.cmake`: install `hipblas-dev` and `rocblas-dev`.

Current Prism Q2_0 correctness status:

```text
Testing q2_0
 q2_0 absolute quantization error:    FAILED (0.008678)
 q2_0 dot product error:              FAILED (0.141111)
2 tests failed
```

This is not a HIP compile failure. The HIP backend builds, including
`mmq-instance-q2_0.cu`; the remaining blocker is the Q2_0 quantization
correctness threshold/implementation in the Prism branch before benchmark
numbers should be trusted.

Performance:

```bash
./build-q2-hip/bin/llama-bench \
  -m /path/to/q2_0-g64.gguf \
  -ngl 99 \
  -fa 1
```

Record:

- commit hash
- ROCm version
- kernel version
- model file and format (`g64` or `g128`)
- prompt-processing tok/s
- token-generation tok/s
- memory footprint
- whether Vulkan, HIP, or CPU was selected

Product integration:

- Serve the model on an OpenAI-compatible endpoint.
- Point `1bit-proxy` at it using `LEMOND_URL`.
- Verify `curl http://127.0.0.1:13306/health`.
- Verify one `/v1/chat/completions` request.

## Prism Bonsai Demo

The Prism Bonsai 1.7B Q2_0 model was downloaded to:

```text
/home/bcloud/models/prism-bonsai/Ternary-Bonsai-1.7B-Q2_0.gguf
sha256: d97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a
```

Persistent local demo endpoints:

```text
llama-server: http://127.0.0.1:18092/v1
1bit proxy:   http://127.0.0.1:13308/v1
```

Start commands:

```bash
setsid -f /bin/bash -lc 'PATH=/opt/rocm/bin:$PATH exec /home/bcloud/prism-llama.cpp/build-q2-hip/bin/llama-server --host 127.0.0.1 --port 18092 -m /home/bcloud/models/prism-bonsai/Ternary-Bonsai-1.7B-Q2_0.gguf -c 4096 -ngl 99 -fa 1 > /tmp/1bit-bonsai-server-18092.log 2>&1'

setsid -f /bin/bash -lc 'LEMOND_URL=http://127.0.0.1:18092 ONEBIT_PROXY_PORT=13308 PROXY_HOST=127.0.0.1 exec node /home/bcloud/1bit-systems/scripts/1bit-proxy.js > /tmp/1bit-bonsai-proxy-13308.log 2>&1'
```

Verified checks:

```bash
curl http://127.0.0.1:13308/health
curl http://127.0.0.1:13308/v1/models
```

Demo request:

```bash
curl http://127.0.0.1:13308/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Ternary-Bonsai-1.7B-Q2_0.gguf","messages":[{"role":"system","content":"Repeat only facts given by the user. Do not invent URLs."},{"role":"user","content":"Demo facts: model=Ternary-Bonsai-1.7B-Q2_0.gguf; backend=ROCm gfx1151 on AMD Strix Halo; server=http://127.0.0.1:18092/v1; proxy=http://127.0.0.1:13308/v1. Return them as four short lines."}],"max_tokens":96,"temperature":0}'
```

Observed demo timing through the proxy:

```text
prompt:     116 tokens, 3850 tok/s
generation: 77 tokens, 219 tok/s
backend:    b8846-d104cf1b6, ROCm gfx1151
```

## Risks

- `g64` and `g128` files should not silently share one ambiguous name.
- Existing g128 Bonsai GGUFs may load incorrectly if upstream Q2_0 becomes g64
  under the same enum/type without migration handling.
- HIP may compile the CUDA Q2_0 path but choose a slow fallback or fail at
  runtime on `gfx1151`; benchmark and logs matter.
- Vulkan may be a more reliable first-run path on some Strix Halo hosts, but
  the upstream Q2_0 contribution should still validate HIP/ROCm explicitly.

## Coordination Message

```text
I will target the upstream Q2_0 g64 layout for Strix Halo HIP/ROCm, and keep
the current Bonsai g128 layout isolated as fork-specific unless there is an
agreed upstream type/name for it.

Once the upstream Q2_0 PR is available, I can rebase the HIP path onto it,
build for gfx1151, and send validation numbers from Strix Halo: correctness,
llama-bench pp/tg, memory, and llama-server OpenAI endpoint checks.
```
