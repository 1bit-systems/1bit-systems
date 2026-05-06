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
