# 1bit-systems

A lean 1-bit inference engine for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, XDNA 2 NPU).

Three lanes behind one OpenAI-compatible endpoint:

| Lane | Backend | Quants | Best at |
|---|---|---|---|
| **GPU (ROCm)** | llama.cpp HIP, Lemonade | Q4_K_M, Q2_K, IQ1_S, TQ1_0, TQ2_0 | prompt-eval, dense models |
| **GPU (Vulkan)** | llama.cpp Vulkan, Lemonade | same | decode on sub-2-bit (best for IQ1_S) |
| **NPU (XDNA 2)** | FastFlowLM | UINT4-AWQ, q4nx | low-power, parallel to GPU |

Talks OpenAI on `:13306` (unified) — `1bit up` starts lemond + flm + a small proxy that fans out to both lanes behind one endpoint. Works with any client that speaks OpenAI: AMD GAIA, Open WebUI, Continue, Hermes, raw `curl`.

## Install

```sh
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh
```

Installs `lemonade-server`, `fastflowlm`, `xrt`, `xrt-plugin-amdxdna`, `rocm-hip-sdk`, writes memlock limits, pulls a default 1-bit model (`Bonsai-1.7B-IQ1_S`, 385 MB), installs the `1bit` CLI to `/usr/local/bin/`.

After install — re-login or reboot once so memlock limits apply for the NPU lane.

## Run

```sh
1bit up           # starts lemond, opens webapp, launches GAIA if installed
1bit status       # show lemond + FLM status
1bit pull qwen3:0.6b   # pull a model (auto-routes: NPU via flm, GPU via lemonade)
1bit npu          # start FLM NPU server
1bit bench        # run the 1-bit / ternary pile bench
1bit down         # stop lemond
```

## Reference numbers

Strix Halo gfx1151, captured 2026-04-27, `-p 512 -n 128 -r 2 -ngl 99`:

| Model | Backend | Quant | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|---:|---:|
| Bonsai-1.7B | Vulkan | IQ1_S | 4624 | **278** |
| Bonsai-1.7B | ROCm   | IQ1_S | 4481 | 194 |
| Qwen3-0.6B  | NPU    | q4nx  | 49 (prefill) | 92 |

Vulkan wins decode on sub-2-bit for now (mainline IQ1_S compute shaders > stock ROCm kernels). NPU is a separate lane — runs in parallel to GPU at low power.

## AMD GAIA integration

[AMD GAIA](https://amd-gaia.ai) is AMD's local desktop UI for AI agents on Ryzen AI MAX+. Install the `.deb`, point it at `http://127.0.0.1:13305/v1/` — done. `1bit up` will launch GAIA automatically if it's on `$PATH`.

## Layout

```
.
├── install.sh             # bootstrap
├── scripts/1bit           # control-plane CLI
├── benchmarks/            # bench-1bit-pile.sh + RESULTS-*.md
├── 1bit-site/             # 1bit.systems landing page (manual deploy: wrangler pages deploy)
└── docs/                  # architecture + integration notes
```

## License

MIT. See `LICENSE`.
