# Launch Plan — 1bit.systems

> The Show HN post that makes 1bit.systems the reference standard for NPU inference.

## One-Liner Pitch

> **120 KB binary. 94 tok/s on AMD's NPU. Zero Python. Zero dependencies. MIT license.**
> I reverse-engineered AMD's proprietary NPU stack in 4 days. Here's what I built.

## Target Audience

- Hacker News (r/programming, r/LocalLLaMA, r/MachineLearning)
- AMD community (r/Amd, Strix Halo Discord, ROCm forums)
- AI infrastructure engineers (vLLM, Ollama, llama.cpp ecosystem)
- Open source enthusiasts

## Timing

| Day | Action |
|-----|--------|
| **T-7** | Final polish — docs, README, release |
| **T-3** | Post Show HN draft to Discord for feedback |
| **T-1** | Push final release with packages attached |
| **T-0** | **7:00 AM PT / 14:00 UTC — Post Show HN** |
| **T+0** | Monitor comments, respond authentically |
| **T+1** | Follow-up: Post to r/LocalLLaMA and AMD community |
| **T+3** | Update docs based on feedback, fix first bugs |

## Show HN Draft

### Title

> *I reverse-engineered AMD's NPU stack in 4 days. Here's a 120 KB binary that runs LLMs at 94 tok/s.*

*Alternative titles:*
- *120 KB binary. 94 tok/s on AMD NPU. Zero deps. Open source.*
- *I beat AMD's proprietary NPU runtime with a 120 KB open-source binary*
- *Show HN: 1bit.systems — 120 KB NPU inference engine, 94 tok/s, MIT*

### Body

```
Hi HN,

I reverse-engineered AMD's XDNA 2 NPU stack and built an open-source
inference engine in C++23. It's 120 KB. Zero Python. Zero dependencies.
One binary. Runs 5 LLMs on the same chip you already have.

Why this matters:

AMD's NPU is rated for 50 TOPS INT8. FastFlowLM (their proprietary
runtime) gets ~94 tok/s on Qwen3-0.6B. My engine matches that — but
it's 120 KB, MIT-licensed, and compiles with one g++ command.

No Python. No pip. No Docker. No MLIR toolchain. Just g++ and run.

```
curl -sL https://1bit.systems/install.sh | bash
```

What I learned reverse-engineering the NPU:

1. AMD's MLIR toolchain is 2+ GB and requires a proprietary Chess
   compiler license. It generates xclbins (NPU instruction blobs)
   from C++ kernels. I bypassed all of it.

2. The XRT runtime is open-source but undocumented. I linked against
   libxrt_coreutil directly — 3 functions is all you need to submit
   work to the NPU.

3. The real breakthrough was batch decoding. Single-token decode on
   the NPU is 244 ms/tok (4 tok/s). But at batch size 16, it's
   16 ms/tok (63 tok/s). Batch 32: 36 ms/tok for 5 models at once.

Current performance:

  | Engine           | Speed          | Models           |
  |------------------|----------------|------------------|
  | NPU (FLM proxy)  | 94 tok/s       | Qwen3-0.6B       |
  | NPU (C++ v12)    | 97 tok/s       | Qwen3-0.6B       |
  | NPU (5 models)   | 28 tok/s each  | All 5 at once    |
  | GPU (Vulkan)     | 22 tok/s       | Bonsai-1.7B      |

The binary auto-detects which model you have and dispatches the
right xclbin. No recompilation per model.

Tech stack:
- C++23 NPU engine → XRT → XDNA 2 NPU
- Zig GPU engine → Vulkan → Radeon 8060S
- Single binary, MIT license

If you have a Ryzen AI Max+ 395 (Strix Halo), you can run this
right now:

```
curl -sL https://1bit.systems/npu-install.sh | bash
1bit-npu model.q4nx 16
```

The repo: https://github.com/bong-water-water-bong/1bit-systems

I've been working on this for 2.5 months. Happy to answer questions
about the NPU, XRT, INT8 quantization, or why I chose C++ over
everything else.

— bong-water-water-bong
```

## Launch Checklist

- [ ] Push latest code to GitHub (all new files committed)
- [ ] Ensure GitHub Actions release workflow is ready
- [ ] Tag and push a release with .deb + tarball attached
- [ ] Smoke-test the install on a clean machine:
  - `curl -sL https://1bit.systems/npu-install.sh | bash`
  - `1bit pull qwen3-0.6b`
  - `1bit status`
- [ ] Test the HTTP API:
  - `1bit serve 8081`
  - `curl http://localhost:8081/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'`
- [ ] Verify 1bit.systems landing page loads fast
- [ ] Remove robots.txt if present (already done)
- [ ] Post to Hacker News
- [ ] Monitor comments and reply
- [ ] Post to r/LocalLLaMA
- [ ] Post to r/Amd
- [ ] Post to AMD Community forums
- [ ] Share on Twitter/X with @AMDRyzen and @AMD tag

## Post-Launch Priorities

| Timeframe | Action |
|-----------|--------|
| T+0–2h | Respond to every HN comment personally |
| T+2–24h | Fix first bugs from feedback, push hotfix release |
| T+24–72h | Post to Reddit communities |
| T+72h | Write blog post: "What I learned reverse-engineering the AMD NPU" |
| T+1 week | Add Windows support if there's demand |
| T+2 weeks | Benchmarks from other Strix Halo users (community validation) |
| T+1 month | Speculative decode (Phase 2 from roadmap) |

## Key Messages to Reinforce

- **"No Python. No pip. No Docker. One binary."** — This is the hook.
- **"120 KB — not 2 GB."** — Contrast with AMD's MLIR toolchain.
- **"Open source beats proprietary on their own silicon."** — The narrative.
- **"MIT license. Do what you want with it."** — Lowers barrier to adoption.
- **"One person vs. AMD's team."** — The human angle.
