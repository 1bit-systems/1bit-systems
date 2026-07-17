# Python → Native/Rust Boundary — Audit & Decision (2026-07-17)

**Author:** agent (Python-boundary audit)
**Status:** findings landed; wiring tasks open
**Convention:** follows `openspec/changes/stack-audit-2026-07-14/`

## Motivating principle

> "As long as there is no Python/pip in the *hotpatch* of the LLM, what happens
> after that — how clients interact with the LLM — is up to the client. If you
> can't get rid of Python, turn it into Rust."

The README brand line is now **"Zero Python"** (it previously also said "No Rust";
that clause is gone, and the repo already ships a Rust HTTP frontend at
`rust/onebit`). So the target architecture is explicitly:

```
  C++/HIP/Zig  →  the LLM hot path (kernels, decode, tokenizer)   — no interpreter
  Rust         →  runtime glue / HTTP frontend / router            — static binary
  Python       →  OFFLINE tooling only (train, convert, kernelgen) — never in serving
```

## What this change does

1. Records a **verified** map of where Python currently sits vs. the LLM hot path.
2. Makes a decision on each Python touch-point (native-C++ / Rust / keep-offline).
3. Ships a **tested reference** Rust port of `unified-router.py` (std-only, zero crates).
4. Flags the native tokenizer as **ready to wire** (proven bit-exact) and the native
   decode engine as **blocked** (live memory-corruption + xclbin/IOMMU bug).

## Decision summary

| Touch-point | Today | Decision | State |
|---|---|---|---|
| Per-request tokenization (`tokenizers` import / tokenize subprocess) | Python | → native C++ `engine/fusion/tokenize.cpp` | **verified bit-exact, ready to wire** |
| `src/server/rest_handler.cpp` `simple_tokenize()` stub | C++ stub | → real native tokenizer (C ABI) | ready to wire |
| NPU decode loop (`npu-cppd.py` → torch venv `npu_runner.py`) | Python + torch | → native `npu-infer/src/npu_engine_stdio.cpp` | **blocked: builds+inits but `free(): invalid size` + v6 xclbin IOMMU hang** |
| `unified-router.py` (request router/proxy) | Python | → Rust (fold into `rust/onebit`) | **reference impl shipped + tested here** |
| `engine/lora/*.py`, `engine/npu/kernel/*.py`, `engine/fusion/hf_to_q4nx.py`, `export_tokenizer.py` | Python | **keep** (offline, not in serving path) | no action |

## Why defer the decode-engine swap

`npu_engine_stdio.cpp` was verified on hardware (Strix Halo, XDNA2). It:
- **builds clean** against system XRT + `engine/npu/src/dequant_q4nx.c` (`-lxrt_coreutil`);
- **initializes** all 4 NPU contexts and dequant+packs weights (~2.8 s);
- with **base `int8_32tile` xclbins** produces a token (~2.2 s/token, CPU-bound lm_head);
- but throws **`free(): invalid size`** (heap corruption) and, with the **`v6`** xclbins,
  hangs the NPU with `AMD-Vi: IO_PAGE_FAULT` + `aie2_config_cu: Lookup GEM object failed`
  (`aie2_hwctx_restart ... ret -22`).

Wiring this in as-is would replace a working (Python) decode path with a crashing
C++ one. Must fix the heap bug and pin a known-good xclbin set first — or retarget the
already-validated FLM engine (94 tok/s). See `findings.md` §3.

## Non-goals

- No edit to `src/server/rest_handler.cpp` in this change (wiring is a follow-up task,
  tracked in `tasks.md`, so it can be reviewed on its own).
- No brand/README rewrite here (the "Zero Python" claim is not yet literally true —
  `daemon/npu-cppd.py` remains; tracked as a task, not silently changed).
