# Why no Python at runtime? (Rule A)

**One-line answer**: Python is fine as a caller, training tool, notebook, or author-time DSL. It is banned from the core serving path. We've been burned too many times by dependency hell, silent import failures, and interpreter-heavy services shipped to run a local model.

## Scope

**Banned in the core serving path**: native runtimes, kernel hot paths, model dispatch, and service glue that is part of serving inference.

**Allowed elsewhere**:
- Caller-side scripts — DSPy, lemonade-python-sdk, user's jupyter notebook.
- Build-time tooling — our `.h1b` requantizer runs once to convert Microsoft's weights. It's a Python + PyTorch one-shot that writes a file. Not shipped.
- Research notebooks — benchmark analysis, paper reproduction. Output is numbers or plots, not services.
- Toolbox backends — Strix Halo llama.cpp toolboxes are allowed as isolated
  inference backends behind `1bit-proxy`, not as the control plane.
- Compatibility UIs — Open WebUI is allowed as an isolated secondary UI/client behind `1bit-proxy`, not as the core engine.
- Author-time NPU DSLs — IRON's Python DSL is allowed before runtime; the shipped artifact is `xclbin` loaded by native code through `libxrt`.

## What we tried first

Gen-1 of 1bit systems was Python (AMD Lemonade + transformers + a custom FastAPI layer). It was the natural starting point — Microsoft's reference is Python, MLX is Python, lemonade is Python, every BitNet community project is Python.

What we hit:

1. **Cold-start latency** — 4-12 seconds to import torch + transformers + numpy before the first inference. On a service meant to serve LAN requests, that's a restart outage. Rust `1bit-server` cold-start is ~200 ms.
2. **Dependency churn** — `pip install` resolvers pick different versions day to day. One morning the Lemonade server wouldn't start because a transitive dep bumped past a compatible minor. Rust cargo's lockfile prevents this class of bug.
3. **Silent failure modes** — Python typing is advisory. JSON parsing fails at the first bad field, which may be six levels deep in an error the user sees as a 500. Rust serde fails at deserialization with a precise path.
4. **Multiple Pythons** — the dev box had `python3.12`, the install script pulled `python3.11`, the AUR package expected `python3.13`. Three stacks. Three venv roots. Three reasons the service won't start on a fresh box.
5. **Binary size** — the gen-1 install weighed 1.2 GB of Python + wheels. Gen-2 Rust is 15 MB of static-ish binaries.
6. **Global interpreter lock** — Python's concurrency story is either subprocess (slow) or asyncio (complicated, doesn't parallelize CPU-bound code). Rust `tokio` handles 1000+ concurrent HTTP connections on one thread.

## The immediate wins of the Rust rewrite

- Repair path: toolbox-backed llama.cpp can serve first while `1bit-proxy`
  keeps app URLs stable.
- Native install time: `./install.sh` is idempotent for the Arch/CachyOS
  Lemonade + FastFlowLM + `1bit-proxy` stack.
- First-request latency: under 250 ms cold, effectively 0 warm.
- Memory footprint at idle: native services stay small compared with the gen-1 Python process.
- Error messages: deserializer-level "expected u32 at `.usage.prompt_tokens`, got string" vs Python's "`KeyError: 'usage'`" with no line number.

## But what about speed of development?

Genuinely, Python's write-time is faster. We accept ~1.5× the implementation time in exchange for ~10× the deploy reliability. For a stack that's going to run unattended in a closet for months, the trade favors Rust.

## "What about the kernels?"

Kernels are C++. Rule B, not Rule A. C++ at the kernel layer is specifically blessed because:
- The GPU compiler toolchain is C++.
- The surface is small (30 extern "C" functions).
- The FFI boundary with Rust is tight and reviewed.

## "What about the Python requantizer?"

Build-time, not runtime. `requantize-h1b.py` reads Microsoft's `.safetensors`, writes our `.h1b` format, exits. Runs once per model release. Never shipped with the binary, never called by a service.

If we ever needed a *runtime* requantize (which we don't — the output is cached to disk), we'd port to Rust before running it under a systemd unit.

## Enforcement

- Review blocks Python in the core serving path.
- Service changes must preserve the current boundary: toolbox/Lemonade/FastFlowLM
  backends stay behind OpenAI-compatible HTTP; caller-side Python remains
  outside the core serving path we own.
- The 1bit-mcp MCP server is native; DSPy, notebooks, Hermes, and other Python examples are callers, not core services.

## Citations

- [`CLAUDE.md`](../../CLAUDE.md) — the formal rule in the repo conventions.
- `feedback_no_python_runtime.md` in the private memory archive — origin of the rule, with the first Python-service-crashes-in-prod anecdote.
