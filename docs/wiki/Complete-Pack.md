# Complete Pack

Current complete-pack docs track the live service stack first. Older package-manifest language from April 2026 is historical.

| Lane | Current surface | Status |
|---|---|---|
| LLM / multimodal | Lemonade on `:13305`, usually through `1bit-proxy :13306` | Live |
| NPU chat / embeddings | FastFlowLM on `:52625`, usually through `1bit-proxy :13306` | Live |
| Primary UI | GAIA pointed at `http://127.0.0.1:13306/api/v1` | Live |
| Secondary UI | Open WebUI on `:3000`, pointed at `http://127.0.0.1:13306/v1` | Live |
| Custom NPU kernels | IRON author-time -> MLIR-AIE -> Peano -> `xclbin` -> `libxrt` | Toolchain lane |
| Website/docs | `1bit-site/` and `docs/wiki/` | Live docs surface |

## Install

```bash
./install.sh
1bit status
```

The installer wires the current Lemonade + FastFlowLM + proxy stack. It does not use the older package-manifest flow as the operator entry point.

## Rule Fit

Rule A through Rule E apply to every lane. Open WebUI, GAIA, notebooks, and other caller-side tools may use their own implementation stacks, but the core serving path remains native and Python-free.

## See Also

- [Development](./Development.md)
- [Installation](./Installation.md)
- [Clients](./Clients.md)
- [NPU status](./Why-No-NPU-Yet.md)
