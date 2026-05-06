# Complete Pack

Current complete-pack docs track the repair path first. Older package-manifest
language from April 2026 is historical, and the finished single control plane
is still roadmap work.

| Lane | Current surface | Status |
|---|---|---|
| LLM / GGUF | Toolbox `llama-server` on `:13305`, usually through `1bit-proxy :13306` | First repair path |
| LLM / multimodal | Lemonade on `:13305`, usually through `1bit-proxy :13306` | Native target lane |
| NPU chat / embeddings | FastFlowLM on `:52625`, usually through `1bit-proxy :13306` | Optional native lane |
| Primary UI | GAIA pointed at `http://127.0.0.1:13306/api/v1` | Product direction |
| Secondary UI | Open WebUI on `:3000`, pointed at `http://127.0.0.1:13306/v1` | Compatibility UI |
| Custom NPU kernels | IRON author-time -> MLIR-AIE -> Peano -> `xclbin` -> `libxrt` | Toolchain lane |
| Website/docs | `1bit-site/` and `docs/wiki/` | Live docs surface |

## Install

```bash
node scripts/1bit-proxy.js
curl -s http://127.0.0.1:13306/v1/models
```

Run a Strix Halo toolbox backend before starting the proxy. On Arch/CachyOS,
`./install.sh` wires the native Lemonade + FastFlowLM + proxy stack. On
Ubuntu/Fedora, use [Toolbox backends](../toolbox-backends.md) first.

## Rule Fit

Rule A through Rule E apply to every lane. Open WebUI, GAIA, notebooks, and other caller-side tools may use their own implementation stacks, but the core serving path remains native and Python-free.

## See Also

- [Development](./Development.md)
- [Installation](./Installation.md)
- [Clients](./Clients.md)
- [NPU status](./Why-No-NPU-Yet.md)
