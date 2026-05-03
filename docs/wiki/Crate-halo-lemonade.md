---
phase: historical
owner: anvil
---

# Historical: 1bit-lemonade

`1bit-lemonade` was the earlier Rust gateway idea. It is no longer the current runtime surface.

Current stack:

- Lemonade Server runs directly on `:13305`.
- FastFlowLM runs on `:52625`.
- `1bit-proxy` runs on `:13306` and unifies Lemonade with FastFlowLM.
- GAIA points at `http://127.0.0.1:13306/api/v1`.
- Generic OpenAI clients point at `http://127.0.0.1:13306/v1`.

Do not add new docs that instruct users to run a `1bit-lemonade :8200` gateway. Use [Lemonade compatibility](./Lemonade-Compat.md) and [Development](./Development.md) for the live statement.
