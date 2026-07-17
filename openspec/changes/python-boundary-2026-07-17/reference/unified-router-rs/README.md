# unified-router (Rust, std-only)

Behavior-identical port of `unified-router.py`. Zero external crates — only the Rust
standard library. Single static binary (~427 KB), no interpreter / pip in the request path.

## Build & run
```bash
cargo build --release
./target/release/unified-router --port 13305 --backend http://127.0.0.1:13305
```

## Routing (identical to unified-router.py)
- `auto` / `user.Unified` → NPU (qwen3-0.6b-FLM) unless content has GPU keywords,
  is >800 chars, or includes `tools` → then GPU (Qwen3-8B-4bit)
- `npu` → SMALL, `gpu` → BIG, any other model → pass-through unchanged
- Non-completion paths (GET, other POST) → transparent proxy

Verified against a mock backend: see ../findings.md §4.

## Note
Long-term this policy should fold into `rust/onebit` (the repo's Rust HTTP frontend)
rather than run as a second binary. This std-only version is the portable reference.
