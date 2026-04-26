# medusa-convert

DEV-ONLY one-shot converter — see `CLAUDE.md` Rule A exception.

Turns the upstream `parrishcorcoran/MedusaBitNet-2B-4T`
`medusa_heads_step2000.pt` checkpoint into the native `.h1b-medusa` v2
binary consumed by `crates/lemond/src/medusa/loader.rs`.

**Not a runtime component.** Never invoke from a systemd unit or HTTP
path. Python is only tolerated here because it runs once on a dev box;
the shipping loader and forward pass are 100% Rust/C++.

## Usage

```bash
python3 tools/medusa-convert/convert.py \
    --in /home/bcloud/halo-ai/models/medusa/medusa_heads_step2000.pt \
    --out /home/bcloud/halo-ai/models/medusa/halo-medusa-heads.h1b-medusa
```

Requires `torch` in the ambient Python environment — the CachyOS dev
box already has it (used by other one-shot converters like
`tools/h1b-sherry/`).

## Output format

Matches `MEDUSA_FORMAT_VERSION = 2` in
`crates/lemond/src/medusa/loader.rs`. See that file's module
docstring for the exact byte layout.

Header: 68 bytes. Per-head payload: `w_in` then `w_out`, each
`fp16[2560, 2560]` (~25 MiB). Total for 4 heads: ~100 MiB.

bf16 → fp16 cast clamps to ±65504 and prints a per-head clamp count to
stderr so outliers don't get silently saturated.
