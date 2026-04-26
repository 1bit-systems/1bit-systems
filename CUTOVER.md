# CUTOVER.md — gen-2 Rust → gen-3 C++23

> **Status: in flight (2026-04-26).** Phase 2 of the stack-architecture
> plan triggered today. The Rust `crates/` workspace is being gutted
> for the C++23 monorepo at `cpp/`. Shadow-burnin compares gen-3 vs
> gen-2 byte-for-byte during the cutover window.

## Generations

| gen | impl | `/v1/*` | retired |
|---|---|---|---|
| gen-1 | C++ `bitnet_decode` | `:8080` | 2026-04-24 (v0.1.0) |
| gen-2 | Rust `crates/` + `1bit-server` | `:8180` | 2026-04-26 |
| gen-3 | C++23 `cpp/` via lemond | `:8180` | current |

`lemond` is unchanged across gen-2 → gen-3; the Engine and tower it
dispatches into flipped.

## Cutover gates (all must pass)

| gate | target |
|---|---|
| PPL wikitext-103 1024 tok | ≤ 9.21 (gen-1 9.1607, gen-2 9.1805) |
| Shadow burnin exact-match | ≥ 95% gen-3 vs gen-2 argmax |
| Shadow burnin total rounds | ≥ 50,000 sustained |
| gen-3 unreachable | < 0.1% |
| p95 latency gap | < 100 ms |
| Median prefix-match length | ≥ 40 chars |
| `ctest --preset release-strix` | green |
| CI on main | green |

Re-run check:

```bash
1bit ppl
1bit bench
1bit doctor
1bit status
```

## Rollback safety

Gen-2 binaries stay under `~/.local/bin/lemond.gen2` (and peers) for
the burn-in window:

```bash
systemctl --user stop 1bit-halo-lemonade.service
cp ~/.local/bin/lemond.gen2 ~/.local/bin/lemond
systemctl --user start 1bit-halo-lemonade.service
```

Under 10 seconds.

## Procedure

1. Verify gates (`1bit doctor && 1bit ppl && 1bit bench | head -15`).
   Abort if any fail.
2. Build:
   ```bash
   cmake --preset release-strix
   cmake --build --preset release-strix
   ctest --preset release-strix
   ```
3. Snapshot gen-2 binaries to `~/.local/bin/*.gen2`.
4. Stop live units, install gen-3 via `1bit install --gen3
   <component>`, restart, smoke:
   ```bash
   curl -sS -H "Authorization: Bearer $TOKEN" \
        https://strixhalo.local/v1/chat/completions \
        -d '{"model":"halo-1bit-2b","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' \
        | jq -c '.choices[0].message.content'
   ```
5. Run shadow-burnin against gen-2 for ≥ 50,000 rounds; logs to
   `~/claude output/shadow-burnin-gen3.jsonl`.
6. Post to #changelog: "cutover done, /v1 now gen-3".

## Post-cutover (within 14 days)

- Delete `crates/`, `Cargo.toml`, `Cargo.lock`, Rust bits in
  `flake.nix`, `target/`.
- Move drafts from `cpp/docs-draft/` into the repo root.
- Drop `*.gen2` once burn-in closes clean. Tag `v0.2.0`.
- Update Caddy config to drop any residual `/v2/*` shadow routes.

## Known breakage

- Rust-specific JSON field ordering may regress; gen-3 matches the
  OpenAI shape exactly. Fix: clients should use a JSON parser, not
  string matching.
- SSE clients **must** handle the `data: [DONE]\n\n` sentinel. lemond
  emits it; ad-hoc curl consumers should check.
- `1bit-hip` / `1bit-server` consumers go away — port to the C++ ABI
  in `cpp/core/` or call lemond's HTTP surface.

## Fire-drill test (recommended before the real cutover)

Run the cutover against a second lemond instance on a non-standard
port, smoke against it, then exercise the rollback. Confirms install +
rollback both work before touching the live unit on `:8180`.
