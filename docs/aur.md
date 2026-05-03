# AUR Packaging

The AUR packages are not the canonical install path right now.

Use the source installer:

```sh
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh
```

Reason: the current stack builds the maintained forked Lemonade and FastFlowLM paths under `/opt/1bit` and wires them through systemd. The old AUR packages still describe upstream `lemonade-server` / `fastflowlm` package dependencies and a Lemonade `flm:npu` pin patch hook. That was useful for the earlier integration pass, but it is now stale relative to the fork-first policy.

## Target Package Shape

When AUR packaging is refreshed, it should match this service layout:

| Component | Port / path |
|---|---|
| GAIA | desktop AppImage + `~/.gaia/venv/bin/gaia` |
| Lemonade | `http://127.0.0.1:13305/api/v1` or `/v1` |
| FastFlowLM | `http://127.0.0.1:52625/v1` |
| 1bit proxy | `http://127.0.0.1:13306/api/v1` or `/v1` |
| Open WebUI | `http://127.0.0.1:3000` |
| systemd | `1bit-stack.target` |

## Required Cleanup Before Publishing

- Replace upstream `lemonade-server` and `fastflowlm` package dependencies with source-built fork packages, or mark the AUR package as a thin wrapper that only installs CLI/docs and requires a completed `install.sh` build.
- Remove the old Lemonade `backend_versions.json` pin patch hook.
- Add `scripts/1bit-omni.py`, `scripts/1bit-proxy.js`, `scripts/1bit-home.html`, and current `scripts/1bit` exactly as installed by `install.sh`.
- Document GAIA as primary and Open WebUI as secondary.
- Pin FLM to `:52625`.
- Regenerate `.SRCINFO` after any PKGBUILD change.

Until that work is done, do not treat `packaging/aur/*` as a clean release surface.
