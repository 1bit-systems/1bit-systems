# 1bit install landing — runbook

## What this installs

`1bit-landing` — the on-box landing page on `:8190`. Same content as
the public 1bit.systems site, served locally so the helm desktop and
remote helpers can hit a known URL without internet.

Bundled in `core` since 2026-04-20; the standalone component exists
as a back-compat alias.

## Prereqs

- `core` installed
- Port `:8190` free
- `httplib::httplib` linked at build time (vendored)

## Install

```sh
1bit install landing
```

Or simply `1bit install core` — landing comes along.

## Verify

```sh
curl -s http://127.0.0.1:8190/_health
# expected: {"status":"ok","version":"<x.y.z>"}
```

## Common errors

- **`bind: permission denied`** — never on `:8190` (>1024) unless
  somebody bound `0.0.0.0:8190` already. `ss -tlnp | grep 8190` to
  find the squatter.
- **Static assets 404** — install dropped them under
  `${HOME}/.local/share/1bit-landing/` but the binary is looking at
  `cwd`. Either run with `cwd=${HOME}/.local/share/1bit-landing` or
  `1bit-landing --root <path>`.

## Logs

```sh
journalctl --user -u strix-landing.service -f
```

## Rollback

```sh
systemctl --user stop strix-landing.service
systemctl --user disable strix-landing.service
rm -f ${HOME}/.local/bin/1bit-landing
rm -rf ${HOME}/.local/share/1bit-landing/
```
