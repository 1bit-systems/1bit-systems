# Why Caddy + user-scope systemd?

**One-line answer**: Caddy handles TLS, HTTP/3, reverse-proxy, and reload semantics with a 20-line config. systemd --user lets every 1bit systems process restart without root. Snapper catches any mistake. No Docker, no k8s, no YAML jungle.

## Why Caddy, not nginx

- **ACME + HTTP/3 + TLS 1.3 out of the box.** Nginx needs certbot + openresty + a reload cron. Caddy has it built in.
- **Config surface we actually touch is 3 directives**: `reverse_proxy`, `tls`, `handle_path`. That's the whole `Caddyfile`.
- **Graceful reload** on `caddy reload` — no dropped connections, no pid juggling.
- **Structured JSON logs** to the systemd journal. `journalctl -u caddy -o json` is grep-able.

We run one `Caddyfile` at `/etc/caddy/Caddyfile` (root-owned because Caddy binds :443). A placeholder copy lives at [`strixhalo/caddy/Caddyfile`](../../strixhalo/caddy/Caddyfile) with `sk-halo-REPLACE_ME` tokens — never replaced in git.

## Why `tls internal` on LAN

All 1bit systems traffic rides Headscale (100.64.0.0/10). No public DNS, no public cert to manage. Caddy's `tls internal` spins up a tiny CA, mints certs for `strixhalo.local`, `landing.strixhalo.local`, etc. Clients trust the Headscale-pinned CA. Zero LetsEncrypt rate-limit risk.

```caddyfile
landing.strixhalo.local {
    tls internal
    reverse_proxy 127.0.0.1:8190
}
```

If we ever expose a public endpoint, Caddy flips from `tls internal` to ACME automatically. One-line change.

## Why systemd --user, not system units

- **No root for user-scope restart.** User services can restart without `sudo`; system services still use the documented systemd path.
- **All state under `$HOME`.** Snapper's rootfs snapshot catches every service change. Rollback to snapshot `#6` undoes a bad unit install without touching `/etc`.
- **Isolation.** Local stack services run as the operator where possible, not as broad root processes.
- **Per-user tuning.** `~/.config/systemd/user/*.service` is editable without a config manager.

Units live at [`strixhalo/systemd/`](../../strixhalo/systemd/) in git. `install.sh` copies them to `~/.config/systemd/user/` and runs `systemctl --user daemon-reload`.

## `loginctl enable-linger`

Without it, user services stop on logout. One command, done forever:

```bash
loginctl enable-linger bcloud
```

Now user-scope helpers survive SSH disconnect, reboot, and TTY switch. The box is headless in a closet, so lingering is non-negotiable.

## Why we do NOT use Docker

Rule A (bare-metal-first, see [`feedback_bare_metal_first.md`](memory-only)) forbids containers in the runtime path. Reasons, in order:

1. **Cold start.** Container boot adds 300-800 ms on top of process start. For a service called by voice pipelines that target <2 s end-to-end, that's a third of the budget.
2. **Storage driver churn.** Overlay2 + Btrfs = unpredictable layer timing under snapper.
3. **Attack surface.** Every daemon (`dockerd`, `containerd`) is another root-owned service to patch. On a home box, we'd rather patch zero.
4. **GPU passthrough.** HIP in a container needs `--device=/dev/kfd` + `--device=/dev/dri` + group mapping. Works, but is a chore. Bare-metal `1bit-server` sees the GPU directly.
5. **Debugging.** `strace` on bare-metal is grep-able; in a container you're layering namespaces.

Caller-side is different — if you want to run 1bit-cli inside Docker on your laptop, fine. The service side stays bare.

## Pointers

- Units: [`strixhalo/systemd/`](../../strixhalo/systemd/)
- Caddyfile (tracked): [`strixhalo/caddy/Caddyfile`](../../strixhalo/caddy/Caddyfile)
- Bootstrap: [`install.sh`](../../install.sh)
