# OPTC troubleshooting

Strix Halo systems can hard-freeze under mixed display + GPU-compute load with
kernel messages like:

```text
REG_WAIT timeout 1us * 100000 tries - optc35_disable_crtc
```

This is an AMDGPU display-controller failure path. When it fires, the compositor,
TTY, SSH, and SysRq may all stop responding, so the only practical recovery is a
power cycle.

## Current production mitigation

Stage this kernel command-line flag:

```text
amdgpu.dcdebugmask=0x410
```

On Limine systems, `install.sh` now stages the flag automatically when it detects
the Strix Halo display path:

1. Creates a Snapper snapshot when `snapper` is installed.
2. Backs up `/etc/default/limine`.
3. Adds `amdgpu.dcdebugmask=0x410`.
4. Runs `limine-install`.

The flag only becomes active after reboot.

Verify:

```bash
1bit-optc-status
```

Expected after reboot:

```text
amdgpu.dcdebugmask=0x410 active
no optc35_disable_crtc events in current boot
```

## DPM performance pin

`install.sh` also installs and enables:

```text
1bit-optc-gpu-perf.service
```

It is a root system service, not a user service, because
`/sys/class/drm/card*/device/power_dpm_force_performance_level` is root-owned on
stock CachyOS. The unit writes `high` to each available AMDGPU DPM control file.

Check it with:

```bash
systemctl status 1bit-optc-gpu-perf.service
```

## Soak test

Run a monitor-only soak:

```bash
1bit-optc-soak 7200
```

To add workload commands while monitoring:

```bash
OPTC_AI_CMD='while true; do curl -fsS http://127.0.0.1:13306/v1/models >/dev/null; sleep 1; done' \
OPTC_GPU_CMD='your-gpu-load-command' \
1bit-optc-soak 7200
```

Artifacts are written to:

```text
~/claude output/optc-soak-<timestamp>/
```

Green verdict:

- `new_optc_events=0`
- `verdict=PASS`

## Live machine finding on 2026-05-03

The wiki note said Tier-3 was current, but the live machine did not actually have
`amdgpu.dcdebugmask=0x410` in `/proc/cmdline`. The current boot also contained
two `optc35_disable_crtc` events. The flag has been staged in
`/etc/default/limine` and will become active after the next reboot.

## What does not help

- `amdgpu.noretry=0`: unrelated hang class.
- `amdgpu.ppfeaturemask=0xffffffff`: does not bypass this display path.
- Rebuilding Mesa or shader tools: orthogonal.
- CPU fallback rendering alone: scanout still crosses the display controller.
