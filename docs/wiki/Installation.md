# Installation

> This page exists because issue #1 linked to it before it was written. For the
> full, maintained instructions see [`docs/getting-started.md`](../getting-started.md)
> (running the server) and [`docs/building.md`](../building.md) (building from
> source). This page is the short version plus the one thing that will actually
> bite you.

## ⚠️ Kernel first (issue #1)

On **Strix Halo (gfx1151)**, Linux **6.19.x** kernels have a reproducible
`amdgpu` OPTC CRTC hang under sustained NPU/GPU load. Use a confirmed-stable
kernel before anything else:

- ✅ **6.18.22-lts** (confirmed stable)
- ✅ **7.x** series (confirmed stable)
- ❌ **6.19.x** (hangs mid-inference)

```bash
uname -r          # check your running kernel
```

`install.sh` warns automatically if it detects 6.19.x.

## Requirements

| Component | Requirement                                    |
|-----------|------------------------------------------------|
| Hardware  | AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151)    |
| OS        | Ubuntu 24.04 LTS or later (Arch/CachyOS work)  |
| Kernel    | 6.18.22-lts or 7.x (not 6.19.x)                |
| ROCm      | 7.2.4                                          |

## Quick install

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh          # warns on a 6.19.x kernel; use --skip-rocm to reuse prebuilt libs
```

## ⚠️ NPU 40-column unlock

On Strix Halo the NPU defaults to **8 columns** (32 tiles). To unlock the
full **40 columns** (160 tiles) without disabling Secure Boot, pass
`amdxdna.aie2_max_col=40` via the kernel command line in GRUB.

See the [full boot configuration guide](boot-configuration.md) for
step-by-step instructions.

## See also

- [Getting Started](../getting-started.md)
- [Building from source](../building.md)
- [Boot Configuration](boot-configuration.md) — 40-column NPU unlock
- [Network Topology](Network-Topology.md)
