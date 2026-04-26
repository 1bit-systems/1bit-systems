# AUR packaging

Two packages publishable to the Arch User Repository:

| Package | Install path | Best for |
|---|---|---|
| `1bit-systems`     | builds from source via cmake      | devs, anyone on Arch rolling with ROCm already tuned |
| `1bit-systems-bin` | extracts the upstream AppImage    | end users who want a 30-second install |

## Publish workflow

```bash
# In a fresh AUR clone (ssh://aur@aur.archlinux.org/1bit-systems.git):
cp /path/to/halo-workspace/packaging/aur/PKGBUILD .
updpkgsums
makepkg --printsrcinfo > .SRCINFO
git add PKGBUILD .SRCINFO
git commit -m "upgpkg: 1bit-systems 0.1.8-1"
git push
```

Same flow for `1bit-systems-bin` in its own AUR clone.

## CachyOS

CachyOS inherits from Arch and will pick up the AUR package directly. No separate Cachy-specific packaging needed.

## Rule A

Neither PKGBUILD bundles a Python runtime. ironenv (NPU kernel authoring) is listed as a dev-only `optdepends` line so pacman users see the flag without pacman pulling python into the standard serving install.
