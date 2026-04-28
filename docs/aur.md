# AUR packaging

`1bit-systems` ships an AUR PKGBUILD so Arch / CachyOS users can install
the entire stack with one `paru -S 1bit-systems-git` instead of cloning
the repo and running `install.sh` manually.

## What's in the repo

```
packaging/aur/1bit-systems-git/
├── PKGBUILD                 # AUR build script — pulls main HEAD, installs to /usr/...
├── 1bit-systems.install     # post_install / post_upgrade / post_remove hooks
└── .SRCINFO                 # generated; required by AUR
```

The package depends on the actual inference stack as `pacman` / AUR
deps — it does not bundle them:

- `lemonade-server` (AUR) — the OpenAI-compat server
- `fastflowlm` (cachyos-extra-znver4) — NPU runtime
- `xrt` + `xrt-plugin-amdxdna` (cachyos-extra-znver4) — XDNA driver plugin
- `rocm-hip-sdk` (extra) — iGPU lane (gfx1151)
- `nodejs`, `curl`, `python` — runtime helpers used by the `1bit` CLI / proxy

## What it installs

| Path | Source | Notes |
|---|---|---|
| `/usr/bin/1bit` | `scripts/1bit` | the control-plane CLI |
| `/usr/share/1bit-systems/1bit-proxy.js` | `scripts/1bit-proxy.js` | invoked by `1bit up` |
| `/usr/share/1bit-systems/1bit-home.html` | `scripts/1bit-home.html` | served by the proxy at `:13306/` |
| `/etc/security/limits.d/99-1bit-systems.conf` | inline in PKGBUILD | `memlock=unlimited` for NPU buffers |
| `/usr/share/licenses/1bit-systems-git/LICENSE` | `LICENSE` | MIT |
| `/usr/share/doc/1bit-systems-git/install.sh` | `install.sh` | reference copy of the manual installer |
| `/usr/share/doc/1bit-systems-git/README.md` | `README.md` | reference copy of the README |

## What the post-install hook does

`1bit-systems.install` runs on `post_install` and `post_upgrade`:

1. Patches `/usr/share/lemonade-server/resources/backend_versions.json`
   so its `flm.npu` field matches the FastFlowLM version actually
   installed on the box (the silent `update_required` gotcha — see
   [`docs/reddit-npu-gate.md`](reddit-npu-gate.md) Correction #1 for
   the receipts). Skipped if `flm` or `python3` aren't on `PATH`.

2. Reminds the user to re-login or reboot once so memlock limits apply.

`post_remove` keeps `99-1bit-systems.conf` in place — other processes on
the box may rely on `memlock=unlimited`. Remove manually if undesired.

## Building locally

```sh
cd packaging/aur/1bit-systems-git
makepkg -s              # builds + installs all deps; produces 1bit-systems-git-*.pkg.tar.zst
sudo pacman -U *.pkg.tar.zst
```

For a clean-room test in a chroot:

```sh
extra-x86_64-build       # devtools must be installed
```

## Submitting to AUR

The AUR is a **git-based repo per package**. To publish:

```sh
# clone the AUR remote (empty on first push)
git clone ssh://aur@aur.archlinux.org/1bit-systems-git.git aur-1bit-systems-git
cd aur-1bit-systems-git

# stage the PKGBUILD + .SRCINFO + .install
cp ../packaging/aur/1bit-systems-git/PKGBUILD .
cp ../packaging/aur/1bit-systems-git/1bit-systems.install .
cp ../packaging/aur/1bit-systems-git/.SRCINFO .

git add PKGBUILD 1bit-systems.install .SRCINFO
git commit -m 'Initial PKGBUILD'
git push
```

Requires:
- AUR account at `https://aur.archlinux.org/`
- SSH key registered on that account

## Updating the AUR pkg

After every relevant change in `main` (anything that affects what gets
installed — e.g., a script edit, dep change, new file in the install
list), bump `pkgrel` in the PKGBUILD and re-commit. The `pkgver` updates
automatically via the `pkgver()` function on each rebuild.

## `1bit-systems-bin` &mdash; the compiled-binary track

A second AUR pkg for users who want a **single static binary** instead of
the bash + Node runtime split. Drops the `nodejs` dep entirely &mdash; the
Bun runtime is baked into the binary.

```
packaging/aur/1bit-systems-bin/
├── PKGBUILD                 # downloads 1bit-x86_64-linux from GitHub Releases
├── 1bit-systems.install     # same post-install hook as -git
└── .SRCINFO
```

Source artifacts come from a GitHub Release (built by `.github/workflows/release.yml`):

- `1bit-x86_64-linux` &mdash; the bun-compiled binary (~62 MB, statically
  linked, embeds `home.html` at build time).
- `LICENSE`, `README.md`, `install.sh` &mdash; pulled from the tagged commit
  via `raw.githubusercontent.com` for inclusion in `/usr/share/doc/`.

`provides=("1bit-systems")` and `conflicts=("1bit-systems-git")` so users pick one.

### How a release is cut

```sh
git tag v2.0.0
git push --tags
```

The `release.yml` workflow runs on tag push, builds the binary on
`ubuntu-24.04`, smoke-tests `--help` and `status`, computes a
sha256, and attaches `1bit-x86_64-linux` + `1bit-x86_64-linux.sha256`
to the GitHub Release. After the release lands:

1. Update `pkgver` in `packaging/aur/1bit-systems-bin/PKGBUILD` to
   match the tag.
2. Replace the four `'SKIP'` entries in `sha256sums=()` with real
   checksums. Compute them with (substitute the tag you're packaging —
   `v2.0.0` is the planned cutover, but the URL only resolves once
   `release.yml` has actually published the assets):
   ```sh
   curl -sSL https://github.com/bong-water-water-bong/1bit-systems/releases/download/v2.0.0/1bit-x86_64-linux.sha256
   ```
3. Regenerate `.SRCINFO`: `cd packaging/aur/1bit-systems-bin && makepkg --printsrcinfo > .SRCINFO`
4. Push to AUR: same flow as `-git` &mdash;
   `git clone ssh://aur@aur.archlinux.org/1bit-systems-bin.git`, copy
   files in, push.

### Stable (non-`-git`, non-`-bin`) PKGBUILD &mdash; deferred

A third sibling `packaging/aur/1bit-systems/PKGBUILD` pinned to a tag
release tarball is the conventional "stable" Arch package. Skipped for
now because:

- `1bit-systems-git` already covers the rolling case
- `1bit-systems-bin` already covers the binary-release case
- A third package that just unpacks a source tarball without compilation
  is redundant with `-git` for a pure-script repo

The current `v1.0.0` tag is 500+ commits behind `main` (pre-cutover, no
`install.sh`, no `scripts/1bit`); not viable as a base anyway. Revisit
after the cutover work tags `v2.0.0`.
