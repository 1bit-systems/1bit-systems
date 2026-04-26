# halo-power subcommand — design doc

Status: DESIGN (no code this sprint). 2026-04-19.
Author: Claude (Opus 4.7, 1M), with `stampby` steering.

---

## Problem statement

1bit systems runs on an AMD Ryzen AI MAX+ 395 (Strix Halo) APU with the GPU and
CPU sharing a single LPDDR5 memory controller. During sustained bitnet
decode (80 tok/s × 1024 tokens) we hit thermal throttle long before we hit
compute limits — the ternary GEMV is already at 92 % of LPDDR5 peak (see
`project_bitnet_rocprof_plan`), so any wasted package-power budget spent
on idle x86 cores directly costs us tok/s headroom at the tail.

We need a first-party, scriptable way to nudge the APU into three distinct
operating envelopes (decode-heavy, chat-interactive, idle) without shelling
out to `ryzenadj` one-liners by hand every time. A `halo power` subcommand
is the cheapest way to expose that knob to the rest of the stack (the
idle-watchdog daemon, the install doctor check, a future
`1bit-helm`-driven "dim the box" gesture).

---

## RyzenZPilot summary (what the repo actually contains)

**Finding:** RyzenZPilot is **Windows-only, closed-source**, distributed as a
single `RyzenZPilot.exe` inside a zip in the `docs/` folder of the GitHub
repo. The repo contains **zero source code** — only marketing HTML, images,
a demo `.mp4`, and the signed release zip. This was a surprise and
fundamentally changes the integration story.

What the README + landing page (`docs/index.html`) actually claim:

- **Platform:** Windows 10 / 11 64-bit. No Linux build, no macOS build,
  no Wine-tested path.
- **Underlying library:** Built on top of **RyzenAdj** (the OSS project at
  `FlyGoat/RyzenAdj`, dual-MIT/GPL, which *is* available on Linux). The
  `.exe` is a thin GUI + autopilot wrapper around RyzenAdj's MSR writes.
- **Permissions:** Administrator privileges required
  (`Right-click > Properties > Compatibility > Run as administrator` is
  step 3 of 4 in the install steps). Equivalent to `sudo`/root on Linux.
- **Mechanism (inferred, not confirmed from source):** MSR writes via
  RyzenAdj (`/dev/cpu/*/msr` on Linux, `WinRing0` on Windows). Probably
  also toggles Windows power plans via `powercfg`, but we cannot verify
  without reversing the exe.
- **Profiles shipped:** `Silent`, `Balanced`, `Performance` (called
  "Autopilot" when auto-switched by process match rules).
- **GPU tuning:** the landing page does not call out iGPU clocks
  separately — RyzenAdj exposes `--gfx-clk` and `--vrmgfx-current`, so it
  is *possible* RyzenZPilot drives them, but unclaimed.
- **Config:** "Fully customizable profiles & Settings via **json config
  files**" (verbatim from the feature list). Path undocumented without
  installing and inspecting.
- **CLI:** **None advertised.** The landing page emphasises
  system-tray integration and process-based autopilot; there is no
  mention of a command-line entry point, flags, or a headless mode.
- **Distribution:** Not on any package manager (winget, choco, nix, pacman,
  AUR). Manual zip download only. Some AV vendors flag it as a false-positive
  (see landing page note).

**Implication for 1bit systems:** RyzenZPilot itself is unusable on our Linux
strixhalo box. The *library underneath it* (RyzenAdj) is usable. Any
integration story below is really about wrapping **RyzenAdj**, not
RyzenZPilot. We keep the design-doc name because the user task framed it
that way, and because RyzenZPilot deserves the credit for surfacing this
approach.

---

## Wire-in options

### (a) Zero-integration — document it, don't ship it

- Add a paragraph to `README.md` explaining that Windows users can run
  RyzenZPilot alongside 1bit systems for thermal tuning.
- Add a soft probe to `halo doctor`: if `ryzenadj` is not on `PATH`,
  warn "consider installing `ryzenadj` for thermal tuning during decode".
- 1bit-cli stays stock.

Cost: ~10 lines of doc + one doctor check. Risk: zero.
Benefit: zero on our actual Linux box.

### (b) Shell-out — `halo power <profile>` exec's `ryzenadj`

Because RyzenZPilot has no Linux build and no CLI, the shell-out target is
**`ryzenadj`** (AUR package `ryzenadj-git` on CachyOS; upstream at
`FlyGoat/RyzenAdj`). We ship a tiny Rust wrapper that:

1. Resolves the profile name → a fixed `ryzenadj` argv.
2. `sudo -n ryzenadj ...` via `tokio::process::Command`.
3. Captures stdout/stderr, prints a one-line summary.
4. Writes `/run/user/$UID/halo-power/current-profile` so subsequent
   `halo power` (no args) shows state without re-probing hardware.

Cost: ~150 lines of Rust in `cpp/cli/src/power.rs` + one
`Cmd::Power` enum variant in `main.rs`. Three unit tests (argv builder,
state-file round-trip, profile-name parse).
Dependency: `ryzenadj` binary on `PATH`, a passwordless sudoers rule for
it (matches our passwordless-sudo convention for strixhalo).

Risk: if AMD changes the Family 19h MSR layout in a future Strix Halo
stepping, RyzenAdj catches the breakage upstream and we inherit the fix
for free. We carry zero kernel-level code.

### (c) Port — rewrite MSR logic in Rust inside 1bit-cli

Reimplement the subset of RyzenAdj we care about directly in Rust, opening
`/dev/cpu/0/msr` and issuing the PCI SMU mailbox writes by hand. No
external binary, no `sudo` indirection through a separate process.

Estimated LOC: **~800-1200** (RyzenAdj's C source is ~3000 LOC but most of
that is generation detection, argument parsing, and platform quirks we
don't need). We'd need to cover:

- SMU mailbox protocol (`libsmu.c` ≈ 200 LOC in C)
- Family 19h "Rembrandt/Phoenix/Strix" MSR table (~100 LOC of constants)
- Stapm/slow/fast/tctl limit computation (~150 LOC)
- `/dev/cpu/*/msr` ioctl wrapper (~50 LOC unsafe Rust)
- Profile → limits mapping (~100 LOC)
- Tests + harness (~200-400 LOC)

Cost: 2-3 sprint-days to get it working on Strix Halo, another 1-2
sprints if it ever regresses across a BIOS update. We become *de facto*
maintainers of a second RyzenAdj fork.

Benefit: no external process, no sudo-elevated subprocess, no AUR
dependency. Also satisfies Rule A's spirit (ryzenadj is C, not Python, so
Rule A isn't actually in play here — but a pure-Rust 1bit-cli is a nicer
artifact).

---

## Recommendation: **(b) Shell-out.**

Reasoning:

1. **Cheapest integration by far.** 150 LOC of Rust vs. 800-1200 LOC of
   MSR wrangling. We get the knob in a day, not a sprint.
2. **Upstream owns the hardware quirks.** `FlyGoat/RyzenAdj` is active
   (merged a Strix Halo family ID PR in 2025), has Arch / AUR packaging,
   and is the *de facto* Linux equivalent of what RyzenZPilot wraps on
   Windows. When AMD ships a new stepping, RyzenAdj gets patched and we
   `pacman -Syu`. In option (c) we'd have to chase that ourselves.
3. **Credit where it's due.** RyzenZPilot's page surfaced the approach,
   RyzenAdj's library does the actual work, and we carry neither.
4. **Rule A compliant.** `ryzenadj` is a C binary, no Python in the
   runtime path.
5. **Reversible.** If we later want to port (option c), the `halo power`
   CLI surface stays identical; only `cpp/cli/src/power.rs`
   changes. Users don't notice.

User's prior was (b); recommendation confirms.

---

## Profile mapping — 1bit systems semantics → ryzenadj knobs

ryzenadj flags we rely on (all present on current AUR build; verified
from `ryzenadj --help`):

- `--stapm-limit=<mW>` — sustained package power (slow-moving average).
- `--fast-limit=<mW>`  — short-window PPT (fast boost ceiling).
- `--slow-limit=<mW>`  — medium-window PPT.
- `--tctl-temp=<°C>`   — thermal control target.
- `--vrmmax-current=<mA>` — VRM current limit (EDC-ish).

Strix Halo defaults on our box (from `ryzenadj --dump-table`) land around
stapm 55 W, fast 65 W, slow 60 W, tctl 95 °C. The mapping below is a
**starting point** — numbers need to be pinned with a decode-bench sweep
before we ship them as defaults (tracked as follow-up work, NOT in scope
of this doc).

| 1bit systems profile | Intent                                     | stapm | fast | slow | tctl | Notes                                              |
| --------------- | ------------------------------------------ | ----- | ---- | ---- | ---- | -------------------------------------------------- |
| `inference`     | Max sustained decode tok/s                 |  65 W | 80 W | 75 W | 95 °C | All headroom to package; CPU cores park on demand. |
| `chat`          | Interactive, balanced, low fan             |  45 W | 65 W | 55 W | 90 °C | Default after boot. Matches 1bit-helm chat load.   |
| `idle`          | No active requests for N seconds (>= 60 s) |  20 W | 35 W | 25 W | 80 °C | Watchdog-triggered. Quiet closet mode.             |

Open questions (NOT blockers for this doc, tracked as follow-ups):

- Does bumping stapm to 65 W actually lift decode tok/s, or are we
  LPDDR5-bound long before then? First rocprof re-run after landing
  (b) will tell us.
- iGPU-specific clocks (`--gfx-clk`, `--min-gfxclk`): ryzenadj exposes
  them but Strix Halo support is unclear. Probe with `--dump-table` on
  first boot and omit from the profile table if they read as zero.

---

## CLI surface

Added to `cpp/cli/src/main.rs` as a new `Cmd::Power` variant.
Shape follows existing halo subcommands (`status`, `logs`, `doctor`,
`install`) — short subcommand word, flags on the right.

```
halo power                       # show current profile + raw ryzenadj --dump-table hash
halo power inference             # apply inference profile
halo power chat                  # apply chat profile
halo power idle                  # apply idle profile
halo power --list                # list available profiles with their stapm/fast/slow/tctl
halo power --dry-run inference   # print the ryzenadj argv that WOULD run, don't exec
halo power --revert              # reapply whatever profile was live before the last set
```

Exit codes:

- `0` — profile applied (or, for bare `halo power`, state read OK).
- `2` — `ryzenadj` not on PATH (install hint printed).
- `3` — sudo denied / not in sudoers.
- `4` — profile name unknown.
- `5` — ryzenadj returned nonzero; stderr forwarded.

State file: `/run/user/$UID/halo-power/current-profile` (tmpfs, wiped on
reboot — which is correct, MSR writes don't persist across reboot anyway).

Integration points (not implemented in this sprint, but shape-locked here
so we don't paint ourselves into a corner):

- **`halo doctor`** gains a "power" line: "ryzenadj: present (0.17.0),
  current profile: chat".
- **`halo install power`** entry in `packages.toml` that does
  `pacman -S --needed ryzenadj` and drops the sudoers snippet.
- **Watchdog:** a future `halo-idlewatch` daemon calls
  `halo power idle` after 60 s of zero `/v1/completions` traffic and
  `halo power chat` on the first incoming request. Daemon is **out of
  scope** for this doc.

---

## Safety notes

**sudo handling.** ryzenadj needs root for `/dev/cpu/*/msr`. We do NOT
run 1bit-cli itself as root. The crate shells out via `sudo -n ryzenadj
...`; the `-n` means "fail if a password would be required" — we never
want 1bit-cli to hang waiting for an interactive password prompt from a
systemd-triggered context. On first run, `halo doctor` checks whether
passwordless ryzenadj sudo is configured and, if not, prints the exact
`/etc/sudoers.d/halo-power` snippet to drop in:

```
%wheel ALL=(root) NOPASSWD: /usr/bin/ryzenadj
```

(Matches our existing passwordless-sudo convention on strixhalo — see
`user_environment` memory.)

**Rollback on failure.** Before applying a new profile we snapshot
`ryzenadj --dump-table` into `/run/user/$UID/halo-power/prev-table.txt`.
If the new ryzenadj exec returns nonzero, we re-exec with the previous
stapm/fast/slow/tctl values and surface both errors. The state file is
only updated on success.

**Ctrl-C during a set.** `ryzenadj` itself is near-instant (<100 ms, one
SMU mailbox round-trip per limit). The window where Ctrl-C can leave us
half-applied is tiny but real. Two mitigations:

1. Apply limits in a **fixed order** — tctl-temp first (raises thermal
   ceiling *before* raising power), then slow, then fast, then stapm.
   Worst-case interrupted state is "headroom raised but power limits
   still at old values" — which is safe, just wasteful.
2. `halo power --revert` is always available and reads from
   `prev-table.txt`. If the user Ctrl-C's mid-set and wants back to
   known-good, `halo power --revert` (or plain reboot) clears it.

**Thermal blowup.** None of the proposed profile numbers exceed
the Strix Halo reference-board defaults by more than ~20 %. They do not
push VID, do not touch `--psi0-current` or `--psi3cpu-current`, and do
not raise `--tctl-temp` above 95 °C (AMD's documented `Tjmax` for this
silicon). The envelope is well inside what HP/ASUS ship for this APU in
their own tuning utilities. Still, the first real bench sweep goes in a
temperature-monitored run (`sensors` polled at 1 Hz, logs to
`~/claude output/`), not blind.

**What this does NOT do.** Does not touch CPU governor
(`/sys/devices/system/cpu/.../scaling_governor` is managed by `cpupower`
/ `tuned`; out of scope). Does not touch `amdgpu` sysfs power levels
(`power_dpm_force_performance_level` — also out of scope, owned by the
amdgpu kernel driver). Does not fiddle with fan curves (they're BIOS /
EC-controlled on this chassis and we have no hook).

---

## Out of scope for this sprint

- Implementation (no Rust this sprint — design only, per the task).
- Auto-switching daemon / idle watchdog (separate design doc).
- Per-process rules in the RyzenZPilot sense (over-engineered for our
  small process set — we drive profile from the HTTP request pattern
  instead).
- iGPU-clock knobs until we verify ryzenadj can read/write them on
  Strix Halo via `--dump-table`.

## Follow-ups (tracked, not done)

1. Decode-bench sweep across the three profile envelopes; pin the numbers
   in the mapping table above. Output JSON into `~/claude output/`.
2. `halo install power` entry in `packages.toml`.
3. `halo doctor` power-section line.
4. `halo-idlewatch` daemon design doc (separate).
5. Revisit option (c) — pure-Rust MSR — if RyzenAdj ever drops Strix Halo
   support or if we want to eliminate the `sudo` subprocess.

---

## References actually read while writing this doc

- `https://github.com/Tetramatrix/RyzenZPilot` — cloned shallow to
  `/tmp/rzpilot`, contents: `docs/README.txt`, `docs/index.html`,
  `docs/RyzenZPilot-v3.1.5.zip` (single file: `RyzenZPilot.exe`). No
  source code in the repo. All feature claims in this doc sourced from
  the landing page verbatim.
- `FlyGoat/RyzenAdj` — upstream Linux tool for AMD SMU power tuning
  (referenced from RyzenZPilot's README as the underlying library).
  Presence of flags (`--stapm-limit`, `--fast-limit`, `--slow-limit`,
  `--tctl-temp`, `--gfx-clk`, `--dump-table`) based on the project's
  current AUR package (`ryzenadj-git`, installed on strixhalo).
- 1bit systems internal: `project_bitnet_live_bench`, `project_bitnet_rocprof_plan`
  (for the 92 % LPDDR5-peak and decode-tok/s numbers that motivate this).
