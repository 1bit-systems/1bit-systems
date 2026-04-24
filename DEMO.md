# DEMO.md — 3-minute 1bit systems cold open

Recording script for a screencast / walkthrough. Times are target, not hard.

## 00:00 — open (15s)

Hero shot on the terminal. Type, don't paste:

```bash
1bit status
```

Expected: 7 green dots (bitnet · strix · sd · whisper · kokoro · lemonade · agent).
Voice-over: "1bit systems runs on this single Strix Halo mini-PC — AMD Radeon 8060S, 128 GB unified memory. Everything on that list is a service running locally, no cloud."

## 00:15 — the page (20s)

Switch to browser, visit **https://strixhalo.local/**.

Voice-over: "Three pillars — HIP kernels, Rust caller, agents + services.
All green on this box right now. The tok/s number in the hero is live —
it's reading /metrics from the Rust server."

Scroll once to show the pillar grid + benchmark table.

## 00:35 — one chat completion (30s)

Back to terminal:

```bash
curl -s http://127.0.0.1:8180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"1bit-monster-2b","messages":[{"role":"user","content":"Explain ternary weights in one sentence."}],"max_tokens":64,"temperature":0,"stream":false}' \
  | jq -r .choices[0].message.content
```

Expected: a coherent one-sentence answer in <1s. Voice-over during: "This
is the Rust 1bit-server on port 8180, hitting hand-written ternary BitNet
kernels on the GPU."

## 01:05 — parity proof (30s)

Prior to v0.1.0 (2026-04-24) this slot showed the live gen-1 (C++, :8080)
vs gen-2 (Rust, :8180) shadow-burnin. The cutover is done — gen-2 owns
`:8180`, gen-1 is retired. Show the archived summary instead:

```bash
/home/bcloud/1bit systems-core/benchmarks/shadow-burnin.sh --summary --archive
```

Point at the numbers: rounds (1500+), exact-match rate 96.66% at
temperature zero, PPL 9.1805 (Rust) vs 9.1607 (C++ baseline).

Voice-over: "Before we cut over, gen-1 C++ and gen-2 Rust agreed
byte-for-byte 96.66 percent of the time at temperature zero. The remaining
~3 percent is sub-ULP FP16 rounding. PPL on wikitext is 9.18 vs Microsoft's
9.16 paper baseline — inside tolerance. Rust owns :8180 as of v0.1.0."

## 01:35 — install story (45s)

Open a second terminal. Fake a fresh box:

```bash
cd /tmp && git clone git@github-bong:bong-water-water-bong/1bit-systems.git
cd 1bit-systems
./install.sh            # or: CI=1 ./install.sh for the syntax-only version
```

Let it talk through the steps (cloning rocm-cpp, building kernels, cargo
install, `1bit install core`). Cut to after `✓ update complete` if it takes
longer than 45s in real time.

## 02:20 — CLI tour (30s)

```bash
1bit doctor          # 0 warn, 0 fail
1bit install --list  # 8 components
1bit logs strix -n 5 # SSE streaming recent events
```

Voice-over: "The halo CLI is one Rust binary. Status, logs, doctor, update,
install — standard ops story."

## 02:50 — close (15s)

Voice-over: "Public repos on GitHub, 90+ workspace tests green, CI on
every push. v0.1.0 shipped 2026-04-24. Next gate is the NPU serve-path
trip — public launch opens when that lands."

End card: 1bit systems wordmark + "strix-ai-rs — built on Strix Halo".

## Pre-roll checklist

- [ ] `1bit doctor` green (0 warn, 0 fail).
- [ ] Landing page loads without cert warnings (accept halo CA once in the
  browser profile you record with).
- [ ] No `~/.cache/`, `target/`, or build noise visible — pinned tmux pane
  with a clean fish prompt.
- [ ] Model file present at `~/1bit systems/models/halo-1bit-2b.h1b`.
- [ ] Second Chrome profile (halo-browser) closed during recording so the
  default profile's bookmarks / tabs aren't in frame.

## Retakes

Each section is independent. If a live number doesn't render (e.g. live
tokps = 0), run a single chat completion against :8180 first to prime the
metrics sliding window, then retake that shot.
