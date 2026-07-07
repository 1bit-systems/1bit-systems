---
name: benchmark-audit
description: Complete diff on stale benchmark numbers across the entire repo. Scans all reading material (docs, site, README, CLAUDE.md, blog posts, badges, changelogs) and cross-references every tok/s, ms/tok, TFLOPS, and model count against the single source of truth in docs/wiki/performance.md. Use when maintaining accuracy, before releases, or when the user asks about stale numbers.
---

# Benchmark Audit — Keep the Single Source of Truth

**Source of truth**: `docs/wiki/performance.md` (verified July 6, 2026)

Every benchmark claim in the repo is checked against this file. If the source of truth was updated but other files weren't, this catches it.

## Usage

```bash
# Full audit — scan all docs, site, README, CLAUDE.md, badges, blog posts
bash ~/.pi/agent/skills/benchmark-audit/audit.sh

# Quick check — only the most visible files (index, README, CLAUDE.md, badges)
bash ~/.pi/agent/skills/benchmark-audit/audit.sh --quick

# Report-only mode — show results without scanning (uses last scan cache at /tmp/benchmark-audit-cache/)
bash ~/.pi/agent/skills/benchmark-audit/audit.sh --report
```

## What It Checks

| Pattern | Example | Source of truth section |
|---------|---------|------------------------|
| `NNN tok/s` | `291 tok/s` | At a Glance, Engine Speed |
| `NN tok/s` | `97 tok/s` | At a Glance, Engine Speed |
| `N.N ms/tok` | `3.4 ms/tok` | Engine Speed, GPU Decode |
| `NN ms/tok` | `36 ms/tok` | Raw C++ Engine |
| `NN.N TFLOPS` | `55.7 TFLOPS` | Raw Silicon GEMM |
| `38 KB` | binary size | Binary size claim |
| `72× speedup` | narrative | Engine Evolution |
| `24× speedup` | narrative | Engine Evolution |
| `NN×` speedups | narrative | Engine Evolution |
| model counts | `73+ models` | At a Glance |
| `XX TOPS` | `50 TOPS` | Hardware spec |
| `NNN tok/s` (GPU models) | `381 tok/s` | GPU 1-Bit Model Benchmarks |
| `NNN TFLOPS` | `55.7 TFLOPS` | Raw Silicon GEMM |

## Files Scanned

| Priority | Files |
|----------|-------|
| **Critical** | `CLAUDE.md`, `README.md`, `site/index.html`, `engine/npu/BENCHMARKS.md` |
| **High** | `site/*.html`, `site/blog/*.html`, `site/*-badge.json` |
| **Medium** | `CHANGELOG.md`, `ROADMAP.md`, `packaging/README.md`, `docs/*.md`, `prompts/1bit.md` |
| **Info** | `docs/wiki/*.md`, `docs/archive/*.md`, `docs/superpowers/**/*.md`, `*.pr_agent.toml` |

## Fix Workflow

1. Run `bash ~/.pi/agent/skills/benchmark-audit/audit.sh`
2. Review the **STALE** section — all numbers that don't match the source of truth
3. For each stale file, update the numbers to match `docs/wiki/performance.md`
4. Run the audit again to confirm clean

## Files

| Path | Purpose |
|------|---------|
| `audit.sh` | Main script — extracts numbers, diffs against source of truth |
