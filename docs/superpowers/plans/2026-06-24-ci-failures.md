# Multi-Repo CI Failures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore green product CI for the active `bong-water-water-bong` repositories and separate non-product review-bot failures from real build/test failures.

**Architecture:** Treat each repository as an independent unit. Fix real product CI first (`ci`, `CI`, deploy/sync workflows), then handle review-agent workflows only where they block required status checks.

**Tech Stack:** GitHub Actions, Python packaging, ruff, mypy, pytest, uv, GitHub CLI.

## Global Constraints

- Do not bundle unrelated repo fixes into one local git repo; each GitHub repo gets its own branch/commit.
- Run the same command that failed in CI before pushing.
- Prefer workflow fixes for workflow defects; prefer code/test fixes for product defects.
- Do not hide real failures by deleting product CI jobs.
- Review-agent workflows that need missing external secrets should be gated or skipped gracefully, not marked failing.

---

### Task 1: `1bit-lemonade` sync workflow token failure

**Files:**
- Modify: `.github/workflows/*sync*.yml` or the workflow file containing `1bit · sync from upstream`

**Root Cause:** `actions/checkout` has `token:` set to a missing required secret, causing `Input required and not supplied: token`.

- [ ] Clone `bong-water-water-bong/1bit-lemonade`.
- [ ] Inspect `.github/workflows/` for checkout steps using a custom token.
- [ ] Replace the missing required token with `${{ secrets.SYNC_TOKEN || github.token }}` if expression context supports it, or split checkout steps so default `GITHUB_TOKEN` is used when the secret is unavailable.
- [ ] Run `git diff` and `gh workflow run` or push branch.

### Task 2: `lemonade-agents` product CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

**Root Cause:** CI runs `pytest --cov=lemonade_agents --cov-report=term-missing` but package install does not include `pytest-cov`; ruff runs `src tests` when `tests/` may not exist in the checkout/layout.

- [ ] Clone `bong-water-water-bong/lemonade-agents`.
- [ ] Add `pytest-cov>=5` to dev/test dependencies.
- [ ] Make ruff/pytest paths match actual repo layout; if there is no `tests/`, remove `tests` from ruff command or add tests.
- [ ] Run `python -m ruff check ...`, `python -m mypy ...`, and `python -m pytest --cov=lemonade_agents --cov-report=term-missing` locally.

### Task 3: Lemonade module ruff-only failures

**Repos:**
- `lemonade-marketeer`
- `lemonade-reports`
- `lemonade-site`
- `lemonade-supplier`
- `lemonade-inventory`

**Files:**
- Modify the exact files reported by ruff in each repo.

**Root Cause:** Strict ruff rules caught one-line `__main__.py`, unsorted imports, unused imports/variables, ambiguous `l` names, and `str, Enum` rules.

- [ ] Clone each repo.
- [ ] Run `python -m ruff check src tests` to reproduce.
- [ ] Run safe formatting/import fixes: `python -m ruff check src tests --fix` and `python -m ruff format src tests`.
- [ ] Manually handle unsafe fixes: replace one-line `from pkg.cli import main; main()` with a normal function call block, remove unused imports, rename `l` loop variables to `line`, import missing `field`, use `StrEnum` only if Python version supports it or add `# noqa: UP042` if compatibility requires `str, Enum`.
- [ ] Run product CI command from each workflow.

### Task 4: `lemonade-security` drift tests

**Files:**
- Modify: test fixtures in `tests/test_drift.py`, `tests/test_cli.py`, `tests/test_sdk_integration.py`, `tests/test_sdk_plugin.py` or production drift validation if stricter envelope validation is incorrect.

**Root Cause:** Fixtures use event type `store.opened`, which is now rejected as `invalid_envelope`; tests expect downstream drift findings (`namespace_violation`, `approval_gate_drift`).

- [ ] Clone `bong-water-water-bong/lemonade-security`.
- [ ] Reproduce with `python -m pytest`.
- [ ] Decide if `store.opened` should be added to allowed envelope types or tests should use a currently valid event type from the store schema.
- [ ] Add a regression test that proves valid-envelope drift still emits namespace/approval findings.
- [ ] Run `python3 -m ruff check src tests`, `python3 -m mypy`, and `python3 -m pytest`.

### Task 5: `lemonade-vision-server` uv workflow

**Files:**
- Modify: `.github/workflows/ci.yml` or equivalent CI workflow.

**Root Cause:** CI runs `uv pip install ...` without a virtualenv and without `--system`, causing `No virtual environment found; run uv venv ... or pass --system`.

- [ ] Clone `bong-water-water-bong/lemonade-vision-server`.
- [ ] Change CI install step to create/use `.venv` with `uv venv` and `uv pip install`, or add `--system` if the workflow intentionally installs into hosted Python.
- [ ] Run the CI commands locally or with `uv`.

### Task 6: Review-agent workflows across repos

**Repos:** Active Lemonade repos with failing `gemini-review`, `codex-review`, `openhands-review`, `pr-agent-review`, `qodo-merge` workflows.

**Root Cause:** These likely require missing API keys or are triggered on `push` without PR context.

- [ ] Inspect each review workflow failure logs.
- [ ] Gate each job with secret checks and PR-only conditions, e.g. do not run paid/external review bots on `push` without required secrets.
- [ ] Ensure skipped review bots do not fail product CI.

## Execution Order

1. Fix product CI in Tasks 1-5.
2. Push one branch/commit per repo.
3. Re-run Actions and confirm green product CI.
4. Handle review-agent workflows separately, unless they are required status checks.
