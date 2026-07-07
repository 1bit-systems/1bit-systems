---
name: quality-gate
description: Automated quality verification after every subagent completes — compile check, test run, diff integrity, and output validation. Blocks bad code from entering the repo. Use after every worker subagent, before commits, and in CI-style gating.
---

# Quality Gate Framework

Automated verification pipeline that runs after every subagent completes.
If a worker says "done" but the code doesn't compile, the gate catches it
before anything gets committed.

## Gate Layers

Every agent output passes through these gates in order:

```
Agent completes → Layer 1: Build → Layer 2: Tests → Layer 3: Diff → Layer 4: Output → ✅ PASS
                                                    ↓               ↓
                                                  FAIL            FAIL
```

### Layer 1 — Build Gate (hard block)

Must pass. No exceptions.

```bash
# For Zig projects
cd /home/bcloud && zig build 2>&1 | tail -20

# For C++ projects (spec-decode)
cd /home/bcloud/spec-decode && cmake --build build 2>&1 | tail -20

# For TypeScript projects
cd /home/bcloud && npx tsc --noEmit 2>&1 | tail -20
```

**Pass:** Exit code 0, no errors in output.
**Fail:** Any compilation error → return to agent with exact errors.

### Layer 2 — Test Gate (soft block)

Run if tests exist. Non-zero exit → warn but don't block unless `--strict`.

```bash
# Zig tests
cd /home/bcloud/engine/fusion && zig build test 2>&1 | tail -30

# C++ tests
cd /home/bcloud/spec-decode/build && ctest --output-on-failure 2>&1 | tail -30

# Python tests
cd /home/bcloud && python3 -m pytest tools/ -x -q 2>&1 | tail -20
```

**Pass:** All tests pass or no tests exist.
**Warn:** Test failures (report to agent, don't block).
**Block (strict):** Any test failure.

### Layer 3 — Diff Integrity (soft block)

Checks that changes are coherent, not destructive.

```bash
# What changed?
git diff --stat HEAD

# Any files deleted unexpectedly?
git diff --diff-filter=D --name-only HEAD

# Any binary files changed? (suspicious)
git diff --numstat HEAD | awk '$1 == "-" && $2 == "-"'

# Line count sanity — did we delete >500 lines without adding similar?
git diff --shortstat HEAD
```

**Checks:**
- No unexpected file deletions (only if task explicitly said "delete")
- No binary file modifications without explicit intent
- No >500 line net deletion without matching task description
- Diff touches only files mentioned in task scope
- No changes to `.gitignore`, `package-lock.json`, `CMakeLists.txt` without intent

### Layer 4 — Output Validation (soft block)

Checks the agent's output for common failure patterns.

**Red flags:**
- "I apologize" / "I couldn't" / "Unfortunately" → agent hit limitations
- "Let me try" repeated >3 times → loop detected
- Output < 20 lines for a task that should produce code → probably failed
- No file paths mentioned → no actual changes made
- "This is a cutover" but code still has legacy patterns → false claim

## Gate Configuration

```json
{
  "quality_gate": {
    "strict": false,
    "layers": {
      "build": { "enabled": true, "block": true },
      "test": { "enabled": true, "block": false },
      "diff": { "enabled": true, "block": false },
      "output": { "enabled": true, "block": false }
    },
    "auto_fix_attempts": 1
  }
}
```

## Gate Protocol

After every `worker` or `subagent` call that modifies files:

```
1. Run Layer 1 (Build)
   ├─ PASS → continue
   └─ FAIL → feed errors back to agent, retry once
              └─ Still FAIL? → escalate with error log

2. Run Layer 2 (Tests)
   ├─ PASS → continue
   └─ FAIL → report warnings, continue (or block if strict)

3. Run Layer 3 (Diff)
   ├─ PASS → continue
   └─ FAIL → report anomalies, require human review

4. Run Layer 4 (Output)
   ├─ PASS → ✅ Agent output verified
   └─ FAIL → mark agent result as suspect, possibly retry
```

## Integration with Other Skills

- **babysitter**: Run quality-gate after babysitter confirms session completion
- **supervisor-orchestration**: Insert gates between pipeline stages
- **scope-decomposer**: Decomposer defines per-task acceptance criteria that the gate checks
- **auto-commits**: Only commit if all gates pass (with strict mode)

## Quick Check

```bash
# One-shot gate on current state
bash ~/.pi/agent/skills/quality-gate/check.sh

# Gate on specific commit
bash ~/.pi/agent/skills/quality-gate/check.sh --against HEAD~1

# Strict mode (test failures block)
bash ~/.pi/agent/skills/quality-gate/check.sh --strict
```
