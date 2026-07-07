---
name: suggested-edits
description: Tab-navigable multi-edit suggestions with inline diff preview, accept/skip/reject controls, and context-aware chunking. Use when making multiple changes across one or more files, or when the user asks for suggested edits, tab navigation, or interactive edit review.
---

# Suggested Edits — Tab-Navigable Multi-Edit Review

When making multiple code changes, this skill structures each edit as a
**numbered suggestion** the user can navigate with Tab. Each suggestion
shows the exact change (old → new), file location, and accepts on Enter,
skips on Tab, or rejects all remaining on Escape.

## Overview

```
┌────────────────────────────────────────────────────────────┐
│  Suggested Edits  (3/6)                          [Tab] ✓   │
│────────────────────────────────────────────────────────────│
│  ›  1. src/engine/npu/worker.ts:54                        │
│      ── Old: if (err) throw err;                          │
│      ┌─ New: if (err) {                                   │
│      │       console.error("[npu] dispatch failed:", err); │
│      │       return null;                                   │
│      │     }                                               │
│                                                             │
│      2. src/engine/npu/worker.ts:89                        │
│      ── Old: maxRetries = 3                                │
│      ┌─ New: maxRetries = 5                                │
│                                                             │
│  [Tab=Accept & Next] [Shift+Tab=Skip & Next] [Esc=Abort]  │
└────────────────────────────────────────────────────────────┘
```

## How It Works

1. **Analyze → Chunk** — The agent breaks changes into atomic edits
   (one logical change per suggestion).

2. **Present** — Each suggestion shows file path, line, old text, new
   text, and sometimes a reason.

3. **Navigate** — The user tabs through suggestions one at a time.

4. **Apply** — Accepted edits are applied to the file immediately.
   Skipped edits are dropped. Aborted edits cancel all remaining.

## Activating the Overlay

Use the TUI overlay component for interactive navigation:

```bash
node ~/.pi/agent/skills/suggested-edits/suggested-edits.js \
  --file "src/example.ts" \
  --old "if (err) throw err;" \
  --new "if (err) { console.error(err); return null; }" \
  --old "maxRetries = 3" \
  --new "maxRetries = 5"
```

Or let the agent invoke it programmatically via the `workflow` tool
when assembling multiple edits (see [Agent Workflow](#agent-workflow)).

## Skill-Guided Agent Behavior

When this skill is active and you need to make multiple edits, follow
this protocol:

### Step 1 — Chunk Edits

Break changes into atomic, single-purpose suggestions. Each suggestion
should be a complete, correct edit the user can understand at a glance.

Good chunking:
```
Suggestion 1: Add input validation to processUser()
  → guard clause for null/undefined

Suggestion 2: Add error logging to processUser()
  → log failures before rethrowing

Suggestion 3: Update call site in handler.ts
  → use try/catch instead of .catch()
```

Bad chunking (too coarse):
```
Suggestion 1: Rewrite processUser() entirely
  → 40-line diff, multiple concerns
```

### Step 2 — Present Suggestions

Present suggestions in a structured format. Show the full suggestion
list up front so the user knows the scope:

````
## Suggested Edits (3 total)

Press **Tab** to accept each suggestion and advance to the next.
Press **Shift+Tab** to skip and advance.
Press **Escape** to abort remaining suggestions.

### Suggestion 1/3 — `src/auth.ts:42`
*Add null guard in `validateToken()`*

```diff
- if (payload.exp < Date.now() / 1000) {
+ if (!payload || payload.exp < Date.now() / 1000) {
    throw new AuthError("Token expired");
  }
```

### Suggestion 2/3 — `src/auth.ts:71`
*Add error context to `verifySignature()`*

```diff
- throw new AuthError("Invalid signature");
+ throw new AuthError("Invalid signature", { 
+   kid: header.kid,
+   alg: header.alg,
+ });
```

### Suggestion 3/3 — `src/handler.ts:15`
*Wrap route handler with error handling*

```diff
- router.post("/auth", authenticate);
+ router.post("/auth", async (req, res) => {
+   try { await authenticate(req, res); }
+   catch (err) { errorHandler(err, req, res); }
+ });
```
````

### Step 3 — Apply Sequentially

Apply accepted suggestions one at a time using the `edit` tool.
After each accepted suggestion, confirm it was applied successfully
and advance to the next.

- **Tab** → Use `edit` to apply the current suggestion, then show
  the next one.
- **"skip" or Tab again without applying** → Move to next without
  changing the file.
- **"stop" or Escape** → Cancel remaining suggestions.

Track state clearly:

```
✓ Applied: Suggestion 1/3 (null guard)
   Pending: Suggestion 2/3 (error context)
   Pending: Suggestion 3/3 (handler wrapper)
```

### Step 4 — Summary

After all suggestions are processed, show a summary:

```
═══ Suggested Edits: Complete ═══
✓ Applied: 2 of 3
⏭ Skipped: 1 (suggestion 3 — handler wrapper)
───
Files modified: src/auth.ts
```

## Interactive Overlay (TUI)

For a richer experience, the agent can launch the TUI overlay component
(`suggested-edits.js`) which renders a live-navigable suggestion list
in the terminal.

### When to use the TUI overlay

- You have **3+ edits** across **1+ files**
- The user is in `pi` interactive TUI mode
- The user explicitly asks for a visual preview

### When to use the bare protocol (Step 2–4)

- Only 1–2 small edits
- The user prefers inline diff format
- Working in headless/non-TUI mode

### How the agent invokes the TUI overlay

The agent uses the `suggested-edits.js` script as a TUI overlay
via `ctx.ui.custom()`. The overlay handles keyboard input and
returns a list of accepted/rejected edits.

The overlay component (`suggested-edits.js`) supports:

- **Tab** — Accept current suggestion, apply it, advance
- **Shift+Tab** — Skip current suggestion, advance
- **Escape** — Abort all remaining suggestions
- **Up/Down** — Navigate suggestions without accepting

## Examples

### Example 1: Rename + update callers (3 edits)

```
You: "Rename `computeHash` to `digest`"

Agent → Suggested Edits (3):
  [1/3] src/crypto.ts:12   — Rename function declaration
  [2/3] src/crypto.ts:45   — Update recursive call
  [3/3] src/auth.ts:88     — Update import + call site

Tab → Tab → Tab → Done
```

### Example 2: Add logging + error handling (5 edits)

```
You: "Add structured logging to the payment flow"

Agent → Suggested Edits (5):
  [1/5] src/payment.ts:22    — Log incoming request
  [2/5] src/payment.ts:45    — Log validation result
  [3/5] src/payment.ts:67    — Log provider response
  [4/5] src/payment.ts:89    — Log retry attempt
  [5/5] src/payment.ts:105   — Log final outcome

Tab → Tab → Skip → Tab → Tab
  (Skipped #3 — provider response was sensitive)
```

## Tips

- **File at a time** — Group suggestions by file when possible
  so the user sees one file's changes together.
- **Line references** — Always include the line number (file:line)
  for context. Use nearest relevant line (function start, not
  inside a comment block).
- **Reason columns** — For large edit sets, add a one-line reason
  per suggestion so the user can decide quickly.
- **Diff context** — Show 1–2 lines of surrounding context so the
  user recognizes the code without searching.
- **Reversible** — If the user undoes an accepted suggestion, make
  the reverse edit as a new suggestion with `[UNDO]` prefix.

## Related

- Use with `auto-commits` skill to commit accepted edits as a
  single cohesive commit.
- Use with `gitnexus-impact-analysis` before making suggested
  edits to check blast radius.
