# IDEAS.md — High-Impact Improvements for 1bit Coding Agent

> Generated from analysis of `/home/bcloud/1bit-agent/packages/coding-agent/` (README, docs/, src/, examples/)
> Date: 2026-07-07

---

## Overview

Pi is an aggressively extensible, minimal coding agent harness. It ships with a powerful extension API, rich TUI component system, SDK for embedding, RPC for process integration, skill system (Agent Skills standard), session tree branching, auto-compaction, and a package ecosystem via npm/git. Its philosophy explicitly avoids built-in sub-agents, MCP, plan mode, permission popups, background bash, and built-in to-dos — instead enabling users to build exactly what they want via extensions, skills, and packages.

After a thorough review, here are **5 high-impact improvement ideas** that align with pi's philosophy of extensibility and minimalism, addressing real pain points without contradicting core design principles.

---

## Idea 1: Extension Permission Declaration System

### Problem

Extensions currently run with **full system access** — they can read/write any file, spawn any process, make any network request, and access any environment variable. The only safeguard is the README warning: _"Review source code before installing third-party packages."_ 

Package installation is frictionless (`pi install npm:foo`), and the package ecosystem is growing. A single malicious or buggy extension can exfiltrate API keys, destroy files, or pivot through the host. Project trust only controls _loading_ — once loaded, extensions have unlimited power.

Pi's philosophy explicitly rejects _per-operation_ permission popups ("No permission popups"), but there's a middle ground: **declarative, upfront permissions.**

### Proposed Solution

Allow extensions to **declare** required capabilities via a manifest, and let pi enforce them transparently:

```typescript
// extensions/my-ext/manifest.ts (or frontmatter in index.ts)
export const permissions = {
  filesystem: { read: true, write: ["./src/**"], deny: ["**/.env", "**/secrets/**"] },
  network: { domains: ["api.github.com", "*.openai.com"] },
  shell: { allow: ["git", "npm", "node", "docker"] },
  env: { read: ["PATH", "HOME"], deny: ["*_API_KEY", "*_SECRET*"] },
};
```

**Implementation layers:**

| Layer | What it does |
|-------|-------------|
| **Declaration** | Extension exports a `permissions` object. Pi validates at load time. |
| **Transparency** | `/extensions` shows permission summary. `pi install` displays permissions before confirming. |
| **Soft enforcement** | Pi warns when an extension exceeds declared permissions (opt-in strict mode blocks). |
| **Tool-scoped** | Extensions that register custom tools can declare per-tool permissions. |
| **Package registry metadata** | `pi` manifest key in package.json can include `permissions` for discovery before install. |

**Key design choices:**

- **Not a sandbox** — this is a transparency and trust mechanism, not a security boundary. Pi's security stance remains the same.
- **No runtime popups** — permissions are declared once, upfront. No interruption during use.
- **Opt-in strict mode** — power users can set `"extensionPermissions": "strict"` to block undeclared operations.
- **Backward compatible** — extensions without a declaration default to "legacy/unrestricted" with a warning.

### Complexity: MEDIUM

- **Extension API change**: Add optional `permissions` export to extension modules (~200 lines)
- **Permission model/types**: Define capability schema (~300 lines)
- **Enforcement hooks**: Intercept tool calls, `node:fs` usage, `fetch`, `child_process` via extension context (~400 lines)
- **TUI display**: Permission summary in extension list and install flow (~200 lines)
- **Testing**: Permutation of capability combinations (~500 lines)

**Total estimated:** ~1,600 lines across 6-8 files. Can ship as a single PR.

### Estimated Impact: HIGH

- **Security**: Reduces attack surface from malicious/buggy extensions
- **Trust**: Makes the package ecosystem safer to browse and install from
- **Auditability**: Users and reviewers can quickly understand what an extension needs
- **Ecosystem growth**: Lower barrier to sharing extensions (reviewers can check permissions at a glance)
- **Aligns with pi philosophy**: No runtime popups, no sandbox pretense — just transparency

---

## Idea 2: Agent Observability & Session Analytics

### Problem

Pi currently has **no built-in observability** beyond per-session token counts and costs in the footer. There is no way to:

- See **what happened** across past sessions (what worked? what failed?)
- Track **tool call success rates** over time (is `bash` erroring more often?)
- Understand **cost trends** per project, per model, per task type
- Identify **compaction patterns** (when does context exhaustion happen?)
- Debug **why an agent made a bad decision** retrospectively

The event system (`agent_start`, `tool_call`, `tool_result`, `turn_end`, `agent_end`, etc.) already captures all the raw data. It just isn't persisted or analyzed. Users are flying blind.

### Proposed Solution

A structured observability layer that **logs and aggregates** agent events across sessions, with an optional TUI dashboard:

```
~/.1bit/agent/analytics/
├── sessions.db          # SQLite database (or JSONL)
├── 2026-07/
│   ├── project-foo.jsonl
│   └── project-bar.jsonl
└── summary.json         # Aggregated stats
```

**Core features:**

| Feature | Description |
|---------|-------------|
| **Event logger** | Extension that subscribes to all agent events and writes structured records to SQLite/JSONL |
| **Session summaries** | Per-session: duration, turns, tool calls, tokens, cost, errors, compaction events |
| **Project dashboards** | Aggregated: cost over time, model usage distribution, tool success rates, common errors |
| **TUI browser** | `/analytics` command opens an interactive dashboard in pi (SelectList + charts via braille/block chars) |
| **Export** | JSON/CSV export for external analysis |
| **Query API** | SDK method: `session.analytics.query({ project, dateRange, metric })` |
| **Cost alerts** | Configurable budget thresholds with notifications |

**Why this fits as an extension/skill (not core):**

- Ships as an **official pi package** (`pi install npm:@1bit/pi-analytics`) or bundled skill
- Uses existing extension hooks (`tool_call`, `tool_result`, `turn_end`, `agent_end`, `session_start`, `session_shutdown`)
- The TUI dashboard uses existing `SelectList`, `SettingsList`, `Container`, `Text` components
- Opt-in, not forced — aligns with minimal core philosophy

### Complexity: MEDIUM

- **Event logging extension**: ~400 lines (subscribe to events, serialize, write to SQLite)
- **SQLite schema**: ~100 lines (sessions, turns, tool_calls, costs, errors tables)
- **Aggregation queries**: ~200 lines (SUM, AVG, GROUP BY over the schema)
- **TUI dashboard**: ~500 lines (reuse existing components: SelectList, Container, Text, DynamicBorder)
- **CLI + SDK API**: ~200 lines

**Total estimated:** ~1,400 lines. Ships as an extension.

### Estimated Impact: HIGH

- **Debugging**: Finally see why an agent failed — replay the exact sequence of tool calls that led to an error
- **Cost optimization**: Identify which projects/models/tasks drive cost, adjust model selection
- **Prompt improvement**: See patterns in compaction/context exhaustion, improve AGENTS.md instructions
- **Team visibility**: Share session summaries for collaborative debugging
- **Pi improvement**: Aggregate telemetry (opt-in) to improve pi's defaults and prompts

---

## Idea 3: Intelligent Context Management (beyond binary compaction)

### Problem

Pi's compaction system is **binary**: either messages are in context or summarized away. There's no middle ground. Specific pain points:

1. **Giant tool outputs pollute context** — a `bash` command that outputs 50KB or a `read` that returns a 10,000-line file immediately consumes thousands of tokens. These stay in context until the next compaction cycle.
2. **Stale context accumulates** — the agent reads file A, then rewrites it. The original read of file A is no longer relevant, but it stays in context.
3. **No output prioritization** — all tool outputs are treated equally. A critical error message has the same weight as a verbose log line.
4. **Compaction is lossy** — when it triggers, it summarizes everything in the compaction window, including context that might have been auto-prunable without summarization.

The result: context windows fill up faster than necessary, compaction triggers more often, and quality degrades in long sessions.

### Proposed Solution

A **context middleware layer** that operates transparently between tool execution and context assembly, applying smart transformations:

```
Tool Result → Context Middleware → LLM Context
                ├─ Auto-truncate large outputs with smart summarization
                ├─ Mark stale reads (file later modified)
                ├─ Collapse redundant outputs
                └─ Prioritize errors/signals over noise
```

**Specific mechanisms:**

| Mechanism | Description |
|-----------|-------------|
| **Auto-truncation** | When tool output exceeds a configurable threshold, replace the full output with: a header (first 500 chars), a summary (model-generated 1-2 sentences), and a "full output available" indicator |
| **Stale read detection** | When `edit` or `write` modifies a file, mark previous `read` results for that file as stale. Inject `[This file has been modified since this read — current content may differ]` into the read output |
| **Output deduplication** | When the same command is run twice or the same file is read twice, collapse the duplicate with `[Same output as previous call #N — omitted]` |
| **Error amplification** | When a tool returns `isError: true` or exit code ≠ 0, boost it with visual emphasis and never auto-truncate |
| **Configurable thresholds** | All behaviors are configurable in `settings.json` |

**Implementation as a context event middleware:**

```typescript
pi.on("tool_result", async (event, ctx) => {
  // Auto-truncate large bash output
  if (event.toolName === "bash" && event.content?.[0]?.text?.length > 10000) {
    const truncated = await summarizeOutput(event.content[0].text, ctx);
    return {
      content: [{ type: "text", text: truncated }],
      details: { ...event.details, autoTruncated: true, originalLength: event.content[0].text.length },
    };
  }
});

pi.on("context", async (event, ctx) => {
  // Mark stale reads
  const modifiedFiles = findModifiedFiles(event.messages);
  return {
    messages: event.messages.map(msg => markStaleReads(msg, modifiedFiles)),
  };
});
```

This is **fully implementable as an extension** using existing `tool_result` and `context` hooks. No core changes needed. It could ship as an official pi package or be included as a built-in skill.

### Complexity: MEDIUM

- **Auto-truncation with LLM summarization**: ~300 lines (call cheap model to summarize output)
- **Stale read detection**: ~200 lines (track file writes/edits, annotate reads)
- **Output deduplication**: ~150 lines (hash-based dedup of tool outputs)
- **Error amplification**: ~100 lines (detect isError/exitCode, inject emphasis)
- **Configuration/settings**: ~100 lines
- **Tests**: ~400 lines

**Total estimated:** ~1,250 lines. Ships as an extension or official package.

### Estimated Impact: HIGH

- **Longer effective sessions**: Context windows last 2-4x longer before compaction needed
- **Better compaction quality**: Less noise to summarize → better summaries
- **Faster response times**: Smaller context = lower latency per turn
- **Lower costs**: Fewer tokens in context = cheaper API calls
- **Smarter agent**: Less distraction from irrelevant/stale context → better decisions

---

## Idea 4: Command Palette with Fuzzy Discovery

### Problem

Pi has a **growing surface area of commands, skills, and shortcuts**, but discovery is fragmented:

- Built-in commands: `/hotkeys` shows keyboard shortcuts, but not commands or skills
- Extension commands: must know the name to invoke (`/mycommand`)
- Skills: `/skill:name` requires knowing the exact skill name
- Prompt templates: `/templatename` with no browse/discovery mechanism
- Settings: `/settings` is a nested menu, not searchable
- Model switching: `/model` or Ctrl+L, no fuzzy search across model names

New users struggle to discover capabilities. Power users waste keystrokes navigating menus. The Ctrl+P model cycling and tab completion for paths show pi already values discoverability — this extends that principle to the entire command surface.

### Proposed Solution

A **unified command palette** (similar to VS Code's Ctrl+Shift+P or macOS Spotlight) that provides fuzzy-searchable access to:

| Category | Items |
|----------|-------|
| **Built-in commands** | `/new`, `/fork`, `/compact`, `/export`, `/share`, `/settings`, etc. |
| **Extension commands** | All registered via `pi.registerCommand()` |
| **Skills** | All discovered skills with descriptions |
| **Prompt templates** | All discovered templates with filenames |
| **Keyboard shortcuts** | All registered shortcuts with descriptions |
| **Settings** | All settings keys with current values |
| **Session history** | Recent prompts/sessions for quick reuse |
| **Models** | Available models (merged with Ctrl+L functionality) |

**UX:**

```
┌─ Command Palette ──────────────────────────────────────────────┐
│                                                                 │
│ > compa                                                         │
│                                                                 │
│   /compact             Summarize older context                  │
│   /compact <instructions>  Manual compact with custom prompt    │
│   compact (setting)     compaction.enabled: true                │
│   session_before_compact  Extension event hook                  │
│                                                                 │
│   ↑↓ navigate   enter execute   esc cancel                     │
└─────────────────────────────────────────────────────────────────┘
```

**Trigger:** Configurable shortcut (e.g., Ctrl+Shift+P or Ctrl+K), type `/palette`, or bind to a key.

**Implementation:** Extension using existing `SelectList`, `ctx.ui.custom()`, and `pi.registerShortcut()`. No core changes.

### Complexity: LOW

- **Fuzzy search engine**: ~150 lines (fzy/ufuzzy-style scoring, or use `fuzzysort` from npm)
- **Data collection**: ~200 lines (gather commands from pi internals, extensions, skills, prompts, settings)
- **TUI renderer**: ~200 lines (reuse SelectList with search, categories, descriptions)
- **Keyboard shortcut registration**: ~50 lines
- **Testing**: ~100 lines

**Total estimated:** ~700 lines. Ships as an extension or bundled into core as a built-in command.

### Estimated Impact: MEDIUM-HIGH

- **Discoverability**: New users can explore all capabilities without reading docs
- **Efficiency**: Power users can invoke anything in 2-3 keystrokes
- **Extension visibility**: Extension commands become as discoverable as built-in commands
- **Skill adoption**: Users can find and use skills without memorizing names
- **Reduced support burden**: Fewer "how do I..." questions

---

## Idea 5: Cost-Aware Model Router

### Problem

Pi supports **20+ providers** and has a sophisticated model registry with cost data, but model selection is entirely manual:

- Users pick one model via `/model` or Ctrl+P
- Model cycling (Ctrl+P) cycles through a user-defined list, but it's still manual
- All prompts go to the same model regardless of complexity

This leads to two inefficiencies:
1. **Simple queries waste money**: "What does git status do?" or "List files in src/" goes to Claude Opus at premium cost
2. **Complex queries use cheap models**: Users forget to switch to a powerful model for hard tasks

Other tools (e.g., Cursor, GitHub Copilot) use model routing to balance cost and quality. Pi has all the data needed (model costs, context size heuristics) but no routing layer.

### Proposed Solution

A **model router extension** that automatically selects the best model for each prompt based on configurable heuristics:

```json
// .1bit/settings.json
{
  "modelRouter": {
    "enabled": true,
    "defaultModel": "anthropic/claude-sonnet-4-20250514:medium",
    "rules": [
      {
        "name": "trivial-query",
        "match": { "promptLength": "< 100", "noContextFiles": true },
        "model": "openai/gpt-4o-mini:off"
      },
      {
        "name": "complex-refactor",
        "match": { "promptPatterns": ["refactor", "architecture", "design"], "contextTokens": "> 50000" },
        "model": "anthropic/claude-opus-4-5:high"
      },
      {
        "name": "code-generation",
        "match": { "skills": ["code-generation"] },
        "model": "anthropic/claude-sonnet-4-20250514:high"
      },
      {
        "name": "quick-read",
        "match": { "onlyTools": ["read", "grep", "find", "ls"] },
        "model": "anthropic/claude-haiku-4-5:off"
      }
    ],
    "costLimit": { "daily": 50, "weekly": 200, "action": "warn" }
  }
}
```

**Routing signals:**

| Signal | Example |
|--------|---------|
| **Prompt length** | Short query → cheap model |
| **Prompt content patterns** | Keywords like "refactor", "debug", "architecture" → powerful model |
| **Context size** | Large context (>50K tokens) → model with large context window |
| **Active tools** | Read-only tools → cheap model; write/edit tools → careful model |
| **Active skills** | Skill declares `suggestedModel` in frontmatter |
| **Time of day / budget remaining** | Near daily limit → force cheap model |
| **Past session outcomes** | High error rate → try a different model |

**Key design choices:**

- **Override on `/model`**: Manual model selection always wins (router steps aside)
- **Transparent**: Each turn shows which model was selected and why (tooltip in footer)
- **Learning mode**: Optional — router tracks success rates per model per task type and adapts
- **Extension-based**: Fully implementable with existing hooks — no core changes

### Complexity: MEDIUM

- **Router engine**: ~300 lines (rule matching, signal extraction, model selection)
- **Rule schema and config**: ~150 lines
- **Cost tracking and budget enforcement**: ~200 lines
- **Footer indicator**: ~100 lines ("router: gpt-4o-mini ← short query")
- **before_agent_start hook**: ~100 lines (intercept before LLM call, swap model)
- **Learning/adaptation** (optional): ~400 lines
- **Tests**: ~300 lines

**Total estimated:** ~850-1,550 lines. Ships as an extension.

### Estimated Impact: HIGH

- **Cost reduction**: 30-60% cost reduction by routing simple queries to cheap models
- **Quality improvement**: Complex tasks always get the right model
- **No UX friction**: Users don't think about model selection — it just works
- **Budget predictability**: Cost limits prevent surprise bills
- **Ecosystem play**: Router could integrate with observability (#2) for data-driven routing

---

## Prioritization Matrix

| # | Idea | Complexity | Impact | Effort/Reward | Aligns with Philosophy |
|---|------|-----------|--------|---------------|----------------------|
| 1 | Extension Permissions | Medium | High | ★★★★★ | ✅ Transparency, no popups |
| 2 | Observability & Analytics | Medium | High | ★★★★★ | ✅ Extension-based, opt-in |
| 3 | Intelligent Context Mgmt | Medium | High | ★★★★★ | ✅ Extension-based, transparent |
| 4 | Command Palette | Low | Med-High | ★★★★☆ | ✅ Improves UX, minimal core changes |
| 5 | Cost-Aware Model Router | Medium | High | ★★★★☆ | ✅ Extension-based, optional |

**Recommended implementation order:** 4 → 3 → 1 → 5 → 2

- **4 (Command Palette)** first: low effort, high UX impact, builds momentum
- **3 (Context Management)** second: high practical impact for daily use
- **1 (Permissions)** third: important for ecosystem trust, but needs design thought
- **5 (Router)** fourth: depends on having good observability (idea #2) for data-driven routing
- **2 (Observability)** last: most valuable long-term, but best built after other ideas generate data to observe

---

## Additional Community Ideas (Lower Priority)

These emerged from analysis but were deprioritized vs. the top 5:

| Idea | Why deprioritized |
|------|-------------------|
| **Session diff/replay UI** | Partially addressed by `/tree` and branch summaries; replay is niche |
| **Skill composition/dependencies** | Skills are standalone by design per Agent Skills spec; composition adds complexity |
| **Extension marketplace in TUI** | `pi install` and npm search already work; TUI browsing is nice-to-have |
| **Automatic AGENTS.md generation** | pi can already do this when asked; automation has quality risks |
| **Multi-terminal/split pane** | Conflicts with pi's "no background bash, use tmux" philosophy |
| **Git integration (auto-commit on turn)** | Already exists as example extension (`git-checkpoint.ts`) |
| **Snippet/template library** | Prompt templates + skills already cover this; a library is organizational |

---

## Appendix: Technical Implementation Notes

### Idea 1: Extension Permissions — Type definitions

```typescript
interface ExtensionPermissions {
  filesystem?: {
    read?: boolean | string[];   // true = all, string[] = glob allowlist
    write?: boolean | string[];  // true = all, string[] = glob allowlist
    deny?: string[];             // glob denylist (overrides allow)
  };
  network?: {
    domains?: string[];          // glob allowlist (e.g., "*.github.com")
    deny?: string[];            // glob denylist
  };
  shell?: {
    allow?: string[];           // command allowlist (e.g., ["git", "npm"])
    deny?: string[];            // command denylist
  };
  env?: {
    read?: string[];            // env var allowlist
    deny?: string[];            // env var denylist
  };
}
```

### Idea 3: Context middleware — stale read detection

```typescript
// Track file modifications
const fileModifications = new Map<string, number>(); // path → last_modified_tool_call_id

pi.on("tool_call", (event) => {
  if (event.toolName === "write" || event.toolName === "edit") {
    fileModifications.set(event.input.path, event.toolCallId);
  }
});

pi.on("context", (event) => {
  event.messages.forEach(msg => {
    if (msg.role === "toolResult") {
      const toolCall = findToolCall(msg);
      if (toolCall?.toolName === "read" && wasLaterModified(fileModifications, toolCall)) {
        msg.content[0].text = `[STALE — file modified after this read]\n${msg.content[0].text}`;
      }
    }
  });
});
```

### Idea 5: Model router — rule evaluation

```typescript
interface RouterRule {
  name: string;
  match: {
    promptLength?: { min?: number; max?: number };
    promptPatterns?: string[];
    contextTokens?: { min?: number; max?: number };
    onlyTools?: string[];
    skills?: string[];
    hasContextFiles?: boolean;
  };
  model: string; // provider/model:thinking
}

function evaluateRule(rule: RouterRule, signal: PromptSignal): boolean {
  if (rule.match.promptLength && !inRange(signal.prompt.length, rule.match.promptLength)) return false;
  if (rule.match.promptPatterns && !matchesAny(signal.prompt, rule.match.promptPatterns)) return false;
  if (rule.match.contextTokens && !inRange(signal.contextTokens, rule.match.contextTokens)) return false;
  if (rule.match.onlyTools && !toolsSubset(signal.activeTools, rule.match.onlyTools)) return false;
  if (rule.match.skills && !skillsIntersect(signal.activeSkills, rule.match.skills)) return false;
  return true;
}
```
