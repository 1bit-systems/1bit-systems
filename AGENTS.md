<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **1bit-systems** (15957 symbols, 27916 relationships, 229 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Bug Fix Workflow

**MUST file a GitHub issue BEFORE making any fix.** The purpose is to document the bug (what's wrong, where, impact) before changing any code. This creates a clear audit trail: issue describes the problem, commit/PR provides the solution.

1. **File the issue** using `github_issue_write` with `method: "create"` — include:
   - Clear title with file name and bug category
   - Summary explaining what's wrong and where
   - Impact assessment (what breaks, when)
   - Suggested fix approach
   - Appropriate labels (`bug`, `security`, `performance`, etc.)
2. **Implement the fix** — edit the code
3. **Close the issue** — after the fix is committed, close the issue with `state: "closed"` and reference the commit/PR in a comment

Exception: trivial typos (comment spelling, formatting) don't need an issue. Every real bug does.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/1bit-systems/context` | Codebase overview, check index freshness |
| `gitnexus://repo/1bit-systems/clusters` | All functional areas |
| `gitnexus://repo/1bit-systems/processes` | All execution flows |
| `gitnexus://repo/1bit-systems/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
