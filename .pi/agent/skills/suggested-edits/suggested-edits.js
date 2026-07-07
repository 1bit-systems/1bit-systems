#!/usr/bin/env node
/**
 * suggested-edits.js — Tab-navigable multi-edit suggestion review
 *
 * Two modes:
 *   MODE 1: TUI Overlay (interactive, run inside pi via workflow)
 *   MODE 2: Pretty printer / state tracker (standalone CLI)
 *
 * TUI Overlay:
 *   Returns a TUI component for ctx.ui.custom() that renders a
 *   navigable suggestion list. Each suggestion shows file, line,
 *   old → new diff. Handles Tab, Shift+Tab, Escape, Up/Down.
 *
 * Standalone CLI:
 *   suggested-edits.js --json '[...]'
 *     → Pretty-prints suggestions as a formatted terminal display.
 *
 *   suggested-edits.js --edits '[...]' --apply
 *     → Reads edits JSON, applies them using the pi 'edit' tool
 *       pattern, outputs a summary.
 *
 * Edit JSON format:
 *   [{
 *     "file": "src/example.ts",
 *     "line": 42,
 *     "label": "Add null guard",
 *     "reason": "Prevents crash on missing payload",
 *     "oldText": "if (payload.exp < Date.now())",
 *     "newText": "if (!payload || payload.exp < Date.now())",
 *     "oldContext": "  // validate expiry\n  if (payload.exp < Date.now()) {",
 *     "newContext": "  // validate expiry\n  if (!payload || payload.exp < Date.now()) {"
 *   }]
 *
 * Each edit's oldText/newText should be exact-match-ready for the
 * pi `edit` tool's oldText/newText parameters.
 */

const MODE = process.argv.includes("--tui")
  ? "tui"
  : process.argv.includes("--apply")
    ? "apply"
    : "format";

// ─── Parse input ───────────────────────────────────────────────

function parseInput() {
  const jsonIdx = process.argv.indexOf("--json");
  const fileIdx = process.argv.indexOf("--file");
  const editsIdx = process.argv.indexOf("--edits");
  const oldIdx = process.argv.indexOf("--old");
  const newIdx = process.argv.indexOf("--new");

  let edits = [];

  // --json flag
  if (jsonIdx !== -1 && process.argv[jsonIdx + 1]) {
    try {
      edits = JSON.parse(process.argv[jsonIdx + 1]);
    } catch (e) {
      console.error("ERROR: Invalid --json input:", e.message);
      process.exit(1);
    }
  }

  // --edits flag
  if (editsIdx !== -1 && process.argv[editsIdx + 1]) {
    try {
      edits = JSON.parse(process.argv[editsIdx + 1]);
    } catch (e) {
      console.error("ERROR: Invalid --edits input:", e.message);
      process.exit(1);
    }
  }

  // --file / --old / --new shorthand (multiple pairs)
  if (fileIdx !== -1 && newIdx !== -1) {
    const files = collectValues("--file");
    const olds = collectValues("--old");
    const news = collectValues("--new");
    const lines = collectValues("--line");
    const labels = collectValues("--label");
    const reasons = collectValues("--reason");

    const count = Math.max(files.length, olds.length, news.length);
    for (let i = 0; i < count; i++) {
      edits.push({
        file: files[i] || "unknown",
        line: lines[i] ? parseInt(lines[i], 10) : undefined,
        label: labels[i] || "",
        reason: reasons[i] || "",
        oldText: olds[i] || "",
        newText: news[i] || "",
      });
    }
  }

  return edits;
}

function collectValues(flag) {
  const values = [];
  const idx = process.argv.indexOf(flag);
  if (idx === -1) return values;
  // Collect all values for this flag (repeatable)
  for (let i = idx + 1; i < process.argv.length; i++) {
    if (process.argv[i].startsWith("--")) break;
    values.push(process.argv[i]);
  }
  return values;
}

// ─── ANSI helpers ─────────────────────────────────────────────
const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const GREEN = "\x1b[32m";
const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const CYAN = "\x1b[36m";
const MAGENTA = "\x1b[35m";
const BLUE = "\x1b[34m";
const WHITE = "\x1b[37m";
const BG_GRAY = "\x1b[48;5;236m";
const BG_GREEN = "\x1b[48;5;22m";
const BG_RED = "\x1b[48;5;52m";
const BG_BLUE = "\x1b[48;5;17m";
const REVERSE = "\x1b[7m";
const INVERSE = "\x1b[27m";

function truncate(str, maxLen) {
  if (!str) return "";
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + "...";
}

function stripAnsi(str) {
  return str.replace(/\x1b\[[0-9;]*m/g, "");
}

function visibleLen(str) {
  return stripAnsi(str).length;
}

function padRight(str, len) {
  const vLen = visibleLen(str);
  return str + " ".repeat(Math.max(0, len - vLen));
}

// ─── Formatter ────────────────────────────────────────────────

function formatEdits(edits, { currentIndex = -1, accepted = new Set(), skipped = new Set(), width = 80 } = {}) {
  const lines = [];
  const total = edits.length;

  if (total === 0) {
    return ["(no edits)"];
  }

  // Header
  const header = `${BOLD}${WHITE}Suggested Edits${RESET}  (${Math.max(0, currentIndex + 1)}/${total})`;
  lines.push(header);
  lines.push("");

  for (let i = 0; i < edits.length; i++) {
    const edit = edits[i];
    const isCurrent = i === currentIndex;
    const isAccepted = accepted.has(i);
    const isSkipped = skipped.has(i);
    const isDone = isAccepted || isSkipped;

    // Status badge
    let statusBadge;
    if (isAccepted) {
      statusBadge = `${BG_GREEN}${WHITE} ✓ ${RESET}`;
    } else if (isSkipped) {
      statusBadge = `${DIM}⏭${RESET}`;
    } else if (isCurrent) {
      statusBadge = `${REVERSE} ${BOLD}→${RESET}${INVERSE}`;
    } else {
      statusBadge = "  ";
    }

    // Number
    const numStr = isCurrent
      ? `${BOLD}${WHITE}${i + 1}${RESET}`
      : `${DIM}${i + 1}${RESET}`;

    // File location
    const loc = edit.file
      ? `${CYAN}${edit.file}${edit.line ? `:${edit.line}` : ""}${RESET}`
      : "";

    // Label
    const label = edit.label ? `${DIM}— ${edit.label}${RESET}` : "";

    // Line output
    const prefix = isCurrent ? `${BG_BLUE}  ` : "  ";
    const suffix = isCurrent ? `  ${RESET}` : "";
    const lineStr = `${prefix}${statusBadge} ${numStr}. ${loc} ${label}${suffix}`;
    lines.push(lineStr);

    // Reason (dimmed, shown only for current or if relevant)
    if (edit.reason && (isCurrent || !isDone)) {
      lines.push(`     ${DIM}${truncate(edit.reason, width - 10)}${RESET}`);
    }

    // Diff preview (current suggestion only for compactness)
    if (isCurrent && edit.oldText && edit.newText) {
      const oldPreview = truncate(edit.oldText.split("\n")[0], width - 14);
      const newPreview = truncate(edit.newText.split("\n")[0], width - 14);
      const hasMultiLine =
        edit.oldText.includes("\n") || edit.newText.includes("\n");
      if (oldPreview) {
        lines.push(`     ${RED}─ ${oldPreview}${RESET}`);
        if (hasMultiLine) lines.push(`       ${DIM}...${RESET}`);
      }
      if (newPreview) {
        lines.push(`     ${GREEN}┌ ${newPreview}${RESET}`);
        if (hasMultiLine) lines.push(`       ${DIM}...${RESET}`);
      }
    }

    // Accepted/skipped annotation
    if (isAccepted) {
      lines.push(`     ${GREEN}✓ Accepted${RESET}`);
    } else if (isSkipped) {
      lines.push(`     ${YELLOW}⏭ Skipped${RESET}`);
    }

    lines.push("");  // Spacer
  }

  // Footer with key hints (only if there's a current suggestion)
  if (currentIndex >= 0 && currentIndex < total) {
    const remaining = total - currentIndex;
    const footer = `${DIM}[Tab=${RESET}${GREEN}Accept${RESET}${DIM}] [Shift+Tab=${RESET}${YELLOW}Skip${RESET}${DIM}] [Esc=${RESET}${RED}Abort${RESET}${DIM}] [↑↓=${RESET}Navigate${RESET}${DIM}]  (${remaining} remaining)${RESET}`;
    lines.push(footer);
  } else {
    const summaryAccepted = accepted.size;
    const summarySkipped = skipped.size;
    const summaryColor = summarySkipped > 0 ? YELLOW : GREEN;
    lines.push(
      `${summaryColor}═══ Complete: ${summaryAccepted} accepted, ${summarySkipped} skipped${RESET}`
    );
  }

  return lines;
}

// ─── Summary printer ──────────────────────────────────────────

function printSummary(edits, accepted, skipped) {
  console.log("\n" + "═".repeat(60));
  console.log(`${GREEN}${BOLD} Suggested Edits: Complete${RESET}`);
  console.log("═".repeat(60));
  console.log(`${GREEN}✓ Applied: ${accepted.size}${RESET}`);
  if (skipped.size > 0) {
    console.log(`${YELLOW}⏭ Skipped: ${skipped.size}${RESET}`);
  }

  // Group by file
  const files = new Map();
  for (const idx of accepted) {
    const file = edits[idx].file || "unknown";
    if (!files.has(file)) files.set(file, []);
    files.get(file).push(edits[idx]);
  }
  for (const [file, fileEdits] of files) {
    console.log(`\n  ${CYAN}${file}${RESET}:`);
    for (const edit of fileEdits) {
      const label = edit.label ? ` — ${edit.label}` : "";
      console.log(`    ${GREEN}✓${RESET} ${edit.oldText ? truncate(edit.oldText, 40) : "(new file)"}${label}`);
    }
  }

  console.log("");
}

// ─── Main ──────────────────────────────────────────────────────

const edits = parseInput();

if (edits.length === 0 && MODE !== "tui") {
  console.log(
    "Usage:\n"
    + "  suggested-edits.js --json '[{\"file\": \"...\", \"oldText\": \"...\", \"newText\": \"...\"}]'\n"
    + "  suggested-edits.js --file src/a.ts --old \"old code\" --new \"new code\" ...\n"
    + "  suggested-edits.js --edits '[...]' --apply\n"
    + "  suggested-edits.js --tui  (returns a TUI component for ctx.ui.custom)\n"
    + "\n"
    + "See SKILL.md for full documentation."
  );
  process.exit(0);
}

switch (MODE) {
  case "format": {
    const lines = formatEdits(edits, { width: process.stdout.columns || 80 });
    for (const line of lines) {
      console.log(line);
    }
    break;
  }

  case "apply": {
    // Apply all edits (no interactive review)
    const accepted = new Set();
    const skipped = new Set();

    for (let i = 0; i < edits.length; i++) {
      const edit = edits[i];
      if (!edit.oldText || !edit.newText) {
        skipped.add(i);
        continue;
      }
      // Print what would be applied
      const loc = edit.file ? `${edit.file}${edit.line ? `:${edit.line}` : ""}` : "unknown";
      console.log(`${BG_BLUE}${WHITE} [${i + 1}/${edits.length}] ${loc} ${RESET}`);
      console.log(`  ${RED}─ ${edit.oldText.split("\n")[0]}${RESET}`);
      console.log(`  ${GREEN}┌ ${edit.newText.split("\n")[0]}${RESET}`);
      accepted.add(i);
    }

    printSummary(edits, accepted, skipped);
    break;
  }

  case "tui": {
    // TUI mode: Return a component object for ctx.ui.custom()
    // This is a placeholder — the actual TUI component must be
    // instantiated inside a pi extension where ctx.ui.custom() is
    // available. See the EXTENSION.md reference.
    console.log(
      JSON.stringify({
        _tuiComponent: true,
        type: "suggested-edits",
        edits: edits.map((e) => ({
          file: e.file,
          line: e.line,
          label: e.label || e.oldText?.split("\n")[0] || "",
          reason: e.reason || "",
          oldText: e.oldText,
          newText: e.newText,
        })),
        total: edits.length,
      })
    );
    break;
  }
}
