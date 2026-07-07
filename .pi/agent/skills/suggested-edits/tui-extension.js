/**
 * suggested-edits TUI Extension — Interactive overlay for pi
 *
 * Reference implementation. Register this with pi via:
 *   pi --skill ~/.pi/agent/skills/suggested-edits
 *   /suggested
 *
 * Or embed the component directly in your own extension.
 *
 * Usage in an extension:
 *   import { SuggestedEditsOverlay } from "./tui-extension.js";
 *
 *   pi.registerCommand("suggested", {
 *     description: "Open suggested edits overlay",
 *     handler: async (args, ctx) => {
 *       const edits = parseSuggestedEdits(args);
 *       const result = await ctx.ui.custom(
 *         (tui, theme, keybindings, done) =>
 *           new SuggestedEditsOverlay(edits, { tui, theme, done })
 *       );
 *       // result.accepted = [...], result.skipped = [...]
 *     }
 *   });
 */

// ─── Imports (requires @earendil-works/pi-tui) ────────────────
// In a real extension, import from the package:
//   import { matchesKey, Key, truncateToWidth } from "@earendil-works/pi-tui";
//   import { Container, Text, Spacer, Markdown } from "@earendil-works/pi-tui";
//
// For this reference, we document the interface and pattern.

/*
 * ─── TUI Component: SuggestedEditsOverlay ─────────────────────
 *
 * A focusable keyboard-navigable overlay for reviewing edit
 * suggestions. Displays one suggestion at a time with a full
 * diff preview.
 *
 * Keyboard controls:
 *   Tab         — Accept current suggestion, apply via edit tool, advance
 *   Shift+Tab   — Skip current suggestion, advance
 *   Escape      — Abort all remaining suggestions
 *   Up          — Go back one suggestion
 *   Down        — Go forward one suggestion
 *   Enter       — Same as Tab (accept and advance)
 *
 * Visual layout (one suggestion at a time, focused view):
 *
 *   ┌──────────────────────────────────────────────────┐
 *   │  Suggested Edits                   3 of 6    [✓] │
 *   │──────────────────────────────────────────────────┤
 *   │                                                    │
 *   │  src/engine/npu/worker.ts:42                       │
 *   │  Add input validation for dispatch()               │
 *   │                                                    │
 *   │  ── if (opts === undefined) {                      │
 *   │  ┌─ if (opts === undefined || opts === null) {     │
 *   │                                                    │
 *   │  Context:                                          │
 *   │    function dispatch(opts?: DispatchOptions) {     │
 *   │                                                    │
 *   │  ├─ Accepted ─────────────────────────────────┤    │
 *   │  │  [1] Add null guard           ✓  applied   │    │
 *   │  │  [2] Add error logging        ✓  applied   │    │
 *   │  ├─ Remaining ────────────────────────────────┤    │
 *   │  │  [3] Update call site          →  current  │    │
 *   │  │  [4] Add retry logic                       │    │
 *   │  │  [5] Add metric emission                   │    │
 *   │  └────────────────────────────────────────────┘    │
 *   │                                                    │
 *   │  [Tab=Accept] [S-Tab=Skip] [Esc=Abort]  [↑↓=Nav] │
 *   └──────────────────────────────────────────────────┘
 *
 * ─── Reference Implementation ───────────────────────────────
 *
 * Below is the key pattern for the component class. This is
 * written as a TypeScript-like reference using pi-tui imports.
 * Copy this pattern into your extension.
 */

/*

import {
  matchesKey,
  Key,
  truncateToWidth,
  visibleWidth,
  Container,
  Text,
  Spacer,
  DynamicBorder,
} from "@earendil-works/pi-tui";

// ─── Types ────────────────────────────────────────────────────

interface SuggestedEdit {
  file: string;
  line?: number;
  label?: string;
  reason?: string;
  oldText: string;
  newText: string;
}

interface OverlayResult {
  accepted: number[];
  skipped: number[];
  aborted: boolean;
}

interface OverlayOptions {
  tui: any;        // TUI instance from ctx.ui.custom callback
  theme: any;      // Theme instance from ctx.ui.custom callback
  done: (result: OverlayResult | null) => void;
}

// ─── Component ─────────────────────────────────────────────────

class SuggestedEditsOverlay {
  private edits: SuggestedEdit[];
  private currentIndex: number;
  private accepted: Set<number>;
  private skipped: Set<number>;
  private aborted: boolean;
  private tui: any;
  private theme: any;
  private done: (result: OverlayResult | null) => void;

  constructor(edits: SuggestedEdit[], opts: OverlayOptions) {
    this.edits = edits;
    this.currentIndex = 0;
    this.accepted = new Set();
    this.skipped = new Set();
    this.aborted = false;
    this.tui = opts.tui;
    this.theme = opts.theme;
    this.done = opts.done;
  }

  handleInput(data: string): void {
    if (matchesKey(data, Key.tab) || matchesKey(data, Key.enter)) {
      // Accept current suggestion
      this.accepted.add(this.currentIndex);
      this.advance();
    } else if (matchesKey(data, Key.shift("tab"))) {
      // Skip current suggestion
      this.skipped.add(this.currentIndex);
      this.advance();
    } else if (matchesKey(data, Key.escape)) {
      // Abort all remaining
      this.aborted = true;
      this.done({
        accepted: [...this.accepted],
        skipped: [...this.skipped],
        aborted: true,
      });
    } else if (matchesKey(data, Key.up)) {
      // Go back
      if (this.currentIndex > 0) {
        this.currentIndex--;
        this.invalidate();
        this.tui.requestRender();
      }
    } else if (matchesKey(data, Key.down)) {
      // Go forward (without accepting)
      if (this.currentIndex < this.edits.length - 1) {
        this.currentIndex++;
        this.invalidate();
        this.tui.requestRender();
      }
    }
  }

  private advance(): void {
    this.currentIndex++;
    if (this.currentIndex >= this.edits.length) {
      // All done
      this.done({
        accepted: [...this.accepted],
        skipped: [...this.skipped],
        aborted: false,
      });
    } else {
      this.invalidate();
      this.tui.requestRender();
    }
  }

  invalidate(): void {
    // Clear cached render state
  }

  render(width: number): string[] {
    const lines: string[] = [];
    const total = this.edits.length;
    const fg = this.theme.fg;
    const bg = this.theme.bg;

    if (total === 0) {
      return [fg("muted", "(no suggestions)")];
    }

    // ── Top border ──
    lines.push(fg("accent", "─".repeat(width)));

    // ── Header ──
    const headerText = ` Suggested Edits   ${this.currentIndex + 1} of ${total}`;
    lines.push(fg("accent", headerText));

    // ── Separator ──
    lines.push(fg("border", "─".repeat(width)));

    // ── Spacer ──
    lines.push("");

    // ── Current suggestion (full view) ──
    const edit = this.edits[this.currentIndex];
    const loc = edit.file + (edit.line ? `:${edit.line}` : "");
    lines.push(fg("accent", " " + loc));
    if (edit.label) {
      lines.push(fg("text", ` ${edit.label}`));
    }
    if (edit.reason) {
      lines.push(fg("muted", `  ${edit.reason}`));
    }

    // Spacer
    lines.push("");

    // Diff view — old text
    if (edit.oldText) {
      const oldLines = edit.oldText.split("\\n");
      for (const ol of oldLines) {
        lines.push(`  ${fg("error", "─ " + ol)}`);
      }
    }

    // Diff view — new text
    if (edit.newText) {
      const newLines = edit.newText.split("\\n");
      for (const nl of newLines) {
        lines.push(`  ${fg("success", "┌ " + nl)}`);
      }
    }

    // ── Spacer ──
    lines.push("");

    // ── Progress panel ──
    const panelWidth = Math.min(60, width - 4);
    const panelPad = "  ";

    // Accepted section
    lines.push(panelPad + fg("border", "├─ Accepted " + "─".repeat(Math.max(0, panelWidth - 8 - 2)) + "┤"));
    if (this.accepted.size === 0) {
      lines.push(panelPad + fg("muted", "  (none yet)"));
    } else {
      for (const idx of this.accepted) {
        const e = this.edits[idx];
        const label = e.label || e.oldText.split("\\n")[0] || "";
        lines.push(panelPad + fg("success", ` ✓ [${idx + 1}] ${truncateToWidth(label, panelWidth - 10)}`));
      }
    }

    // Remaining section
    const remaining = total - this.accepted.size - this.skipped.size;
    if (remaining > 1 || this.skipped.size > 0) {
      lines.push(panelPad + fg("border", "├─ Remaining " + "─".repeat(Math.max(0, panelWidth - 3 - 2)) + "┤"));
      for (let i = this.currentIndex; i < total; i++) {
        if (this.accepted.has(i) || this.skipped.has(i)) continue;
        const e = this.edits[i];
        const label = e.label || e.oldText.split("\\n")[0] || "";
        const marker = i === this.currentIndex ? fg("accent", "→") : " ";
        lines.push(panelPad + `${marker}  [${i + 1}] ${truncateToWidth(label, panelWidth - 10)}`);
      }
    }

    // ── Footer ──
    lines.push("");
    lines.push(fg("muted", `  [Tab=${fg("success", "Accept")}${fg("muted", "")}] [S-Tab=${fg("warning", "Skip")}${fg("muted", "")}] [Esc=${fg("error", "Abort")}${fg("muted", "")}] [↑↓=${fg("accent", "Nav")}${fg("muted", "")}]`));
    lines.push(fg("accent", "─".repeat(width)));

    return lines;
  }
}

*/

console.log("Reference TUI extension — see source for the SuggestedEditsOverlay component pattern.");
console.log("To register as a pi command, import and use pi.registerCommand('suggested', ...).");
