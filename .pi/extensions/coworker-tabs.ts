/**
 * Coworker Tabs Extension — Multi-Agent Sessions in pi
 *
 * Like Claude Code Coworker: run multiple independent sessions in one terminal.
 * Each tab is a fully isolated session with its own context window.
 * All sessions auto-save to disk — close the terminal safely.
 *
 * Features:
 *   - Tab bar showing all coworkers above the editor
 *   - Quick switch between sessions (/cw <n>, ← → arrows, Alt+1..9)
 *   - Handoff: transfer context from one coworker to a new one
 *   - State survives restarts (stored in .pi/coworkers.json)
 *
 * Usage:
 *   /cw <n>                Switch to coworker N
 *   /cw new [name]         Start a new coworker
 *   /cw handoff [name] [goal]  Transfer context + create new coworker
 *   /cw close              Close current coworker
 *   /cw next|prev          Navigate (alias: n/p)
 *   /cw rename <name>      Rename current coworker
 *   /cw list               Show all coworkers
 *
 * Keyboard (same as Claude Code coworker):
 *   ← / →       - Previous/next coworker (when input empty)
 *   Alt+1..9    - Jump to coworker N
 *   Ctrl+n      - New coworker
 *   Ctrl+w      - Close coworker
 *   Alt+h       - Handoff (context transfer)
 *
 * State file: .pi/coworkers.json (per project)
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { complete, type Message } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionContext, SessionEntry } from "@earendil-works/pi-coding-agent";
import {
	BorderedLoader,
	CONFIG_DIR_NAME,
	convertToLlm,
	CustomEditor,
	serializeConversation,
} from "@earendil-works/pi-coding-agent";
import { matchesKey, Key } from "@earendil-works/pi-tui";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";

// ─── Coworker-aware editor ────────────────────────────────────────────
// Intercepts ←/→ when the input is empty to navigate coworkers
// (same behavior as Claude Code coworker)

class CoworkerEditor extends CustomEditor {
	public onCoworkerNav?: (direction: "prev" | "next") => void;

	constructor(
		tui: ConstructorParameters<typeof CustomEditor>[0],
		theme: ConstructorParameters<typeof CustomEditor>[1],
		keybindings: ConstructorParameters<typeof CustomEditor>[2],
	) {
		super(tui, theme, keybindings);

		// This runs BEFORE the editor's own key handling.
		// When the editor is empty, ←/→ navigate coworkers.
		// When the editor has text, ←/→ move the cursor normally.
		this.onExtensionShortcut = (data: string) => {
			if (this.getText().length === 0) {
				if (matchesKey(data, Key.left)) {
					this.onCoworkerNav?.("prev");
					return true; // consumed
				}
				if (matchesKey(data, Key.right)) {
					this.onCoworkerNav?.("next");
					return true; // consumed
				}
			}
			return false; // let editor handle it normally
		};
	}
}

// ─── Types ────────────────────────────────────────────────────────────────

interface Coworker {
	name: string;
	sessionFile: string;
	parentFile?: string;     // Handoff parent — which session this was forked from
	createdAt: number;
	lastActiveAt: number;
}

interface CoworkerState {
	coworkers: Coworker[];
	activeIndex: number;
}

// ─── Handoff — context transfer prompt ─────────────────────────────────

const HANDOFF_SYSTEM_PROMPT = `You are a context transfer assistant. Given a conversation history and the user's goal for a new thread, generate a focused prompt that:

1. Summarizes relevant context from the conversation (decisions made, approaches taken, key findings)
2. Lists any relevant files that were discussed or modified
3. Clearly states the next task based on the user's goal
4. Is self-contained - the new thread should be able to proceed without the old conversation

Format your response as a prompt the user can send to start the new thread. Be concise but include all necessary context. Do not include any preamble like "Here's the prompt" - just output the prompt itself.

Example output format:
## Context
We've been working on X. Key decisions:
- Decision 1
- Decision 2

Files involved:
- path/to/file1.ts
- path/to/file2.ts

## Task
[Clear description of what to do next based on user's goal]`;

function entryToMessage(entry: SessionEntry): AgentMessage | undefined {
	if (entry.type === "message") {
		return entry.message;
	}
	if (entry.type === "compaction") {
		return {
			role: "compactionSummary",
			summary: entry.summary,
			tokensBefore: entry.tokensBefore,
			timestamp: new Date(entry.timestamp).getTime(),
		};
	}
	return undefined;
}

function getHandoffMessages(branch: SessionEntry[]): AgentMessage[] {
	let compactionIndex = -1;
	for (let i = branch.length - 1; i >= 0; i--) {
		if (branch[i].type === "compaction") {
			compactionIndex = i;
			break;
		}
	}
	if (compactionIndex < 0) {
		return branch.map(entryToMessage).filter((m) => m !== undefined);
	}
	const compaction = branch[compactionIndex];
	const firstKeptIndex =
		compaction.type === "compaction"
			? branch.findIndex((entry) => entry.id === compaction.firstKeptEntryId)
			: -1;
	const compactedBranch = [
		compaction,
		...(firstKeptIndex >= 0 ? branch.slice(firstKeptIndex, compactionIndex) : []),
		...branch.slice(compactionIndex + 1),
	];
	return compactedBranch.map(entryToMessage).filter((m) => m !== undefined);
}

// ─── Extension ────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	let state: CoworkerState = { coworkers: [], activeIndex: 0 };
	let cwd = "";

	// ─── State persistence ─────────────────────────────────────────────

	function getStatePath(): string {
		return join(cwd, CONFIG_DIR_NAME, "coworkers.json");
	}

	function loadState() {
		try {
			const path = getStatePath();
			if (existsSync(path)) {
				state = JSON.parse(readFileSync(path, "utf-8"));
				state.coworkers = state.coworkers.filter(
					(c) => c.sessionFile && existsSync(c.sessionFile),
				);
				if (state.activeIndex >= state.coworkers.length) {
					state.activeIndex =
						state.coworkers.length > 0 ? state.coworkers.length - 1 : 0;
				}
			}
		} catch {
			// Fresh state
		}
	}

	function saveState() {
		try {
			const path = getStatePath();
			mkdirSync(dirname(path), { recursive: true });
			writeFileSync(path, JSON.stringify(state, null, 2));
		} catch {
			// Best effort
		}
	}

	// ─── Tab bar widget ─────────────────────────────────────────────────

	function renderTabBar(ctx: ExtensionContext) {
		if (ctx.mode !== "tui") return;

		ctx.ui.setWidget("coworker-tabs", (_tui, theme) => {
			const parts: string[] = [];
			const active = state.coworkers[state.activeIndex];

			for (let i = 0; i < state.coworkers.length; i++) {
				const c = state.coworkers[i];
				const isActive = i === state.activeIndex;
				const tabNum = theme.fg("dim", `${i + 1}`);
				const name = isActive
					? theme.fg("accent", theme.bold(c.name))
					: theme.fg("text", c.name);
				const body = `${tabNum}:${name}`;
				parts.push(isActive ? theme.fg("accent", `▸ ${body}`) : `  ${body}`);
			}

			parts.push(theme.fg("success", "+new"));

			// Parent indicator for the active coworker
			const parentLine =
				active?.parentFile
					? theme.fg("dim", `  ← handoff from ${findParentName(active.parentFile)}`)
					: "";

			return {
				render: () => [
					parts.join(` ${theme.fg("dim", "│")} `),
					`${theme.fg("dim", "  /cw <n> · ← → arrows · alt+1..9 jump · alt+n new · alt+w close · alt+h handoff")}${parentLine}`,
				],
				invalidate: () => {},
			};
		});
	}

	function findParentName(sessionFile: string): string {
		const c = state.coworkers.find((c) => c.sessionFile === sessionFile);
		return c ? c.name : "?";
	}

	// ─── Core switch logic ──────────────────────────────────────────────

	async function switchToCoworker(index: number, ctx: ExtensionContext) {
		if (index < 0 || index >= state.coworkers.length) return;
		if (index === state.activeIndex) return;
		const target = state.coworkers[index];
		state.activeIndex = index;
		saveState();
		const result = await ctx.switchSession(target.sessionFile, {
			withSession: async (replCtx) => {
				state.activeIndex = index;
				saveState();
				pi.setSessionName(target.name);
				renderTabBar(replCtx);
				replCtx.ui.notify(`Switched to: ${target.name}`, "info");
			},
		});
		if (result.cancelled) {
			ctx.ui.notify("Switch cancelled", "info");
		}
	}

	// ─── Events ─────────────────────────────────────────────────────────

	pi.on("session_start", async (_event, ctx) => {
		cwd = ctx.cwd;
		loadState();

		const currentFile = ctx.sessionManager.getSessionFile();
		if (!currentFile) {
			renderTabBar(ctx);
			return;
		}

		const existingIdx = state.coworkers.findIndex(
			(c) => c.sessionFile === currentFile,
		);
		if (existingIdx >= 0) {
			state.activeIndex = existingIdx;
			state.coworkers[existingIdx].lastActiveAt = Date.now();
		} else {
			const name =
				pi.getSessionName() || `cw${state.coworkers.length + 1}`;
			state.coworkers.push({
				name,
				sessionFile: currentFile,
				createdAt: Date.now(),
				lastActiveAt: Date.now(),
			});
			state.activeIndex = state.coworkers.length - 1;
		}

		const active = state.coworkers[state.activeIndex];
		if (active) pi.setSessionName(active.name);
		saveState();

		// Replace editor with cow worker-aware version
		// ←/→ navigate coworkers when editor is empty (Claude Code style)
		ctx.ui.setEditorComponent((tui, theme, kb) => {
			const editor = new CoworkerEditor(tui, theme, kb);
			editor.onCoworkerNav = (dir) => {
				const cur = state.activeIndex;
				const next =
					dir === "next"
						? (cur + 1) % state.coworkers.length
						: (cur - 1 + state.coworkers.length) % state.coworkers.length;
				if (next !== cur && next < state.coworkers.length) {
					pi.sendUserMessage(`/cw ${next + 1}`, {
						deliverAs: "steer",
						triggerTurn: true,
					});
				}
			};
			return editor;
		});

		renderTabBar(ctx);
	});

	pi.on("session_shutdown", async () => {
		const active = state.coworkers[state.activeIndex];
		if (active) active.lastActiveAt = Date.now();
		saveState();
	});

	// ─── Handoff logic ──────────────────────────────────────────────────

	async function doHandoff(
		goal: string,
		newName: string,
		ctx: ExtensionContext,
	): Promise<void> {
		if (ctx.mode !== "tui") {
			ctx.ui.notify("Handoff requires interactive mode", "error");
			return;
		}
		if (!ctx.model) {
			ctx.ui.notify("No model selected", "error");
			return;
		}

		// Gather conversation context
		const messages = getHandoffMessages(ctx.sessionManager.getBranch());
		if (messages.length === 0) {
			ctx.ui.notify("No conversation to hand off", "error");
			return;
		}

		const llmMessages = convertToLlm(messages);
		const conversationText = serializeConversation(llmMessages);
		const currentSessionFile = ctx.sessionManager.getSessionFile();

		// Generate handoff prompt with loader UI
		const result = await ctx.ui.custom<string | null>((tui, theme, _kb, done) => {
			const loader = new BorderedLoader(
				tui,
				theme,
				`Generating handoff prompt for "${newName}"...`,
			);
			loader.onAbort = () => done(null);

			const doGenerate = async () => {
				const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model!);
				if (!auth.ok || !auth.apiKey) {
					throw new Error(
						auth.ok ? `No API key for ${ctx.model!.provider}` : auth.error,
					);
				}
				const userMessage: Message = {
					role: "user",
					content: [
						{
							type: "text",
							text: `## Conversation History\n\n${conversationText}\n\n## User's Goal for New Thread\n\n${goal}`,
						},
					],
					timestamp: Date.now(),
				};
				const response = await complete(
					ctx.model!,
					{ systemPrompt: HANDOFF_SYSTEM_PROMPT, messages: [userMessage] },
					{
						apiKey: auth.apiKey,
						headers: auth.headers,
						env: auth.env,
						signal: loader.signal,
					},
				);
				if (response.stopReason === "aborted") return null;
				return response.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");
			};

			doGenerate()
				.then(done)
				.catch((err) => {
					console.error("Handoff generation failed:", err);
					done(null);
				});
			return loader;
		});

		if (result === null) {
			ctx.ui.notify("Cancelled", "info");
			return;
		}

		// Let user edit the generated prompt
		const editedPrompt = await ctx.ui.editor(
			`Edit handoff prompt for "${newName}"`,
			result,
		);
		if (editedPrompt === undefined) {
			ctx.ui.notify("Cancelled", "info");
			return;
		}

		// Add new coworker entry to state BEFORE switching
		const newSessionResult = await ctx.newSession({
			parentSession: currentSessionFile ?? undefined,
			withSession: async (replCtx) => {
				const sessionFile = replCtx.sessionManager.getSessionFile();
				if (sessionFile) {
					state.coworkers.push({
						name: newName,
						sessionFile,
						parentFile: currentSessionFile ?? undefined,
						createdAt: Date.now(),
						lastActiveAt: Date.now(),
					});
					state.activeIndex = state.coworkers.length - 1;
					saveState();
					pi.setSessionName(newName);
					renderTabBar(replCtx);
					replCtx.ui.setEditorText(editedPrompt);
					replCtx.ui.notify(
						`✨ Handoff to "${newName}" ready. Submit when ready.`,
						"success",
					);
				}
			},
		});

		if (newSessionResult.cancelled) {
			ctx.ui.notify("Handoff cancelled", "info");
		}
	}

	// ─── Commands ───────────────────────────────────────────────────────

	pi.registerCommand("cw", {
		description: "Coworker: multi-session tabs + handoff",
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/);
			const cmd = parts[0]?.toLowerCase();

			// ── Numeric switch: /cw 2 ────────────────
			const numArg = parseInt(cmd);
			if (!isNaN(numArg) && cmd === String(numArg)) {
				const idx = numArg - 1;
				if (idx >= 0 && idx < state.coworkers.length) {
					await switchToCoworker(idx, ctx);
				} else {
					ctx.ui.notify(
						`Invalid index. Use 1–${state.coworkers.length}`,
						"error",
					);
				}
				return;
			}

			switch (cmd) {
				// ── /cw new [name] ────────────────────
				case "new": {
					const name =
						parts.slice(1).join(" ").trim() ||
						`cw${state.coworkers.length + 1}`;
					const result = await ctx.newSession({
						withSession: async (replCtx) => {
							const sessionFile =
								replCtx.sessionManager.getSessionFile();
							if (sessionFile) {
								state.coworkers.push({
									name,
									sessionFile,
									createdAt: Date.now(),
									lastActiveAt: Date.now(),
								});
								state.activeIndex = state.coworkers.length - 1;
								saveState();
								pi.setSessionName(name);
								renderTabBar(replCtx);
								replCtx.ui.notify(
									`✨ New coworker: ${name}`,
									"success",
								);
							}
						},
					});
					if (result.cancelled) {
						ctx.ui.notify("Cancelled", "info");
					}
					break;
				}

				// ── /cw handoff [name] [goal...] ─────
				case "handoff":
				case "h": {
					// Parse: /cw handoff "name" goal text
					// or /cw handoff goal text (auto-name)
					const rest = parts.slice(1).join(" ").trim();
					if (!rest) {
						ctx.ui.notify(
							"Usage: /cw handoff [name] <goal for new thread>",
							"error",
						);
						return;
					}

					// Try to extract a name from the beginning (if quoted or first word is short)
					let handoffName: string;
					let handoffGoal: string;

					// Check for quoted name: /cw handoff "my task" do X
					const quotedMatch = rest.match(/^"([^"]+)"\s+(.*)$/);
					if (quotedMatch) {
						handoffName = quotedMatch[1];
						handoffGoal = quotedMatch[2];
					} else {
						// Auto-generate name from goal
						const goalPreview = rest.length > 20 ? rest.slice(0, 20) + "…" : rest;
						handoffName = `hf-${state.coworkers.length + 1}-${goalPreview.replace(/\s+/g, "-").toLowerCase()}`;
						handoffGoal = rest;
					}

					await doHandoff(handoffGoal, handoffName, ctx);
					break;
				}

				// ── /cw close ─────────────────────────
				case "close":
				case "x": {
					if (state.coworkers.length <= 1) {
						ctx.ui.notify(
							"Cannot close the last coworker",
							"warning",
						);
						return;
					}
					const closingIdx = state.activeIndex;
					const closingName = state.coworkers[closingIdx].name;
					state.coworkers.splice(closingIdx, 1);
					const newIdx = Math.min(
						closingIdx,
						state.coworkers.length - 1,
					);
					state.activeIndex = newIdx;
					saveState();
					const target = state.coworkers[newIdx];
					const result = await ctx.switchSession(target.sessionFile, {
						withSession: async (replCtx) => {
							state.activeIndex = newIdx;
							saveState();
							pi.setSessionName(target.name);
							renderTabBar(replCtx);
							replCtx.ui.notify(
								`Closed: ${closingName} · Now on: ${target.name}`,
								"info",
							);
						},
					});
					if (result.cancelled) ctx.ui.notify("Cancelled", "info");
					break;
				}

				// ── /cw next / /cw prev ───────────────
				case "next":
				case "n": {
					const next =
						(state.activeIndex + 1) % state.coworkers.length;
					await switchToCoworker(next, ctx);
					break;
				}
				case "prev":
				case "p": {
					const prev =
						(state.activeIndex - 1 + state.coworkers.length) %
						state.coworkers.length;
					await switchToCoworker(prev, ctx);
					break;
				}

				// ── /cw rename ────────────────────────
				case "rename":
				case "name": {
					const name = parts.slice(1).join(" ").trim();
					if (!name) {
						ctx.ui.notify(
							"Usage: /cw rename <new name>",
							"error",
						);
						return;
					}
					const active = state.coworkers[state.activeIndex];
					if (active) {
						const oldName = active.name;
						active.name = name;
						pi.setSessionName(name);
						saveState();
						renderTabBar(ctx);
						ctx.ui.notify(`Renamed: ${oldName} → ${name}`, "info");
					}
					break;
				}

				// ── /cw list ──────────────────────────
				case "list":
				case "ls": {
					const lines = state.coworkers
						.map((c, i) => {
							const marker =
								i === state.activeIndex ? "← " : "  ";
							const parentInfo = c.parentFile
								? ` (from ${findParentName(c.parentFile)})`
								: "";
							return `${marker}${i + 1}. ${c.name}${parentInfo}`;
						})
						.join("\n");
					ctx.ui.notify(`Coworkers:\n${lines}`, "info");
					break;
				}

				default: {
					ctx.ui.notify(
						`Coworker — multi-session tabs with handoff

  /cw <n>                 Switch to coworker N
  /cw new [name]          Start a new coworker
  /cw handoff [name] <goal>  Context transfer to new coworker
  /cw close               Close current coworker
  /cw next|prev           Navigate (aliases: n/p)
  /cw rename <name>       Rename current coworker
  /cw list                Show all coworkers

  ← → arrows · alt+1..9 · alt+n new · alt+w close · alt+h handoff`,
						"info",
					);
				}
			}
		},
	});

	// `/coworker` alias
	pi.registerCommand("coworker", {
		description: "Alias for /cw — multi-session tabs",
		handler: async (args, ctx) => {
			pi.sendUserMessage(`/cw ${args.trim()}`, {
				deliverAs: "steer",
				triggerTurn: true,
			});
		},
	});

	// ─── Keyboard shortcuts ────────────────────────────────────────────

	for (let i = 1; i <= 9; i++) {
		const idx = i - 1;
		pi.registerShortcut(`alt+${i}`, {
			description: `Switch to coworker ${i}`,
			handler: async (_ctx) => {
				if (idx < state.coworkers.length && idx !== state.activeIndex) {
					pi.sendUserMessage(`/cw ${i}`, {
						deliverAs: "steer",
						triggerTurn: true,
					});
				}
			},
		});
	}

	pi.registerShortcut("alt+n", {
		description: "New coworker",
		handler: async (_ctx) => {
			pi.sendUserMessage("/cw new", {
				deliverAs: "steer",
				triggerTurn: true,
			});
		},
	});

	pi.registerShortcut("alt+w", {
		description: "Close current coworker",
		handler: async (_ctx) => {
			pi.sendUserMessage("/cw close", {
				deliverAs: "steer",
				triggerTurn: true,
			});
		},
	});

	pi.registerShortcut("alt+h", {
		description: "Handoff — context transfer to new coworker",
		handler: async (_ctx) => {
			pi.sendUserMessage("/cw handoff", {
				deliverAs: "steer",
				triggerTurn: true,
			});
		},
	});
}
