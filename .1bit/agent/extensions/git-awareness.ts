/**
 * git-awareness.ts — Cross-Agent Change Awareness Extension
 *
 * Makes every agent aware of what other agents have committed/pushed.
 *
 * Features:
 *   1. Session-start notification — shows changes since this agent last ran
 *   2. `check_codebase_changes` custom tool — agents call to inspect recent changes
 *   3. `/awareness` command — manual check with widget display
 *   4. Real-time awareness via trigger file (fs.watch for running sessions)
 *   5. Tracks agent identity from git config for accurate attribution
 *
 * Awareness data stored in: ~/.1bit/agent/awareness.json
 * Trigger file: /tmp/1bit-agent-awareness-trigger.txt
 */

import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI } from "@1bit/1bit-coding-agent";
import { Type } from "typebox";

// ── Constants ──────────────────────────────────────────────────────────────

const AWARENESS_FILE = path.join(
	process.env.HOME || "/root",
	".1bit",
	"agent",
	"awareness.json",
);

const TRIGGER_FILE = "/tmp/1bit-agent-awareness-trigger.txt";

// ── Types ──────────────────────────────────────────────────────────────────

interface AwarenessEvent {
	id: number;
	timestamp: string;
	type: "commit" | "checkout" | "merge";
	agent: string;
	branch: string;
	commit: string;
	message: string;
	files?: string[];
	oldBranch?: string;
}

interface AwarenessData {
	events: AwarenessEvent[];
	lastSeen: Record<string, string>;
	agents: Record<string, { lastCommit: string; lastSeen: string; branch: string }>;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function readAwareness(): AwarenessData | null {
	try {
		const raw = fs.readFileSync(AWARENESS_FILE, "utf-8");
		return JSON.parse(raw) as AwarenessData;
	} catch {
		return null;
	}
}

/** Return the git user name (lowercased) for this repo */
function getGitUser(): string {
	try {
		const envName =
			process.env.GIT_AUTHOR_NAME || process.env.GIT_COMMITTER_NAME || "";
		if (envName) return envName.toLowerCase();
		const output = execSync("git config user.name", {
			encoding: "utf-8",
			cwd: process.cwd(),
		});
		return (output?.trim() || "unknown").toLowerCase();
	} catch {
		return "unknown";
	}
}

/** Get the current HEAD commit hash */
function getCurrentHead(): string | null {
	try {
		const output = execSync("git rev-parse HEAD", {
			encoding: "utf-8",
			cwd: process.cwd(),
		});
		return output?.trim() || null;
	} catch {
		return null;
	}
}

/** Get the current branch name */
function getCurrentBranch(): string | null {
	try {
		const output = execSync("git rev-parse --abbrev-ref HEAD", {
			encoding: "utf-8",
			cwd: process.cwd(),
		});
		return output?.trim() || null;
	} catch {
		return null;
	}
}

/**
 * Build a summary of changes from OTHER agents since this agent last checked.
 */
function generateChangeSummary(
	data: AwarenessData,
	agentName: string,
): { hasChanges: boolean; summary: string; newEvents: AwarenessEvent[] } {
	const lastSeenCommit = data.lastSeen?.[agentName];
	const headCommit = getCurrentHead();

	// Gather events from other agents, newest first
	let candidates = data.events.filter(
		(e) => e.type === "commit" && e.agent !== agentName,
	);

	if (lastSeenCommit) {
		// Collect events until we find the last-seen commit
		const sliced: AwarenessEvent[] = [];
		for (const evt of candidates) {
			if (evt.commit === lastSeenCommit) break;
			sliced.push(evt);
		}
		candidates = sliced.length > 0 ? sliced : candidates.slice(0, 5);
	} else {
		// Never seen anything — just show last 5
		candidates = candidates.slice(0, 5);
	}

	// Deduplicate by commit hash
	const seen = new Set<string>();
	const unique = candidates.filter((e) => {
		if (seen.has(e.commit)) return false;
		seen.add(e.commit);
		return true;
	});

	if (unique.length === 0) {
		return { hasChanges: false, summary: "No new changes found.", newEvents: [] };
	}

	const lines = unique.map(
		(e) =>
			`  • [${e.agent}] ${e.message} (${e.branch}, ${e.commit.slice(0, 7)})`,
	);

	return {
		hasChanges: true,
		summary: [
			`📢 ${unique.length} new change${unique.length > 1 ? "s" : ""} since you last worked:`,
			...lines,
		].join("\n"),
		newEvents: unique,
	};
}

/** Record that the given agent has seen the latest HEAD */
function recordLastSeen(agentName: string): void {
	try {
		const head = getCurrentHead();
		if (!head) return;

		const raw = fs.readFileSync(AWARENESS_FILE, "utf-8");
		const data = JSON.parse(raw) as AwarenessData;
		data.lastSeen = data.lastSeen || {};
		data.lastSeen[agentName] = head;
		fs.writeFileSync(AWARENESS_FILE, JSON.stringify(data, null, 2) + "\n");
	} catch {
		// Awareness file might not exist yet
	}
}

// ── Extension Export ───────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	// ── 1. Session-Start Notification ───────────────────────────────────────
	pi.on("session_start", () => {
		const data = readAwareness();
		const agentName = getGitUser();

		if (!data || data.events.length === 0) return;

		const { hasChanges, summary } = generateChangeSummary(data, agentName);

		if (hasChanges) {
			pi.sendMessage(
				{
					customType: "git-awareness",
					content: summary,
					display: true,
				},
				{ triggerTurn: false },
			);
		}

		// Mark the agent as having seen the latest state
		recordLastSeen(agentName);
	});

	// ── 2. Custom Tool: check_codebase_changes ────────────────────────────
	pi.registerTool({
		name: "check_codebase_changes",
		label: "Check Codebase Changes",
		description:
			"Check what files have been changed recently by other agents or contributors. " +
			"Use this before starting work to see if anything relevant has changed since you last ran.",
		parameters: Type.Object({
			agent: Type.Optional(
				Type.String({
					description:
						"Filter changes by agent name (e.g., 'vulkan', 'npu', 'worker')",
				}),
			),
			branch: Type.Optional(
				Type.String({
					description:
						"Filter changes by branch (e.g., 'main', 'zero-copy-dmabuf')",
				}),
			),
			count: Type.Optional(
				Type.Integer({
					description: "Number of recent changes to return (default: 10, max: 50)",
					minimum: 1,
					maximum: 50,
					default: 10,
				}),
			),
			sinceCommit: Type.Optional(
				Type.String({
					description:
						"Show changes since a specific commit hash. Leave empty for all recent changes.",
				}),
			),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			const data = readAwareness();
			if (!data || data.events.length === 0) {
				return {
					content: [
						{
							type: "text" as const,
							text: "No awareness data found. The awareness system may not be set up yet.",
						},
					],
					details: {},
				};
			}

			let events = [...data.events];

			// Filter by agent
			if (params.agent) {
				const agentFilter = (params.agent as string).toLowerCase();
				events = events.filter((e) =>
					e.agent.toLowerCase().includes(agentFilter),
				);
			}

			// Filter by branch
			if (params.branch) {
				events = events.filter((e) => e.branch === params.branch);
			}

			// Filter since commit
			if (params.sinceCommit) {
				let found = false;
				const filtered: AwarenessEvent[] = [];
				for (const evt of events) {
					if (evt.commit === params.sinceCommit) {
						found = true;
						break;
					}
					filtered.push(evt);
				}
				if (found) events = filtered;
			}

			// Limit
			const count = (params.count as number) || 10;
			events = events.slice(0, count);

			if (events.length === 0) {
				return {
					content: [
						{ type: "text" as const, text: "No matching changes found." },
					],
					details: {},
				};
			}

			const lines = events.map((e) => {
				const filesInfo = e.files?.length
					? `\n       Files: ${e.files.join(", ")}`
					: "";
				const commitShort = e.commit?.slice(0, 7) || "???????";
				return `  [${e.id}] ${e.timestamp.slice(0, 19)} | ${e.agent} | ${e.branch}\n       ${e.message}${filesInfo}  (${commitShort})`;
			});

			return {
				content: [
					{
						type: "text" as const,
						text: [
							`## Recent Codebase Changes (${events.length} event${events.length > 1 ? "s" : ""})`,
							"",
							...lines,
							"",
							"Use `git log --oneline -20` or `git diff HEAD~5..HEAD --stat` for more detail.",
						].join("\n"),
					},
				],
				details: {
					changeCount: events.length,
					totalEvents: data.events.length,
					newest: events[0]?.commit?.slice(0, 7) || "",
				},
			};
		},
	});

	// ── 3. Command: /awareness ─────────────────────────────────────────────
	pi.registerCommand("awareness", {
		description:
			"Show recent codebase changes from other agents. Use /awareness [count] for more entries.",
		handler: (args, ctx) => {
			const count = parseInt(args || "10", 10) || 10;
			const data = readAwareness();

			if (!data || data.events.length === 0) {
				ctx.ui.notify("No awareness data found.", "warn");
				return;
			}

			const events = data.events.slice(0, Math.min(count, 50));
			const lines = events.map((e) => {
				const commitShort = e.commit?.slice(0, 7) || "???????";
				const ts = e.timestamp.slice(0, 19).padEnd(20);
				const agent = e.agent.padEnd(12);
				const branch = e.branch.padEnd(20);
				return `${ts} ${agent} ${branch} ${commitShort}  ${e.message.slice(0, 55)}`;
			});

			ctx.ui.setWidget("git-awareness", [
				"Recent Codebase Changes:",
				...lines.slice(0, 8).map((l) => `  ${l}`),
			]);
			ctx.ui.notify(
				`${events.length} recent changes — see widget above`,
				"info",
			);
		},
	});

	// ── 4. Trigger File Watcher (Real-time Awareness) ─────────────────────
	// External processes (CI, git hooks) write to TRIGGER_FILE to push
	// messages into a running agent session.
	try {
		if (!fs.existsSync(TRIGGER_FILE)) {
			fs.writeFileSync(TRIGGER_FILE, "");
		}

		fs.watch(TRIGGER_FILE, () => {
			try {
				const content = fs.readFileSync(TRIGGER_FILE, "utf-8").trim();
				if (content) {
					pi.sendMessage(
						{
							customType: "git-awareness",
							content: `📢 Awareness: ${content}`,
							display: true,
						},
						{ triggerTurn: true },
					);
					// Clear the trigger immediately
					fs.writeFileSync(TRIGGER_FILE, "");
				}
			} catch {
				// File might be momentarily unavailable
			}
		});
	} catch {
		// fs.watch might not be available in all environments
	}

	// ── 5. Track Agent Activity ────────────────────────────────────────────
	pi.on("tool_call", (event) => {
		if (event.toolName === "write" || event.toolName === "edit") {
			try {
				const agentName = getGitUser();
				const head = getCurrentHead();
				if (!head) return;

				const raw = fs.readFileSync(AWARENESS_FILE, "utf-8");
				const data = JSON.parse(raw) as AwarenessData;
				data.lastSeen = data.lastSeen || {};
				data.lastSeen[agentName] = head;

				const branch = getCurrentBranch();
				if (branch && data.agents) {
					data.agents[agentName] = data.agents[agentName] || {
						lastCommit: "",
						lastSeen: "",
						branch: "",
					};
					data.agents[agentName].branch = branch;
					data.agents[agentName].lastSeen =
						new Date().toISOString().replace("T", " ").slice(0, 19) + "Z";
				}

				fs.writeFileSync(AWARENESS_FILE, JSON.stringify(data, null, 2) + "\n");
			} catch {
				// Awareness file might not exist yet
			}
		}
	});

	// ── 6. Status Indicator ────────────────────────────────────────────────
	pi.on("session_start", (_event, ctx) => {
		if (ctx.hasUI) {
			ctx.ui.setStatus("git-awareness", "awareness: active");
		}
	});
}
