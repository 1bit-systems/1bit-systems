/**
 * Agent View — Multi-Agent Dashboard for pi
 *
 * Inspired by Claude Code's agent view and open-source alternatives (Chloe, AoE, OpenRig).
 *
 * Key design rules:
 *   - ctx is NEVER captured in async callbacks (prevents stale-context crashes)
 *   - TUI mode is stored as a simple boolean flag
 *   - All LLM-facing tools are safe in all modes (TUI, JSON, print, RPC)
 *   - State persists in .pi/agent-view/state.json
 *
 * Commands:
 *   /av              Open agent view panel
 *   /av spawn <task> Spawn a new background agent
 *   /av list         List all agents
 *   /av close <id>   Stop an agent
 *   /av output <id>  Show agent output
 *
 * LLM Tools:
 *   spawn_agent      Spawn a background agent teammate
 *   list_agents      List all agents and their status
 *   get_agent_output Get results from a specific agent
 */

import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI, ToolResult } from "@1bit/1bit-coding-agent";
import { CONFIG_DIR_NAME } from "@1bit/1bit-coding-agent";
import { Type } from "typebox";

// ─── Types ────────────────────────────────────────────────────────────────

type AgentStatus = "spawning" | "running" | "idle" | "done" | "error" | "stopped";

interface StoredAgent {
	id: number;
	name: string;
	status: AgentStatus;
	model: string;
	createdAt: number;
	lastActiveAt: number;
	exitCode: number | null;
	parentId?: number;
	usage: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		cost: number;
		turns: number;
	};
	outputFile: string;
}

interface LiveAgent extends StoredAgent {
	task: string;
	errorOutput: string;
	process: ChildProcess | null;
}

// ─── Constants ────────────────────────────────────────────────────────────

const MAX_CONCURRENT = 4;

// ─── State Manager ────────────────────────────────────────────────────────

class AgentStore {
	private agents = new Map<number, LiveAgent>();
	private nextId = 1;
	private stateDir = "";
	private onDirty: (() => void) | null = null;

	init(cwd: string) {
		this.stateDir = path.join(cwd, CONFIG_DIR_NAME, "agent-view");
		fs.mkdirSync(this.stateDir, { recursive: true });
	}

	onChange(fn: () => void) {
		this.onDirty = fn;
	}

	private dirty() {
		this.persist();
		this.onDirty?.();
	}

	// ─── Persistence ──────────────────────────────────────────────────

	async restore(cwd: string) {
		this.init(cwd);
		const statePath = path.join(this.stateDir, "state.json");
		if (!fs.existsSync(statePath)) return;
		try {
			const raw = fs.readFileSync(statePath, "utf-8");
			const saved: { agents: StoredAgent[]; nextId: number } = JSON.parse(raw);
			this.nextId = saved.nextId || 1;
			for (const a of saved.agents) {
				const output = this.readOutput(a.outputFile);
				this.agents.set(a.id, {
					...a,
					task: "(restored)",
					errorOutput: "",
					process: null,
				});
			}
		} catch {
			/* ignore corrupt state */
		}
	}

	private outputPath(id: number): string {
		return path.join(this.stateDir, `agent-${id}.jsonl`);
	}

	private readOutput(filePath: string): string {
		try {
			return fs.readFileSync(filePath, "utf-8");
		} catch {
			return "";
		}
	}

	private persist() {
		const statePath = path.join(this.stateDir, "state.json");
		const stored: StoredAgent[] = [];
		for (const a of this.agents.values()) {
			if (a.status === "stopped" && a.exitCode === 0) continue; // prune stopped
			if (a.status === "done") continue; // prune completed (auto-clean)
			stored.push(this.toStored(a));
		}
		fs.writeFileSync(statePath, JSON.stringify({ agents: stored, nextId: this.nextId }, null, 2));
	}

	private toStored(a: LiveAgent): StoredAgent {
		return {
			id: a.id,
			name: a.name,
			status: a.status,
			model: a.model,
			createdAt: a.createdAt,
			lastActiveAt: a.lastActiveAt,
			exitCode: a.exitCode,
			parentId: a.parentId,
			usage: { ...a.usage },
			outputFile: this.outputPath(a.id),
		};
	}

	// ─── Accessors ───────────────────────────────────────────────────

	getAll(): LiveAgent[] {
		return Array.from(this.agents.values()).sort((a, b) => a.id - b.id);
	}

	get(id: number): LiveAgent | undefined {
		return this.agents.get(id);
	}

	getActiveCount(): number {
		let n = 0;
		for (const a of this.agents.values()) {
			if (a.status === "running" || a.status === "spawning") n++;
		}
		return n;
	}

	// ─── Spawn ──────────────────────────────────────────────────────

	spawn(name: string, task: string, model?: string): LiveAgent {
		const id = this.nextId++;
		const agent: LiveAgent = {
			id,
			name,
			status: "spawning",
			model: model || "",
			createdAt: Date.now(),
			lastActiveAt: Date.now(),
			task,
			errorOutput: "",
			exitCode: null,
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 0 },
			process: null,
			outputFile: this.outputPath(id),
		};
		this.agents.set(id, agent);
		this.dirty();
		this.launchProcess(id);
		return agent;
	}

	// ─── Subprocess ────────────────────────────────────────────────

	private getPiInvocation(args: string[]): { command: string; args: string[] } {
		const script = process.argv[1];
		const isBun = script?.startsWith("/$bunfs/root/");
		if (script && !isBun && fs.existsSync(script)) {
			return { command: process.execPath, args: [script, ...args] };
		}
		return { command: "pi", args };
	}

	private launchProcess(id: number) {
		const agent = this.agents.get(id);
		if (!agent) return;

		const args: string[] = ["--mode", "json", "-p", "--no-session"];
		if (agent.model) args.push("--model", agent.model);

		// Append system prompt shortcuts for status reporting
		const task = agent.task;
		args.push(task);

		const inv = this.getPiInvocation(args);
		const proc = spawn(inv.command, inv.args, {
			cwd: process.cwd(),
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
		});

		agent.process = proc;
		agent.status = "running";
		this.dirty();

		let buf = "";
		const outStream = fs.createWriteStream(this.outputPath(id), { flags: "a" });

		const handleLine = (line: string) => {
			if (!line.trim()) return;
			try {
				const ev = JSON.parse(line);
				if (ev.type === "message_end" && ev.message) {
					const msg = ev.message;
					if (msg.role === "assistant") {
						agent.usage.turns++;
						const u = msg.usage;
						if (u) {
							agent.usage.input += u.input || 0;
							agent.usage.output += u.output || 0;
							agent.usage.cacheRead += u.cacheRead || 0;
							agent.usage.cacheWrite += u.cacheWrite || 0;
							agent.usage.cost += u.cost?.total || 0;
						}
						if (!agent.model && msg.model) agent.model = msg.model;
					}
					agent.lastActiveAt = Date.now();
					outStream.write(JSON.stringify(msg) + "\n");
				}
			} catch {
				/* skip partial lines */
			}
		};

		proc.stdout!.on("data", (data: Buffer) => {
			buf += data.toString();
			const lines = buf.split("\n");
			buf = lines.pop() || "";
			for (const line of lines) handleLine(line);
		});

		proc.stderr!.on("data", (data: Buffer) => {
			agent.errorOutput += data.toString();
		});

		proc.on("close", (code) => {
			if (buf.trim()) handleLine(buf);
			if (agent.status !== "stopped") {
				agent.status = code === 0 ? "done" : "error";
				agent.exitCode = code;
			}
			outStream.end();
			this.dirty();
		});

		proc.on("error", () => {
			if (agent.status !== "stopped") {
				agent.status = "error";
				agent.exitCode = 1;
			}
			this.dirty();
		});
	}

	// ─── Controls ──────────────────────────────────────────────────

	stop(id: number): boolean {
		const agent = this.agents.get(id);
		if (!agent) return false;
		if (agent.process && !agent.process.killed) {
			agent.process.kill("SIGTERM");
			setTimeout(() => {
				if (agent.process && !agent.process.killed) agent.process.kill("SIGKILL");
			}, 3000);
		}
		agent.status = "stopped";
		this.dirty();
		return true;
	}

	stopAll() {
		for (const a of this.agents.values()) this.stop(a.id);
	}

	// ─── Output helpers ────────────────────────────────────────────

	getFinalOutput(id: number): string {
		const agent = this.agents.get(id);
		if (!agent) return "(not found)";
		try {
			const raw = this.readOutput(this.outputPath(id));
			const lines = raw.trim().split("\n").filter(Boolean);
			for (let i = lines.length - 1; i >= 0; i--) {
				try {
					const msg = JSON.parse(lines[i]);
					if (msg.role === "assistant") {
						for (const part of msg.content || []) {
							if (part.type === "text" && part.text?.trim()) return part.text;
						}
					}
				} catch {
					/* skip */
				}
			}
		} catch {
			/* ignore */
		}
		return "(no output)";
	}
}

// ─── Format helpers ────────────────────────────────────────────────────

function fmtTokens(n: number): string {
	if (n < 1000) return String(n);
	if (n < 10_000) return `${(n / 1000).toFixed(1)}k`;
	return `${Math.round(n / 1000)}k`;
}

function statusIcon(s: AgentStatus): string {
	switch (s) {
		case "spawning": return "⏳";
		case "running": return "●";
		case "idle": return "○";
		case "done": return "✓";
		case "error": return "✗";
		case "stopped": return "⊘";
	}
}

// ─── Extension ────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	const store = new AgentStore();
	let isTui = false;

	// ─── Session lifecycle ──────────────────────────────────────────

	pi.on("session_start", async (_event, ctx) => {
		isTui = ctx.mode === "tui";
		await store.restore(ctx.cwd);
		store.onChange(() => {
			/* widget will be refreshed on next user interaction */
		});

		// Set up the widget once
		if (isTui) {
			renderWidget(ctx);
		}
	});

	pi.on("session_shutdown", async () => {
		isTui = false;
		// Don't stop agents - they persist
	});

	// ─── Widget ────────────────────────────────────────────────────

	function renderWidget(ctx: any) {
		if (!isTui) return;

		const active = store.getActiveCount();
		const allAgents = store.getAll();

		// Only show non-done agents in the widget (running, spawning, errored)
		const visibleAgents = allAgents.filter(a => a.status !== "done");

		if (visibleAgents.length === 0) {
			// Nothing to show — clear the widget entirely
			ctx.ui.setWidget("agent-view", undefined);
			return;
		}

		// Show only active/errored agents
		ctx.ui.setWidget("agent-view", (_tui: any, theme: any) => {
			const lines: string[] = [];

			// Header
			lines.push(
				theme.fg("accent", "Agent View") +
				theme.fg("dim", ` ${visibleAgents.length} agents · ${active} active`),
			);

			// Agent list
			for (const a of visibleAgents) {
				const icon = statusIcon(a.status);
				const name = theme.fg("text", a.name);
				const idLabel = theme.fg("dim", `[${a.id}]`);

				let right = "";
				if (a.status === "running" || a.status === "spawning") {
					right = theme.fg("dim", ` ${a.usage.turns > 0 ? `${a.usage.turns}t` : "..."}`);
				} else if (a.status === "error") {
					right = theme.fg("error", ` exit:${a.exitCode ?? "?"}`);
				}

				lines.push(` ${icon} ${idLabel} ${name}${right}`);
			}

			lines.push(theme.fg("dim", "  /av spawn|list|close|output · /av <id> for detail"));

			return {
				render: () => lines,
				invalidate: () => {},
			};
		});
	}

	// ─── Help text ─────────────────────────────────────────────────

	function formatHelp(): string {
		return `Agent View — Multi-Agent Dashboard

  COMMANDS:
    /av spawn <task>              Spawn a background agent
    /av list                      List all agents
    /av <id>                      Show agent detail
    /av output <id>               View agent output
    /av close <id>                Stop an agent
    /av help                      Show this help

  LLM TOOLS:
    spawn_agent      Spawn a background agent teammate
    list_agents      Check on all running agents
    get_agent_output Get results from a specific agent

  HOW IT WORKS:
    Each agent runs as a separate pi subprocess with
    its own context window — truly parallel execution.`;
	}

	// ─── Command: /av ──────────────────────────────────────────────

	pi.registerCommand("av", {
		description: "Agent View: multi-agent dashboard",
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/);
			const cmd = parts[0]?.toLowerCase() || "";

			// Numeric shortcut: /av 1 -> detail
			const num = parseInt(cmd);
			if (!isNaN(num) && cmd === String(num)) {
				const agent = store.get(num);
				if (agent) {
					await showDetail(ctx, num);
				} else {
					ctx.ui.notify(`Agent ${num} not found`, "error");
				}
				return;
			}

			switch (cmd) {
				case "spawn":
				case "new":
				case "create": {
					const name = parts[1] || `agent-${store.getAll().length + 1}`;
					const task = parts.slice(cmd === "new" ? 2 : 2).join(" ").trim() ||
						(await ctx.ui.input("Task:", ""));
					if (!task) return;
					const a = store.spawn(name, task);
					ctx.ui.notify(`✨ Spawned agent #${a.id}: ${a.name}`, "success");
					renderWidget(ctx);
					return;
				}

				case "list":
				case "ls": {
					const agents = store.getAll();
					if (agents.length === 0) {
						ctx.ui.notify("No agents. Use /av spawn <name> <task> to start one.", "info");
						return;
					}
					const lines = agents.map((a) => {
						return `${statusIcon(a.status)} [${a.id}] ${a.name} — ${a.task.slice(0, 80)}`;
					});
					ctx.ui.notify(`Agents:\n${lines.join("\n")}`, "info");
					return;
				}

				case "close":
				case "stop":
				case "kill":
				case "x": {
					const id = parseInt(parts[1]);
					if (isNaN(id)) {
						ctx.ui.notify("Usage: /av close <agent-id>", "error");
						return;
					}
					if (store.stop(id)) {
						ctx.ui.notify(`Stopped agent #${id}`, "info");
						renderWidget(ctx);
					} else {
						ctx.ui.notify(`Agent #${id} not found`, "error");
					}
					return;
				}

				case "output":
				case "out":
				case "show": {
					const id = parseInt(parts[1]);
					if (isNaN(id)) {
						ctx.ui.notify("Usage: /av output <agent-id>", "error");
						return;
					}
					await showDetail(ctx, id);
					return;
				}

				case "help":
				case "?":
				default: {
					ctx.ui.notify(formatHelp(), "info");
				}
			}
		},
	});

	// ─── Agent detail overlay ─────────────────────────────────────

	async function showDetail(ctx: any, id: number) {
		const agent = store.get(id);
		if (!agent) {
			ctx.ui.notify(`Agent ${id} not found`, "error");
			return;
		}

		if (ctx.mode === "tui" && ctx.hasUI) {
			await ctx.ui.custom<void>((_tui: any, theme: any, _kb: any, done: () => void) => {
				const { Container, Text, Spacer } = require("@1bit/1bit-tui");
				const container = new Container();

				const hdr = `${statusIcon(agent.status)} ${theme.fg("accent", agent.name)}` +
					theme.fg("dim", ` [${agent.id}] · ${agent.model || "default"}`);
				container.addChild(new Text(hdr, 1, 0));
				container.addChild(new Text(theme.fg("muted", `Task: ${agent.task}`), 1, 0));

				if (agent.status === "error") {
					container.addChild(new Text(theme.fg("error", `Failed (exit: ${agent.exitCode ?? "?"})`), 1, 0));
				} else if (agent.status === "done") {
					container.addChild(new Text(theme.fg("success", "✓ Completed"), 1, 0));
				} else if (agent.status === "running") {
					container.addChild(new Text(theme.fg("warning", "● Running..."), 1, 0));
				}

				if (agent.usage.turns > 0) {
					const u = agent.usage;
					container.addChild(new Text(
						theme.fg("dim", `Turns: ${u.turns} · Input: ${fmtTokens(u.input)} · Output: ${fmtTokens(u.output)} · Cost: $${u.cost.toFixed(4)}`),
						1, 0,
					));
				}

				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("muted", "─── Output ───"), 1, 0));

				const output = store.getFinalOutput(id);
				if (output.trim()) {
					container.addChild(new Text(theme.fg("toolOutput", output.slice(0, 2000)), 1, 0));
				} else {
					container.addChild(new Text(
						theme.fg("dim", agent.status === "running" ? "Waiting for output..." : "(no output)"),
						1, 0,
					));
				}

				container.addChild(new Spacer(1));
				container.addChild(new Text(theme.fg("dim", "press any key to close"), 1, 0));

				return {
					render: (w: number) => container.render(w),
					invalidate: () => container.invalidate(),
					handleInput: () => done(),
				};
			}, { overlay: true });
		} else {
			// Non-TUI mode: print to stdout
			const output = store.getFinalOutput(id);
			ctx.ui.notify(
				`Agent [${agent.id}] ${agent.name} — ${agent.status}\n\n` +
				`Task: ${agent.task}\n\n${output}`,
				"info",
			);
		}
	}

	// ─── LLM Tool: spawn_agent ────────────────────────────────────

	pi.registerTool({
		name: "spawn_agent",
		label: "Spawn Agent",
		description: [
			"Spawn a new background agent to work on a task in parallel.",
			"Each agent runs as a separate pi instance with its own context window.",
			"Agents work independently and simultaneously.",
			"Use this when the task benefits from parallel work.",
			"Check progress with list_agents and get_agent_output.",
		].join(" "),
		parameters: Type.Object({
			name: Type.String({ description: "Short descriptive name (e.g. 'security-review', 'api-tester')" }),
			task: Type.String({ description: "Full task description with context and expected output" }),
			model: Type.Optional(Type.String({ description: "Optional model override" })),
		}),
		async execute(_toolCallId, params, _signal, onUpdate, ctx) {
			const name = params.name || `agent-${store.getAll().length + 1}`;
			const agent = store.spawn(name, params.task, params.model);

			if (isTui) renderWidget(ctx);

			onUpdate?.({
				content: [{ type: "text", text: `⏳ Agent "${name}" (ID: ${agent.id}) spawned and working...` }],
				details: { agentId: agent.id, status: "running", agentName: name },
			});

			return {
				content: [{
					type: "text",
					text: [
						`✅ Agent #${agent.id} "${name}" is working on: ${params.task.slice(0, 200)}${params.task.length > 200 ? "..." : ""}`,
						"",
						`Use /av output ${agent.id} to view progress.`,
						`Use /av close ${agent.id} to terminate early.`,
						"The agent runs in its own context window in the background.",
					].join("\n"),
				}],
				details: { agentId: agent.id, agentName: name, status: "running", task: params.task },
			};
		},
	});

	// ─── LLM Tool: list_agents ─────────────────────────────────────

	pi.registerTool({
		name: "list_agents",
		label: "List Agents",
		description: "List all background agents, their status, and progress. Use to check on spawned agents.",
		parameters: Type.Object({}),
		async execute() {
			const agents = store.getAll();
			if (agents.length === 0) {
				return { content: [{ type: "text", text: "No agents running." }], details: { agents: [] } };
			}
			const summary = agents.map((a) => {
				const t = a.usage.turns > 0 ? ` ${a.usage.turns}t` : "";
				const c = a.usage.cost > 0 ? ` $${a.usage.cost.toFixed(4)}` : "";
				return `[${a.id}] ${a.name}: ${a.status}${t}${c} — ${a.task.slice(0, 80)}`;
			});
			return {
				content: [{ type: "text", text: `**${agents.length} agent(s):**\n\n${summary.join("\n")}` }],
				details: { agents: agents.map((a) => ({ id: a.id, name: a.name, status: a.status, task: a.task })) },
			};
		},
	});

	// ─── LLM Tool: get_agent_output ────────────────────────────────

	pi.registerTool({
		name: "get_agent_output",
		label: "Get Agent Output",
		description: "Get the full output and results from a specific background agent.",
		parameters: Type.Object({
			agentId: Type.Number({ description: "The ID of the agent to get output from" }),
		}),
		async execute(_toolCallId, params) {
			const agent = store.get(params.agentId);
			if (!agent) {
				return {
					content: [{ type: "text", text: `Agent #${params.agentId} not found.` }],
					details: { error: "not_found" },
					isError: true,
				};
			}
			const output = store.getFinalOutput(params.agentId);
			return {
				content: [{
					type: "text",
					text: [
						`**Agent [${agent.id}] ${agent.name}** — ${agent.status}`,
						agent.status === "running" ? "(still running, showing partial output)" : "",
						`Task: ${agent.task}`,
						"",
						`**Output:**`,
						output,
						"",
						agent.usage.turns > 0
							? `Turns: ${agent.usage.turns} | Cost: $${agent.usage.cost.toFixed(4)} | Model: ${agent.model || "default"}`
							: "",
					].filter(Boolean).join("\n"),
				}],
				details: {
					agentId: agent.id,
					agentName: agent.name,
					status: agent.status,
					task: agent.task,
					output,
					usage: agent.usage,
				},
			};
		},
	});
}
