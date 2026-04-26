// MCP tool registry + invocation, brokered through halo-agent (which
// already speaks MCP server-side). The agent exposes a flat
// /agent/tools list and an /agent/invoke RPC. Both are reachable on
// the same lemonade :8180 host (halo-agent runs in-process today).

const AGENT_BASE = "http://127.0.0.1:8180/agent";

export interface MCPToolArg {
  readonly name: string;
  readonly type: "string" | "number" | "boolean";
  readonly required: boolean;
  readonly description?: string;
}

export interface MCPTool {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly args: ReadonlyArray<MCPToolArg>;
  readonly source: "builtin" | "external";
}

export interface MCPInvokeResult {
  readonly ok: boolean;
  readonly output: string;
  readonly error?: string;
}

export async function listTools(signal?: AbortSignal): Promise<MCPTool[]> {
  const r = await fetch(`${AGENT_BASE}/tools`, { signal });
  if (!r.ok) throw new Error(`/agent/tools ${r.status}`);
  const data = (await r.json()) as { tools: MCPTool[] };
  return data.tools;
}

export async function invokeTool(
  id: string,
  args: Readonly<Record<string, string | number | boolean>>,
  signal?: AbortSignal,
): Promise<MCPInvokeResult> {
  const r = await fetch(`${AGENT_BASE}/invoke`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool: id, args }),
    signal,
  });
  if (!r.ok) throw new Error(`/agent/invoke ${r.status}`);
  return (await r.json()) as MCPInvokeResult;
}
