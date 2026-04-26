// lemonade :8180 wrappers. All calls hit 127.0.0.1:8180 (gated by the
// Tauri allowlist). No remote endpoints, no telemetry.

const LEMONADE_BASE = "http://127.0.0.1:8180";

export interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

export interface SystemStats {
  readonly tokens_per_sec: number;
  readonly prompt_tokens_per_sec: number;
  readonly gpu_mem_mb: number;
  readonly system_ram_mb: number;
  readonly kv_cache_mb: number;
}

export interface HealthStatus {
  readonly ok: boolean;
  readonly model: string;
  readonly recipe: string;
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  const r = await fetch(`${LEMONADE_BASE}/health`, { signal });
  if (!r.ok) throw new Error(`/health ${r.status}`);
  return (await r.json()) as HealthStatus;
}

export async function fetchSystemStats(
  signal?: AbortSignal,
): Promise<SystemStats> {
  const r = await fetch(`${LEMONADE_BASE}/system-stats`, { signal });
  if (!r.ok) throw new Error(`/system-stats ${r.status}`);
  return (await r.json()) as SystemStats;
}

// SSE stream over /v1/chat/completions. Yields incremental content
// deltas; caller is responsible for joining them into the final
// assistant turn. Aborts cleanly on `signal`.
export async function* streamChat(
  messages: readonly ChatMessage[],
  signal: AbortSignal,
): AsyncGenerator<string, void, void> {
  const body = JSON.stringify({
    model: "1bit-systems/halo-1bit-2b",
    messages,
    stream: true,
  });
  const r = await fetch(`${LEMONADE_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`/v1/chat/completions ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        const parsed = JSON.parse(payload) as {
          choices?: ReadonlyArray<{ delta?: { content?: string } }>;
        };
        const delta = parsed.choices?.[0]?.delta?.content;
        if (delta) yield delta;
      } catch {
        // ignore malformed SSE chunk
      }
    }
  }
}
