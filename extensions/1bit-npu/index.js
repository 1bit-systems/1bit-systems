/**
 * 1bit NPU Extension — pi-compatible extension with NPU tools.
 *
 * Provides tools for managing the NPU inference stack from within
 * the pi/1bit agent: status checks, start/stop, model queries.
 */

/** @param {import('@earendil-works/pi-agent-core').ExtensionApi} pi */
export default function activate(pi) {
  pi.registerTool({
    name: "npu_status",
    description: "Check if the NPU inference stack is running (ports 9090, 13305)",
    parameters: {
      type: "object",
      properties: {},
    },
    execute: async () => {
      const results = [];
      const ports = [
        { port: 9090, name: "1bit API Bridge" },
        { port: 13305, name: "Lemond (Chat UI)" },
      ];

      for (const { port, name } of ports) {
        try {
          const { execSync } = await import("child_process");
          execSync(`ss -tlnp | grep -q ":${port} "`, { stdio: "ignore" });
          results.push(`${name} (port ${port}): \u2705 running`);
        } catch {
          results.push(`${name} (port ${port}): \u274c not running`);
        }
      }

      // Check API
      try {
        const res = await fetch("http://127.0.0.1:9090/v1/models", {
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          const data = await res.json();
          const count = data?.data?.length ?? 0;
          results.push(`NPU API: \u2705 responding (${count} models)`);
        } else {
          results.push(`NPU API: \u26a0\ufe0f error ${res.status}`);
        }
      } catch {
        results.push("NPU API: \u274c not reachable");
      }

      return results.join("\n");
    },
  });

  pi.registerTool({
    name: "npu_up",
    description: "Start the 1bit NPU inference stack (API bridge + Lemond)",
    parameters: {
      type: "object",
      properties: {},
    },
    execute: async () => {
      const { execSync, spawn } = await import("child_process");
      const { existsSync } = await import("fs");
      const { resolve } = await import("path");

      const HOME = process.env.HOME || "/home/bcloud";
      const started = [];

      // API bridge
      const bridgeScript = resolve(HOME, "npu-sandbox/npu-infer/1bit/dist/server.cjs");
      if (existsSync(bridgeScript)) {
        try {
          execSync(`ss -tlnp | grep -q ":9090 "`, { stdio: "ignore" });
          started.push("API bridge already running");
        } catch {
          const bridge = spawn("node", [bridgeScript], {
            stdio: "ignore", detached: true,
            env: { ...process.env, PORT: "9090" },
          });
          bridge.unref();
          started.push("API bridge started on port 9090");
        }
      } else {
        started.push("API bridge script not found at " + bridgeScript);
      }

      // Lemond
      try {
        execSync("which lemond", { stdio: "ignore" });
        try {
          execSync(`ss -tlnp | grep -q ":13305 "`, { stdio: "ignore" });
          started.push("Lemond already running");
        } catch {
          const lemond = spawn("lemond", ["--port", "13305"], {
            stdio: "ignore", detached: true,
          });
          lemond.unref();
          started.push("Lemond started on port 13305");
        }
      } catch {
        started.push("lemond not found in PATH");
      }

      return started.join("\n");
    },
  });

  pi.registerTool({
    name: "npu_down",
    description: "Stop the 1bit NPU inference stack",
    parameters: {
      type: "object",
      properties: {},
    },
    execute: async () => {
      const { execSync } = await import("child_process");
      const ports = [13305, 9090, 9000];
      for (const port of ports) {
        try {
          execSync(`sudo fuser -k ${port}/tcp 2>/dev/null`, { stdio: "ignore" });
          execSync(`fuser -k ${port}/tcp 2>/dev/null`, { stdio: "ignore" });
        } catch {
          // ok
        }
      }
      for (const proc of ["lemond", "node.*server\\.cjs"]) {
        try {
          execSync(`pkill -f "${proc}" 2>/dev/null`, { stdio: "ignore" });
        } catch {
          // ok
        }
      }
      return "NPU stack stopped";
    },
  });

  pi.registerTool({
    name: "npu_query",
    description: "Send a prompt to the NPU inference endpoint and get a response",
    parameters: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "The prompt to send" },
        model: {
          type: "string",
          description: "Model name (default: qwen3-0.6b-FLM)",
          default: "qwen3-0.6b-FLM",
        },
      },
      required: ["prompt"],
    },
    execute: async ({ prompt, model = "qwen3-0.6b-FLM" }) => {
      const res = await fetch("http://127.0.0.1:9090/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          stream: false,
        }),
        signal: AbortSignal.timeout(120000),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`NPU API error (${res.status}): ${text}`);
      }

      const data = await res.json();
      return data?.choices?.[0]?.message?.content || "[no response]";
    },
  });
}
