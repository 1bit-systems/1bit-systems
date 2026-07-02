import { createInterface } from "readline";
import { loadSettings } from "../branding/config.js";

const VERSION = "2026.07.02";
const NPU_ENDPOINT = "http://127.0.0.1:9090/v1";

function printBanner(): void {
  const banner = `
  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║    ████     ██████╗  ██╗  ████████╗        ║
  ║     ██      ██╔══██╗  ██║  ╚══██╔══╝       ║
  ║     ██      ██████╔╝  ██║     ██║          ║
  ║     ██      ██╔══██╗  ██║     ██║          ║
  ║   ██████    ██████╔╝  ██║     ██║          ║
  ║   ╚════╝    ╚═════╝   ╚═╝     ╚═╝          ║
  ║                                          ║
  ║         NPU-native coding agent          ║
  ║      50 TOPS · 94 tok/s · 0 cloud        ║
  ║                v${VERSION}               ║
  ╚══════════════════════════════════════════╝
  `;
  console.log(banner);
}

async function checkNpuStack(): Promise<boolean> {
  try {
    const res = await fetch(`${NPU_ENDPOINT}/models`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

async function queryNpu(prompt: string): Promise<string> {
  const settings = loadSettings();
  const endpoint = settings.npuEndpoint || NPU_ENDPOINT;

  const res = await fetch(`${endpoint}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen3-0.6b-FLM",
      messages: [{ role: "user", content: prompt }],
      stream: false,
    }),
    signal: AbortSignal.timeout(60000),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`NPU API error (${res.status}): ${text}`);
  }

  const data = await res.json() as { choices?: { message?: { content?: string } }[] };
  return data?.choices?.[0]?.message?.content?.trim() || "[no response]";
}

async function handleCommand(cmd: string, rl: ReturnType<typeof createInterface>): Promise<boolean | null> {
  const parts = cmd.trim().split(/\s+/);
  const verb = parts[0]?.toLowerCase();

  switch (verb) {
    case "/help":
      console.log(`
  Commands:
    /help              Show this help
    /status            Check NPU stack status
    /up                Start NPU stack
    /down              Stop NPU stack
    /clear             Clear the screen
    /exit              Exit 1bit chat
  `);
      return true;

    case "/status": {
      const running = await checkNpuStack();
      if (running) {
        console.log("  ✅ NPU stack is running (localhost:9090)");
      } else {
        console.log("  ❌ NPU stack is not running. Start with /up");
      }
      return true;
    }

    case "/up": {
      const { startUp } = await import("./up.js");
      await startUp();
      return true;
    }

    case "/down": {
      const { shutDown } = await import("./down.js");
      await shutDown();
      return true;
    }

    case "/clear":
      console.clear();
      printBanner();
      return true;

    case "/exit":
    case "/quit":
      console.log("  Goodbye.\n");
      rl.close();
      return false;

    default:
      // Slash commands we don't recognize
      if (verb?.startsWith("/")) {
        console.log(`  Unknown command: ${verb}. Try /help`);
        return true;
      }
      // Non-slash input — send to NPU
      return null; // sentinel: not a command, forward to NPU
  }
}

export async function startChat(): Promise<void> {
  printBanner();

  const npuRunning = await checkNpuStack();
  if (!npuRunning) {
    console.log("  ℹ️  NPU stack is not running. Type /up to start it, or just ask a question.");
    console.log("     (I'll wait for the stack to be available.)\n");
  } else {
    console.log("  ✅ NPU stack is online — ask me anything.\n");
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "1bit> ",
  });

  rl.prompt();

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      rl.prompt();
      continue;
    }

    const result = await handleCommand(trimmed, rl);
    if (result === false) {
      break;
    }

    if (result === null) {
      // Send to NPU
      if (!await checkNpuStack()) {
        console.log("  ⚠️  NPU stack not running. Type /up to start, or /help for commands.");
        rl.prompt();
        continue;
      }

      console.log("  🤔 Thinking...");
      try {
        const response = await queryNpu(trimmed);
        console.log(`\n  ${response}\n`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.log(`  ⚠️  Error: ${msg}`);
      }
    }

    rl.prompt();
  }
}
