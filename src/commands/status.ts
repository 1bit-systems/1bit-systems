import { execSync } from "child_process";

interface PortInfo {
  port: number;
  name: string;
  pid?: string;
  process?: string;
}

function getPortInfo(port: number): PortInfo | null {
  try {
    const output = execSync(
      `ss -tlnp 2>/dev/null | grep ":${port} " || true`,
      { encoding: "utf-8" }
    );
    if (!output.trim()) return null;

    // Extract PID if available
    const pidMatch = output.match(/pid=(\d+)/);
    return {
      port,
      name: "",
      pid: pidMatch?.[1],
      process: output.trim(),
    };
  } catch {
    return null;
  }
}

function getProcessInfo(pid: string): string {
  try {
    const cmd = execSync(`ps -p ${pid} -o comm= 2>/dev/null`, {
      encoding: "utf-8",
    }).trim();
    return cmd;
  } catch {
    return "unknown";
  }
}

export async function showStatus(): Promise<void> {
  console.log("  ┌─ 1bit NPU Stack Status ──────────────────────────┐\n");

  // Check ports
  const ports: { port: number; name: string }[] = [
    { port: 13305, name: "Lemond (Chat UI)" },
    { port: 9090, name: "1bit API Bridge" },
    { port: 9000, name: "Lemond WebSocket" },
  ];

  let allRunning = true;
  for (const { port, name } of ports) {
    const info = getPortInfo(port);
    if (info) {
      const proc = info.pid ? getProcessInfo(info.pid) : "unknown";
      console.log(`  ✅  ${name.padEnd(25)} port ${port}  (${proc})`);
    } else {
      console.log(`  ❌  ${name.padEnd(25)} port ${port}  (not running)`);
      allRunning = false;
    }
  }

  console.log("");

  // Check NPU API health
  try {
    const res = await fetch("http://127.0.0.1:9090/v1/models", {
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      const data = await res.json() as { data?: Array<{ id: string }> };
      const models = data?.data ?? [];
      console.log(`  ✅  NPU API responding (${models.length} models)`);
      if (models.length > 0) {
        console.log(`      Models: ${models.slice(0, 5).map((m: { id: string }) => m.id).join(", ")}${models.length > 5 ? "..." : ""}`);
      }
    } else {
      console.log(`  ⚠️  NPU API error: ${res.status}`);
    }
  } catch {
    console.log("  ❌  NPU API not reachable");
  }

  try {
    const res2 = await fetch("http://127.0.0.1:13305/v1/models", {
      signal: AbortSignal.timeout(3000),
    });
    if (res2.ok) {
      console.log("  ✅  Lemond API responding");
    } else {
      console.log(`  ⚠️  Lemond API error: ${res2.status}`);
    }
  } catch {
    // Already shown above
  }

  console.log("\n  └──────────────────────────────────────────────────┘");
  console.log(`  📍 Chat UI:   http://127.0.0.1:13305/`);
  console.log(`  📍 Terminal:  http://127.0.0.1:9090/\n`);

  if (allRunning) {
    console.log("  ✨ All services running.\n");
  }
}
