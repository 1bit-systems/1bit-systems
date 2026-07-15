import { execSync, spawn } from "child_process";
import { existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DAEMON = resolve(REPO_ROOT, "daemon", "npu-gpu-cpud.py");

/**
 * Check if a service is already running on a port.
 */
function isPortInUse(port: number): boolean {
  try {
    execSync(`ss -tlnp | grep -q ":${port} "`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

/**
 * Start the NPU stack — npu-gpu-cpud daemon (port 9090) + Lemond (port 13305).
 */
export async function startUp(): Promise<void> {
  console.log("  🚀 Starting 1bit NPU stack...\n");

  // --- npu-gpu-cpud daemon (port 9090) ---
  if (isPortInUse(9090)) {
    console.log("  ✅ npu-gpu-cpud daemon already running on port 9090");
  } else if (existsSync(DAEMON)) {
    console.log("  Starting npu-gpu-cpud daemon (requires sudo for NPU access)...");
    try {
      const daemon = spawn("sudo", ["python3", DAEMON, "--port", "9090"], {
        stdio: "inherit",
        detached: true,
      });
      daemon.unref();
      console.log("  ✅ Started npu-gpu-cpud daemon (port 9090)");
    } catch {
      console.log("  ⚠️  Failed to start daemon. Run manually:");
      console.log(`     sudo python3 ${DAEMON} --port 9090`);
    }
  } else {
    console.log("  ⚠️  Daemon not found at", DAEMON);
    console.log("     Clone the repo and ensure daemon/npu-gpu-cpud.py exists.");
  }

  // --- Lemond v2 (port 13305) ---
  if (isPortInUse(13305)) {
    console.log("  ✅ Lemond already running on port 13305");
  } else {
    try {
      execSync("which lemond", { stdio: "ignore" });
      const lemond = spawn("lemond", ["--port", "13305"], {
        stdio: "ignore",
        detached: true,
      });
      lemond.unref();
      console.log("  ✅ Started Lemond v2 (port 13305)");
    } catch {
      console.log("  ⚠️  lemond not found in PATH. Install lemond or start manually.");
    }
  }

  console.log("\n  📍 API:       http://127.0.0.1:9090/v1");
  console.log("  📍 Health:    http://127.0.0.1:9090/v1/health");
  console.log("  📍 Chat UI:   http://127.0.0.1:13305/\n");
}

/**
 * Build the C++ NPU engine from source.
 */
export async function startBuild(): Promise<void> {
  const buildDir = resolve(REPO_ROOT, "engine", "npu");
  if (!existsSync(buildDir)) {
    console.log(`  ⚠️  Build directory not found: ${buildDir}`);
    return;
  }

  const buildScript = resolve(buildDir, "build_npu.sh");
  if (existsSync(buildScript)) {
    console.log("  🔨 Building NPU engine from source...\n");
    try {
      execSync(`bash "${buildScript}"`, {
        cwd: buildDir,
        stdio: "inherit",
      });
      console.log("\n  ✅ NPU engine build complete");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`\n  ❌ Build failed: ${msg}`);
    }
    return;
  }

  // Fallback: direct g++ compile
  console.log("  🔨 Building NPU engine (g++)...\n");
  const src = resolve(buildDir, "src", "npu_engine_all.cpp");
  const dequant = resolve(buildDir, "build", "dequant_q4nx.o");
  const out = resolve(buildDir, "build", "npu_engine_all");

  if (!existsSync(src)) {
    console.log(`  ⚠️  Engine source not found: ${src}`);
    return;
  }

  try {
    execSync(
      `g++ -std=c++23 -O3 -march=native -fopenmp -ffast-math -o "${out}" "${src}" "${dequant}" -lxrt_coreutil -luuid -lm -ldl`,
      { cwd: buildDir, stdio: "inherit" }
    );
    console.log(`\n  ✅ NPU engine built: ${out}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`\n  ❌ Build failed: ${msg}`);
  }
}
