import { execSync, spawn } from "child_process";
import { existsSync } from "fs";
import { resolve } from "path";
const HOME = process.env.HOME || "/home/bcloud";
/**
 * Check if a service is already running on a port.
 */
function isPortInUse(port) {
    try {
        execSync(`ss -tlnp | grep -q ":${port} "`, { stdio: "ignore" });
        return true;
    }
    catch {
        return false;
    }
}
/**
 * Start the NPU stack — API bridge (port 9090) + Lemond (port 13305).
 */
export async function startUp() {
    console.log("  🚀 Starting 1bit NPU stack...\n");
    // --- 1bit API bridge (port 9090) ---
    const bridgeScript = resolve(HOME, "npu-sandbox/npu-infer/1bit/dist/server.cjs");
    const bridgeFallback = resolve(HOME, "npu-sandbox/npu-infer/1bit", "package.json");
    if (isPortInUse(9090)) {
        console.log("  ✅ 1bit API bridge already running on port 9090");
    }
    else if (existsSync(bridgeScript)) {
        const bridge = spawn("node", [bridgeScript], {
            stdio: "ignore",
            detached: true,
            env: { ...process.env, PORT: "9090" },
        });
        bridge.unref();
        console.log("  ✅ Started 1bit API bridge (port 9090)");
    }
    else if (existsSync(bridgeFallback)) {
        // Try building first
        console.log("  ⚠️  bridge not built. Building...");
        try {
            execSync("npm run build", {
                cwd: resolve(HOME, "npu-sandbox/npu-infer/1bit"),
                stdio: "inherit",
            });
            const bridge2 = spawn("node", [bridgeScript], {
                stdio: "ignore",
                detached: true,
                env: { ...process.env, PORT: "9090" },
            });
            bridge2.unref();
            console.log("  ✅ Started 1bit API bridge (port 9090)");
        }
        catch {
            console.log("  ⚠️  Could not build bridge. Run build manually.");
        }
    }
    else {
        console.log("  ⚠️  1bit API bridge not found at", bridgeScript);
    }
    // --- Lemond v2 (port 13305) ---
    if (isPortInUse(13305)) {
        console.log("  ✅ Lemond already running on port 13305");
    }
    else {
        try {
            // Check if lemond exists
            execSync("which lemond", { stdio: "ignore" });
            const lemond = spawn("lemond", ["--port", "13305"], {
                stdio: "ignore",
                detached: true,
            });
            lemond.unref();
            console.log("  ✅ Started Lemond v2 (port 13305)");
        }
        catch {
            console.log("  ⚠️  lemond not found in PATH. Install lemond or start manually.");
        }
    }
    console.log("\n  📍 API:       http://127.0.0.1:9090/v1");
    console.log("  📍 Chat UI:    http://127.0.0.1:13305/");
    console.log("  📍 Terminal:   http://127.0.0.1:9090/\n");
}
/**
 * Build the C++ NPU engine from source.
 */
export async function startBuild() {
    const buildDir = resolve(HOME, "npu-sandbox/npu-infer");
    if (!existsSync(buildDir)) {
        console.log(`  ⚠️  Build directory not found: ${buildDir}`);
        return;
    }
    const buildScript = resolve(buildDir, "build/build_npu.sh");
    if (!existsSync(buildScript)) {
        console.log(`  ⚠️  Build script not found: ${buildScript}`);
        console.log("  Falling back to CMake in", buildDir);
        try {
            execSync("cmake --build build --target npu_engine_mt", {
                cwd: buildDir,
                stdio: "inherit",
            });
        }
        catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            console.log(`  ❌ Build failed: ${msg}`);
        }
        return;
    }
    console.log("  🔨 Building NPU engine from source...\n");
    try {
        execSync(`bash "${buildScript}"`, {
            cwd: buildDir,
            stdio: "inherit",
        });
        console.log("\n  ✅ NPU engine build complete");
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.log(`\n  ❌ Build failed: ${msg}`);
    }
}
//# sourceMappingURL=up.js.map