import { execSync } from "child_process";
/**
 * Stop the NPU stack — kill 1bit API bridge (port 9090) + Lemond (port 13305).
 */
export async function shutDown() {
    console.log("  🛑 Stopping 1bit NPU stack...\n");
    // Kill processes on the NPU ports
    const ports = [13305, 9090, 9000];
    let anyKilled = false;
    for (const port of ports) {
        try {
            execSync(`sudo fuser -k ${port}/tcp 2>/dev/null`, { stdio: "ignore" });
            // Also try without sudo as fallback
            execSync(`fuser -k ${port}/tcp 2>/dev/null`, { stdio: "ignore" });
            anyKilled = true;
        }
        catch {
            // Port not open or no permission — skip
        }
    }
    // Also kill any orphaned lemond or server processes
    for (const proc of ["lemond", "node.*server\\.cjs"]) {
        try {
            execSync(`pkill -f "${proc}" 2>/dev/null`, { stdio: "ignore" });
        }
        catch {
            // ok
        }
    }
    if (anyKilled) {
        console.log("  ✅ NPU stack stopped");
    }
    else {
        console.log("  ℹ️  No NPU processes were running");
    }
    // Verify
    setTimeout(() => {
        try {
            const result = execSync("ss -tlnp 2>/dev/null | grep -E '1330[0-9]|9090|9000' || true", {
                encoding: "utf-8",
            });
            if (result.trim()) {
                console.log("  ⚠️  Some ports still in use:\n" + result);
            }
        }
        catch {
            // ignore
        }
    }, 500);
}
//# sourceMappingURL=down.js.map