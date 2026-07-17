import { execSync } from "child_process";
import { existsSync } from "fs";
import { resolve } from "path";
import { loadSettings, saveSettings } from "../branding/config.js";

const PROJECT_ROOT = resolve(import.meta.dirname, "../..");

/**
 * Update 1bit itself and optionally a specific package.
 */
export async function runUpdate(spec?: string): Promise<void> {
  if (spec) {
    // Update a specific package (from settings)
    const settings = loadSettings();
    const pkg = settings.packages.find((p) => {
      const display = p.includes(":") ? p.split(":")[1] : p;
      return p === spec || display === spec;
    });

    if (!pkg) {
      console.log(`  ℹ️  Package not found: ${spec}`);
      console.log(`      Installed: ${settings.packages.join(", ") || "none"}`);
      return;
    }
    console.log(`  ℹ️  Package updates coming soon: ${pkg}`);
    return;
  }

  // Update 1bit itself
  console.log("  🔄 Updating 1bit...\n");

  // Check if git repo
  if (existsSync(resolve(PROJECT_ROOT, ".git"))) {
    try {
      console.log("  → Pulling latest from git...");
      execSync("git pull --ff-only", {
        cwd: PROJECT_ROOT,
        stdio: "inherit",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  ⚠️  Git pull failed: ${msg}`);
      console.log("       Continuing with npm update...");
    }
  } else {
    console.log("  ℹ️  Not a git repo, skipping git pull");
  }

  // npm install
  try {
    console.log("  → Installing dependencies...");
    execSync("npm install", { cwd: PROJECT_ROOT, stdio: "inherit" });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`  ⚠️  npm install failed: ${msg}`);
  }

  // tsc build
  try {
    console.log("  → Building...");
    execSync("npm run build", { cwd: PROJECT_ROOT, stdio: "inherit" });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`  ⚠️  Build failed: ${msg}`);
  }

  console.log("\n  ✅ 1bit updated");
}
