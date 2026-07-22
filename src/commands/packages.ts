import { execFileSync } from "child_process";
import { loadSettings, saveSettings } from "../branding/config.js";

/**
 * Validate a package spec string.
 * Accepted forms: npm:<pkg>, git:<repo-url>, or a local path.
 */
function validateSpec(spec: string): string | null {
  spec = spec.trim();

  if (spec.startsWith("npm:") || spec.startsWith("git:")) {
    const rest = spec.slice(4).trim();
    if (!rest) {
      console.log(`  ⚠️  Empty ${spec.slice(0, 3)} spec`);
      return null;
    }
    return spec;
  }

  // Treat as npm: prefix if no colon and looks like a package name
  if (!spec.includes(":")) {
    if (!/^[@a-zA-Z0-9._/-]+$/.test(spec)) {
      console.log(`  ⚠️  Invalid package name: "${spec}"`);
      return null;
    }
    return `npm:${spec}`;
  }

  // Raw URL
  if (spec.startsWith("https://") || spec.startsWith("http://") || spec.startsWith("ssh://")) {
    return spec;
  }

  console.log(`  ⚠️  Unknown package format: "${spec}"`);
  console.log("       Use npm:<pkg>, git:<url>, <url>, or just <pkg>");
  return null;
}

function findPi(): string {
  try {
    return execSync("which pi 2>/dev/null", { encoding: "utf-8" }).trim();
  } catch {
    return "";
  }
}

export async function installPackage(spec: string): Promise<void> {
  const parsed = validateSpec(spec);
  if (!parsed) return;

  const settings = loadSettings();
  if (settings.packages.includes(parsed)) {
    console.log(`  ℹ️  Package already installed: ${parsed}`);
    console.log(`      Run: pi install ${parsed}  (or pi update ${parsed})`);
    return;
  }

  const piPath = findPi();
  if (!piPath) {
    // Fallback: just track it
    settings.packages.push(parsed);
    saveSettings(settings);
    console.log(`  ✅ Recorded: ${parsed}`);
    console.log("     (pi not found — package recorded but not installed)");
    return;
  }

  console.log(`  → Installing ${parsed} via pi...\n`);
  try {
    execFileSync(piPath, ["install", parsed, "--no-approve"], {
      stdio: "inherit",
      timeout: 120000,
    });
    // Add to 1bit's package list too
    if (!settings.packages.includes(parsed)) {
      settings.packages.push(parsed);
      saveSettings(settings);
    }
    console.log(`\n  ✅ Installed: ${parsed}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`\n  ⚠️  pi install failed: ${msg}`);
    console.log("     Available: pi list  (to see installed packages)");
  }
}

export async function removePackage(name: string): Promise<void> {
  // Remove from 1bit settings
  const settings = loadSettings();
  const before = settings.packages.length;
  settings.packages = settings.packages.filter((p) => {
    const display = p.includes(":") ? p.split(":")[1] : p;
    return p !== name && display !== name;
  });

  const changed = settings.packages.length !== before;
  if (changed) {
    saveSettings(settings);
  }

  // Also try pi remove
  const piPath = findPi();
  if (piPath) {
    try {
      execFileSync(piPath, ["remove", name, "--no-approve"], {
        stdio: "inherit",
        timeout: 30000,
      });
      if (!changed) {
        // Wasn't in 1bit settings but pi removed it
        console.log(`  ✅ Removed: ${name} (via pi)`);
        return;
      }
      console.log(`  ✅ Removed: ${name}`);
      return;
    } catch {
      // pi remove failed — still remove from 1bit settings
    }
  }

  if (changed) {
    console.log(`  ✅ Removed from 1bit: ${name}`);
  } else {
    console.log(`  ℹ️  Package not found: ${name}`);
    console.log(`      Installed: ${settings.packages.join(", ") || "none"}`);
  }
}

export async function listPackages(): Promise<void> {
  const settings = loadSettings();
  const pkgs = settings.packages;

  console.log(`  📦 1bit packages (${pkgs.length}):\n`);
  if (pkgs.length > 0) {
    for (const pkg of pkgs) {
      const [type, value] = pkg.includes(":") ? pkg.split(":") : ["npm", pkg];
      console.log(`     • ${type === "npm" ? "" : `${type}:`}${value}`);
    }
  } else {
    console.log("     (none installed)");
  }

  // Also show pi's packages
  const piPath = findPi();
  if (piPath) {
    console.log("");
    try {
      const piOutput = execFileSync(piPath, ["list", "--no-approve"], {
        encoding: "utf-8",
        timeout: 10000,
      });
      console.log(`  🔌 pi packages:\n`);
      for (const line of piOutput.trim().split("\n")) {
        // Show package lines (starting with "  npm:" or "  git:") and their paths
        if (line.startsWith("  npm:") || line.startsWith("  git:") || line.startsWith("  https://")) {
          console.log(`     ${line.trim()}`);
        }
      }
    } catch {
      console.log("  🔌 pi: could not list packages");
    }
  } else {
    console.log("\n  🔌 pi not found — install pi.dev for full package management");
  }

  console.log(`\n  Use: 1bit install <pkg>   or  1bit remove <pkg>
  See: pi.dev/packages\n`);
}

export async function updatePackage(spec?: string): Promise<void> {
  const piPath = findPi();
  if (!piPath) {
    console.log("  ⚠️  pi not found — cannot update packages");
    return;
  }

  if (spec) {
    console.log(`  → Updating ${spec}...\n`);
    try {
      execFileSync(piPath, ["update", spec, "--no-approve"], {
        stdio: "inherit",
        timeout: 120000,
      });
      console.log(`  ✅ Updated: ${spec}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`\n  ⚠️  Update failed: ${msg}`);
    }
    return;
  }

  // Update all packages
  console.log("  → Updating all packages via pi...\n");
  try {
    execFileSync(piPath, ["update", "--extensions", "--no-approve"], {
      stdio: "inherit",
      timeout: 120000,
    });
    console.log("  ✅ Packages updated");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`\n  ⚠️  Update failed: ${msg}`);
  }
}
