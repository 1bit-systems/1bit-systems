import { loadSettings, saveSettings, SETTINGS_PATH, type Settings } from "../branding/config.js";

function showAll(settings: Settings): void {
  console.log("  1bit Configuration\n");
  console.log(`  Theme:               ${settings.theme}`);
  console.log(`  Default provider:    ${settings.defaultProvider}`);
  console.log(`  Default model:       ${settings.defaultModel}`);
  console.log(`  Thinking level:      ${settings.defaultThinkingLevel}`);
  console.log(`  NPU endpoint:        ${settings.npuEndpoint}`);
  console.log(`  Packages:            ${settings.packages.length > 0 ? settings.packages.join(", ") : "none"}`);
  console.log(`  API keys set:        ${Object.keys(settings.apiKeys).length} provider(s)`);

  if (Object.keys(settings.apiKeys).length > 0) {
    console.log("");
    for (const [provider, key] of Object.entries(settings.apiKeys)) {
      const masked = key.length > 8 ? key.slice(0, 4) + "…" + key.slice(-4) : "****";
      console.log(`    ${provider}: ${masked}`);
    }
  }

  console.log(`\n  Config file: ${SETTINGS_PATH}\n`);
}

export async function manageConfig(args: string[]): Promise<void> {
  const settings = loadSettings();

  if (args.length === 0) {
    // Show all config
    showAll(settings);
    return;
  }

  // Single arg: key=value  (set) or key (get)
  for (const arg of args) {
    const eqIdx = arg.indexOf("=");
    let key: string;
    let value: string | undefined;

    if (eqIdx === -1) {
      // Get
      key = arg;
      value = undefined;
    } else {
      // Set
      key = arg.slice(0, eqIdx);
      value = arg.slice(eqIdx + 1);
    }

    const validKeys: (keyof Settings)[] = [
      "theme", "defaultProvider", "defaultModel", "defaultThinkingLevel",
      "npuEndpoint", "packages",
    ];

    if (!validKeys.includes(key as keyof Settings)) {
      console.log(`  ⚠️  Unknown setting: ${key}`);
      console.log(`      Valid keys: ${validKeys.join(", ")}`);
      continue;
    }

    if (value === undefined) {
      // Get
      const val = settings[key as keyof Settings];
      const display = Array.isArray(val) ? val.join(", ") : String(val);
      console.log(`  ${key} = ${display}`);
    } else {
      // Set
      let parsed: unknown = value;
      if (key === "packages") {
        parsed = value.split(",").map((s) => s.trim()).filter(Boolean);
      }

      (settings as unknown as Record<string, unknown>)[key] = parsed;
      saveSettings(settings);
      console.log(`  ✅ ${key} = ${value}`);
    }
  }
}
