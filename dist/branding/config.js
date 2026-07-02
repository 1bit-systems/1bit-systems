import { homedir } from "os";
import { join } from "path";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
/** Path to 1bit's agent state directory (~/.1bit/) */
export const ONE_BIT_DIR = join(homedir(), ".1bit");
export const AGENT_DIR = join(ONE_BIT_DIR, "agent");
export const SETTINGS_PATH = join(AGENT_DIR, "settings.json");
export function defaultSettings() {
    return {
        theme: "1bit",
        defaultProvider: "npu",
        defaultModel: "npu-local",
        defaultThinkingLevel: "medium",
        packages: [],
        apiKeys: {},
        npuEndpoint: "http://127.0.0.1:9090/v1",
    };
}
export function loadSettings() {
    try {
        const raw = readFileSync(SETTINGS_PATH, "utf-8");
        return { ...defaultSettings(), ...JSON.parse(raw) };
    }
    catch {
        return defaultSettings();
    }
}
export function saveSettings(settings) {
    if (!existsSync(AGENT_DIR)) {
        mkdirSync(AGENT_DIR, { recursive: true });
    }
    writeFileSync(SETTINGS_PATH, JSON.stringify(settings, null, 2) + "\n");
}
//# sourceMappingURL=config.js.map