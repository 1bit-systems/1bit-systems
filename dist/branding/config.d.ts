/** Path to 1bit's agent state directory (~/.1bit/) */
export declare const ONE_BIT_DIR: string;
export declare const AGENT_DIR: string;
export declare const SETTINGS_PATH: string;
export interface Settings {
    theme: string;
    defaultProvider: string;
    defaultModel: string;
    defaultThinkingLevel: string;
    packages: string[];
    apiKeys: Record<string, string>;
    npuEndpoint: string;
}
export declare function defaultSettings(): Settings;
export declare function loadSettings(): Settings;
export declare function saveSettings(settings: Settings): void;
//# sourceMappingURL=config.d.ts.map