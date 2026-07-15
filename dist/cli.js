#!/usr/bin/env node
"use strict";
const VERSION = "2026.07.02";
function printHelp() {
    process.stdout.write(`\n  1bit — your NPU-native coding agent  v${VERSION}\n\n`);
    process.stdout.write("  Usage: 1bit <command> [options]\n\n");
    process.stdout.write("  Agent commands:\n");
    process.stdout.write("    chat              Start interactive agent session (default)\n");
    process.stdout.write("    install <pkg>     Install a package (npm:, git:, or path)\n");
    process.stdout.write("    remove <pkg>      Remove an installed package\n");
    process.stdout.write("    list              List installed packages\n");
    process.stdout.write("    update [pkg]      Update 1bit and packages\n");
    process.stdout.write("    config [key=val]  View or set configuration\n\n");
    process.stdout.write("  NPU stack commands:\n");
    process.stdout.write("    up        Start NPU stack (npu-gpu-cpud daemon + FLM proxy)\n");
    process.stdout.write("    down      Stop NPU stack\n");
    process.stdout.write("    status    Show NPU stack status\n");
    process.stdout.write("    build     Build NPU engine from source\n\n");
    process.stdout.write("  Other:\n");
    process.stdout.write("    help      Show this help\n");
    process.stdout.write("    version   Show version\n\n");
    process.stdout.write("  Visit http://1bit.systems\n");
}
async function main() {
    const cmd = process.argv[2] || "chat";
    try {
        switch (cmd) {
            case "chat": {
                const { startChat } = await import("./commands/chat.js");
                await startChat();
                break;
            }
            case "up": {
                const { startUp } = await import("./commands/up.js");
                await startUp();
                break;
            }
            case "down": {
                const { shutDown } = await import("./commands/down.js");
                await shutDown();
                break;
            }
            case "status": {
                const { showStatus } = await import("./commands/status.js");
                await showStatus();
                break;
            }
            case "build": {
                const { startBuild } = await import("./commands/up.js");
                await startBuild();
                break;
            }
            case "install": {
                const { installPackage } = await import("./commands/packages.js");
                await installPackage(process.argv.slice(3).join(" "));
                break;
            }
            case "remove": {
                const { removePackage } = await import("./commands/packages.js");
                await removePackage(process.argv[3]);
                break;
            }
            case "list": {
                const { listPackages } = await import("./commands/packages.js");
                await listPackages();
                break;
            }
            case "update": {
                const pkg = process.argv[3];
                if (pkg && (pkg.startsWith("npm:") || pkg.startsWith("git:") || !pkg.includes(":"))) {
                    // Package update — delegate to packages.ts
                    const { updatePackage } = await import("./commands/packages.js");
                    await updatePackage(pkg);
                }
                else {
                    // Self update
                    const { runUpdate } = await import("./commands/update.js");
                    await runUpdate(pkg);
                }
                break;
            }
            case "config": {
                const { manageConfig } = await import("./commands/config.js");
                await manageConfig(process.argv.slice(3));
                break;
            }
            case "help":
            case "--help":
            case "-h":
                printHelp();
                break;
            case "version":
            case "--version":
            case "-v":
                process.stdout.write(VERSION + "\n");
                break;
            default:
                process.stderr.write(`Unknown command: ${cmd}\n`);
                printHelp();
                process.exit(1);
        }
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        process.stderr.write(`Error in command '${cmd}': ${msg}\n`);
        process.exit(1);
    }
}
main().catch((err) => {
    const msg = err instanceof Error ? err.message : String(err);
    process.stderr.write(`Fatal error: ${msg}\n`);
    process.exit(1);
});
//# sourceMappingURL=cli.js.map