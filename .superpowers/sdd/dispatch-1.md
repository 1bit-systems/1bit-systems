You are implementing Task 1: Scaffold — package.json, tsconfig, CLI bin, shim

Read your task brief first: /home/bcloud/1bit-systems/.superpowers/sdd/task-1-brief.md

## Context

This is the root task for building the 1bit CLI — a cloned/rebranded pi.dev agent harness. All subsequent tasks depend on this one working. The project lives at /home/bcloud/1bit-systems/.

## Your Job

1. Create package.json and tsconfig.json exactly as specified in the brief
2. Create src/cli.ts with the full CLI entry point that dispatches to all commands via dynamic imports
3. Create bin/1bit as the Node.js shim entry point
4. Run npm install in the project directory
5. Verify the CLI works: npx tsx src/cli.ts help prints 1bit-branded help
6. Verify tsc compilation: npm run build compiles without errors
7. Verify node bin/1bit help works after build
8. Commit your work with a meaningful message
9. Write your report to /home/bcloud/1bit-systems/.superpowers/sdd/task-1-report.md

Work from: /home/bcloud/1bit-systems

Important: src/cli.ts must use dynamic imports for every command module (chat, up, down, status, build, install, remove, list, update, config). Each command module resolves to ./commands/<name>.js (the .js extension is required for ESM). The default command when no args provided is "chat".
