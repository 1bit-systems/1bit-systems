### Task 1: Scaffold

Create `package.json`, `tsconfig.json`, `src/cli.ts`, `bin/1bit`.

`package.json`:
```json
{
  "name": "1bit-systems",
  "version": "2026.07.02",
  "description": "1bit — your NPU-native coding agent. Fork of pi.dev, wired to local NPU.",
  "type": "module",
  "bin": { "1bit": "dist/cli.js" },
  "main": "./dist/index.js",
  "files": ["dist", "extensions", "skills", "prompts", "themes", "docs"],
  "scripts": { "build": "tsc", "dev": "tsx src/cli.ts", "watch": "tsc --watch", "start": "node dist/cli.js" },
  "keywords": ["1bit", "coding-agent", "ai", "npu", "pi-package"],
  "dependencies": {
    "@earendil-works/pi-agent-core": "^0.80.3",
    "@earendil-works/pi-ai": "^0.80.3",
    "@earendil-works/pi-tui": "^0.80.3",
    "yaml": "^2.9.0", "minimatch": "^10.2.5", "chalk": "^5.6.2", "semver": "^7.8.0"
  },
  "devDependencies": {
    "typescript": "^5.9.3", "@types/node": "^24.12.4", "tsx": "^4.19.0"
  },
  "engines": { "node": ">=22.19.0" },
  "license": "MIT"
}
```

`tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2024", "module": "ESNext", "moduleResolution": "bundler",
    "strict": true, "esModuleInterop": true, "skipLibCheck": true,
    "outDir": "dist", "rootDir": "src", "declaration": true, "declarationMap": true, "sourceMap": true
  },
  "include": ["src/**/*"], "exclude": ["node_modules", "dist"]
}
```

`src/cli.ts` — CLI entry that prints 1bit-branded help, routes commands:
```
Commands: chat, up, down, status, build, install, remove, list, update, config, help, version
```
Default command: `chat`. Uses dynamic imports for each command module.

`bin/1bit` — Node.js shim that uses dist/cli.js if built, else tsx src/cli.ts in dev.

Steps: create files, `npm install`, verify `npx tsx src/cli.ts help` prints 1bit help.

