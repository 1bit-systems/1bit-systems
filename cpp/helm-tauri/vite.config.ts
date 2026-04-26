import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri's recommended Vite config: fixed dev port + no HMR over network.
// See https://v2.tauri.app/start/frontend/vite/.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "127.0.0.1",
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    target: "es2022",
    minify: "esbuild",
    sourcemap: false,
  },
});
