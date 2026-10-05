import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev and preview servers proxy /demo (the routes the five screens call) to the backend,
// so the browser makes same-origin requests. The backend address is VITE_BACKEND_URL, default
// the API_PORT of .env.example. In production the two are separate deployables (Phase 19).
const backend = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8010";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/demo": backend },
  },
  preview: {
    proxy: { "/demo": backend },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    target: "es2022",
  },
});
