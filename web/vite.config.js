// vite.config.js - settings for the dev server.
// The "proxy" forwards every request that starts with /api to our Python server (port 8000),
// so the React code can simply call fetch("/api/...") without caring where Python runs.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
});
