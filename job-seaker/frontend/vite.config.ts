import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: true,
    port: 3000,
    // Docker Desktop on Windows doesn't propagate host filesystem change
    // events into the Linux container over the bind mount, so Vite's
    // default watcher never fires — fall back to polling.
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
});
