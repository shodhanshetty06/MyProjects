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
  build: {
    rollupOptions: {
      output: {
        // Split out react/react-dom into their own chunk. They're loaded on
        // every route (there's no route that doesn't need them) and change
        // far less often than app code, so isolating them lets browsers keep
        // a long-lived cache hit across deploys instead of re-downloading
        // them every time any app file changes. Deliberately narrow (just
        // these two packages) so this doesn't pull async-only dependencies
        // like recharts/framer-motion — which route-level lazy() already
        // isolates into their own on-demand chunks — back into a
        // synchronously-loaded bundle.
        manualChunks(id) {
          if (id.includes("node_modules/react-dom/") || id.includes("node_modules/react/")) {
            return "vendor-react";
          }
        },
      },
    },
  },
});
