import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
// During `npm run dev` the API + metrics are proxied to the FastAPI server on
// :8080, so the frontend can call same-origin paths in both dev and prod.
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": "http://localhost:8080",
            "/metrics": "http://localhost:8080",
            "/health": "http://localhost:8080",
        },
    },
    build: {
        outDir: "dist",
    },
});
