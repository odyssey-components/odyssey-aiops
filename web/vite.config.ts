import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import { appearanceBootstrapVitePlugin } from "@odyssey/ui/patterns/admin-appearance-vite";
import { appearance } from "./src/appearance.ts";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const loopback = { host: "127.0.0.1", port: 4175, strictPort: true, cors: false };
const runtimeDist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web/wasm"));

const configuredAPI = process.env.AIOPS_API_ORIGIN ?? process.env.ODYMAZE_API_ORIGIN;
const apiOrigin = new URL(configuredAPI ?? "http://127.0.0.1:4176");
if (apiOrigin.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(apiOrigin.hostname)
  || apiOrigin.username || apiOrigin.password || apiOrigin.pathname !== "/" || apiOrigin.search || apiOrigin.hash) {
  throw new Error("AIOPS_API_ORIGIN must be a loopback HTTP origin");
}

const base = process.env.AIOPS_WEB_BASE ?? "/";
if (base !== "/" && base !== "/aiops/") throw new Error("AIOPS_WEB_BASE must be / or /aiops/");
const proxy = {
  [`${base}api`]: {
    target: apiOrigin.origin,
    changeOrigin: false,
    rewrite: (path: string) => path.slice(base.length - 1),
  },
};

export default defineConfig({
  base,
  resolve: { alias: [
    { find: "odymaze-onnx-proto", replacement: join(runtimeDist, "../lib/onnxjs/ort-schema/protobuf/onnx.js") },
    { find: /^odymaze-ort-wasm(?=\?|$)/, replacement: join(runtimeDist, "ort-wasm-simd-threaded.wasm") },
    { find: /^odymaze-ort-loader(?=\?|$)/, replacement: join(runtimeDist, "ort-wasm-simd-threaded.mjs") },
  ] },
  optimizeDeps: { exclude: ["odymaze-ort-loader", "odymaze-ort-loader?url"] },
  plugins: [appearanceBootstrapVitePlugin(appearance), tailwindcss(), tanstackRouter({ target: "react", autoCodeSplitting: true }), react()],
  server: { ...loopback, proxy },
  preview: { ...loopback, proxy: configuredAPI ? proxy : {} },
});
