import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative asset paths so the build works from any sub-path (GitHub Pages).
  base: "./",
  build: {
    target: "es2022",
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
