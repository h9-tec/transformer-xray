import { defineConfig } from '@playwright/test';

// E2E runs against the exported out/ directory — the real deployment
// artifact, not the dev server. BASE_PATH (e.g. /transformer-xray) serves
// that export under the same prefix GitHub Pages will, so prefix bugs fail
// here instead of in production. Specs use RELATIVE gotos ('en/playground/')
// so the prefix in baseURL is kept.
//
// Two projects: `default` is the suite CI runs (headless shell, no WebGPU);
// `gpu` runs only e2e/gpu/** in full Chromium with WebGPU enabled and is the
// authoritative local check on a machine with a GPU. Its specs skip
// themselves when the browser reports no adapter.
const BASE_PATH = (process.env.BASE_PATH ?? '').replace(/\/+$/, '');
const PORT = 4930;

export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  use: {
    baseURL: `http://localhost:${PORT}${BASE_PATH}/`,
  },
  projects: [
    { name: 'default', testIgnore: /e2e\/gpu\//, use: { browserName: 'chromium' } },
    {
      name: 'gpu',
      testMatch: /e2e\/gpu\/.*\.spec\.ts/,
      timeout: 180_000,
      use: {
        browserName: 'chromium',
        channel: 'chromium',
        launchOptions: {
          args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan', '--use-angle=vulkan'],
        },
      },
    },
  ],
  webServer: {
    command: `node scripts/serve-static.mjs --dir out --port ${PORT} --prefix "${BASE_PATH}"`,
    port: PORT,
    // A stale un-prefixed server would mask prefix bugs.
    reuseExistingServer: false,
  },
});
