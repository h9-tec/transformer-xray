import { defineConfig } from '@playwright/test';

// E2E runs against the exported out/ directory — the real deployment
// artifact, not the dev server. BASE_PATH (e.g. /transformer-xray) serves
// that export under the same prefix GitHub Pages will, so prefix bugs fail
// here instead of in production. Specs use RELATIVE gotos ('en/playground/')
// so the prefix in baseURL is kept.
const BASE_PATH = (process.env.BASE_PATH ?? '').replace(/\/+$/, '');
const PORT = 4930;

export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  use: {
    baseURL: `http://localhost:${PORT}${BASE_PATH}/`,
  },
  webServer: {
    command: `node scripts/serve-static.mjs --dir out --port ${PORT} --prefix "${BASE_PATH}"`,
    port: PORT,
    // A stale un-prefixed server would mask prefix bugs.
    reuseExistingServer: false,
  },
});
