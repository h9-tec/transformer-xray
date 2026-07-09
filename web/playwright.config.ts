import { defineConfig } from '@playwright/test';

// E2E runs against the exported out/ directory — the real deployment
// artifact, not the dev server.
export default defineConfig({
  testDir: 'e2e',
  timeout: 60_000,
  use: {
    baseURL: 'http://localhost:4930',
  },
  webServer: {
    command: 'npx serve -l 4930 out',
    port: 4930,
    reuseExistingServer: true,
  },
});
