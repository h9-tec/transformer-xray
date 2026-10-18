import { expect, test } from '@playwright/test';

// Static export has no root page; public/index.html meta-refreshes to en/.
// Relative goto on purpose: keeps the Pages prefix carried by baseURL.
test('site root redirects to the English home', async ({ page }) => {
  await page.goto('./');
  await page.waitForURL(/\/en\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Transformer');
});
