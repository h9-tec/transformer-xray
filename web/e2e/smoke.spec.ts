/** Smoke suite against the static export: the engine boots, generates
 * deterministically, the inspectors trace, the depth/phase toggles gate
 * content, training reduces loss, and /ar is honestly RTL. */

import { expect, test } from '@playwright/test';


async function waitForModel(page: import('@playwright/test').Page) {
  await page.waitForFunction(() => window.__stores?.model.getState().status === 'ready', {
    timeout: 30_000,
  });
}

test('playground generates with a seed, deterministically', async ({ page }) => {
  await page.goto('en/playground/');
  await page.getByRole('button', { name: 'generate' }).click();
  const stream = page.locator('[aria-label^="Generated text"]');
  await expect(stream.locator('button').nth(199)).toBeVisible({ timeout: 30_000 });

  const first = await stream.textContent();
  await page.getByRole('button', { name: 'generate' }).click();
  await expect(stream.locator('button').nth(199)).toBeVisible({ timeout: 30_000 });
  expect(await stream.textContent()).toBe(first); // same seed → same text

  // clicking a token opens its distribution
  await stream.locator('button').first().click();
  await expect(page.locator('figure svg')).toBeVisible();
});

test('attention inspector traces and mirrors into gradients', async ({ page }) => {
  await page.goto('en/inspect/attention/');
  await waitForModel(page);
  await page.waitForFunction(() => window.__stores.trace.getState().status === 'ready', {
    timeout: 30_000,
  });

  await expect(page.locator('canvas').first()).toBeVisible();
  await expect(page.getByText('the four steps', { exact: false })).toBeVisible();

  await page.getByRole('button', { name: /Backward/ }).click();
  await expect(page.getByText('∂L/∂(softmax)')).toBeVisible();

  const loss = await page.evaluate(() => window.__stores.trace.getState().loss);
  expect(loss).toBeGreaterThan(0.5);
  expect(loss).toBeLessThan(3);
});

test('depth toggle reveals the senior derivation', async ({ page }) => {
  await page.goto('en/inspect/attention/');
  const senior = page.locator('.senior-only').first();
  await expect(senior).toBeHidden();
  await page.getByRole('button', { name: 'Full depth' }).click();
  await expect(senior).toBeVisible();
  await expect(page.locator('.junior-only').first()).toBeHidden();
});

test('watch-it-learn reduces the loss from random init', async ({ page }) => {
  await page.goto('en/train/');
  await waitForModel(page);
  await page.getByRole('button', { name: 'start training' }).click();

  // First recorded losses ≈ ln(65) ≈ 4.17 from noise; wait for real progress.
  await page.waitForFunction(
    () => {
      const text = document.body.textContent ?? '';
      const m = text.match(/step (\d+)/);
      return m && Number(m[1]) >= 30;
    },
    { timeout: 60_000 },
  );
  const lastLoss = await page.evaluate(() => {
    const m = (document.body.textContent ?? '').match(/loss (\d+\.\d+)/);
    return m ? Number(m[1]) : null;
  });
  expect(lastLoss).not.toBeNull();
  expect(lastLoss!).toBeLessThan(3.9); // meaningfully below the 4.17 start
});

test('/ar renders RTL with Arabic chrome', async ({ page }) => {
  await page.goto('ar/');
  await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
  await expect(page.locator('html')).toHaveAttribute('lang', 'ar');
  await expect(page.getByText('دعه يكتب')).toBeVisible();
});

test('new identity: display type and copper accent are live', async ({ page }) => {
  await page.goto('en/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveCSS('font-family', /Syne/);
  const accent = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim(),
  );
  expect(accent).toBe('#ff9a4d');
});
