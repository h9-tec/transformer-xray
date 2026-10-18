import { expect, test } from '@playwright/test';

/** gpu project: the playground's "your GPU" path produces the same seeded
 * text as the CPU path and shows a WebGPU timing. */

async function waitReady(page: import('@playwright/test').Page) {
  await page.waitForFunction(() => window.__stores?.model.getState().status === 'ready', { timeout: 60_000 });
}

async function generateText(page: import('@playwright/test').Page): Promise<string> {
  await page.getByRole('button', { name: 'generate' }).click();
  const stream = page.locator('[aria-label^="Generated text"]');
  await expect(stream.locator('button').nth(199)).toBeVisible({ timeout: 90_000 });
  return (await stream.textContent()) ?? '';
}

test('your GPU generates the same seeded text as the CPU and reports a timing', async ({ page }) => {
  await page.goto('en/playground/');
  await waitReady(page);
  const gpuButton = page.getByRole('button', { name: /your GPU/i });
  const available = await gpuButton.isEnabled();
  test.skip(!available, 'no WebGPU adapter in this browser');

  const cpuText = await generateText(page);
  await gpuButton.click();
  await expect(page.getByTestId('backend-active')).toHaveText(/gpu/i, { timeout: 30_000 });
  const gpuText = await generateText(page);
  expect(gpuText).toBe(cpuText);
  await expect(page.getByTestId('backend-ms')).toContainText(/ms/);

  await page.getByRole('button', { name: /^CPU$/ }).click();
  await expect(page.getByTestId('backend-active')).toHaveText(/cpu/i);
});
