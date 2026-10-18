import { expect, test } from '@playwright/test';


test('/learn lists all lessons', async ({ page }) => {
  await page.goto('en/learn/');
  await expect(page.getByRole('list', { name: 'Lesson map' }).getByRole('link')).toHaveCount(12);
});

test('a lesson with a kernel shows its real source', async ({ page }) => {
  await page.goto('en/learn/tiled-matmul/');
  await expect(page.getByTestId('kernel-source')).toContainText('matmul_tiled');
  await expect(page.getByRole('link', { name: /in the model/i })).toHaveAttribute('href', /\/en\/inspect\/mlp\/$/);
});

test('a lesson without a kernel renders without the kernel block', async ({ page }) => {
  await page.goto('en/learn/hello-thread/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Hello, thread');
  await expect(page.getByTestId('kernel-source')).toHaveCount(0);
});

test('progress persists across reload', async ({ page }) => {
  await page.goto('en/learn/');
  await page.waitForFunction(() => Boolean(window.__stores?.learn));
  await page.evaluate(() => window.__stores.learn.getState().markPassed('hello-thread'));
  await page.reload();
  const first = page.getByRole('list', { name: 'Lesson map' }).getByRole('link').first();
  await expect(first).toHaveAttribute('data-status', 'passed');
  await page.goto('en/learn/tiled-matmul/');
  await expect(page.getByText('1 of 12 passed')).toBeVisible();
});

test('English lesson text is isolated as LTR inside the Arabic site', async ({ page }) => {
  await page.goto('ar/learn/');
  const title = page.getByRole('list', { name: 'خريطة الدروس' }).getByRole('heading', { level: 3 }).first();
  await expect(title).toHaveAttribute('dir', 'ltr');
  await expect(title).toHaveAttribute('lang', 'en');
});

test('slot copy never points at a kernel that is not there', async ({ page }) => {
  await page.goto('en/learn/hello-thread/');
  const text = await page.locator('main').innerText();
  expect(text).not.toMatch(/kernel below/i);
  expect(text).not.toMatch(/inspector link/i);
});
