import { expect, test } from '@playwright/test';


async function openLesson(page: import('@playwright/test').Page, slug: string) {
  await page.goto(`en/learn/${slug}/`);
  await page.waitForFunction((s) => Boolean(window.__cm?.[s]), slug);
}

test('starter fails, solution passes and lights the tile', async ({ page }) => {
  await openLesson(page, 'hello-thread');
  await page.getByRole('button', { name: /run on the gpu/i }).click();
  const correct = page.getByTestId('target-correct');
  await expect(correct).toHaveAttribute('data-pass', 'false', { timeout: 30_000 });
  await page.evaluate(() => window.__cm!['hello-thread'].setText(window.__exercise!.solution));
  await page.getByRole('button', { name: /run on the gpu/i }).click();
  await expect(correct).toHaveAttribute('data-pass', 'true', { timeout: 30_000 });
  await expect(page.getByTestId('exercise-passed')).toBeVisible();
  await page.goto('en/learn/');
  await expect(page.getByRole('list', { name: 'Lesson map' }).getByRole('link').first()).toHaveAttribute('data-status', 'passed');
});

test('a syntax error highlights the line and explains', async ({ page }) => {
  await openLesson(page, 'hello-thread');
  await page.evaluate(() =>
    window.__cm!['hello-thread'].setText('__global__ void vector_add(const float* a, const float* b, float* y, int n) {\n  int i = 0;\n  y[0] = ;\n}'),
  );
  await page.getByRole('button', { name: /run on the gpu/i }).click();
  const err = page.getByTestId('run-error');
  await expect(err).toBeVisible({ timeout: 30_000 });
  await expect(err).toContainText(/line 3/i);
  await expect(page.locator('.cm-error-line')).toHaveCount(1);
});

test('an infinite loop times out and Run is usable again', async ({ page }) => {
  await openLesson(page, 'hello-thread');
  await page.evaluate(() =>
    window.__cm!['hello-thread'].setText('__global__ void vector_add(const float* a, const float* b, float* y, int n) {\n  while (true) {}\n}'),
  );
  const run = page.getByRole('button', { name: /run on the gpu/i });
  await run.click();
  await expect(page.getByTestId('run-error')).toContainText(/budget|terminating/i, { timeout: 60_000 });
  await expect(run).toBeEnabled();
});

test('lesson page at 360px keeps the editor inside and the page unscrolled', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 780 });
  await openLesson(page, 'tiled-matmul');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await expect(page.getByTestId('editor')).toBeVisible();
});

test('show solution then hide restores the draft, and the draft survives reload', async ({ page }) => {
  await openLesson(page, 'hello-thread');
  const draft = '// my draft\n' + window_starter();
  await page.evaluate((d) => window.__cm!['hello-thread'].setText(d), draft);
  await page.getByRole('button', { name: /run on the gpu/i }).click();
  await expect(page.getByTestId('target-correct')).toHaveAttribute('data-pass', 'false', { timeout: 30_000 });
  await page.getByRole('button', { name: /show solution/i }).click();
  expect(await page.evaluate(() => window.__cm!['hello-thread'].getText())).toBe(await page.evaluate(() => window.__exercise!.solution));
  await page.getByRole('button', { name: /back to my code/i }).click();
  expect(await page.evaluate(() => window.__cm!['hello-thread'].getText())).toBe(draft);
  await page.reload();
  await page.waitForFunction(() => Boolean(window.__cm?.['hello-thread']));
  expect(await page.evaluate(() => window.__cm!['hello-thread'].getText())).toBe(draft);
});

function window_starter(): string {
  return '__global__ void vector_add(const float* a, const float* b, float* y, int n) {\n  int i = 0;\n  y[i] = a[i] + b[i];\n}\n';
}

test('results are isolated as LTR on the Arabic page', async ({ page }) => {
  await page.goto('ar/learn/hello-thread/');
  await page.waitForFunction(() => Boolean(window.__cm?.['hello-thread']));
  await expect(page.getByTestId('run-results')).toHaveAttribute('dir', 'ltr');
});
