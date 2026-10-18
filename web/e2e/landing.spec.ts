import { expect, test } from '@playwright/test';

test('landing shows the 12-lesson map linking to /learn pages', async ({ page }) => {
  await page.goto('en/');
  const tiles = page.getByRole('list', { name: 'Lesson map' }).getByRole('link');
  await expect(tiles).toHaveCount(12);
  await expect(tiles.first()).toHaveAttribute('href', /\/en\/learn\/hello-thread\/$/);
});

test('warp instrument recomputes sectors per access pattern', async ({ page }) => {
  await page.goto('en/');
  const sectors = page.getByTestId('sectors');
  await expect(sectors).toHaveText('4');
  await page.getByRole('button', { name: /column/ }).click();
  await expect(sectors).toHaveText('32');
});

for (const path of ['en/', 'en/learn/', 'en/learn/tiled-matmul/', 'en/playground/', 'ar/']) {
  test(`no horizontal scroll at 360px on ${path}`, async ({ page }) => {
    await page.setViewportSize({ width: 360, height: 780 });
    await page.goto(path);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true);
    await expect(page.getByRole('navigation', { name: 'Primary' })).toBeVisible();
  });
}

test('warp instrument draws exactly as many sector cells as it reports', async ({ page }) => {
  await page.goto('en/');
  await page.getByRole('button', { name: /column/ }).click();
  await expect(page.getByTestId('sectors')).toHaveText('32');
  await expect(page.getByTestId('warp-canvas')).toHaveAttribute('data-cells', '32');
  await page.getByRole('button', { name: /scattered/ }).click();
  const n = await page.getByTestId('sectors').textContent();
  await expect(page.getByTestId('warp-canvas')).toHaveAttribute('data-cells', n ?? '');
});

test('landing says plainly what opens with the next release', async ({ page }) => {
  await page.goto('en/');
  await expect(page.getByText(/opens? with the next release/i).first()).toBeVisible();
});

test.describe('reduced motion', () => {
  test.use({ contextOptions: { reducedMotion: 'reduce' } });
  test('still renders a populated instrument', async ({ page }) => {
    await page.goto('en/');
    await expect(page.getByTestId('sectors')).toHaveText('4');
  });
});

test('landing carries the primary navigation and a side gutter', async ({ page }) => {
  await page.goto('en/');
  await expect(page.getByRole('navigation', { name: 'Primary' })).toBeVisible();
  const left = await page.getByRole('heading', { level: 1 }).evaluate((el) => el.getBoundingClientRect().left);
  expect(left).toBeGreaterThanOrEqual(16);
});
