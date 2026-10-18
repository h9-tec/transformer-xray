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

test('landing has no horizontal scroll at phone width', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('en/');
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBe(true);
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
