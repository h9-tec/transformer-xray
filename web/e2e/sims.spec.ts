import { expect, test } from '@playwright/test';

const SLUGS = [
  'hello-thread',
  'bounds-grid-stride',
  'warps',
  'memory-hierarchy',
  'coalescing',
  'shared-memory-barriers',
  'bank-conflicts',
  'tiled-matmul',
  'reductions-shuffles',
  'atomics',
  'fusing-attention',
  'roofline-float4',
];

for (const slug of SLUGS) {
  test(`lesson ${slug} renders its simulator`, async ({ page }) => {
    await page.goto(`en/learn/${slug}/`);
    const sim = page.getByTestId('sim');
    await expect(sim).toBeVisible();
    await expect(sim.locator('canvas, svg').first()).toBeVisible();
  });
}

test('the bank-conflict simulator reacts to padding', async ({ page }) => {
  await page.goto('en/learn/bank-conflicts/');
  const worst = page.getByTestId('sim-worst');
  await expect(worst).toHaveText('32');
  await page.getByLabel(/padding/i).fill('1');
  await expect(worst).toHaveText('1');
});
