import { expect, test } from '@playwright/test';
import { report } from './helpers';

test('the GPU forward pass matches the CPU engine on the real weights', async ({ page }) => {
  const r = await report(page, 'forward');
  test.skip(r.adapter === null, r.reason ?? 'no WebGPU adapter in this browser');
  expect(r.forwardMaxDiff, r.reason).toBeDefined();
  expect(r.forwardMaxDiff!).toBeLessThan(1e-3);
  expect(r.argmaxMatch).toBe(true);
});
