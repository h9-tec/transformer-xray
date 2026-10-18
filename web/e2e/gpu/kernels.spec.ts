import { expect, test } from '@playwright/test';
import { report } from './helpers';

/** Runs only in the `gpu` Playwright project (Chromium with WebGPU flags).
 * Skips, visibly, when the browser has no adapter. */

test('every forward kernel matches the TS engine op at the model shapes', async ({ page }) => {
  const r = await report(page, 'kernels');
  test.skip(r.adapter === null, r.reason ?? 'no WebGPU adapter in this browser');
  expect(r.kernels).toBeDefined();
  for (const [name, k] of Object.entries(r.kernels!)) {
    expect(k.maxDiff, name).toBeLessThan(1e-4);
  }
  expect(Object.keys(r.kernels!)).toEqual(
    expect.arrayContaining(['embedding_gather', 'layernorm_warp', 'matmul_tiled', 'matmul_tiled_TB', 'matmul_tiled_TA', 'causal_softmax', 'gelu_forward', 'residual_add', 'split_heads', 'merge_heads']),
  );
});
