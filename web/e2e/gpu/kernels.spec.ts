import { expect, test } from '@playwright/test';

/** Runs only in the `gpu` Playwright project (Chromium with WebGPU flags).
 * Skips, visibly, when the browser has no adapter. */

export interface GpuReport {
  adapter: string | null;
  reason?: string;
  kernels?: Record<string, { maxDiff: number; ms: number }>;
  forwardMaxDiff?: number;
  argmaxMatch?: boolean;
  trainMaxDiff?: number;
  lossGpu?: number;
  lossCpu?: number;
}

export async function report(page: import('@playwright/test').Page, level: string): Promise<GpuReport> {
  await page.goto(`en/debug/gpu/?level=${level}`);
  const pre = page.getByTestId('gpu-report');
  await expect(pre).toHaveAttribute('data-done', 'true', { timeout: 120_000 });
  return JSON.parse((await pre.textContent()) ?? '{}') as GpuReport;
}

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
