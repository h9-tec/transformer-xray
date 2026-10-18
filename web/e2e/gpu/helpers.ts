import { expect } from '@playwright/test';

/** Shared by the gpu-project specs: fetch the debug page's JSON report. */

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

