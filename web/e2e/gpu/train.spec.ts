import { expect, test } from '@playwright/test';
import { report } from './helpers';

test('one GPU training step matches the CPU engine (params and loss)', async ({ page }) => {
  const r = await report(page, 'train');
  test.skip(r.adapter === null, r.reason ?? 'no WebGPU adapter in this browser');
  expect(r.trainMaxDiff, r.reason).toBeDefined();
  expect(r.trainMaxDiff!).toBeLessThan(5e-3);
  expect(Math.abs((r.lossGpu ?? 0) - (r.lossCpu ?? 1))).toBeLessThan(1e-3);
});

test('training on your GPU lowers the loss and the x-ray sees the trained weights', async ({ page }) => {
  await page.goto('en/train/');
  await page.waitForFunction(() => window.__stores?.model.getState().status === 'ready', { timeout: 60_000 });
  const gpuButton = page.getByRole('button', { name: /your GPU/i });
  test.skip(!(await gpuButton.isEnabled()), 'no WebGPU adapter in this browser');
  await gpuButton.click();
  await expect(page.getByTestId('backend-active')).toHaveText(/gpu/i, { timeout: 30_000 });
  await page.getByRole('radio', { name: /random noise/i }).click();
  await page.getByRole('button', { name: /start training/i }).click();
  const stepText = page.getByTestId('train-step');
  await expect(stepText).toHaveText(/\b(2\d|[3-9]\d)\b/, { timeout: 90_000 });
  await page.getByRole('button', { name: /pause/i }).click();
  const losses = await page.evaluate(() => window.__stores.train.getState().lossHistory as number[]);
  expect(losses.length).toBeGreaterThan(15);
  expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
  await page.getByRole('button', { name: /x-ray/i }).click();
  await page.waitForFunction(() => window.__stores?.trace.getState().status === 'ready', { timeout: 60_000 });
  // The inspector traces a different text than the training batch, so the
  // losses need not match; what must hold is that the trace used the
  // GPU-trained weights, not the untrained CPU copy (whose loss is ~ln 65).
  const traceLoss = await page.evaluate(() => window.__stores.trace.getState().loss);
  expect(traceLoss).not.toBeNull();
  expect(traceLoss!).toBeLessThan(losses[0] - 0.3);
});
