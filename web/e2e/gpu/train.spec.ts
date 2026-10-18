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

async function greedy(page: import('@playwright/test').Page): Promise<string> {
  const stream = page.locator('[aria-label^="Generated text"]');
  await page.getByRole('button', { name: 'generate' }).click();
  await expect(stream.locator('button').nth(49)).toBeVisible({ timeout: 90_000 });
  return (await stream.textContent()) ?? '';
}

test('weights trained on the GPU are what the CPU generates with after switching', async ({ page }) => {
  await page.goto('en/train/');
  await page.waitForFunction(() => window.__stores?.model.getState().status === 'ready', { timeout: 60_000 });
  const gpuButton = page.getByRole('button', { name: /your GPU/i });
  test.skip(!(await gpuButton.isEnabled()), 'no WebGPU adapter in this browser');
  await gpuButton.click();
  await expect(page.getByTestId('backend-active')).toHaveText(/gpu/i, { timeout: 30_000 });
  await page.getByRole('radio', { name: /random noise/i }).click();
  await page.getByRole('button', { name: /start training/i }).click();
  await expect(page.getByTestId('train-step')).toHaveText(/\b(1\d|[2-9]\d)\b/, { timeout: 90_000 });
  await page.getByRole('button', { name: /pause/i }).click();
  // x-ray on the GPU path, then switch to the CPU and x-ray again: same weights, same loss
  await page.getByRole('button', { name: /x-ray/i }).click();
  await page.waitForFunction(() => window.__stores?.trace.getState().status === 'ready', { timeout: 60_000 });
  const lossGpu = await page.evaluate(() => window.__stores.trace.getState().loss);
  await page.getByRole('button', { name: /^CPU$/ }).click();
  await expect(page.getByTestId('backend-active')).toHaveText(/cpu/i, { timeout: 30_000 });
  await page.getByRole('button', { name: /x-ray/i }).click();
  await page.waitForFunction(() => window.__stores?.trace.getState().status === 'ready', { timeout: 60_000 });
  const lossCpu = await page.evaluate(() => window.__stores.trace.getState().loss);
  expect(Math.abs((lossCpu ?? 0) - (lossGpu ?? 1))).toBeLessThan(1e-3);
});

test('weights trained on the CPU are what the GPU generates with after switching', async ({ page }) => {
  await page.goto('en/train/');
  await page.waitForFunction(() => window.__stores?.model.getState().status === 'ready', { timeout: 60_000 });
  const gpuButton = page.getByRole('button', { name: /your GPU/i });
  test.skip(!(await gpuButton.isEnabled()), 'no WebGPU adapter in this browser');
  // touch the GPU once (uploads the checkpoint), then train on the CPU
  await gpuButton.click();
  await expect(page.getByTestId('backend-active')).toHaveText(/gpu/i, { timeout: 30_000 });
  await page.getByRole('button', { name: /^CPU$/ }).click();
  await expect(page.getByTestId('backend-active')).toHaveText(/cpu/i, { timeout: 30_000 });
  await page.getByRole('radio', { name: /random noise/i }).click();
  await page.getByRole('button', { name: /start training/i }).click();
  await expect(page.getByTestId('train-step')).toHaveText(/\b(1\d|[2-9]\d)\b/, { timeout: 90_000 });
  await page.getByRole('button', { name: /pause/i }).click();
  await page.getByRole('button', { name: /x-ray/i }).click();
  await page.waitForFunction(() => window.__stores?.trace.getState().status === 'ready', { timeout: 60_000 });
  const lossCpu = await page.evaluate(() => window.__stores.trace.getState().loss);
  await gpuButton.click();
  await expect(page.getByTestId('backend-active')).toHaveText(/gpu/i, { timeout: 30_000 });
  // the x-ray after the switch reads the GPU copy: it must be the CPU-trained weights
  await page.getByRole('button', { name: /x-ray/i }).click();
  await page.waitForFunction(() => window.__stores?.trace.getState().status === 'ready', { timeout: 60_000 });
  const lossGpu = await page.evaluate(() => window.__stores.trace.getState().loss);
  expect(lossCpu).not.toBeNull();
  expect(Math.abs((lossGpu ?? 0) - lossCpu!), `cpu ${lossCpu} gpu ${lossGpu}`).toBeLessThan(1e-3);
  // and training continues from there: a few GPU steps stay in the trained regime
  const before = await page.evaluate(() => window.__stores.train.getState().lossHistory.length);
  await page.getByRole('button', { name: /resume/i }).click();
  await page.waitForFunction((n) => window.__stores.train.getState().lossHistory.length > n + 4, before, { timeout: 90_000 });
  await page.getByRole('button', { name: /pause/i }).click();
  const hist = await page.evaluate(() => window.__stores.train.getState().lossHistory as number[]);
  const tail = hist.slice(before, before + 5);
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  expect(mean, `history ${hist.map((x) => x.toFixed(2)).join(' ')}`).toBeLessThan(4.1);
});
