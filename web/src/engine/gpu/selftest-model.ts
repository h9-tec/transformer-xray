/** Model-level parity: the GPU forward pass against the CPU engine on the
 * real checkpoint (the e2e smoke prompt), and later one training step. */

import { forward } from '@/engine/gpt';
import { loadModel } from '@/lib/loadModel';
import type { Gpu } from './device';
import { gpuForward } from './forward';
import { uploadModel } from './model';
import { maxAbsDiff, type GpuReport } from './selftest';

const PROMPT = 'ROMEO:\nFor nothing marry, ';

export async function extendReport(gpu: Gpu, report: GpuReport, level: 'forward' | 'train'): Promise<GpuReport> {
  const m = await loadModel();
  const ids = m.tokenizer.encode(PROMPT);
  const cpu = forward(m.weights, m.cfg, ids).logits.data;
  const model = uploadModel(gpu, m.weights, m.cfg);
  try {
    const { logits, ms } = await gpuForward(gpu, model, m.cfg, ids);
    report.forwardMaxDiff = maxAbsDiff(logits, cpu);
    report.kernels = { ...report.kernels, forward_pass: { maxDiff: report.forwardMaxDiff, ms } };
    const V = m.cfg.vocabSize;
    let match = true;
    for (let t = 0; t < ids.length && match; t++) {
      const a = argmax(logits.subarray(t * V, (t + 1) * V));
      const b = argmax(cpu.subarray(t * V, (t + 1) * V));
      if (a !== b) match = false;
    }
    report.argmaxMatch = match;
    if (level === 'train') report.reason = 'train-level parity is not implemented yet';
  } finally {
    model.destroy();
  }
  return report;
}

function argmax(x: ArrayLike<number>): number {
  let bi = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[bi]) bi = i;
  return bi;
}
