/** Model-level parity: the GPU forward pass against the CPU engine on the
 * real checkpoint, and one training step (params + loss) on the same batch. */

import { adamwInit } from '@/engine/adamw';
import { forward, toMutableParams, trainStep } from '@/engine/gpt';
import { loadModel } from '@/lib/loadModel';
import type { Gpu } from './device';
import { gpuForward } from './forward';
import { uploadModel } from './model';
import { maxAbsDiff, type GpuReport } from './selftest';
import { GpuTrainer } from './train';

const PROMPT = 'ROMEO:\nFor nothing marry, ';
const HP = { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 };

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
      if (argmax(logits.subarray(t * V, (t + 1) * V)) !== argmax(cpu.subarray(t * V, (t + 1) * V))) match = false;
    }
    report.argmaxMatch = match;
  } finally {
    model.destroy();
  }
  if (level !== 'train') return report;

  // one step, identical init and batch, on both engines
  const text = 'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n';
  const seq = m.tokenizer.encode(text);
  const T = 48;
  const x = seq.subarray(0, T);
  const y = seq.subarray(1, T + 1);
  const cpuParams = toMutableParams(m.weights);
  const res = trainStep(cpuParams, m.cfg, x, y, adamwInit(), HP);
  const gpuModel = uploadModel(gpu, toMutableParams(m.weights), m.cfg);
  const trainer = new GpuTrainer(gpu, gpuModel, m.cfg, HP);
  try {
    const t0 = performance.now();
    const { loss } = await trainer.step(x, y);
    const after = await gpuModel.download();
    let worst = 0;
    for (const [name, t] of cpuParams) {
      const d = maxAbsDiff(after.get(name)!.data, t.data);
      if (Number.isNaN(d) || d > worst) worst = d;
    }
    report.trainMaxDiff = worst;
    report.lossGpu = loss;
    report.lossCpu = res.loss;
    report.kernels = { ...report.kernels, train_step: { maxDiff: worst, ms: performance.now() - t0 } };
  } finally {
    trainer.destroy();
    gpuModel.destroy();
  }
  return report;
}

function argmax(x: ArrayLike<number>): number {
  let bi = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[bi]) bi = i;
  return bi;
}
