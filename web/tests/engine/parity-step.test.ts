/** Optimizer parity, isolated: feed AdamW *PyTorch's own* gradients (from the
 * golden trace) so the comparison tests exactly one thing — the update rule,
 * its order of operations, and bias correction. With identical inputs the
 * only residual is fp32 arithmetic, so the tolerance is tight (5e-6).
 *
 * (Gradient correctness is parity-backward's job; composing the two in one
 * test made near-zero grads ill-conditioned — m̂/√v̂ normalizes any tiny
 * gradient to ±1, amplifying sub-tolerance grad noise into full-size
 * update flips.) */

import { describe, expect, it } from 'vitest';
import { adamwInit, adamwStep } from '@/engine/adamw';
import { toMutableParams } from '@/engine/gpt';
import { loadArtifacts, loadGolden, maxAbsDiff } from './helpers';

const TOL = 5e-6;

describe('adamw step parity (t4, golden grads)', () => {
  const { weights } = loadArtifacts();
  const golden = loadGolden('t4');
  const hp = golden.meta.adamw!;

  const goldenGrads = new Map<string, Float32Array>();
  for (const t of golden.meta.tensors) {
    if (t.name.startsWith('grad:param:')) {
      goldenGrads.set(t.name.slice('grad:param:'.length), golden.tensors.get(t.name)!.data);
    }
  }

  const params = toMutableParams(weights);
  adamwStep(params, goldenGrads, adamwInit(), {
    lr: hp.lr,
    beta1: hp.beta1,
    beta2: hp.beta2,
    eps: hp.eps,
    weightDecay: hp.weight_decay,
  });

  const stepped = golden.meta.tensors.map((t) => t.name).filter((n) => n.startsWith('step:param:'));

  it('golden includes stepped params and grads', () => {
    expect(stepped.length).toBeGreaterThan(0);
    expect(goldenGrads.size).toBe(stepped.length);
  });

  it.each(stepped)('%s', (tensorName) => {
    const paramName = tensorName.slice('step:param:'.length);
    const got = params.get(paramName);
    expect(got, `no param ${paramName}`).toBeDefined();
    const g = golden.tensors.get(tensorName)!;
    expect(maxAbsDiff(got!.data, g.data)).toBeLessThan(TOL);
  });
});
