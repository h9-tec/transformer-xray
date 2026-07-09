/** Optimizer parity: after exactly one AdamW step on the T=4 gradients,
 * every parameter must match PyTorch within 1e-5. */

import { describe, expect, it } from 'vitest';
import { adamwInit, adamwStep } from '@/engine/adamw';
import { forward, lossBackward, toMutableParams } from '@/engine/gpt';
import { CharTokenizer } from '@/engine/tokenizer';
import { loadArtifacts, loadGolden, maxAbsDiff } from './helpers';

const TOL = 1e-5;

describe('adamw step parity (t4)', () => {
  const { cfg, weights, chars } = loadArtifacts();
  const golden = loadGolden('t4');
  const tok = new CharTokenizer(chars);
  const hp = golden.meta.adamw!;

  const params = toMutableParams(weights);
  const ctx = forward(params, cfg, tok.encode(golden.meta.x));
  const res = lossBackward(params, cfg, ctx, tok.encode(golden.meta.y));
  adamwStep(params, res.grads, adamwInit(), {
    lr: hp.lr,
    beta1: hp.beta1,
    beta2: hp.beta2,
    eps: hp.eps,
    weightDecay: hp.weight_decay,
  });

  const stepped = golden.meta.tensors.map((t) => t.name).filter((n) => n.startsWith('step:param:'));

  it('golden includes stepped params', () => {
    expect(stepped.length).toBeGreaterThan(0);
  });

  it.each(stepped)('%s', (tensorName) => {
    const paramName = tensorName.slice('step:param:'.length);
    const got = params.get(paramName);
    expect(got, `no param ${paramName}`).toBeDefined();
    const g = golden.tensors.get(tensorName)!;
    expect(maxAbsDiff(got!.data, g.data)).toBeLessThan(TOL);
  });
});
