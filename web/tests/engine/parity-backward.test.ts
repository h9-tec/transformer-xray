/** Backward parity: every activation gradient and every parameter gradient
 * must match PyTorch autograd within 5e-4 (gradient chains accumulate more
 * fp32 noise than the forward pass). */

import { describe, expect, it } from 'vitest';
import { forward, lossBackward } from '@/engine/gpt';
import { Tape } from '@/engine/tape';
import { CharTokenizer } from '@/engine/tokenizer';
import { loadArtifacts, loadGolden, maxAbsDiff } from './helpers';

const TOL = 5e-4;

// PyTorch retains grads for tensors our fused/closed-form backward never
// materializes; they carry no parity information of their own.
const SKIP = [/^grad:.*\.(mean|rstd)$/, /^grad:loss$/, /^grad:loss\.per_token$/];

describe.each(['t4', 't64'] as const)('backward parity (%s)', (name) => {
  const { cfg, weights, chars } = loadArtifacts();
  const golden = loadGolden(name);
  const tok = new CharTokenizer(chars);

  const tape = new Tape();
  const ctx = forward(weights, cfg, tok.encode(golden.meta.x), tape);
  const res = lossBackward(weights, cfg, ctx, tok.encode(golden.meta.y), tape);

  const gradNames = golden.meta.tensors
    .map((t) => t.name)
    .filter((n) => n.startsWith('grad:') && !n.startsWith('grad:param:'))
    .filter((n) => !SKIP.some((re) => re.test(n)));

  it.each(gradNames)('%s', (tensorName) => {
    const node = tape.get(tensorName);
    expect(node, `no trace node for golden grad ${tensorName}`).toBeDefined();
    const g = golden.tensors.get(tensorName)!;
    expect(node!.shape).toEqual(g.shape);
    expect(maxAbsDiff(node!.data!, g.data)).toBeLessThan(TOL);
  });

  const paramGradNames = golden.meta.tensors
    .map((t) => t.name)
    .filter((n) => n.startsWith('grad:param:'));

  if (paramGradNames.length > 0) {
    it.each(paramGradNames)('%s', (tensorName) => {
      const paramName = tensorName.slice('grad:param:'.length);
      const got = res.grads.get(paramName);
      expect(got, `no computed grad for param ${paramName}`).toBeDefined();
      const g = golden.tensors.get(tensorName)!;
      expect(maxAbsDiff(got!, g.data)).toBeLessThan(TOL);
    });
  }
});
