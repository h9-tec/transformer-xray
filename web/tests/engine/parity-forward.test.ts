/** The credibility cornerstone: every forward tensor must match PyTorch
 * within 1e-4. Walks the golden manifest and compares node-for-node. */

import { describe, expect, it } from 'vitest';
import { forward, lossBackward } from '@/engine/gpt';
import { Tape } from '@/engine/tape';
import { CharTokenizer } from '@/engine/tokenizer';
import { loadArtifacts, loadGolden, maxAbsDiff } from './helpers';

const TOL = 1e-4;

describe.each(['t4', 't64'] as const)('forward parity (%s)', (name) => {
  const { cfg, weights, chars } = loadArtifacts();
  const golden = loadGolden(name);
  const tok = new CharTokenizer(chars);

  const tape = new Tape();
  const ctx = forward(weights, cfg, tok.encode(golden.meta.x), tape);
  const res = lossBackward(weights, cfg, ctx, tok.encode(golden.meta.y), tape);

  it('loss matches', () => {
    expect(Math.abs(res.loss - golden.meta.loss)).toBeLessThan(1e-5);
  });

  const forwardNames = golden.meta.tensors
    .map((t) => t.name)
    .filter((n) => !n.startsWith('grad:') && !n.startsWith('step:'));

  it.each(forwardNames)('%s', (tensorName) => {
    const node = tape.get(tensorName);
    expect(node, `no trace node for golden tensor ${tensorName}`).toBeDefined();
    const g = golden.tensors.get(tensorName)!;
    expect(node!.data, `node ${tensorName} has no data`).not.toBeNull();
    expect(node!.shape).toEqual(g.shape);
    expect(maxAbsDiff(node!.data!, g.data)).toBeLessThan(TOL);
  });
});
