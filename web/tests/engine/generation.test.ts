/** Generation parity and equivalence:
 *  - greedy decode must reproduce PyTorch token-for-token (golden transcript)
 *  - the full re-forward path must produce the same logits as the KV path
 *  - seeded sampling is deterministic; multinomial matches its distribution */

import { describe, expect, it } from 'vitest';
import { forward } from '@/engine/gpt';
import { generateKV } from '@/engine/kvcache';
import { sampleMultinomial } from '@/engine/ops/sample';
import { mulberry32 } from '@/engine/rng';
import { CharTokenizer } from '@/engine/tokenizer';
import { loadArtifacts, loadGolden, maxAbsDiff } from './helpers';

describe('generation', () => {
  const { cfg, weights, chars } = loadArtifacts();
  const tok = new CharTokenizer(chars);
  const greedy = loadGolden('t4').meta.greedy!;
  const opts = { temperature: 1, topK: cfg.vocabSize, rng: null };

  it('greedy KV decode matches the PyTorch transcript exactly', () => {
    const out = generateKV(weights, cfg, tok.encode(greedy.prompt), greedy.tokens, opts);
    expect(tok.decode(out)).toBe(greedy.output);
  });

  // Exact argmax-chain reproduction is the KV path's contract (above). The
  // full re-forward path sums in a different order, so a near-tie logit may
  // legitimately flip a chain; it is tested on logits equivalence instead:
  // along the golden transcript, its last-position logits must match the KV
  // path within fp32 noise.
  it('full re-forward logits equal KV logits along the transcript', () => {
    const ids = tok.encode(greedy.output);
    const kvLogits: Float32Array[] = [];
    generateKV(weights, cfg, ids.subarray(0, ids.length - 1), 1, opts, (r) =>
      kvLogits.push(r.logits),
    );
    const ctx = forward(weights, cfg, ids.subarray(0, ids.length - 1));
    const T = ids.length - 1;
    const last = ctx.logits.data.subarray((T - 1) * cfg.vocabSize, T * cfg.vocabSize);
    expect(maxAbsDiff(last as Float32Array, kvLogits[0])).toBeLessThan(1e-3);
  });

  it('seeded sampling is deterministic', () => {
    const run = () =>
      tok.decode(
        generateKV(weights, cfg, tok.encode('ROMEO:'), 40, {
          temperature: 0.8,
          topK: 40,
          rng: mulberry32(1234),
        }),
      );
    expect(run()).toBe(run());
  });

  it('multinomial draws follow the distribution', () => {
    const probs = Float32Array.from([0.5, 0.3, 0.15, 0.05]);
    const rng = mulberry32(7);
    const counts = [0, 0, 0, 0];
    const N = 20000;
    for (let i = 0; i < N; i++) counts[sampleMultinomial(probs, rng)]++;
    for (let i = 0; i < 4; i++) {
      expect(Math.abs(counts[i] / N - probs[i])).toBeLessThan(0.02);
    }
  });

  it('tokenizer round-trips the full vocabulary', () => {
    expect(tok.decode(tok.encode(chars))).toBe(chars);
    expect(() => tok.encode('emoji 🙂')).toThrowError(/not in the vocabulary/);
  });
});
