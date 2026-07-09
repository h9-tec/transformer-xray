/** Trace invariants: node counts follow the closed form, every provenance
 * ref resolves, every backward node pairs with its forward twin. */

import { describe, expect, it } from 'vitest';
import { forward, lossBackward } from '@/engine/gpt';
import { Tape } from '@/engine/tape';
import { CharTokenizer } from '@/engine/tokenizer';
import { loadArtifacts } from './helpers';

describe('trace invariants', () => {
  const { cfg, weights, chars } = loadArtifacts();
  const tok = new CharTokenizer(chars);
  const tape = new Tape();
  const ctx = forward(weights, cfg, tok.encode('ROME'), tape);
  lossBackward(weights, cfg, ctx, tok.encode('OMEO'), tape);

  it('node counts match the closed form', () => {
    const fwd = tape.nodes.filter((n) => n.phase === 'forward').length;
    const bwd = tape.nodes.filter((n) => n.phase === 'backward').length;
    // forward: 3 emb + 21/block + 3 final LN + 1 logits + 2 loss
    expect(fwd).toBe(3 + 21 * cfg.nLayer + 3 + 1 + 2);
    // backward: 2 head-side + 17/block + 3 embedding-side
    expect(bwd).toBe(2 + 17 * cfg.nLayer + 3);
  });

  it('every inputs ref resolves to an earlier node or a weight', () => {
    const seen = new Set<string>();
    for (const node of tape.nodes) {
      for (const ref of node.inputs) {
        if (ref.node) {
          expect(seen.has(ref.node), `${node.id} references ${ref.node} before it exists`).toBe(
            true,
          );
        }
        if (ref.weight) expect(weights.has(ref.weight), `unknown weight ${ref.weight}`).toBe(true);
      }
      seen.add(node.id);
    }
  });

  it('every backward node pairs with an existing forward node', () => {
    for (const node of tape.nodes) {
      if (node.phase !== 'backward') continue;
      expect(node.gradOf, `${node.id} missing gradOf`).toBeDefined();
      const twin = tape.get(node.gradOf!);
      expect(twin, `${node.id} → missing twin ${node.gradOf}`).toBeDefined();
      expect(twin!.phase).toBe('forward');
      if (node.data && twin!.data) expect(node.shape).toEqual(twin!.shape);
    }
  });

  it('stats are computed and finite-aware', () => {
    const masked = tape.get('block0.attn.masked')!;
    expect(masked.stats.nonFinite).toBeGreaterThan(0); // the −∞ triangle
    expect(Number.isFinite(masked.stats.min)).toBe(true);
    const logits = tape.get('logits')!;
    expect(logits.stats.nonFinite).toBe(0);
    expect(logits.stats.absmax).toBeGreaterThan(0);
  });
});
