'use client';

/** M2 proof page: the engine generating and training-stepping in the browser.
 * Not a product surface — the playground replaces this for users. */

import { useEffect, useRef, useState } from 'react';
import { adamwInit } from '@/engine/adamw';
import { forward, lossBackward, toMutableParams, trainStep } from '@/engine/gpt';
import { generateKV } from '@/engine/kvcache';
import { mulberry32 } from '@/engine/rng';
import { Tape } from '@/engine/tape';
import { loadModel, type LoadedModel } from '@/lib/loadModel';

export default function DebugPage() {
  const model = useRef<LoadedModel | null>(null);
  const [status, setStatus] = useState('loading model…');
  const [output, setOutput] = useState('');
  const [lines, setLines] = useState<string[]>([]);

  useEffect(() => {
    loadModel()
      .then((m) => {
        model.current = m;
        setStatus(
          `loaded ${m.cfgJson.n_params.toLocaleString()} params | val loss ${m.cfgJson.val_loss.toFixed(4)}`,
        );
      })
      .catch((e) => setStatus(`load failed: ${e.message}`));
  }, []);

  const log = (s: string) => setLines((prev) => [...prev, s]);

  function generate() {
    const m = model.current;
    if (!m) return;
    const t0 = performance.now();
    const out = generateKV(m.weights, m.cfg, m.tokenizer.encode('ROMEO:'), 200, {
      temperature: 0.8,
      topK: 40,
      rng: mulberry32(Date.now() & 0xffff),
    });
    const ms = performance.now() - t0;
    setOutput(m.tokenizer.decode(out));
    log(`generateKV: 200 tokens in ${ms.toFixed(0)} ms (${((200 / ms) * 1000).toFixed(0)} tok/s)`);
  }

  function tracedForward() {
    const m = model.current;
    if (!m) return;
    const prompt = 'First Citizen:\nBefore we proceed any further, hear me speak.\nAll'.slice(0, 64);
    const x = m.tokenizer.encode(prompt);
    const y = m.tokenizer.encode(prompt); // dummy targets: same length
    const tape = new Tape();
    const t0 = performance.now();
    const ctx = forward(m.weights, m.cfg, x, tape);
    const res = lossBackward(m.weights, m.cfg, ctx, y, tape);
    const ms = performance.now() - t0;
    const bytes = tape.nodes.reduce((a, n) => a + (n.data?.byteLength ?? 0), 0);
    log(
      `traced fwd+bwd @T=${x.length}: ${tape.nodes.length} nodes, ${(bytes / 1e6).toFixed(1)} MB, ` +
        `loss ${res.loss.toFixed(4)}, ${ms.toFixed(0)} ms`,
    );
  }

  function oneTrainStep() {
    const m = model.current;
    if (!m) return;
    const text = 'O God, O God! that e\'er this tongue of mine';
    const x = m.tokenizer.encode(text.slice(0, -1));
    const y = m.tokenizer.encode(text.slice(1));
    const params = toMutableParams(m.weights);
    const t0 = performance.now();
    const res = trainStep(params, m.cfg, x, y, adamwInit(), {
      lr: 3e-4,
      beta1: 0.9,
      beta2: 0.95,
      eps: 1e-8,
      weightDecay: 0.1,
    });
    const ms = performance.now() - t0;
    log(`trainStep @T=${x.length}: loss ${res.loss.toFixed(4)}, ${ms.toFixed(0)} ms (fwd+bwd+adamw)`);
  }

  return (
    <main className="mx-auto max-w-3xl space-y-4 px-6 py-10">
      <h1 className="font-display text-2xl">engine debug</h1>
      <p className="nums text-sm text-muted">{status}</p>
      <div className="flex gap-3">
        <button onClick={generate} className="rounded border border-fwd px-4 py-1.5 text-fwd hover:bg-fwd hover:text-bg">
          generate
        </button>
        <button onClick={tracedForward} className="rounded border border-accent px-4 py-1.5 text-accent hover:bg-accent hover:text-bg">
          traced fwd+bwd
        </button>
        <button onClick={oneTrainStep} className="rounded border border-bwd px-4 py-1.5 text-bwd hover:bg-bwd hover:text-bg">
          train step
        </button>
      </div>
      <div className="nums space-y-1 text-xs text-muted">
        {lines.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
      {output && (
        <pre className="nums whitespace-pre-wrap rounded border border-line bg-surface p-4 text-sm text-fwd">
          {output}
        </pre>
      )}
    </main>
  );
}
