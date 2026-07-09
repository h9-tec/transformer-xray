/** The engine's home: all heavy math runs here, off the main thread. */

import { forward, lossBackward, type Weights } from '../gpt';
import { generateKV } from '../kvcache';
import { mulberry32 } from '../rng';
import { Tape, type TraceNode } from '../tape';
import { CharTokenizer } from '../tokenizer';
import { configFromJson, parseBlob, type Config, type ModelConfigJson } from '../weights';
import type { WireNode, WorkerRequest, WorkerResponse } from './protocol';

let weights: Weights | null = null;
let cfg: Config | null = null;
let tokenizer: CharTokenizer | null = null;

console.log('[engine.worker] booted');

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(msg, transfer);
}

function handleInit(id: number, cfgJson: ModelConfigJson, chars: string, buf: ArrayBuffer): void {
  cfg = configFromJson(cfgJson);
  weights = parseBlob(buf, cfgJson.tensors);
  tokenizer = new CharTokenizer(chars);
  post({ id, type: 'ready', cfgJson, chars });
}

function handleGenerate(id: number, p: WorkerRequest & { type: 'generate' }): void {
  if (!weights || !cfg || !tokenizer) throw new Error('worker not initialized');
  const { prompt, maxNewTokens, temperature, topK, seed } = p.params;
  let index = 0;
  const out = generateKV(
    weights,
    cfg,
    tokenizer.encode(prompt),
    maxNewTokens,
    { temperature, topK, rng: seed === null ? null : mulberry32(seed) },
    ({ token, logits }) => {
      const buf = logits.buffer as ArrayBuffer;
      post({ id, type: 'token', index: index++, token, logits: buf }, [buf]);
    },
  );
  post({ id, type: 'generated', text: tokenizer.decode(out) });
}

function toWire(n: TraceNode): { node: WireNode; transfer: ArrayBuffer | null } {
  const buffer = n.data ? (n.data.buffer as ArrayBuffer) : null;
  return {
    node: {
      id: n.id,
      step: n.step,
      phase: n.phase,
      kind: n.kind,
      layer: n.layer,
      head: n.head,
      gradOf: n.gradOf,
      inputs: n.inputs,
      shape: n.shape,
      buffer,
      meta: n.meta,
      stats: n.stats,
      costs: n.costs,
    },
    transfer: buffer,
  };
}

function handleTrace(id: number, text: string, targets?: string): void {
  if (!weights || !cfg || !tokenizer) throw new Error('worker not initialized');
  const tape = new Tape();
  const ctx = forward(weights, cfg, tokenizer.encode(text), tape);
  let loss: number | undefined;
  if (targets !== undefined) {
    loss = lossBackward(weights, cfg, ctx, tokenizer.encode(targets), tape).loss;
  }
  const wires = tape.nodes.map(toWire);
  post(
    { id, type: 'traced', nodes: wires.map((w) => w.node), loss },
    wires.flatMap((w) => (w.transfer ? [w.transfer] : [])),
  );
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') handleInit(msg.id, msg.cfgJson, msg.chars, msg.weightsBuffer);
    else if (msg.type === 'generate') handleGenerate(msg.id, msg);
    else if (msg.type === 'trace') handleTrace(msg.id, msg.text, msg.targets);
  } catch (err) {
    post({ id: msg.id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
