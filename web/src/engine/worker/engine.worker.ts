/** The engine's home: all heavy math runs here, off the main thread.
 * Holds two parameter sets: the pretrained checkpoint (read-only) and,
 * once training starts, a live mutable copy — generate/trace always use
 * the live one if it exists, so the inspectors can x-ray a model mid-training. */

import { adamwInit, type AdamWState } from '../adamw';
import { forward, lossBackward, toMutableParams, trainStep, type Weights } from '../gpt';
import { initWeights } from '../init';
import { generateKV } from '../kvcache';
import { mulberry32, type Rng } from '../rng';
import { Tape, type TraceNode } from '../tape';
import { CharTokenizer } from '../tokenizer';
import { configFromJson, parseBlob, type Config, type ModelConfigJson } from '../weights';
import type { TrainParams, WireNode, WorkerRequest, WorkerResponse } from './protocol';

let pretrained: Weights | null = null;
let cfg: Config | null = null;
let tokenizer: CharTokenizer | null = null;

let trainParams: Weights | null = null;
let adamState: AdamWState | null = null;
let corpusIds: Int32Array | null = null;
let trainRng: Rng = mulberry32(0);
let trainStepCount = 0;
let stopRequested = false;
let training = false;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(msg, transfer);
}

function active(): Weights {
  if (trainParams) return trainParams;
  if (!pretrained) throw new Error('worker not initialized');
  return pretrained;
}

function handleInit(id: number, cfgJson: ModelConfigJson, chars: string, buf: ArrayBuffer): void {
  cfg = configFromJson(cfgJson);
  pretrained = parseBlob(buf, cfgJson.tensors);
  tokenizer = new CharTokenizer(chars);
  post({ id, type: 'ready', cfgJson, chars });
}

function handleGenerate(id: number, p: WorkerRequest & { type: 'generate' }): void {
  if (!cfg || !tokenizer) throw new Error('worker not initialized');
  const { prompt, maxNewTokens, temperature, topK, seed } = p.params;
  let index = 0;
  const out = generateKV(
    active(),
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
  if (!cfg || !tokenizer) throw new Error('worker not initialized');
  const weights = active();
  const tape = new Tape();
  const ctx = forward(weights, cfg, tokenizer.encode(text), tape);
  let loss: number | undefined;
  const paramGrads: { name: string; shape: number[]; buffer: ArrayBuffer }[] = [];
  if (targets !== undefined) {
    const res = lossBackward(weights, cfg, ctx, tokenizer.encode(targets), tape);
    loss = res.loss;
    for (const [name, g] of res.grads) {
      const copy = g.slice();
      paramGrads.push({
        name,
        shape: [...(weights.get(name)?.shape ?? [g.length])],
        buffer: copy.buffer as ArrayBuffer,
      });
    }
  }
  const wires = tape.nodes.map(toWire);
  post(
    {
      id,
      type: 'traced',
      nodes: wires.map((w) => w.node),
      loss,
      paramGrads,
      trainedParams: trainParams !== null,
    },
    [
      ...wires.flatMap((w) => (w.transfer ? [w.transfer] : [])),
      ...paramGrads.map((g) => g.buffer),
    ],
  );
}

function handleTrainInit(
  id: number,
  fromScratch: boolean,
  seed: number,
  corpus: string,
): void {
  if (!cfg || !tokenizer || !pretrained) throw new Error('worker not initialized');
  trainRng = mulberry32(seed);
  trainParams = fromScratch ? initWeights(cfg, trainRng) : toMutableParams(pretrained);
  adamState = adamwInit();
  corpusIds = tokenizer.encode(corpus);
  trainStepCount = 0;
  post({ id, type: 'trainReady', fromScratch });
}

async function handleTrainRun(id: number, p: TrainParams): Promise<void> {
  if (!cfg || !tokenizer || !trainParams || !adamState || !corpusIds) {
    throw new Error('call trainInit first');
  }
  if (training) throw new Error('already training');
  training = true;
  stopRequested = false;
  const T = Math.min(p.seqLen, cfg.blockSize);
  const hp = { lr: p.lr, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 };

  for (let i = 0; i < p.steps; i++) {
    if (stopRequested) break;
    const start = Math.floor(trainRng() * (corpusIds.length - T - 1));
    const x = corpusIds.subarray(start, start + T);
    const y = corpusIds.subarray(start + 1, start + T + 1);
    const res = trainStep(trainParams, cfg, x, y, adamState, hp);
    trainStepCount++;

    let sample: string | undefined;
    if (p.sampleEvery > 0 && trainStepCount % p.sampleEvery === 0) {
      const out = generateKV(trainParams, cfg, tokenizer.encode('\n'), 80, {
        temperature: 0.8,
        topK: 40,
        rng: mulberry32(trainStepCount),
      });
      sample = tokenizer.decode(out);
    }
    post({ id, type: 'trainProgress', step: trainStepCount, loss: res.loss, sample });
    // Yield to the event loop so trainStop can interleave.
    await new Promise((r) => setTimeout(r, 0));
  }
  training = false;
  post({ id, type: 'trainDone', step: trainStepCount });
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init':
        handleInit(msg.id, msg.cfgJson, msg.chars, msg.weightsBuffer);
        break;
      case 'generate':
        handleGenerate(msg.id, msg);
        break;
      case 'trace':
        handleTrace(msg.id, msg.text, msg.targets);
        break;
      case 'trainInit':
        handleTrainInit(msg.id, msg.fromScratch, msg.seed, msg.corpus);
        break;
      case 'trainRun':
        await handleTrainRun(msg.id, msg.params);
        break;
      case 'trainStop':
        stopRequested = true;
        break;
      case 'trainReset':
        trainParams = null;
        adamState = null;
        trainStepCount = 0;
        break;
    }
  } catch (err) {
    post({ id: msg.id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
