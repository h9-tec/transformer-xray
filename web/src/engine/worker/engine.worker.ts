/** The engine's home: all heavy math runs here, off the main thread.
 * Holds two parameter sets: the pretrained checkpoint (read-only) and,
 * once training starts, a live mutable copy — generate/trace always use
 * the live one if it exists, so the inspectors can x-ray a model mid-training. */

import { adamwInit, type AdamWState } from '../adamw';
import { backendAfterLoss, chooseBackend, type BackendName, type BackendStatus } from '../backend';
import { getGpu, gpuUnavailableReason, resetGpu, type Gpu } from '../gpu/device';
import { gpuGenerate } from '../gpu/forward';
import { uploadModel, type GpuModel } from '../gpu/model';
import { GpuTrainer } from '../gpu/train';
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

let gpu: Gpu | null = null;
let gpuModel: GpuModel | null = null;
let gpuTrainer: GpuTrainer | null = null;

/** Live params for trace/generate: GPU-trained weights are downloaded first. */
async function liveWeights(): Promise<Weights> {
  if (gpuTrainer) {
    trainParams = await gpuTrainer.download();
    return trainParams;
  }
  return active();
}
let backend: BackendStatus = { active: 'cpu' };

const HP = { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 };

/** The GPU is gone: drop every GPU object and continue on the CPU with
 * whatever weights the CPU copy holds (GPU training progress since the last
 * sync is lost, and the reason says so). */
function onGpuLost(info: string): void {
  const hadTrainer = gpuTrainer !== null;
  gpu = null;
  gpuModel = null;
  gpuTrainer = null;
  resetGpu();
  backend = backendAfterLoss(info + (hadTrainer ? '; training progress on the GPU was lost' : ''));
  post({ id: 0, type: 'backend', ...backend });
}

/** Weight residency is explicit: the engine that is about to run always
 * gets the live weights. GPU → CPU downloads; CPU → GPU re-uploads. A
 * training session continues on the new engine (optimizer moments restart). */
async function handleSetBackend(id: number, requested: BackendName): Promise<void> {
  if (!cfg) throw new Error('worker not initialized');
  if (requested === 'gpu' && !gpu) {
    const g = await getGpu();
    if (g) {
      gpu = g;
      void g.lost.then((info) => onGpuLost(info.message || info.reason || 'unknown'));
    }
  }
  const next = chooseBackend(requested, gpu ? { ok: true, adapter: gpu.adapterInfo } : { ok: false, reason: gpuUnavailableReason });
  if (next.active === 'cpu' && backend.active === 'gpu') {
    if (gpuTrainer) {
      trainParams = await gpuTrainer.download();
      adamState = adamwInit();
      gpuTrainer.destroy();
      gpuTrainer = null;
    }
    gpuModel?.destroy();
    gpuModel = null;
  }
  if (next.active === 'gpu' && gpu) {
    gpuModel?.destroy();
    gpuModel = uploadModel(gpu, active(), cfg);
    gpuTrainer?.destroy();
    gpuTrainer = trainParams ? new GpuTrainer(gpu, gpuModel, cfg, { ...HP }) : null;
  }
  backend = next;
  post({ id, type: 'backend', ...backend });
}

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

async function handleGenerate(id: number, p: WorkerRequest & { type: 'generate' }): Promise<void> {
  if (!cfg || !tokenizer) throw new Error('worker not initialized');
  const { prompt, maxNewTokens, temperature, topK, seed } = p.params;
  let index = 0;
  if (backend.active === 'gpu' && gpu && gpuModel) {
    try {
      const outGpu = await gpuGenerate(
        gpu,
        gpuModel,
        cfg,
        tokenizer.encode(prompt),
        maxNewTokens,
        { temperature, topK, rng: seed === null ? null : mulberry32(seed) },
        ({ token, logits, ms }) => {
          const buf = logits.buffer as ArrayBuffer;
          post({ id, type: 'token', index: index++, token, logits: buf, ms }, [buf]);
        },
      );
      post({ id, type: 'generated', text: tokenizer.decode(outGpu) });
      return;
    } catch (e) {
      // the device went away mid-run: finish on the CPU and say so
      onGpuLost(e instanceof Error ? e.message : String(e));
      index = 0;
    }
  }
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

async function handleTrace(id: number, text: string, targets?: string): Promise<void> {
  if (!cfg || !tokenizer) throw new Error('worker not initialized');
  const weights = await liveWeights();
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
  gpuTrainer?.destroy();
  gpuTrainer = null;
  if (backend.active === 'gpu' && gpu) {
    gpuModel?.destroy();
    gpuModel = uploadModel(gpu, trainParams, cfg);
    gpuTrainer = new GpuTrainer(gpu, gpuModel, cfg, { ...HP });
  }
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
    let loss: number;
    let stepMs: number | undefined;
    if (backend.active === 'gpu' && gpuTrainer && gpu && gpuModel) {
      gpuTrainer.hp.lr = p.lr;
      const r = await gpuTrainer.step(x, y);
      loss = r.loss;
      stepMs = r.ms;
    } else {
      loss = trainStep(trainParams, cfg, x, y, adamState, hp).loss;
    }
    trainStepCount++;

    let sample: string | undefined;
    if (p.sampleEvery > 0 && trainStepCount % p.sampleEvery === 0) {
      const sampleOpts = { temperature: 0.8, topK: 40, rng: mulberry32(trainStepCount) };
      const out =
        backend.active === 'gpu' && gpuTrainer && gpu && gpuModel
          ? await gpuGenerate(gpu, gpuModel, cfg, tokenizer.encode('\n'), 80, sampleOpts)
          : generateKV(trainParams, cfg, tokenizer.encode('\n'), 80, sampleOpts);
      sample = tokenizer.decode(out);
    }
    post({ id, type: 'trainProgress', step: trainStepCount, loss, sample, ms: stepMs });
    // Yield to the event loop so trainStop can interleave, then serve the
    // requests that arrived during this step (x-ray, backend switch, generate).
    await new Promise((r) => setTimeout(r, 0));
    while (deferred.length) await handle(deferred.shift()!);
  }
  training = false;
  post({ id, type: 'trainDone', step: trainStepCount });
  while (deferred.length) await handle(deferred.shift()!);
}

// Handlers are async (the GPU path awaits readbacks), so requests are run
// one at a time: a backend switch can never land in the middle of a
// generation that is still using the GPU buffers.
let chain: Promise<void> = Promise.resolve();
/** Requests that arrived while a training run holds the chain; served between steps. */
const deferred: WorkerRequest[] = [];

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'trainStop') {
    stopRequested = true; // must not queue behind the running training loop
    return;
  }
  if (training && (msg.type === 'trace' || msg.type === 'setBackend' || msg.type === 'generate')) {
    deferred.push(msg);
    return;
  }
  chain = chain.then(() => handle(msg)).catch(() => undefined);
};

async function handle(msg: WorkerRequest): Promise<void> {
  try {
    switch (msg.type) {
      case 'init':
        handleInit(msg.id, msg.cfgJson, msg.chars, msg.weightsBuffer);
        break;
      case 'generate':
        await handleGenerate(msg.id, msg);
        break;
      case 'setBackend':
        await handleSetBackend(msg.id, msg.backend);
        break;
      case 'trace':
        await handleTrace(msg.id, msg.text, msg.targets);
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
        gpuTrainer?.destroy();
        gpuTrainer = null;
        if (gpu && gpuModel && cfg) {
          gpuModel.destroy();
          gpuModel = uploadModel(gpu, active(), cfg);
        }
        break;
    }
  } catch (err) {
    post({ id: msg.id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
}
