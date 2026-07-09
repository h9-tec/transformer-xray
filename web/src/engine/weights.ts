/** Loader for the blob format written by model/blobio.py:
 * raw little-endian fp32 tensors + a JSON manifest of {name, shape, offset, size}. */

import type { Tensor } from './tensor';
import { numel } from './tensor';

export interface ManifestEntry {
  name: string;
  shape: number[];
  offset: number; // bytes
  size: number; // floats
}

export interface ModelConfigJson {
  model: {
    n_layer: number;
    n_embd: number;
    n_head: number;
    head_dim: number;
    block_size: number;
    vocab_size: number;
    ln_eps: number;
  };
  val_loss: number;
  trained_iters: number;
  n_params: number;
  tensors: ManifestEntry[];
}

export interface Config {
  nLayer: number;
  nEmbd: number;
  nHead: number;
  headDim: number;
  blockSize: number;
  vocabSize: number;
  lnEps: number;
}

export function configFromJson(j: ModelConfigJson): Config {
  const m = j.model;
  return {
    nLayer: m.n_layer,
    nEmbd: m.n_embd,
    nHead: m.n_head,
    headDim: m.head_dim,
    blockSize: m.block_size,
    vocabSize: m.vocab_size,
    lnEps: m.ln_eps,
  };
}

export function parseBlob(buf: ArrayBuffer, manifest: ManifestEntry[]): Map<string, Tensor> {
  const out = new Map<string, Tensor>();
  for (const e of manifest) {
    if (e.size !== numel(e.shape)) {
      throw new Error(`manifest entry ${e.name}: size ${e.size} != shape ${e.shape.join('x')}`);
    }
    // A view, not a copy — weights are read-only; training clones what it mutates.
    out.set(e.name, { data: new Float32Array(buf, e.offset, e.size), shape: e.shape });
  }
  return out;
}
