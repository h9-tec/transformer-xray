/** Test helpers: load the real exported artifacts + golden traces from
 * model/artifacts (vitest runs in node, reads the repo directly). */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Tensor } from '@/engine/tensor';
import {
  configFromJson,
  parseBlob,
  type Config,
  type ManifestEntry,
  type ModelConfigJson,
} from '@/engine/weights';

const dir = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(dir, '../../..'); // repo root

function readBuf(rel: string): ArrayBuffer {
  const b = fs.readFileSync(path.join(ROOT, rel));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

export interface Artifacts {
  cfg: Config;
  cfgJson: ModelConfigJson;
  weights: Map<string, Tensor>;
  chars: string;
}

let artifacts: Artifacts | null = null;

export function loadArtifacts(): Artifacts {
  if (artifacts) return artifacts;
  const cfgJson = JSON.parse(
    fs.readFileSync(path.join(ROOT, 'model/artifacts/config.json'), 'utf8'),
  ) as ModelConfigJson;
  const weights = parseBlob(readBuf('model/artifacts/weights.bin'), cfgJson.tensors);
  const chars = (
    JSON.parse(fs.readFileSync(path.join(ROOT, 'model/artifacts/tokenizer.json'), 'utf8')) as {
      chars: string;
    }
  ).chars;
  artifacts = { cfg: configFromJson(cfgJson), cfgJson, weights, chars };
  return artifacts;
}

export interface GoldenMeta {
  x: string;
  y: string;
  loss: number;
  adamw?: { lr: number; beta1: number; beta2: number; eps: number; weight_decay: number };
  greedy?: { prompt: string; tokens: number; output: string };
  tensors: ManifestEntry[];
}

export interface Golden {
  meta: GoldenMeta;
  tensors: Map<string, Tensor>;
}

const goldens = new Map<string, Golden>();

export function loadGolden(name: 't4' | 't64'): Golden {
  const cached = goldens.get(name);
  if (cached) return cached;
  const meta = JSON.parse(
    fs.readFileSync(path.join(ROOT, `model/artifacts/golden/golden-${name}.json`), 'utf8'),
  ) as GoldenMeta;
  const tensors = parseBlob(readBuf(`model/artifacts/golden/golden-${name}.bin`), meta.tensors);
  const g = { meta, tensors };
  goldens.set(name, g);
  return g;
}

/** Max |a−b| treating equal infinities (masked −∞ cells) as zero diff. */
export function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Infinity;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i];
    const bv = b[i];
    if (av === bv) continue; // covers −Inf === −Inf
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return Infinity;
    const d = Math.abs(av - bv);
    if (d > max) max = d;
  }
  return max;
}
