/** Fetch + parse the exported model artifacts (browser side). */

import type { Weights } from '@/engine/gpt';
import { withBase } from '@/lib/basePath';
import { CharTokenizer } from '@/engine/tokenizer';
import {
  configFromJson,
  parseBlob,
  type Config,
  type ModelConfigJson,
} from '@/engine/weights';

export interface LoadedModel {
  cfg: Config;
  cfgJson: ModelConfigJson;
  weights: Weights;
  tokenizer: CharTokenizer;
  /** the raw fetched bytes — callers may copy it into a worker */
  rawBuffer: ArrayBuffer;
}

async function fetchOk(url: string): Promise<Response> {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`fetch failed for ${url}: ${msg}`); // errors must name their URL
  }
}

export async function loadModel(base = withBase('/model')): Promise<LoadedModel> {
  const [cfgJson, tokJson, weightsBuf] = await Promise.all([
    fetchOk(`${base}/config.json`).then((r) => r.json() as Promise<ModelConfigJson>),
    fetchOk(`${base}/tokenizer.json`).then((r) => r.json() as Promise<{ chars: string }>),
    fetchOk(`${base}/weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  return {
    cfg: configFromJson(cfgJson),
    cfgJson,
    weights: parseBlob(weightsBuf, cfgJson.tensors),
    tokenizer: new CharTokenizer(tokJson.chars),
    rawBuffer: weightsBuf,
  };
}
