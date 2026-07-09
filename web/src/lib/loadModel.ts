/** Fetch + parse the exported model artifacts (browser side). */

import type { Weights } from '@/engine/gpt';
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
}

async function fetchOk(url: string): Promise<Response> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`failed to fetch ${url}: ${r.status}`);
  return r;
}

export async function loadModel(base = '/model'): Promise<LoadedModel> {
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
  };
}
