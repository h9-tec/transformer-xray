/** The model's weights resident on the GPU, uploaded once per session and
 * downloadable again (after GPU training) for the CPU inspectors. */

import type { Weights } from '@/engine/gpt';
import type { Config } from '@/engine/weights';
import { Buf, type Gpu } from './device';

export class GpuModel {
  readonly bufs = new Map<string, Buf>();
  readonly shapes = new Map<string, number[]>();

  constructor(
    readonly gpu: Gpu,
    readonly cfg: Config,
    weights: Weights,
  ) {
    for (const [name, t] of weights) {
      this.bufs.set(name, Buf.from(gpu, t.data, name));
      this.shapes.set(name, [...t.shape]);
    }
  }

  get(name: string): Buf {
    const b = this.bufs.get(name);
    if (!b) throw new Error(`GPU model has no weight '${name}'`);
    return b;
  }

  /** Copy every weight back to the CPU (e.g. to x-ray GPU-trained params). */
  async download(): Promise<Weights> {
    const out: Weights = new Map();
    for (const [name, b] of this.bufs) out.set(name, { data: await b.read(), shape: [...this.shapes.get(name)!] });
    return out;
  }

  destroy(): void {
    for (const b of this.bufs.values()) b.destroy();
    this.bufs.clear();
  }
}

export function uploadModel(gpu: Gpu, weights: Weights, cfg: Config): GpuModel {
  return new GpuModel(gpu, cfg, weights);
}
