import fs from 'node:fs';
import path from 'node:path';
import { mulberry32 } from '@/engine/rng';

export const FILES = [
  '01_embedding.cu',
  '02_layernorm.cu',
  '03_matmul.cu',
  '04_attention.cu',
  '05_elementwise.cu',
] as const;

/** The teaching part of a real kernel file: everything before the bench harness. */
export function teachingSource(file: string): string {
  const full = fs.readFileSync(path.join(process.cwd(), '..', 'kernels', file), 'utf8');
  const cut = full.indexOf('void run_');
  return cut === -1 ? full : full.slice(0, cut);
}

export function randFill(n: number, seed: number, scale = 1): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rng() - 0.5) * 2 * scale;
  return out;
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
