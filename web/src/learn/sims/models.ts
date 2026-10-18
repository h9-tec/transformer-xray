/** Pure models behind the lesson simulators. Everything the canvases draw
 * is computed here, so it can be unit-tested and reused by the quizzes. */

import { bankConflicts } from '@/lib/memory';

export const LANES = 32;

// ─── lesson 1 ───
export function threadGrid(n: number, block: number): { blocks: number; lastBlockActive: number; idle: number } {
  const blocks = Math.ceil(n / block);
  const lastBlockActive = n - (blocks - 1) * block;
  return { blocks, lastBlockActive, idle: blocks * block - n };
}

// ─── lesson 2 ───
export function gridStrideCoverage(
  n: number,
  grid: number,
  block: number,
): { total: number; maxIters: number; covered: boolean; itersFor: (thread: number) => number } {
  const total = grid * block;
  const itersFor = (t: number) => (t >= n ? 0 : Math.floor((n - 1 - t) / total) + 1);
  return { total, maxIters: Math.ceil(n / total), covered: total > 0, itersFor };
}

// ─── lesson 3 ───
export type DivergencePattern = 'none' | 'lane-parity' | 'half' | 'warp-parity';

export interface WarpInstruction {
  label: string;
  /** active mask as a 32-bit int; -1 = all lanes */
  active: number;
}

/** A warp running: setup, a two-arm branch under `pattern`, then a join. */
export function divergenceTimeline(pattern: DivergencePattern, lanes = LANES): {
  instructions: WarpInstruction[];
  divergentFraction: number;
} {
  const full = lanes === 32 ? -1 : (1 << lanes) - 1;
  let armA = full;
  switch (pattern) {
    case 'lane-parity':
      armA = 0;
      for (let l = 0; l < lanes; l++) if (l % 2 === 1) armA |= 1 << l;
      break;
    case 'half':
      armA = 0;
      for (let l = 0; l < lanes / 2; l++) armA |= 1 << l;
      break;
    case 'warp-parity': // uniform within this warp (warp 0 takes arm A)
    case 'none':
      armA = full;
  }
  const armB = (full & ~armA) | 0;
  const instructions: WarpInstruction[] = [
    { label: 'i = blockIdx.x * blockDim.x + threadIdx.x', active: full },
    { label: 'load a[i]', active: full },
  ];
  if (armA !== 0) {
    instructions.push({ label: 'arm A: y = a * 2', active: armA }, { label: 'arm A: store', active: armA });
  }
  if (armB !== 0) {
    instructions.push({ label: 'arm B: y = a + 1', active: armB }, { label: 'arm B: store', active: armB });
  }
  instructions.push({ label: 'join', active: full });
  const divergent = instructions.filter((i) => i.active !== full).length;
  return { instructions, divergentFraction: divergent / instructions.length };
}

// ─── lessons 6 and 8 ───
export function tileTraffic(M: number, N: number, K: number, tile: number): { naiveBytes: number; tiledBytes: number; ratio: number } {
  const naiveBytes = 2 * M * N * K * 4;
  const tiledBytes = naiveBytes / tile;
  return { naiveBytes, tiledBytes, ratio: naiveBytes / tiledBytes };
}

// ─── lesson 7 ───
/** A column read of a [32][32 + pad] tile: lane l reads word l·(stride+pad). */
export function bankMap(strideWords: number, padWords: number): { banks: number[]; worst: number } {
  const addrs = Array.from({ length: LANES }, (_, l) => l * (strideWords + padWords) * 4);
  return { banks: addrs.map((a) => (a / 4) % 32), worst: bankConflicts(addrs) };
}

// ─── lesson 9 ───
export function reductionSteps(values: number[]): { offset: number; lanes: number[] }[] {
  let cur = values.slice(0, LANES);
  const steps: { offset: number; lanes: number[] }[] = [];
  for (let offset = 16; offset > 0; offset >>= 1) {
    cur = cur.map((v, l) => (l + offset < LANES ? v + cur[l + offset] : v));
    steps.push({ offset, lanes: cur.slice() });
  }
  return steps;
}

// ─── lesson 11 ───
export function attentionWork(T: number): number[] {
  return Array.from({ length: T }, (_, i) => i + 1);
}

// ─── lesson 12 ───
export interface RooflineEntry {
  name: string;
  flops: number;
  bytes: number;
  ms: number;
}
export function rooflinePoints(entries: RooflineEntry[]): { name: string; intensity: number; gflops: number }[] {
  return entries.map((e) => ({ name: e.name, intensity: e.flops / e.bytes, gflops: e.flops / e.ms / 1e6 }));
}
