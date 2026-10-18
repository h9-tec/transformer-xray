/** Memory-access arithmetic for one warp-wide load, as the hardware counts
 * it. Used by the landing hero now and by the CUDA interpreter's
 * instrumentation later, so the two can never disagree.
 *
 * All addresses are BYTE addresses. */

export type AccessPattern = 'row' | 'col' | 'scatter';

export interface LaneAddressOpts {
  /** lanes in the warp (32) */
  lanes?: number;
  /** bytes per element (4 = fp32) */
  elemBytes?: number;
  /** elements between consecutive lanes for the 'col' pattern */
  rowStride?: number;
  /** seed for the 'scatter' pattern */
  seed?: number;
}

/** Distinct 32-byte sectors a warp's addresses fall into. 32 contiguous
 * floats = 4 sectors (perfect coalescing); 32 scattered floats = up to 32. */
export function sectorsTouched(byteAddrs: number[], sectorBytes = 32): number {
  const seen = new Set<number>();
  for (const a of byteAddrs) seen.add(Math.floor(a / sectorBytes));
  return seen.size;
}

/** Worst-case lanes competing for one shared-memory bank. Identical
 * addresses are a broadcast and count once. 1 = conflict-free. */
export function bankConflicts(byteAddrs: number[], banks = 32, wordBytes = 4): number {
  const perBank = new Map<number, Set<number>>();
  for (const a of byteAddrs) {
    const word = Math.floor(a / wordBytes);
    const bank = word % banks;
    let words = perBank.get(bank);
    if (!words) perBank.set(bank, (words = new Set()));
    words.add(word);
  }
  let worst = 0;
  for (const words of perBank.values()) worst = Math.max(worst, words.size);
  return worst;
}

/** Byte addresses one warp touches for a single load instruction. */
export function laneAddresses(pattern: AccessPattern, opts: LaneAddressOpts): number[] {
  const lanes = opts.lanes ?? 32;
  const elemBytes = opts.elemBytes ?? 4;
  const out = new Array<number>(lanes);
  if (pattern === 'row') {
    for (let l = 0; l < lanes; l++) out[l] = l * elemBytes;
  } else if (pattern === 'col') {
    const stride = opts.rowStride ?? 192;
    for (let l = 0; l < lanes; l++) out[l] = l * stride * elemBytes;
  } else {
    const next = mulberry32(opts.seed ?? 1);
    const range = 480 * elemBytes;
    for (let l = 0; l < lanes; l++) out[l] = Math.floor(next() * (range / elemBytes)) * elemBytes;
  }
  return out;
}

/** Same generator as the engine's rng and kernels/common.cuh fill_random. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
