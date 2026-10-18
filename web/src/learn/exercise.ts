/** The exercise contract and grader. An exercise is data: starter and
 * solution source, a launch, deterministic inputs with a TS reference, and
 * targets expressed on the interpreter's stats so grading is reproducible. */

import {
  compileProgram,
  launch,
  type ArgValue,
  type LaunchResult,
  type LaunchStats,
  type RaceReport,
} from '@/cuda';

export interface TargetCtx {
  stats: LaunchStats;
  races: RaceReport[];
  maxDiff: number;
}

export interface Target {
  id: string;
  label: string;
  pass: (ctx: TargetCtx) => boolean;
  detail: (ctx: TargetCtx) => string;
}

export interface Built {
  args: ArgValue[];
  /** indices into args that are outputs, in the order of `reference` */
  outputs: number[];
  reference: Float32Array[];
  tolerance: number;
}

export interface Exercise {
  slug: string;
  kernel: string;
  starter: string;
  solution: string;
  launch: { grid: [number, number, number]; block: [number, number, number]; sharedBytes?: number };
  build: (seed: number) => Built;
  targets: Target[];
  /** a nudge shown after a failed run, derived from what failed */
  hint?: (ctx: TargetCtx & { firstMismatch: Mismatch | null }) => string | null;
}

export interface Mismatch {
  output: number;
  index: number;
  got: number;
  want: number;
}

export interface GradeResult {
  ok: boolean;
  maxDiff: number;
  firstMismatch: Mismatch | null;
  targets: { id: string; label: string; pass: boolean; detail: string }[];
  stats: LaunchStats;
  races: RaceReport[];
  printf: string[];
  hint: string | null;
}

// ─── standard targets ───

export const correct: Target = {
  id: 'correct',
  label: 'Result matches the reference',
  pass: (c) => c.maxDiff === c.maxDiff && c.maxDiff <= tolerancePlaceholder.value,
  detail: (c) => (Number.isNaN(c.maxDiff) ? 'NaN in output' : `max diff ${fmt(c.maxDiff)}`),
};
// `correct` reads the current exercise's tolerance through this cell so the
// standard target can stay a shared constant.
const tolerancePlaceholder = { value: 1e-5 };

export const coalesced = (maxSectorsPerRequest: number): Target => ({
  id: 'coalesced',
  label: 'Global loads and stores are coalesced',
  pass: (c) => c.stats.globalRequests === 0 || c.stats.globalSectors / c.stats.globalRequests <= maxSectorsPerRequest,
  detail: (c) =>
    c.stats.globalRequests === 0
      ? 'no global traffic'
      : `${(c.stats.globalSectors / c.stats.globalRequests).toFixed(1)} sectors per request (target ≤ ${maxSectorsPerRequest})`,
});

export const noRaces = (): Target => ({
  id: 'races',
  label: 'No data races',
  pass: (c) => c.races.length === 0,
  detail: (c) =>
    c.races.length === 0
      ? 'none detected'
      : `race on ${c.races[0].buffer}, line ${c.races[0].firstLine} vs ${c.races[0].secondLine}${c.races.length > 1 ? ` (+${c.races.length - 1} more)` : ''}`,
});

export const noBankConflicts = (): Target => ({
  id: 'banks',
  label: 'No shared-memory bank conflicts',
  pass: (c) => c.stats.sharedWorstConflict <= 1,
  detail: (c) => (c.stats.sharedWorstConflict <= 1 ? 'conflict-free' : `${c.stats.sharedWorstConflict}-way conflict`),
});

export const noDivergence = (maxDivergent: number): Target => ({
  id: 'divergence',
  label: 'Warps stay converged',
  pass: (c) => c.stats.divergentInstructions <= maxDivergent,
  detail: (c) => `${c.stats.divergentInstructions} divergent warp instructions (target ≤ ${maxDivergent})`,
});

export const trafficAtMost = (bytes: number, why: string): Target => ({
  id: 'traffic',
  label: `Global traffic ≤ ${fmtBytes(bytes)} (${why})`,
  pass: (c) => c.stats.globalBytes <= bytes,
  detail: (c) => `${fmtBytes(c.stats.globalBytes)} moved`,
});

export const requestsAtMost = (n: number, why: string): Target => ({
  id: 'requests',
  label: `At most ${n} memory requests (${why})`,
  pass: (c) => c.stats.globalRequests <= n,
  detail: (c) => `${c.stats.globalRequests} requests`,
});

// ─── grading ───

function compare(built: Built, args: ArgValue[]): { maxDiff: number; firstMismatch: Mismatch | null } {
  let maxDiff = 0;
  let first: Mismatch | null = null;
  built.outputs.forEach((argIdx, oi) => {
    const a = args[argIdx];
    if (a.kind !== 'buffer') return;
    const got = a.data;
    const want = built.reference[oi];
    for (let i = 0; i < want.length; i++) {
      const d = Math.abs(got[i] - want[i]);
      if (Number.isNaN(d)) {
        maxDiff = Number.NaN;
        if (!first) first = { output: oi, index: i, got: got[i], want: want[i] };
        return;
      }
      if (d > maxDiff) maxDiff = d;
      if (!first && d > built.tolerance) first = { output: oi, index: i, got: got[i], want: want[i] };
    }
  });
  return { maxDiff, firstMismatch: first };
}

function finish(ex: Exercise, built: Built, args: ArgValue[], result: LaunchResult): GradeResult {
  const { maxDiff, firstMismatch } = compare(built, args);
  tolerancePlaceholder.value = built.tolerance;
  const ctx: TargetCtx = { stats: result.stats, races: result.races, maxDiff };
  const targets = [correct, ...ex.targets].map((t) => ({ id: t.id, label: t.label, pass: t.pass(ctx), detail: t.detail(ctx) }));
  const ok = targets.every((t) => t.pass);
  const hint = ok ? null : (ex.hint?.({ ...ctx, firstMismatch }) ?? null);
  return { ok, maxDiff, firstMismatch, targets, stats: result.stats, races: result.races, printf: result.printf, hint };
}

/** Compile + run on this thread. Throws CudaError for compile/runtime errors. */
export function gradeExercise(ex: Exercise, source: string, seed = 1): GradeResult {
  const built = ex.build(seed);
  const program = compileProgram(source);
  const result = launch(program, ex.kernel, { ...ex.launch, args: built.args, instrument: true });
  return finish(ex, built, built.args, result);
}

/** Grade from a worker response: `buffers` are the returned ArrayBuffers in
 * the order of the buffer args. */
export function gradeFromWire(ex: Exercise, built: Built, buffers: ArrayBuffer[], result: LaunchResult): GradeResult {
  let bi = 0;
  const args: ArgValue[] = built.args.map((a) => {
    if (a.kind !== 'buffer') return a;
    const buf = buffers[bi++];
    return { ...a, data: a.data instanceof Int32Array ? new Int32Array(buf) : new Float32Array(buf) };
  });
  return finish(ex, built, args, result);
}

function fmt(x: number): string {
  if (x === 0) return '0';
  return x < 1e-3 || x > 1e4 ? x.toExponential(1) : x.toFixed(4);
}
export function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(2)} MB`;
}

/** Deterministic input helpers for exercise builders. */
export function rand(n: number, seed: number, scale = 1): Float32Array {
  let a = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const r = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    out[i] = (r - 0.5) * 2 * scale;
  }
  return out;
}
export const buf = (name: string, data: Float32Array | Int32Array): ArgValue => ({ kind: 'buffer', name, data });
export const int = (value: number): ArgValue => ({ kind: 'int', value });
export const flt = (value: number): ArgValue => ({ kind: 'float', value });
