/** Pure helpers behind the builtin functions: fp32 math, warp shuffles,
 * printf formatting. Atomics and memory live in compile.ts because they
 * need the store machinery. */

import { LANES, has } from './runtime';

const fr = Math.fround;

export const MATH1: Record<string, (x: number) => number> = {
  expf: (x) => fr(Math.exp(x)),
  logf: (x) => fr(Math.log(x)),
  sqrtf: (x) => fr(Math.sqrt(x)),
  rsqrtf: (x) => fr(1 / Math.sqrt(x)),
  tanhf: (x) => fr(Math.tanh(x)),
  fabsf: (x) => Math.abs(x),
  floorf: (x) => Math.floor(x),
  ceilf: (x) => Math.ceil(x),
};

export const MATH2: Record<string, (a: number, b: number) => number> = {
  fmaxf: (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b)),
  fminf: (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.min(a, b)),
  powf: (a, b) => fr(Math.pow(a, b)),
};

export type ShuffleKind = 'sync' | 'up' | 'down' | 'xor';

/** out[l] = v[src(l)] for active lanes; inactive source lanes still supply
 * their last value, as the hardware does for lanes named in the mask. */
export function shuffle(
  kind: ShuffleKind,
  v: Float64Array,
  param: Float64Array,
  out: Float64Array,
  active: number,
): void {
  for (let l = 0; l < LANES; l++) {
    if (!has(active, l)) continue;
    const p = param[l] | 0;
    let src: number;
    switch (kind) {
      case 'sync':
        src = p & 31;
        break;
      case 'up':
        src = l - p >= 0 ? l - p : l;
        break;
      case 'down':
        src = l + p < LANES ? l + p : l;
        break;
      case 'xor':
        src = l ^ p;
        break;
    }
    out[l] = v[src];
  }
}

export type PrintfKind = 'f' | 'i' | 'u';

/** C-ish printf for one lane. Supports %d %i %u %f %g %e %x %s %% with
 * optional width/precision; unknown conversions are echoed. */
export function formatPrintf(fmt: string, args: number[], kinds: PrintfKind[]): string {
  let out = '';
  let ai = 0;
  for (let i = 0; i < fmt.length; i++) {
    const c = fmt[i];
    if (c !== '%') {
      out += c;
      continue;
    }
    let j = i + 1;
    let spec = '';
    while (j < fmt.length && /[-+ 0-9.#l]/.test(fmt[j])) spec += fmt[j++];
    const conv = fmt[j] ?? '';
    i = j;
    if (conv === '%') {
      out += '%';
      continue;
    }
    const precM = /\.(\d+)/.exec(spec);
    const prec = precM ? Number(precM[1]) : 6;
    const widthM = /^(\d+)/.exec(spec.replace(/^[-+ 0]+/, ''));
    const width = widthM ? Number(widthM[1]) : 0;
    const v = args[ai] ?? 0;
    const k = kinds[ai] ?? 'f';
    ai++;
    let s: string;
    switch (conv) {
      case 'd':
      case 'i':
        s = String(k === 'f' ? Math.trunc(v) : v | 0);
        break;
      case 'u':
        s = String(v >>> 0);
        break;
      case 'x':
        s = (v >>> 0).toString(16);
        break;
      case 'f':
        s = v.toFixed(prec);
        break;
      case 'e':
        s = v.toExponential(prec);
        break;
      case 'g': {
        const p = precM ? Math.max(1, prec) : 6;
        s = String(Number(v.toPrecision(p)));
        break;
      }
      case 's':
        s = String(v);
        break;
      default:
        s = `%${spec}${conv}`;
    }
    out += spec.startsWith('-') ? s.padEnd(width) : s.padStart(width);
  }
  return out;
}
