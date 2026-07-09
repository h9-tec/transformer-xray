/** 256-entry RGBA lookup tables for the tensor heatmaps.
 *
 * diverging  — signed tensors (activations, gradients): cool blue for
 *              negative, neutral near-surface dark at zero, warm amber for
 *              positive. Blue↔amber survives every CVD type; zero fades
 *              into the background so sparse tensors read as sparse.
 * sequential — magnitudes in [0, max] (attention probabilities): surface
 *              dark up to phosphor green.
 * Green/magenta are reserved for forward/backward chrome, not data values.
 */

export type ColormapMode = 'diverging' | 'sequential';

interface Rgb {
  r: number;
  g: number;
  b: number;
}

function hex(c: string): Rgb {
  return {
    r: parseInt(c.slice(1, 3), 16),
    g: parseInt(c.slice(3, 5), 16),
    b: parseInt(c.slice(5, 7), 16),
  };
}

function lerp(a: Rgb, b: Rgb, t: number): Rgb {
  return { r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t, b: a.b + (b.b - a.b) * t };
}

function buildLut(stops: [number, Rgb][]): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s = 0;
    while (s < stops.length - 2 && t > stops[s + 1][0]) s++;
    const [t0, c0] = stops[s];
    const [t1, c1] = stops[s + 1];
    const local = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
    const c = lerp(c0, c1, Math.min(1, Math.max(0, local)));
    lut[i * 4] = c.r;
    lut[i * 4 + 1] = c.g;
    lut[i * 4 + 2] = c.b;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

const COOL = hex('#7aa2ff');
const WARM = hex('#ffb454');
const MID = hex('#14161d'); // near-surface: zero disappears
const LOW = hex('#101218');
const HIGH = hex('#3ddc97');

export const LUTS: Record<ColormapMode, Uint8ClampedArray> = {
  diverging: buildLut([
    [0, COOL],
    [0.5, MID],
    [1, WARM],
  ]),
  sequential: buildLut([
    [0, LOW],
    [1, HIGH],
  ]),
};

/** Map a value to a LUT index given the tensor's normalization range. */
export function lutIndex(v: number, mode: ColormapMode, scale: number): number {
  if (!Number.isFinite(v)) return mode === 'diverging' ? 128 : 0; // −∞ masked cells → neutral
  if (mode === 'diverging') {
    const t = Math.max(-1, Math.min(1, v / (scale || 1)));
    return Math.round(128 + t * 127);
  }
  const t = Math.max(0, Math.min(1, v / (scale || 1)));
  return Math.round(t * 255);
}
