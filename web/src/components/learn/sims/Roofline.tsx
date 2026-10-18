'use client';

/** Lesson 12: the measured kernels from kernels/results placed on a
 * roofline. FLOPs and bytes come from the bench shapes; the achieved
 * GFLOP/s from the measured milliseconds. Peak numbers are labeled as the
 * ballpark they are. */

import results from '@/content/gpu-results.json';
import { rooflinePoints, type RooflineEntry } from '@/learn/sims/models';

const T = 128;
const D = 192;
const HD = 32;

function entry(kernel: string, ms: number, shape?: string): RooflineEntry | null {
  switch (kernel) {
    case 'embedding_gather':
      return { name: kernel, flops: 1, bytes: T * D * 8, ms };
    case 'layernorm_naive':
    case 'layernorm_warp':
      return { name: kernel, flops: T * D * 8, bytes: T * D * 8, ms };
    case 'matmul_naive':
    case 'matmul_tiled': {
      const [m, k, n] = (shape ?? '128x192x576').split('x').map(Number);
      const flops = 2 * m * k * n;
      const bytes = kernel === 'matmul_naive' ? 2 * m * n * k * 4 : (2 * m * n * k * 4) / 32 + m * n * 4;
      return { name: `${kernel} ${shape ?? ''}`.trim(), flops, bytes, ms };
    }
    case 'attention_row':
      return { name: kernel, flops: 2 * T * T * HD, bytes: 3 * T * HD * 4 + T * HD * 4, ms };
    case 'gelu_forward':
    case 'gelu_forward_f4': {
      const n = 4 * T * D;
      return { name: kernel, flops: 20 * n, bytes: 8 * n, ms };
    }
    default:
      return null;
  }
}

// ballpark ceilings for a laptop RTX class part
const PEAK_GFLOPS = 20000;
const BW_GBS = 400;

export function Roofline() {
  const entries = (results.results as { kernel: string; ms: number; shape?: string }[])
    .map((r) => entry(r.kernel, r.ms, r.shape))
    .filter((e): e is RooflineEntry => !!e);
  const pts = rooflinePoints(entries);
  const x0 = 60;
  const x1 = 700;
  const y0 = 20;
  const y1 = 250;
  const lx = (i: number) => x0 + ((Math.log10(Math.max(i, 0.01)) + 2) / 5) * (x1 - x0); // 0.01 .. 1000 FLOP/B
  const ly = (g: number) => y1 - ((Math.log10(Math.max(g, 0.1)) + 1) / 6) * (y1 - y0); // 0.1 .. 1e5 GFLOP/s
  const ridge = PEAK_GFLOPS / BW_GBS;
  return (
    <div>
      <svg viewBox="0 0 720 300" className="block h-auto w-full rounded-xl bg-[#080b14]" role="img" aria-label="Roofline of the measured kernels">
        {[0.01, 0.1, 1, 10, 100, 1000].map((i) => (
          <g key={i}>
            <line x1={lx(i)} y1={y0} x2={lx(i)} y2={y1} stroke="rgba(255,255,255,.05)" />
            <text x={lx(i)} y={y1 + 14} textAnchor="middle" fill="#5f6a85" fontSize="10" fontFamily="JetBrains Mono, monospace">{i}</text>
          </g>
        ))}
        {[0.1, 1, 10, 100, 1000, 10000, 100000].map((g) => (
          <g key={g}>
            <line x1={x0} y1={ly(g)} x2={x1} y2={ly(g)} stroke="rgba(255,255,255,.05)" />
            <text x={x0 - 6} y={ly(g) + 3} textAnchor="end" fill="#5f6a85" fontSize="10" fontFamily="JetBrains Mono, monospace">{g >= 1000 ? `${g / 1000}k` : g}</text>
          </g>
        ))}
        <polyline points={`${lx(0.01)},${ly(0.01 * BW_GBS)} ${lx(ridge)},${ly(PEAK_GFLOPS)} ${lx(1000)},${ly(PEAK_GFLOPS)}`} fill="none" stroke="#ff7ab6" strokeWidth="1.5" />
        <text x={lx(ridge) + 6} y={ly(PEAK_GFLOPS) - 6} fill="#ff7ab6" fontSize="11" fontFamily="Instrument Sans, system-ui">ridge at {ridge} FLOP/byte (ballpark peaks: {PEAK_GFLOPS / 1000} TFLOP/s, {BW_GBS} GB/s)</text>
        {pts.map((p, i) => (
          <g key={p.name}>
            <circle cx={lx(p.intensity)} cy={ly(p.gflops)} r="5" fill={p.intensity < ridge ? '#74c7ff' : '#ff9a4d'} />
            <text x={lx(p.intensity) + 8} y={ly(p.gflops) + 4 + (i % 2) * 10} fill="#e8ecf7" fontSize="10" fontFamily="JetBrains Mono, monospace">{p.name}</text>
          </g>
        ))}
        <text x={x1} y={y1 + 28} textAnchor="end" fill="#9aa5bf" fontSize="11" fontFamily="Instrument Sans, system-ui">arithmetic intensity (FLOP per byte)</text>
        <text x={x0} y={12} fill="#9aa5bf" fontSize="11" fontFamily="Instrument Sans, system-ui">achieved GFLOP/s, measured on {results.gpu}</text>
      </svg>
      <p className="m-0 mt-2 text-xs text-faint">
        Every elementwise kernel sits far left of the ridge: bandwidth-bound. Points are measured milliseconds from the repo’s results file; the roof is a ballpark for this GPU class, not a measurement.
      </p>
    </div>
  );
}
