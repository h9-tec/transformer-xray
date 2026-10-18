'use client';

/** Lesson 11: the causal triangle as work per block. Row i touches i + 1
 * columns, so blocks finish at very different times. */

import { useState } from 'react';
import { attentionWork } from '@/learn/sims/models';

export function AttentionRows() {
  const [T, setT] = useState(16);
  const work = attentionWork(T);
  const total = work.reduce((a, b) => a + b, 0);
  const balanced = total / T;
  const cell = Math.min(18, 300 / T);
  return (
    <div>
      <svg viewBox="0 0 720 320" className="block h-auto w-full rounded-xl bg-[#080b14]" role="img" aria-label="Causal attention work per row">
        <text x="12" y="16" fill="#9aa5bf" fontSize="12" fontFamily="Instrument Sans, system-ui">scores s[row][j] computed (lit) vs skipped by the loop bound j ≤ row</text>
        {work.map((n, row) =>
          Array.from({ length: T }, (_, j) => (
            <rect key={`${row}-${j}`} x={12 + j * cell} y={28 + row * cell} width={cell - 1} height={cell - 1} fill={j <= row ? (j === row ? '#ff9a4d' : 'rgba(116,199,255,.5)') : 'rgba(255,255,255,.04)'} />
          )),
        )}
        {work.map((n, row) => (
          <g key={row}>
            <rect x={360} y={28 + row * cell} width={(n / T) * 330} height={cell - 1} fill={n > balanced ? 'rgba(255,154,77,.7)' : 'rgba(116,199,255,.6)'} />
          </g>
        ))}
        <line x1={360 + (balanced / T) * 330} y1="26" x2={360 + (balanced / T) * 330} y2={30 + T * cell} stroke="#ff7ab6" strokeDasharray="3 3" />
        <text x={364 + (balanced / T) * 330} y={38 + T * cell} fill="#ff7ab6" fontSize="11" fontFamily="JetBrains Mono, monospace">balanced: {balanced.toFixed(1)} columns per block</text>
        <text x="360" y="16" fill="#9aa5bf" fontSize="12" fontFamily="Instrument Sans, system-ui">work per block (one block per query row)</text>
        <text x="12" y={56 + T * cell} fill="#9aa5bf" fontSize="12" fontFamily="Instrument Sans, system-ui">
          last row does {T}× the work of the first. Total {total} of {T * T} score cells: the mask saves {Math.round((1 - total / (T * T)) * 100)}% of the work but leaves it uneven.
        </text>
      </svg>
      <label className="mt-3 flex flex-col gap-1 text-sm text-muted">
        Sequence length T <span className="nums text-ink">{T}</span>
        <input type="range" min={4} max={32} step={1} value={T} onChange={(e) => setT(Number(e.target.value))} className="accent-[#ff9a4d]" />
      </label>
    </div>
  );
}
