'use client';

/** Lesson 3: one warp's instruction stream as rows, 32 lanes as columns.
 * A lit cell is a lane doing work; a dim cell is a lane masked off. */

import { useState } from 'react';
import { divergenceTimeline, type DivergencePattern } from '@/learn/sims/models';
import { C, FONT, useCanvas } from './useCanvas';

const PATTERNS: { id: DivergencePattern; label: string }[] = [
  { id: 'lane-parity', label: 'if (i % 2)' },
  { id: 'half', label: 'if (threadIdx.x < 16)' },
  { id: 'warp-parity', label: 'if ((i / 32) % 2)' },
  { id: 'none', label: 'no branch' },
];

export function WarpTimeline() {
  const [pattern, setPattern] = useState<DivergencePattern>('lane-parity');
  const { instructions, divergentFraction } = divergenceTimeline(pattern);

  const ref = useCanvas(
    (ctx, w, h, frame) => {
      const left = 170;
      const cell = (w - left - 12) / 32;
      const rowH = Math.min(26, (h - 30) / instructions.length);
      ctx.font = FONT;
      const cur = frame < 0 ? instructions.length : Math.floor(frame / 30) % (instructions.length + 2);
      instructions.forEach((ins, r) => {
        const y = 20 + r * rowH;
        ctx.fillStyle = r < cur ? C.muted : 'rgba(154,165,191,.35)';
        ctx.fillText(ins.label, 8, y + rowH * 0.65);
        for (let l = 0; l < 32; l++) {
          const on = ((ins.active >>> l) & 1) === 1;
          ctx.fillStyle = r < cur ? (on ? (ins.active === -1 ? C.ice : C.copper) : C.faint) : 'rgba(255,255,255,.03)';
          ctx.fillRect(left + l * cell, y + 3, cell - 2, rowH - 6);
        }
      });
      ctx.fillStyle = C.muted;
      ctx.fillText('lanes 0 to 31', left, 12);
    },
    [pattern],
  );

  return (
    <div>
      <canvas ref={ref} className="block h-[240px] w-full rounded-xl bg-[#080b14]" aria-hidden />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {PATTERNS.map((p) => (
          <button
            key={p.id}
            type="button"
            aria-pressed={pattern === p.id}
            onClick={() => setPattern(p.id)}
            className={`nums rounded-full border px-3 py-1 text-[13px] ${pattern === p.id ? 'border-fwd/50 bg-fwd/10 text-ink' : 'border-line text-muted hover:text-ink'}`}
          >
            {p.label}
          </button>
        ))}
        <span className="nums ms-auto text-sm text-muted">
          divergent: <b className={divergentFraction ? 'text-accent' : 'text-ok'}>{Math.round(divergentFraction * 100)}%</b> of instructions
        </span>
      </div>
    </div>
  );
}
