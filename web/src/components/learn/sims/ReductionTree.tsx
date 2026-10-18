'use client';

/** Lesson 9: the five shuffle-down steps of a warp sum, one row per step. */

import { useState } from 'react';
import { reductionSteps } from '@/learn/sims/models';
import { C, FONT, useCanvas } from './useCanvas';

export function ReductionTree() {
  const [seed, setSeed] = useState(1);
  const values = Array.from({ length: 32 }, (_, i) => ((i * 7 + seed * 13) % 17) + 1);
  const steps = reductionSteps(values);
  const total = values.reduce((a, b) => a + b, 0);

  const ref = useCanvas(
    (ctx, w, h, frame) => {
      const cell = (w - 60) / 32;
      const rowH = (h - 40) / 6;
      const shown = frame < 0 ? 5 : Math.floor(frame / 45) % 7;
      ctx.font = FONT;
      const drawRow = (vals: number[], r: number, live: number, label: string) => {
        const y = 24 + r * rowH;
        ctx.fillStyle = C.muted;
        ctx.fillText(label, 6, y + rowH * 0.6);
        for (let l = 0; l < 32; l++) {
          const x = 52 + l * cell;
          const isLive = l < live;
          ctx.fillStyle = isLive ? (r === 0 ? C.iceSoft : 'rgba(255,154,77,.3)') : C.faint;
          ctx.fillRect(x, y + 4, cell - 2, rowH - 8);
          ctx.fillStyle = isLive ? '#e8ecf7' : 'rgba(154,165,191,.4)';
          ctx.font = '10px JetBrains Mono, monospace';
          ctx.fillText(String(vals[l]), x + 2, y + rowH * 0.62);
          ctx.font = FONT;
        }
      };
      drawRow(values, 0, 32, 'v');
      steps.forEach((s, i) => {
        if (i + 1 > shown) return;
        drawRow(s.lanes, i + 1, s.offset, `+${s.offset}`);
        // arrows from lane l+offset to lane l
        ctx.strokeStyle = 'rgba(255,154,77,.35)';
        for (let l = 0; l < s.offset; l++) {
          const y0 = 24 + i * rowH + rowH - 4;
          const y1 = 24 + (i + 1) * rowH + 4;
          ctx.beginPath();
          ctx.moveTo(52 + (l + s.offset) * cell + cell / 2, y0);
          ctx.lineTo(52 + l * cell + cell / 2, y1);
          ctx.stroke();
        }
      });
      ctx.fillStyle = C.muted;
      ctx.fillText(shown >= 5 ? `lane 0 holds the total: ${total}. Then __shfl_sync(mask, v, 0) broadcasts it.` : 'each step halves the live lanes: 16, 8, 4, 2, 1', 6, h - 6);
    },
    [seed],
  );

  return (
    <div>
      <canvas ref={ref} className="block h-[300px] w-full rounded-xl bg-[#080b14]" aria-hidden />
      <button type="button" onClick={() => setSeed((s) => s + 1)} className="mt-3 rounded-full border border-line px-3 py-1 text-[13px] text-muted hover:text-ink">
        New values
      </button>
    </div>
  );
}
