'use client';

/** Lessons 1 and 2: the grid of blocks and threads over an array. In
 * `stride` mode each thread walks forward by the grid size and the passes
 * are drawn as rows. */

import { useState } from 'react';
import { gridStrideCoverage, threadGrid } from '@/learn/sims/models';
import { C, FONT, useCanvas } from './useCanvas';

export function ThreadGrid({ mode }: { mode: 'index' | 'stride' }) {
  const [block, setBlock] = useState(256);
  const [n, setN] = useState(mode === 'stride' ? 5000 : 4096);
  const grid = mode === 'stride' ? 4 : Math.ceil(n / block);
  const tg = threadGrid(n, block);
  const cov = gridStrideCoverage(n, grid, block);
  const [hover, setHover] = useState<number | null>(null);

  const ref = useCanvas(
    (ctx, w, h, frame) => {
      const cols = 64;
      const cell = (w - 24) / cols;
      const rows = Math.ceil(n / cols);
      const rowH = Math.min(10, (h - 40) / rows);
      ctx.font = FONT;
      ctx.fillStyle = C.muted;
      ctx.fillText(mode === 'index' ? `${n} elements, ${tg.blocks} blocks × ${block} threads` : `${n} elements, ${grid} blocks × ${block} threads, ${cov.maxIters} passes`, 12, 16);
      const total = grid * block;
      const t = frame < 0 ? total : (frame * 24) % (total + 60);
      for (let i = 0; i < n; i++) {
        const x = 12 + (i % cols) * cell;
        const y = 28 + Math.floor(i / cols) * rowH;
        let color = C.faint;
        if (mode === 'index') {
          const b = Math.floor(i / block);
          const lit = i < t;
          color = lit ? (b % 2 ? C.copper : C.ice) : C.faint;
          if (hover !== null && b === hover) color = C.ok;
        } else {
          const pass = Math.floor(i / total);
          const thread = i % total;
          const lit = thread < t;
          const palette = [C.copper, C.ice, C.rose, C.ok, '#b59bff'];
          color = lit ? palette[pass % palette.length] : C.faint;
        }
        ctx.fillStyle = color;
        ctx.fillRect(x, y, Math.max(1, cell - 1), Math.max(1, rowH - 1));
      }
      ctx.fillStyle = C.muted;
      if (mode === 'index') ctx.fillText(tg.idle ? `last block: ${tg.lastBlockActive} active, ${tg.idle} idle threads` : 'every thread owns exactly one element', 12, h - 8);
      else ctx.fillText(`thread 0 handles ${cov.itersFor(0)} elements, thread ${total - 1} handles ${cov.itersFor(total - 1)}`, 12, h - 8);
    },
    [block, n, hover, mode],
  );

  return (
    <div>
      <canvas ref={ref} className="block h-[260px] w-full rounded-xl bg-[#080b14]" aria-hidden />
      <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
        <label className="flex flex-col gap-1 text-muted">
          Threads per block <span className="nums text-ink">{block}</span>
          <input type="range" min={32} max={1024} step={32} value={block} onChange={(e) => setBlock(Number(e.target.value))} className="accent-[#ff9a4d]" />
        </label>
        <label className="flex flex-col gap-1 text-muted">
          Array length n <span className="nums text-ink">{n}</span>
          <input type="range" min={256} max={8192} step={64} value={n} onChange={(e) => setN(Number(e.target.value))} className="accent-[#74c7ff]" />
        </label>
      </div>
      {mode === 'index' && (
        <div className="mt-2 flex flex-wrap gap-1" onMouseLeave={() => setHover(null)}>
          {Array.from({ length: Math.min(tg.blocks, 32) }, (_, b) => (
            <span key={b} onMouseEnter={() => setHover(b)} className="nums cursor-default rounded border border-line px-1.5 py-0.5 text-[11px] text-muted hover:text-ink">
              block {b}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
