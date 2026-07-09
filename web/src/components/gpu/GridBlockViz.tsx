'use client';

/** The launch-geometry schematic: how THIS op's real output matrix tiles
 * into thread blocks. The grid dims come from the same ⌈N/TILE⌉ arithmetic
 * as the kernel launch; the slider steps blockIdx through the grid. Honest
 * by construction: it's a schematic of real geometry, not a simulation. */

import { useState } from 'react';

const TILE = 32;

export function GridBlockViz({ M, N, opLabel }: { M: number; N: number; opLabel: string }) {
  const gx = Math.ceil(N / TILE);
  const gy = Math.ceil(M / TILE);
  const total = gx * gy;
  const [block, setBlock] = useState(0);
  const bx = block % gx;
  const by = Math.floor(block / gx);

  const W = 360;
  const scale = W / (gx * TILE);
  const H = gy * TILE * scale;

  return (
    <div className="space-y-2">
      <p className="nums text-xs text-muted">
        {opLabel}: output [{M} × {N}] → dim3 grid({gx}, {gy}) · dim3 block({TILE}, {TILE}) ·{' '}
        {total} blocks × {TILE * TILE} threads
      </p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full max-w-sm"
        role="img"
        aria-label={`Thread-block tiling of the ${M} by ${N} output matrix`}
      >
        {Array.from({ length: gy }, (_, y) =>
          Array.from({ length: gx }, (_, x) => {
            const isActive = x === bx && y === by;
            const w = Math.min(TILE, N - x * TILE) * scale;
            const h = Math.min(TILE, M - y * TILE) * scale;
            return (
              <rect
                key={`${x}-${y}`}
                x={x * TILE * scale + 1}
                y={y * TILE * scale + 1}
                width={w - 2}
                height={h - 2}
                rx={2}
                fill={isActive ? 'var(--color-fwd)' : 'var(--color-raised)'}
                stroke="var(--color-line)"
                opacity={isActive ? 1 : 0.9}
              />
            );
          }),
        )}
      </svg>
      <label className="flex items-center gap-3 text-xs text-muted">
        <span className="nums">
          blockIdx = ({bx}, {by})
        </span>
        <input
          type="range"
          min={0}
          max={total - 1}
          value={block}
          onChange={(e) => setBlock(Number(e.target.value))}
          className="w-48 accent-(--color-fwd)"
          aria-label="Step through thread blocks"
        />
        <span className="nums text-faint">
          computes C[{by * TILE}..{Math.min(M, (by + 1) * TILE) - 1}, {bx * TILE}..
          {Math.min(N, (bx + 1) * TILE) - 1}]
        </span>
      </label>
    </div>
  );
}
