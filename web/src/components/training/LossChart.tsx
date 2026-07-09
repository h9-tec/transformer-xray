'use client';

/** Streaming loss curve. Single series (no legend — the title names it),
 * crosshair hover, dashed reference line at ln(65) = what uniform guessing
 * scores. Watching this drop from 4.17 IS the product moment. */

import { useMemo, useState } from 'react';

const UNIFORM = Math.log(65); // ≈ 4.174

export function LossChart({ losses }: { losses: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const H = 220;
  const pad = { l: 42, r: 12, t: 12, b: 26 };

  const { pts, maxY } = useMemo(() => {
    const maxY = Math.max(UNIFORM * 1.05, ...losses);
    const n = Math.max(losses.length, 2);
    const x = (i: number) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
    const y = (v: number) => pad.t + (1 - v / maxY) * (H - pad.t - pad.b);
    return { pts: losses.map((v, i) => `${x(i)},${y(v)}`), maxY, x, y };
  }, [losses]);

  const x = (i: number) => pad.l + (i / Math.max(losses.length - 1, 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / maxY) * (H - pad.t - pad.b);

  return (
    <figure>
      <figcaption className="mb-1 text-xs text-muted">
        training loss (nats/char){' '}
        {hover !== null && losses[hover] !== undefined && (
          <span className="nums text-ink">
            — step {hover + 1}: {losses[hover].toFixed(4)}
          </span>
        )}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full max-w-3xl"
        role="img"
        aria-label="Training loss per step"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const fx = ((e.clientX - rect.left) / rect.width) * W;
          const i = Math.round(((fx - pad.l) / (W - pad.l - pad.r)) * (losses.length - 1));
          setHover(i >= 0 && i < losses.length ? i : null);
        }}
        onMouseLeave={() => setHover(null)}
      >
        {/* axis + gridline at the uniform-guess baseline */}
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--color-line)" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--color-line)" />
        <line
          x1={pad.l}
          y1={y(UNIFORM)}
          x2={W - pad.r}
          y2={y(UNIFORM)}
          stroke="var(--color-faint)"
          strokeDasharray="4 4"
        />
        <text x={W - pad.r} y={y(UNIFORM) - 4} textAnchor="end" fontSize="10" fill="var(--color-faint)" className="nums">
          uniform guess · ln 65 ≈ 4.17
        </text>
        {[1, 2, 3, 4].map((v) => (
          <text key={v} x={pad.l - 6} y={y(v) + 3} textAnchor="end" fontSize="10" fill="var(--color-faint)" className="nums">
            {v}
          </text>
        ))}
        {losses.length > 1 && (
          <polyline points={pts.join(' ')} fill="none" stroke="var(--color-bwd)" strokeWidth="2" />
        )}
        {hover !== null && losses[hover] !== undefined && (
          <g>
            <line x1={x(hover)} y1={pad.t} x2={x(hover)} y2={H - pad.b} stroke="var(--color-faint)" />
            <circle cx={x(hover)} cy={y(losses[hover])} r="4" fill="var(--color-bwd)" />
          </g>
        )}
      </svg>
    </figure>
  );
}
