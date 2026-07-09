'use client';

/** The full output distribution — all 65 vocabulary bars, no truncation.
 * Bars show the actual sampling probabilities for the selected step
 * (softmax of logits/T after top-k, exactly what the sampler drew from).
 * The chosen token is amber; top-5 probabilities get direct labels. */

import { useMemo, useState } from 'react';
import { softmax1d, topKFilter } from '@/engine/ops/sample';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };

export interface LogitsBarsProps {
  logits: Float32Array;
  chars: string;
  chosen: number;
  temperature: number;
  topK: number;
  greedy: boolean;
}

export function LogitsBars({ logits, chars, chosen, temperature, topK, greedy }: LogitsBarsProps) {
  const [hover, setHover] = useState<number | null>(null);

  const probs = useMemo(() => {
    if (greedy) return softmax1d(logits);
    const t = Math.max(temperature, 1e-6);
    const scaled = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / t;
    return softmax1d(topKFilter(scaled, topK));
  }, [logits, temperature, topK, greedy]);

  const top5 = useMemo(
    () =>
      Array.from(probs)
        .map((p, i) => [p, i] as const)
        .sort((a, b) => b[0] - a[0])
        .slice(0, 5)
        .map(([, i]) => i),
    [probs],
  );

  const V = probs.length;
  const W = 980;
  const H = 180;
  const pad = { top: 24, bottom: 22, left: 8, right: 8 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;
  const pitch = plotW / V;
  const barW = Math.max(2, pitch - 2); // 2px surface gap between bars
  const maxP = Math.max(...probs, 1e-6);

  const y = (p: number) => pad.top + plotH * (1 - p / maxP);

  return (
    <figure className="relative">
      <figcaption className="mb-1 text-xs text-muted">
        sampling distribution · p(next character){' '}
        {hover !== null && (
          <span className="nums text-ink">
            — {JSON.stringify(GLYPH[chars[hover]] ?? chars[hover]).slice(1, -1)} · p ={' '}
            {(probs[hover] * 100).toFixed(2)}% · logit {logits[hover].toFixed(3)}
          </span>
        )}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Probability of each vocabulary character for the selected step"
      >
        {/* baseline */}
        <line
          x1={pad.left}
          x2={W - pad.right}
          y1={pad.top + plotH}
          y2={pad.top + plotH}
          stroke="var(--color-line)"
        />
        {Array.from(probs).map((p, i) => {
          const isChosen = i === chosen;
          const isHover = i === hover;
          const barH = Math.max(p > 0 ? 1.5 : 0, plotH * (p / maxP));
          return (
            <g key={i}>
              {/* generous hit target */}
              <rect
                x={pad.left + i * pitch}
                y={pad.top}
                width={pitch}
                height={plotH + pad.bottom}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              />
              <rect
                x={pad.left + i * pitch + (pitch - barW) / 2}
                y={y(p)}
                width={barW}
                height={barH}
                rx={2}
                fill={isChosen ? 'var(--color-accent)' : 'var(--color-fwd)'}
                opacity={isChosen || isHover ? 1 : 0.55}
                pointerEvents="none"
              />
              {(top5.includes(i) || isChosen) && p > 0.005 && (
                <text
                  x={pad.left + i * pitch + pitch / 2}
                  y={y(p) - 5}
                  textAnchor="middle"
                  className="nums"
                  fontSize={10}
                  fill={isChosen ? 'var(--color-accent)' : 'var(--color-muted)'}
                >
                  {(p * 100).toFixed(0)}%
                </text>
              )}
              <text
                x={pad.left + i * pitch + pitch / 2}
                y={H - 8}
                textAnchor="middle"
                fontSize={9}
                fill={isChosen ? 'var(--color-accent)' : 'var(--color-faint)'}
                className="nums"
                pointerEvents="none"
              >
                {GLYPH[chars[i]] ?? chars[i]}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
