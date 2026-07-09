'use client';

/** Where learning starts: −ln p(truth) per position, and the famous
 * (p − onehot)/T gradient that kicks off the whole backward pass. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import { useModelStore } from '@/stores/modelStore';
import { currentTrace, useTraceStore } from '@/stores/traceStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };
const label = (ch: string) => GLYPH[ch] ?? ch;

/** Diverging bars around a zero baseline: amber above, blue below. */
function DivergingBars({
  values,
  chars,
  highlight,
}: {
  values: Float32Array;
  chars: string;
  highlight: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const V = values.length;
  const W = 980;
  const H = 170;
  const pad = { top: 14, bottom: 22 };
  const mid = pad.top + (H - pad.top - pad.bottom) / 2;
  const amp = (H - pad.top - pad.bottom) / 2;
  let max = 0;
  for (const v of values) max = Math.max(max, Math.abs(v));
  const pitch = W / V;
  const barW = Math.max(2, pitch - 2);

  return (
    <figure>
      <figcaption className="mb-1 text-xs text-muted">
        ∂L/∂logits at this position — every bet gets a nudge{' '}
        {hover !== null && (
          <span className="nums text-ink">
            — {label(chars[hover])}: {values[hover].toExponential(2)}
          </span>
        )}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Gradient of the loss with respect to each logit">
        <line x1={0} x2={W} y1={mid} y2={mid} stroke="var(--color-line)" />
        {Array.from(values).map((v, i) => {
          const h = max === 0 ? 0 : (Math.abs(v) / max) * amp;
          return (
            <g key={i}>
              <rect
                x={i * pitch}
                y={0}
                width={pitch}
                height={H}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              />
              <rect
                x={i * pitch + (pitch - barW) / 2}
                y={v >= 0 ? mid - h : mid}
                width={barW}
                height={Math.max(h, Math.abs(v) > 0 ? 1 : 0)}
                rx={1.5}
                fill={v >= 0 ? '#ffb454' : '#7aa2ff'}
                opacity={i === highlight || i === hover ? 1 : 0.6}
                pointerEvents="none"
              />
              <text
                x={i * pitch + pitch / 2}
                y={H - 6}
                textAnchor="middle"
                fontSize={9}
                fill={i === highlight ? 'var(--color-accent)' : 'var(--color-faint)'}
                className="nums"
              >
                {label(chars[i])}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

export function LossView() {
  const chars = useModelStore((s) => s.chars);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const loss = useTraceStore((s) => s.loss);
  const [pos, setPos] = useState<number | null>(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const t = currentTrace();
    return t
      ? { perToken: t.byId.get('loss.per_token'), gradLogits: t.byId.get('grad:logits') }
      : null;
  }, [version]);

  if (!nodes?.perToken?.data || !chars) return <p className="text-sm text-faint">…</p>;

  const T = nodes.perToken.shape[0];
  const x = text.slice(0, -1);
  const y = text.slice(1);
  // Default to the most surprising position — where the model was most wrong.
  let argmax = 0;
  for (let i = 1; i < T; i++) if (nodes.perToken.data[i] > nodes.perToken.data[argmax]) argmax = i;
  const sel = pos ?? argmax;
  const target = chars.indexOf(y[sel]);
  const V = chars.length;
  const gradRow = nodes.gradLogits?.data?.subarray(sel * V, (sel + 1) * V) as Float32Array | undefined;

  return (
    <div className="space-y-6">
      <p className="nums text-sm text-muted">
        mean loss = <span className="text-ink">{loss?.toFixed(4)}</span> · uniform guessing would
        be ln(65) = <span className="text-ink">4.174</span>
      </p>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          −ln p(truth) per position — bright = surprised; click a column
        </h3>
        <TensorHeatmap
          data={nodes.perToken.data}
          rows={1}
          cols={T}
          mode="sequential"
          colLabels={Array.from(x).map(label)}
          maxCellPx={14}
          ariaLabel="Per-position cross-entropy loss"
          onCell={(_, c) => setPos(c)}
        />
      </div>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          position {sel}: after{' '}
          <span className="nums text-ink">
            “…{x.slice(Math.max(0, sel - 24), sel + 1).replace(/\n/g, '↵')}”
          </span>{' '}
          truth was <span className="nums text-accent">{label(y[sel])}</span> · loss{' '}
          <span className="nums text-ink">{nodes.perToken.data[sel].toFixed(3)}</span>
        </h3>
        {gradRow && <DivergingBars values={gradRow} chars={chars} highlight={target} />}
        <p className="mt-2 text-xs text-faint">
          Amber bars (p&gt;0 for wrong guesses) push those logits down; the single blue bar at the
          truth (p−1 &lt; 0) pulls it up. Their sum is exactly zero — softmax only cares about
          differences.
        </p>
      </div>
    </div>
  );
}
