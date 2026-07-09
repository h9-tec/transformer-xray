'use client';

/** The residual stream — one vector per position flowing through the whole
 * network, with attention and MLP writing small updates into it. The norm
 * chart shows the stream growing as layers add information. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import type { TraceNode } from '@/engine/tape';
import { useModelStore } from '@/stores/modelStore';
import { currentTrace, useTraceStore } from '@/stores/traceStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };

function meanL2(node: TraceNode): number {
  const [T, D] = node.shape;
  let total = 0;
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let i = 0; i < D; i++) s += node.data![t * D + i] ** 2;
    total += Math.sqrt(s);
  }
  return total / T;
}

function NormChart({ points, labels }: { points: number[]; labels: string[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 520;
  const H = 130;
  const pad = { l: 34, r: 10, t: 12, b: 26 };
  const max = Math.max(...points) * 1.1;
  const x = (i: number) => pad.l + (i / (points.length - 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / max) * (H - pad.t - pad.b);
  return (
    <figure>
      <figcaption className="mb-1 text-xs text-muted">
        mean ‖x‖₂ per stage{' '}
        {hover !== null && (
          <span className="nums text-ink">
            — {labels[hover]}: {points[hover].toFixed(2)}
          </span>
        )}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-xl" role="img" aria-label="Residual stream norm per stage">
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--color-line)" />
        <polyline
          points={points.map((p, i) => `${x(i)},${y(p)}`).join(' ')}
          fill="none"
          stroke="var(--color-fwd)"
          strokeWidth="2"
        />
        {points.map((p, i) => (
          <g key={i}>
            <rect
              x={x(i) - 14}
              y={0}
              width={28}
              height={H}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
            <circle cx={x(i)} cy={y(p)} r={hover === i ? 5 : 3.5} fill="var(--color-fwd)" />
            <text x={x(i)} y={H - 10} textAnchor="middle" fontSize="9" fill="var(--color-faint)" className="nums">
              {labels[i]}
            </text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

export function ResidualView() {
  const chars = useModelStore((s) => s.chars);
  const cfgJson = useModelStore((s) => s.cfgJson);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const nLayer = cfgJson?.model.n_layer ?? 4;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const stream = useMemo(() => {
    const t = currentTrace();
    if (!t) return null;
    const stages: { id: string; short: string; node: TraceNode }[] = [];
    const add = (id: string, short: string) => {
      const node = t.byId.get(id);
      if (node?.data) stages.push({ id, short, node });
    };
    add('emb.sum', 'emb');
    for (let b = 0; b < nLayer; b++) {
      add(`block${b}.res1`, `${b}·attn`);
      add(`block${b}.res2`, `${b}·mlp`);
    }
    return stages;
  }, [version, nLayer]);

  if (!stream || stream.length === 0 || !chars) return <p className="text-sm text-faint">…</p>;

  const xChars = Array.from(text.slice(0, -1)).map((c) => GLYPH[c] ?? c);
  const norms = stream.map((s) => meanL2(s.node));

  return (
    <div className="space-y-6">
      <NormChart points={norms} labels={stream.map((s) => s.short)} />
      <div className="space-y-3">
        {stream.map((s) => (
          <div key={s.id}>
            <h3 className="nums mb-1 text-xs text-muted">{s.id} — mean ‖x‖₂ {meanL2(s.node).toFixed(2)}</h3>
            <TensorHeatmap
              data={s.node.data!}
              rows={s.node.shape[0]}
              cols={s.node.shape[1]}
              mode="diverging"
              rowLabels={xChars}
              maxCellPx={4}
              ariaLabel={`Residual stream at ${s.id}`}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
