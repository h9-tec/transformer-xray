'use client';

/** Attention, told as its actual four-step transformation:
 * raw scores → ÷√d_k → causal mask → softmax, per head, on real numbers.
 * Then: all six heads side by side — same input, six different strategies. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import type { TraceNode } from '@/engine/tape';
import { useModelStore } from '@/stores/modelStore';
import { currentTrace, useTraceStore } from '@/stores/traceStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };
const label = (ch: string) => GLYPH[ch] ?? ch;

function headSlice(node: TraceNode, h: number): { data: Float32Array; rows: number; cols: number } {
  const [, T, C] = node.shape;
  return { data: node.data!.subarray(h * T * C, (h + 1) * T * C) as Float32Array, rows: T, cols: C };
}

function Selector({
  value,
  count,
  onChange,
  name,
  color,
}: {
  value: number;
  count: number;
  onChange: (v: number) => void;
  name: string;
  color: string;
}) {
  return (
    <div role="group" aria-label={name} className="flex items-center gap-1">
      <span className="me-1 text-xs text-muted">{name}</span>
      {Array.from({ length: count }, (_, i) => (
        <button
          key={i}
          type="button"
          aria-pressed={value === i}
          onClick={() => onChange(i)}
          className={`nums h-7 w-7 rounded border text-xs transition-colors ${
            value === i ? `border-transparent ${color} text-bg` : 'border-line text-muted hover:text-ink'
          }`}
        >
          {i}
        </button>
      ))}
    </div>
  );
}

export function AttentionView() {
  const chars = useModelStore((s) => s.chars);
  const cfgJson = useModelStore((s) => s.cfgJson);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);

  const nLayer = cfgJson?.model.n_layer ?? 4;
  const nHead = cfgJson?.model.n_head ?? 6;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const t = currentTrace();
    if (!t) return null;
    const p = `block${layer}.attn`;
    return {
      raw: t.byId.get(`${p}.scores_raw`),
      scores: t.byId.get(`${p}.scores`),
      masked: t.byId.get(`${p}.masked`),
      softmax: t.byId.get(`${p}.softmax`),
      q: t.byId.get(`${p}.q`),
      k: t.byId.get(`${p}.k`),
      wv: t.byId.get(`${p}.wv`),
    };
  }, [version, layer]);

  if (!nodes?.softmax?.data || !chars) return <p className="text-sm text-faint">…</p>;

  const xChars = Array.from(text.slice(0, -1)).map(label);
  const stages = [
    { key: 'raw', node: nodes.raw, title: 'q·kᵀ (raw)', mode: 'diverging' as const },
    { key: 'scores', node: nodes.scores, title: '÷ √d_k', mode: 'diverging' as const },
    { key: 'masked', node: nodes.masked, title: '+ causal mask', mode: 'diverging' as const },
    { key: 'softmax', node: nodes.softmax, title: 'softmax', mode: 'sequential' as const },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-4">
        <Selector value={layer} count={nLayer} onChange={setLayer} name="layer" color="bg-accent" />
        <Selector value={head} count={nHead} onChange={setHead} name="head" color="bg-fwd" />
      </div>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          the four steps, layer {layer} head {head} — same matrix, transformed
        </h3>
        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          {stages.map((s) => {
            if (!s.node?.data) return null;
            const { data, rows, cols } = headSlice(s.node, head);
            return (
              <figure key={s.key}>
                <figcaption className="nums mb-1 text-xs text-muted">{s.title}</figcaption>
                <TensorHeatmap
                  data={data}
                  rows={rows}
                  cols={cols}
                  mode={s.mode}
                  rowLabels={xChars}
                  colLabels={xChars}
                  maxCellPx={6}
                  ariaLabel={`${s.title} matrix for layer ${layer} head ${head}`}
                />
              </figure>
            );
          })}
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          all {nHead} heads of layer {layer} after softmax — six different strategies over the same
          input
        </h3>
        <div className="grid grid-cols-3 gap-3 xl:grid-cols-6">
          {Array.from({ length: nHead }, (_, h) => {
            const { data, rows, cols } = headSlice(nodes.softmax!, h);
            return (
              <button
                key={h}
                type="button"
                onClick={() => setHead(h)}
                aria-pressed={head === h}
                className={`rounded p-1 text-start ${head === h ? 'ring-1 ring-fwd' : ''}`}
              >
                <span className="nums text-xs text-muted">head {h}</span>
                <TensorHeatmap
                  data={data}
                  rows={rows}
                  cols={cols}
                  mode="sequential"
                  maxCellPx={4}
                  ariaLabel={`Attention pattern of head ${h}`}
                />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
