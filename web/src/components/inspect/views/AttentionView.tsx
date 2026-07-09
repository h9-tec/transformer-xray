'use client';

/** Attention, told as its actual four-step transformation:
 * raw scores → ÷√d_k → causal mask → softmax, per head, on real numbers.
 * Then: all six heads side by side — same input, six different strategies. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import type { TraceNode } from '@/engine/tape';
import { useModelStore } from '@/stores/modelStore';
import { nodeForPhase, useTraceStore } from '@/stores/traceStore';
import { useUiStore } from '@/stores/uiStore';

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

  const phase = useUiStore((s) => s.phase);
  const bwd = phase === 'backward';

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const p = `block${layer}.attn`;
    const get = (s: string) => nodeForPhase(`${p}.${s}`, phase);
    return {
      raw: get('scores_raw'),
      scores: get('scores'),
      masked: get('masked'),
      softmax: get('softmax'),
    };
  }, [version, layer, phase]);

  if (!nodes?.softmax?.data || !chars) return <p className="text-sm text-faint">…</p>;

  const xChars = Array.from(text.slice(0, -1)).map(label);
  const d = (t: string) => (bwd ? `∂L/∂(${t})` : t);
  const stages = [
    { key: 'raw', node: nodes.raw, title: d('q·kᵀ'), mode: 'diverging' as const },
    { key: 'scores', node: nodes.scores, title: d('÷ √d_k'), mode: 'diverging' as const },
    { key: 'masked', node: nodes.masked, title: d('+ causal mask'), mode: 'diverging' as const },
    {
      key: 'softmax',
      node: nodes.softmax,
      title: d('softmax'),
      mode: bwd ? ('diverging' as const) : ('sequential' as const),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-4">
        <Selector value={layer} count={nLayer} onChange={setLayer} name="layer" color="bg-accent" />
        <Selector value={head} count={nHead} onChange={setHead} name="head" color="bg-fwd" />
      </div>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          {bwd ? (
            <>
              <span className="text-bwd">gradients</span> flowing back through the four steps,
              layer {layer} head {head}
            </>
          ) : (
            <>the four steps, layer {layer} head {head} — same matrix, transformed</>
          )}
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
          {bwd
            ? `∂L/∂softmax for all ${nHead} heads of layer ${layer}`
            : `all ${nHead} heads of layer ${layer} after softmax — six different strategies over the same input`}
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
                className={`rounded p-1 text-start ${head === h ? (bwd ? 'ring-1 ring-bwd' : 'ring-1 ring-fwd') : ''}`}
              >
                <span className="nums text-xs text-muted">head {h}</span>
                <TensorHeatmap
                  data={data}
                  rows={rows}
                  cols={cols}
                  mode={bwd ? 'diverging' : 'sequential'}
                  maxCellPx={4}
                  ariaLabel={`${bwd ? 'Gradient of attention' : 'Attention'} pattern of head ${h}`}
                />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
