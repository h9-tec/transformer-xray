'use client';

/** The MLP: widen ×4, gate with GELU, project back. The gelu heatmap
 * visibly zeroes the negative half of fc — the nonlinearity at work. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import { useModelStore } from '@/stores/modelStore';
import { currentTrace, useTraceStore } from '@/stores/traceStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };

function GeluCurve() {
  const pts: string[] = [];
  for (let i = 0; i <= 120; i++) {
    const x = -4 + (8 * i) / 120;
    const y = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    pts.push(`${20 + ((x + 4) / 8) * 200},${64 - y * 14}`);
  }
  return (
    <svg viewBox="0 0 240 90" className="w-56" role="img" aria-label="GELU function curve">
      <line x1="20" y1="64" x2="220" y2="64" stroke="var(--color-line)" />
      <line x1="120" y1="8" x2="120" y2="82" stroke="var(--color-line)" />
      <polyline points={pts.join(' ')} fill="none" stroke="var(--color-fwd)" strokeWidth="2" />
      <text x="222" y="68" fontSize="9" fill="var(--color-faint)">x</text>
      <text x="124" y="14" fontSize="9" fill="var(--color-faint)">gelu(x)</text>
    </svg>
  );
}

export function MlpView() {
  const chars = useModelStore((s) => s.chars);
  const cfgJson = useModelStore((s) => s.cfgJson);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const [layer, setLayer] = useState(0);
  const nLayer = cfgJson?.model.n_layer ?? 4;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const t = currentTrace();
    if (!t) return null;
    const p = `block${layer}.mlp`;
    return {
      fc: t.byId.get(`${p}.fc`),
      gelu: t.byId.get(`${p}.gelu`),
      proj: t.byId.get(`${p}.proj`),
    };
  }, [version, layer]);

  if (!nodes?.fc?.data || !chars) return <p className="text-sm text-faint">…</p>;

  const xChars = Array.from(text.slice(0, -1)).map((c) => GLYPH[c] ?? c);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4">
        <div role="group" aria-label="layer" className="flex items-center gap-1">
          <span className="me-1 text-xs text-muted">layer</span>
          {Array.from({ length: nLayer }, (_, i) => (
            <button
              key={i}
              type="button"
              aria-pressed={layer === i}
              onClick={() => setLayer(i)}
              className={`nums h-7 w-7 rounded border text-xs ${
                layer === i ? 'border-transparent bg-fwd text-bg' : 'border-line text-muted hover:text-ink'
              }`}
            >
              {i}
            </button>
          ))}
        </div>
        <GeluCurve />
      </div>

      {(
        [
          ['fc', nodes.fc, 'x·W_fc + b — widened ×4'],
          ['gelu', nodes.gelu, 'gelu(fc) — negatives gated toward zero'],
          ['proj', nodes.proj, 'gelu·W_proj + b — back to the stream'],
        ] as const
      ).map(([key, node, title]) =>
        node?.data ? (
          <div key={key}>
            <h3 className="mb-2 text-sm text-muted">
              {key} [{node.shape[0]} × {node.shape[1]}] — {title}
            </h3>
            <TensorHeatmap
              data={node.data}
              rows={node.shape[0]}
              cols={node.shape[1]}
              mode="diverging"
              rowLabels={xChars}
              maxCellPx={key === 'proj' ? 7 : 3}
              ariaLabel={`${key} tensor`}
            />
          </div>
        ) : null,
      )}
    </div>
  );
}
