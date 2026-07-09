'use client';

/** LayerNorm before/after, with the actual per-row mean and rstd the
 * backward pass will reuse. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import { useModelStore } from '@/stores/modelStore';
import { currentTrace, nodeForPhase, useTraceStore } from '@/stores/traceStore';
import { useUiStore } from '@/stores/uiStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };

const SITES = [
  { id: (b: number) => `block${b}.ln1`, input: (b: number) => (b === 0 ? 'emb.sum' : `block${b - 1}.res2`), name: 'ln1 (pre-attention)' },
  { id: (b: number) => `block${b}.ln2`, input: (b: number) => `block${b}.res1`, name: 'ln2 (pre-MLP)' },
  { id: () => 'final.ln', input: (b: number) => `block${b}.res2`, name: 'final (pre-logits)' },
] as const;

export function LayerNormView() {
  const chars = useModelStore((s) => s.chars);
  const cfgJson = useModelStore((s) => s.cfgJson);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const [layer, setLayer] = useState(0);
  const [site, setSite] = useState(0);
  const phase = useUiStore((s) => s.phase);
  const bwd = phase === 'backward';

  const nLayer = cfgJson?.model.n_layer ?? 4;
  const lastLayer = nLayer - 1;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const t = currentTrace();
    if (!t) return null;
    const s = SITES[site];
    const b = site === 2 ? lastLayer : layer;
    const id = s.id(b);
    return {
      input: nodeForPhase(s.input(b), phase),
      out: nodeForPhase(id, phase),
      mean: t.byId.get(`${id}.mean`),
      rstd: t.byId.get(`${id}.rstd`),
    };
  }, [version, layer, site, lastLayer, phase]);

  if (!nodes?.out?.data || !chars) return <p className="text-sm text-faint">…</p>;

  const xChars = Array.from(text.slice(0, -1)).map((c) => GLYPH[c] ?? c);
  const [T, D] = nodes.out.shape;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <div role="group" aria-label="site" className="flex gap-1">
          {SITES.map((s, i) => (
            <button
              key={s.name}
              type="button"
              aria-pressed={site === i}
              onClick={() => setSite(i)}
              className={`rounded border px-3 py-1 text-xs ${
                site === i ? 'border-transparent bg-accent text-bg' : 'border-line text-muted hover:text-ink'
              }`}
            >
              {s.name}
            </button>
          ))}
        </div>
        {site !== 2 && (
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
        )}
      </div>

      {nodes.input?.data && (
        <div>
          <h3 className="mb-2 text-sm text-muted">
            {bwd ? `∂L/∂x [${T} × ${D}] — gradient entering below` : `input x [${T} × ${D}] — drifting scales`}
          </h3>
          <TensorHeatmap
            data={nodes.input.data}
            rows={T}
            cols={D}
            mode="diverging"
            rowLabels={xChars}
            maxCellPx={7}
            ariaLabel="LayerNorm input"
          />
        </div>
      )}

      <div>
        <h3 className="mb-2 text-sm text-muted">
          {bwd
            ? '∂L/∂output — the closed-form backward reuses the cached μ and 1/σ below'
            : 'output — every row re-centered to μ=0, σ=1, then scaled by g and shifted by b'}
        </h3>
        <TensorHeatmap
          data={nodes.out.data}
          rows={T}
          cols={D}
          mode="diverging"
          rowLabels={xChars}
          maxCellPx={7}
          ariaLabel="LayerNorm output"
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {nodes.mean?.data && (
          <div>
            <h3 className="mb-1 text-sm text-muted">μ per position (cached for backward)</h3>
            <TensorHeatmap
              data={nodes.mean.data}
              rows={1}
              cols={T}
              mode="diverging"
              colLabels={xChars}
              maxCellPx={12}
              ariaLabel="Per-position mean"
            />
          </div>
        )}
        {nodes.rstd?.data && (
          <div>
            <h3 className="mb-1 text-sm text-muted">1/σ per position (cached for backward)</h3>
            <TensorHeatmap
              data={nodes.rstd.data}
              rows={1}
              cols={T}
              mode="sequential"
              colLabels={xChars}
              maxCellPx={12}
              ariaLabel="Per-position reciprocal standard deviation"
            />
          </div>
        )}
      </div>
    </div>
  );
}
