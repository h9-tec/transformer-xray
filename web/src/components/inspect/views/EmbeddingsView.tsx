'use client';

/** The full 65×192 embedding matrix — the reason this model is char-level:
 * every learned vector fits on one screen. Click a row to see which other
 * characters the model considers similar (cosine), then the live trace of
 * the inspected text's embeddings. */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import { loadedModel, useModelStore } from '@/stores/modelStore';
import { currentTrace, nodeForPhase, useTraceStore } from '@/stores/traceStore';
import { useUiStore } from '@/stores/uiStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };
const label = (ch: string) => GLYPH[ch] ?? ch;

function cosineNeighbors(wte: Float32Array, d: number, row: number, k: number): [number, number][] {
  const V = wte.length / d;
  const norms = new Float32Array(V);
  for (let v = 0; v < V; v++) {
    let n = 0;
    for (let i = 0; i < d; i++) n += wte[v * d + i] ** 2;
    norms[v] = Math.sqrt(n) || 1;
  }
  const sims: [number, number][] = [];
  for (let v = 0; v < V; v++) {
    if (v === row) continue;
    let dot = 0;
    for (let i = 0; i < d; i++) dot += wte[row * d + i] * wte[v * d + i];
    sims.push([dot / (norms[row] * norms[v]), v]);
  }
  return sims.sort((a, b) => b[0] - a[0]).slice(0, k);
}

export function EmbeddingsView() {
  const chars = useModelStore((s) => s.chars);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const [picked, setPicked] = useState<number | null>(null);
  const phase = useUiStore((s) => s.phase);
  const bwd = phase === 'backward';

  const model = loadedModel();
  const wte = model?.weights.get('wte');
  const d = model?.cfg.nEmbd ?? 192;

  const neighbors = useMemo(
    () => (wte && picked !== null ? cosineNeighbors(wte.data, d, picked, 5) : null),
    [wte, picked, d],
  );

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const embSum = useMemo(() => nodeForPhase('emb.sum', phase), [version, phase]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const wteGrad = useMemo(
    () => (bwd ? currentTrace()?.paramGrads.get('wte') : undefined),
    [version, bwd],
  );
  const xChars = Array.from(text.slice(0, -1)).map(label);

  if (!wte || !chars) return <p className="text-sm text-faint">…</p>;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm text-muted">
          {bwd && wteGrad
            ? `∂L/∂wte [${chars.length} × ${d}] — rows light up only for characters in the batch (scatter-add) plus the tied-head contribution`
            : `wte — the whole token embedding matrix [${chars.length} × ${d}]; click a row`}
        </h3>
        <TensorHeatmap
          data={bwd && wteGrad ? wteGrad.data : wte.data}
          rows={chars.length}
          cols={d}
          mode="diverging"
          rowLabels={Array.from(chars).map(label)}
          maxCellPx={8}
          ariaLabel={bwd ? 'Gradient of the token embedding matrix' : 'Token embedding matrix, one row per character'}
          onCell={(r) => setPicked(r)}
        />
      </div>

      {picked !== null && neighbors && (
        <div className="rounded-lg border border-line bg-surface p-4">
          <p className="mb-2 text-sm">
            <span className="nums rounded bg-accent px-1.5 py-0.5 text-bg">
              {label(chars[picked])}
            </span>{' '}
            <span className="text-muted">nearest by cosine — the geometry it learned:</span>
          </p>
          <div className="flex flex-wrap gap-2">
            {neighbors.map(([sim, v]) => (
              <span key={v} className="nums rounded border border-line px-2 py-1 text-xs">
                <span className="text-ink">{label(chars[v])}</span>{' '}
                <span className="text-faint">{sim.toFixed(3)}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {embSum?.data && (
        <div>
          <h3 className="mb-2 text-sm text-muted">
            {bwd
              ? `∂L/∂emb.sum — the gradient reaching all the way back to the embeddings`
              : `emb.sum — token + position embeddings for the inspected text [T × ${d}], live from the trace`}
          </h3>
          <TensorHeatmap
            data={embSum.data}
            rows={embSum.shape[0]}
            cols={embSum.shape[1]}
            mode="diverging"
            rowLabels={xChars}
            maxCellPx={8}
            ariaLabel="Summed embeddings for the inspected text"
          />
        </div>
      )}
    </div>
  );
}
