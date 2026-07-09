'use client';

/** Every row of the logits matrix is the same question asked at a different
 * position: "given everything so far, what comes next?" Click a row to see
 * that position's full distribution against what actually came next. */

import { useMemo, useState } from 'react';
import { LogitsBars } from '@/components/playground/LogitsBars';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import { useModelStore } from '@/stores/modelStore';
import { currentTrace, nodeForPhase, useTraceStore } from '@/stores/traceStore';
import { useUiStore } from '@/stores/uiStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };
const label = (ch: string) => GLYPH[ch] ?? ch;

export function LogitsView() {
  const chars = useModelStore((s) => s.chars);
  const version = useTraceStore((s) => s.version);
  const text = useTraceStore((s) => s.text);
  const [pos, setPos] = useState<number | null>(null);
  const phase = useUiStore((s) => s.phase);
  const bwd = phase === 'backward';

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const t = currentTrace();
    return t
      ? { logits: nodeForPhase('logits', phase), perToken: t.byId.get('loss.per_token') }
      : null;
  }, [version, phase]);

  if (!nodes?.logits?.data || !chars) return <p className="text-sm text-faint">…</p>;

  const [T, V] = nodes.logits.shape;
  const x = text.slice(0, -1);
  const y = text.slice(1);
  const sel = pos !== null ? Math.min(pos, T - 1) : T - 1;
  const rowLogits = nodes.logits.data.subarray(sel * V, (sel + 1) * V) as Float32Array;
  const target = chars.indexOf(y[sel]);
  const lossAt = nodes.perToken?.data?.[sel];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm text-muted">
          {bwd
            ? `∂L/∂logits [${T} × ${V}] = (p − onehot)/T — amber blame everywhere the model bet, blue credit at the truth`
            : `logits [${T} × ${V}] — one next-character question per row; click a row`}
        </h3>
        <TensorHeatmap
          data={nodes.logits.data}
          rows={T}
          cols={V}
          mode="diverging"
          rowLabels={Array.from(x).map(label)}
          colLabels={Array.from(chars).map(label)}
          maxCellPx={10}
          ariaLabel="Logits for every position and vocabulary character"
          onCell={(r) => setPos(r)}
        />
      </div>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          position {sel}: after{' '}
          <span className="nums text-ink">
            “…{x.slice(Math.max(0, sel - 24), sel + 1).replace(/\n/g, '↵')}”
          </span>{' '}
          — truth: <span className="nums text-accent">{label(y[sel])}</span>
          {lossAt !== undefined && (
            <>
              {' '}
              · loss −ln p(truth) = <span className="nums text-ink">{lossAt.toFixed(3)}</span>
            </>
          )}
        </h3>
        <LogitsBars
          logits={rowLogits}
          chars={chars}
          chosen={target}
          temperature={1}
          topK={V}
          greedy
        />
      </div>
    </div>
  );
}
