'use client';

/** One AdamW step, dissected on a real parameter tensor: the weight, its
 * gradient from the current trace, and the exact update AdamW would apply
 * (t=1 from a fresh optimizer — first-step Adam is ≈ sign descent). */

import { useMemo, useState } from 'react';
import { TensorHeatmap } from '@/components/heatmap/TensorHeatmap';
import { loadedModel } from '@/stores/modelStore';
import { currentTrace, useTraceStore } from '@/stores/traceStore';

const HP = { lr: 3e-4, beta1: 0.9, beta2: 0.95, eps: 1e-8, wd: 0.1 };

function firstStepDelta(p: Float32Array, g: Float32Array): Float32Array {
  // t=1: m̂ = g, v̂ = g² (bias correction cancels the (1−β) factors exactly)
  const d = new Float32Array(p.length);
  for (let i = 0; i < p.length; i++) {
    d[i] = -HP.lr * HP.wd * p[i] - (HP.lr * g[i]) / (Math.abs(g[i]) + HP.eps);
  }
  return d;
}

export function OptimizerView() {
  const version = useTraceStore((s) => s.version);
  const [name, setName] = useState('block0.attn.qkv.w');

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const data = useMemo(() => {
    const t = currentTrace();
    const model = loadedModel();
    if (!t || !model) return null;
    const grad = t.paramGrads.get(name);
    const param = model.weights.get(name);
    if (!grad || !param) return null;
    const [rows, cols] = param.shape.length === 2 ? param.shape : [1, param.shape[0]];
    return {
      names: Array.from(t.paramGrads.keys()),
      param,
      grad,
      delta: firstStepDelta(param.data, grad.data),
      rows,
      cols,
    };
  }, [version, name]);

  if (!data) return <p className="text-sm text-faint">…</p>;

  const panels = [
    { key: 'p', d: data.param.data, title: 'the parameter θ (from the checkpoint)' },
    { key: 'g', d: data.grad.data, title: '∂L/∂θ — this trace’s gradient' },
    {
      key: 'delta',
      d: data.delta,
      title: 'Δθ AdamW would apply — notice: ≈ −lr · sign(g), magnitude nearly uniform',
    },
  ] as const;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs text-muted">
          parameter{' '}
          <select
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="nums ms-2 rounded border border-line bg-surface px-2 py-1 text-ink"
          >
            {data.names.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <p className="nums text-xs text-faint">
          lr {HP.lr} · β₁ {HP.beta1} · β₂ {HP.beta2} · ε {HP.eps} · wd {HP.wd} · t = 1
        </p>
      </div>

      {panels.map((p) => (
        <div key={p.key}>
          <h3 className="mb-2 text-sm text-muted">{p.title}</h3>
          <TensorHeatmap
            data={p.d}
            rows={data.rows}
            cols={data.cols}
            mode="diverging"
            maxCellPx={data.rows === 1 ? 12 : 5}
            ariaLabel={p.title}
          />
        </div>
      ))}
    </div>
  );
}
