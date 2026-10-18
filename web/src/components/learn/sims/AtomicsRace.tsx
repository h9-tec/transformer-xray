'use client';

/** Lesson 10: run the plain `+=` scatter and the atomicAdd scatter on the
 * in-browser GPU and compare what the race detector sees. Honest about the
 * simulator: blocks run one after another here, so the plain version may
 * still sum correctly; the race report is the evidence. */

import { useCallback, useState } from 'react';
import { CudaClient } from '@/cuda/worker/client';
import type { WireArg } from '@/cuda/worker/protocol';
import { gradeFromWire, type GradeResult } from '@/learn/exercise';
import { EXERCISES } from '@/learn/exercises';

let client: CudaClient | null = null;
const getClient = () => (client ??= new CudaClient());

type Outcome = { races: number; maxDiff: number; ms: number } | { error: string };

async function runVariant(source: string): Promise<Outcome> {
  const ex = EXERCISES.atomics;
  const built = ex.build(1);
  const args: WireArg[] = built.args.map((a) =>
    a.kind === 'buffer'
      ? { kind: 'buffer', name: a.name, dtype: a.data instanceof Int32Array ? 'i32' : 'f32', buffer: a.data.buffer.slice(0) as ArrayBuffer }
      : { kind: a.kind, value: a.value },
  );
  const res = await getClient().run({ source, kernel: ex.kernel, grid: ex.launch.grid, block: ex.launch.block, args, instrument: true });
  if (res.type === 'error') return { error: res.error.message };
  const g: GradeResult = gradeFromWire(ex, built, res.buffers, res.result);
  return { races: g.races.reduce((a, r) => a + r.count, 0), maxDiff: g.maxDiff, ms: res.ms };
}

export function AtomicsRace() {
  const [plain, setPlain] = useState<Outcome | null>(null);
  const [atomic, setAtomic] = useState<Outcome | null>(null);
  const [busy, setBusy] = useState(false);
  const run = useCallback(async () => {
    setBusy(true);
    setPlain(await runVariant(EXERCISES.atomics.starter));
    setAtomic(await runVariant(EXERCISES.atomics.solution));
    setBusy(false);
  }, []);

  const ids = Array.from({ length: 16 }, (_, i) => (i * 3 + 1) % 5);
  const counts = [0, 1, 2, 3, 4].map((v) => ids.filter((x) => x === v).length);

  return (
    <div>
      <svg viewBox="0 0 720 150" className="block h-auto w-full rounded-xl bg-[#080b14]" role="img" aria-label="Token collisions">
        <text x="12" y="16" fill="#9aa5bf" fontSize="12" fontFamily="Instrument Sans, system-ui">16 positions, 5 distinct tokens: every row of dWte below receives several additions</text>
        {ids.map((v, t) => (
          <g key={t}>
            <rect x={12 + t * 43} y={30} width={38} height={24} rx={4} fill="rgba(116,199,255,.18)" stroke="rgba(116,199,255,.5)" />
            <text x={31 + t * 43} y={46} textAnchor="middle" fill="#e8ecf7" fontSize="11" fontFamily="JetBrains Mono, monospace">t{t}:{v}</text>
            <line x1={31 + t * 43} y1={54} x2={80 + v * 130} y2={100} stroke="rgba(255,154,77,.35)" />
          </g>
        ))}
        {counts.map((c, v) => (
          <g key={v}>
            <rect x={30 + v * 130} y={100} width={100} height={30} rx={5} fill={c > 1 ? 'rgba(255,107,122,.18)' : 'rgba(94,230,168,.15)'} stroke={c > 1 ? '#ff6b7a' : '#5ee6a8'} />
            <text x={80 + v * 130} y={119} textAnchor="middle" fill="#e8ecf7" fontSize="11" fontFamily="JetBrains Mono, monospace">dWte row {v}: {c} adds</text>
          </g>
        ))}
      </svg>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
        <button type="button" onClick={() => void run()} disabled={busy} className="btn-primary !py-1.5 text-sm disabled:opacity-50">
          {busy ? 'Running both kernels…' : 'Run plain += and atomicAdd'}
        </button>
        <Card title="plain +=" o={plain} />
        <Card title="atomicAdd" o={atomic} />
      </div>
    </div>
  );
}

function Card({ title, o }: { title: string; o: Outcome | null }) {
  if (!o) return <span className="nums rounded-lg border border-line px-3 py-1.5 text-muted">{title}: not run</span>;
  if ('error' in o) return <span className="nums rounded-lg border border-danger/40 px-3 py-1.5 text-danger">{title}: {o.error}</span>;
  return (
    <span className={`nums rounded-lg border px-3 py-1.5 ${o.races ? 'border-danger/40 text-danger' : 'border-ok/40 text-ok'}`}>
      {title}: {o.races} race{o.races === 1 ? '' : 's'}, max diff {o.maxDiff.toExponential(1)}
    </span>
  );
}
