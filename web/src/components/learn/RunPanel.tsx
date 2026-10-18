'use client';

/** The exercise: editor, Run, and the graded result. Runs the learner's
 * kernel on the CUDA worker, grades the returned buffers against the TS
 * reference and the instrumentation targets, and marks the lesson passed. */

import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CudaClient } from '@/cuda/worker/client';
import type { CudaResponse, WireArg, WireError } from '@/cuda/worker/protocol';
import { gradeFromWire, type GradeResult } from '@/learn/exercise';
import { EXERCISES } from '@/learn/exercises';
import { useLearnStore } from '@/stores/learnStore';
import { Editor } from './Editor';

let client: CudaClient | null = null;
const getClient = () => (client ??= new CudaClient());

type Status = 'idle' | 'running' | 'done' | 'error';

declare global {
  interface Window {
    __exercise?: { slug: string; starter: string; solution: string };
  }
}

export function RunPanel({ slug }: { slug: string }) {
  const t = useTranslations('learn');
  const ex = EXERCISES[slug];
  const markPassed = useLearnStore((s) => s.markPassed);
  const storageKey = `xray-ex-${slug}`;

  const [code, setCode] = useState(ex.starter);
  const [status, setStatus] = useState<Status>('idle');
  const [result, setResult] = useState<GradeResult | null>(null);
  const [error, setError] = useState<WireError | null>(null);
  const [ms, setMs] = useState(0);
  const [ran, setRan] = useState(false);
  const [showingSolution, setShowingSolution] = useState(false);
  const draftBeforeSolution = useRef<string | null>(null);

  // restore the learner's draft
  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) setCode(saved);
    } catch {
      /* private mode: start from the starter */
    }
    window.__exercise = { slug, starter: ex.starter, solution: ex.solution };
  }, [storageKey, slug, ex.starter, ex.solution]);

  const onChange = useCallback(
    (text: string) => {
      setCode(text);
      setShowingSolution(false);
      try {
        localStorage.setItem(storageKey, text);
      } catch {
        /* ignore */
      }
    },
    [storageKey],
  );

  const run = useCallback(async () => {
    setStatus('running');
    setError(null);
    setResult(null);
    const built = ex.build(1);
    // copies: the worker takes ownership of what we send
    const args: WireArg[] = built.args.map((a) =>
      a.kind === 'buffer'
        ? { kind: 'buffer', name: a.name, dtype: a.data instanceof Int32Array ? 'i32' : 'f32', buffer: a.data.buffer.slice(0) as ArrayBuffer }
        : { kind: a.kind, value: a.value },
    );
    let res: CudaResponse;
    try {
      res = await getClient().run({
        source: code,
        kernel: ex.kernel,
        grid: ex.launch.grid,
        block: ex.launch.block,
        sharedBytes: ex.launch.sharedBytes,
        args,
        instrument: true,
      });
    } catch (e) {
      setError({ kind: 'runtime', message: e instanceof Error ? e.message : String(e), line: 1, col: 1 });
      setStatus('error');
      setRan(true);
      return;
    }
    setMs(res.ms);
    setRan(true);
    if (res.type === 'error') {
      setError(res.error);
      setStatus('error');
      return;
    }
    const graded = gradeFromWire(ex, built, res.buffers, res.result);
    setResult(graded);
    setStatus('done');
    if (graded.ok) markPassed(slug);
  }, [code, ex, markPassed, slug]);

  const cancel = useCallback(() => {
    getClient().cancel();
    setStatus('idle');
  }, []);

  const reset = useCallback(() => {
    onChange(ex.starter);
    setResult(null);
    setError(null);
    setStatus('idle');
  }, [ex.starter, onChange]);

  const toggleSolution = useCallback(() => {
    if (showingSolution) {
      setCode(draftBeforeSolution.current ?? ex.starter);
      setShowingSolution(false);
    } else {
      draftBeforeSolution.current = code;
      setCode(ex.solution);
      setShowingSolution(true);
    }
  }, [code, ex.solution, ex.starter, showingSolution]);

  const errorLine = error?.line ?? null;
  const launchInfo = useMemo(() => {
    const g = ex.launch.grid.filter((x, i) => i === 0 || x > 1).join('×');
    const b = ex.launch.block.filter((x, i) => i === 0 || x > 1).join('×');
    return `${ex.kernel}<<<${g}, ${b}${ex.launch.sharedBytes ? `, ${ex.launch.sharedBytes} B` : ''}>>>`;
  }, [ex]);

  return (
    <div className="panel overflow-hidden" data-testid="run-panel">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3 text-sm text-muted">
        <b className="nums text-ink" dir="ltr">
          {launchInfo}
        </b>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={reset} className="rounded-full border border-line px-3 py-1 text-[13px] hover:text-ink">
            {t('reset')}
          </button>
          {ran && (
            <button type="button" onClick={toggleSolution} className="rounded-full border border-line px-3 py-1 text-[13px] hover:text-ink">
              {showingSolution ? t('hideSolution') : t('showSolution')}
            </button>
          )}
        </div>
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <Editor id={slug} value={code} onChange={onChange} errorLine={errorLine} />

        <div className="flex min-w-0 flex-col gap-3 border-t border-line p-4 lg:border-s lg:border-t-0">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={status === 'running' ? cancel : () => void run()}
              className={status === 'running' ? 'btn-ghost !py-2' : 'btn-primary !py-2'}
              aria-busy={status === 'running'}
            >
              {status === 'running' ? t('cancel') : t('run')}
            </button>
            {status === 'done' && result && (
              <span className="nums text-xs text-faint">{ms.toFixed(0)} ms</span>
            )}
          </div>

          {status === 'running' && <p className="m-0 text-sm text-muted">{t('running')}</p>}

          {error && (
            <div className="rounded-[10px] border border-danger/40 bg-danger/10 p-3 text-sm" data-testid="run-error" dir="ltr">
              <b className="text-danger">{t('errorAt', { line: error.line })}</b>
              <p className="m-0 mt-1 text-ink">{error.message}</p>
              {error.thread && (
                <p className="nums m-0 mt-1 text-xs text-muted">
                  block ({error.thread.block.join(',')}) thread ({error.thread.thread.join(',')})
                </p>
              )}
            </div>
          )}

          {result && (
            <>
              {result.ok && (
                <div className="rounded-[10px] border border-ok/40 bg-ok/10 px-3 py-2 text-sm font-semibold text-ok" data-testid="exercise-passed">
                  {t('passed')}
                </div>
              )}
              <ul className="m-0 flex list-none flex-col gap-2 p-0">
                {result.targets.map((tg) => (
                  <li
                    key={tg.id}
                    data-testid={`target-${tg.id}`}
                    data-pass={tg.pass}
                    className="grid grid-cols-[22px_1fr] items-start gap-2.5 rounded-[10px] border border-line bg-white/[.03] px-3 py-2 text-sm"
                  >
                    <span
                      className={`mt-0.5 grid h-5 w-5 place-items-center rounded-full text-xs font-bold ${
                        tg.pass ? 'bg-ok/20 text-ok' : 'bg-danger/20 text-danger'
                      }`}
                      aria-label={tg.pass ? 'pass' : 'fail'}
                    >
                      {tg.pass ? '✓' : '×'}
                    </span>
                    <span className="min-w-0">
                      <span className="block">{tg.label}</span>
                      <span className={`nums block text-xs ${tg.pass ? 'text-ok' : 'text-danger'}`} dir="ltr">
                        {tg.detail}
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
              {result.firstMismatch && !Number.isNaN(result.maxDiff) && (
                <p className="nums m-0 text-xs text-muted" dir="ltr">
                  {t('firstMismatch')}: [{result.firstMismatch.index}] got {fmtNum(result.firstMismatch.got)}, want{' '}
                  {fmtNum(result.firstMismatch.want)}
                </p>
              )}
              {result.hint && (
                <p className="m-0 rounded-[10px] border border-dashed border-accent/40 bg-accent/5 p-3 text-[13.5px] text-muted">
                  <b className="text-accent">{t('hint')}: </b>
                  {result.hint}
                </p>
              )}
              <div className="grid grid-cols-2 gap-2 text-xs" data-testid="instr-card">
                <Stat label={t('sectorsPerReq')} value={result.stats.globalRequests ? (result.stats.globalSectors / result.stats.globalRequests).toFixed(1) : '0'} />
                <Stat label={t('bankWorst')} value={String(result.stats.sharedWorstConflict || 0)} />
                <Stat label={t('divergent')} value={String(result.stats.divergentInstructions)} />
                <Stat label={t('racesN')} value={String(result.races.reduce((a, r) => a + r.count, 0))} />
              </div>
              {result.printf.length > 0 && (
                <pre className="nums m-0 max-h-40 overflow-auto rounded-[10px] bg-[#080b14] p-3 text-xs text-muted" dir="ltr">
                  {result.printf.join('\n')}
                </pre>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[8px] border border-line bg-white/[.03] px-2.5 py-1.5">
      <span className="block text-faint">{label}</span>
      <span className="nums text-sm text-ink">{value}</span>
    </div>
  );
}

function fmtNum(x: number): string {
  if (Number.isInteger(x)) return String(x);
  return Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4 ? x.toExponential(3) : x.toFixed(4);
}
