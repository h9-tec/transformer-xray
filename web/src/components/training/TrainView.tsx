'use client';

/** Watch it learn: a real training loop (fwd + bwd + AdamW, ~2-3 steps/s)
 * running in the worker, from random noise or from the checkpoint. */

import { useEffect } from 'react';
import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { useModelStore } from '@/stores/modelStore';
import { useTrainStore } from '@/stores/trainStore';
import { LossChart } from './LossChart';

export function TrainView() {
  const t = useTranslations('train');
  const model = useModelStore();
  const s = useTrainStore();

  useEffect(() => {
    void model.load();
  }, [model.load]);

  const busy = model.status !== 'ready';

  return (
    <div className="space-y-6">
      <div className="panel flex flex-wrap items-center gap-4 p-4">
        <div role="radiogroup" aria-label={t('source')} className="flex gap-1">
          {(['scratch', 'checkpoint'] as const).map((src) => (
            <button
              key={src}
              type="button"
              role="radio"
              aria-checked={s.source === src}
              disabled={s.initialized}
              onClick={() => s.set({ source: src })}
              className={`rounded-full border px-4 py-1.5 text-sm transition-colors disabled:opacity-40 ${
                s.source === src
                  ? 'border-transparent bg-bwd text-bg'
                  : 'border-line text-muted hover:text-ink'
              }`}
            >
              {src === 'scratch' ? t('fromScratch') : t('fromCheckpoint')}
            </button>
          ))}
        </div>

        <label className="text-xs text-muted">
          lr{' '}
          <select
            value={s.lr}
            disabled={s.running}
            onChange={(e) => s.set({ lr: Number(e.target.value) })}
            className="nums ms-1 rounded border border-line bg-bg px-2 py-1 text-ink"
          >
            <option value={3e-3}>3e-3</option>
            <option value={1e-3}>1e-3</option>
            <option value={3e-4}>3e-4</option>
          </select>
        </label>

        <div className="ms-auto flex gap-2">
          {!s.running ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => void s.start()}
              className="rounded-full bg-bwd px-5 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-30"
            >
              {s.initialized ? t('resume') : t('start')}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => s.stop()}
              className="rounded-full border border-bwd px-5 py-1.5 text-sm text-bwd hover:bg-bwd hover:text-bg"
            >
              {t('pause')}
            </button>
          )}
          <button
            type="button"
            disabled={!s.initialized}
            onClick={() => s.reset()}
            className="rounded-full border border-line px-4 py-1.5 text-sm text-muted hover:text-ink disabled:opacity-30"
          >
            {t('reset')}
          </button>
        </div>
      </div>

      {s.error && (
        <p role="alert" className="text-sm text-danger">
          {s.error}
        </p>
      )}

      <div className="nums flex flex-wrap gap-6 text-sm text-muted">
        <span>
          {t('step')} <span className="text-ink">{s.step}</span>
        </span>
        {s.lossHistory.length > 0 && (
          <span>
            {t('loss')}{' '}
            <span className="text-ink">{s.lossHistory[s.lossHistory.length - 1].toFixed(4)}</span>
          </span>
        )}
        {s.initialized && (
          <button
            type="button"
            onClick={() => void s.inspectNow()}
            className="text-update underline-offset-4 hover:underline"
          >
            {t('inspectNow')} →
          </button>
        )}
        {s.initialized && (
          <Link href="/inspect/residual" className="text-update underline-offset-4 hover:underline">
            {t('openInspectors')}
          </Link>
        )}
      </div>

      <LossChart losses={s.lossHistory} />

      {s.sample && (
        <div>
          <h3 className="mb-1 text-sm text-muted">{t('sampleTitle', { step: s.step })}</h3>
          <pre className="nums panel whitespace-pre-wrap p-4 text-sm text-bwd">
            {s.sample}
          </pre>
        </div>
      )}
    </div>
  );
}
