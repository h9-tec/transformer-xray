'use client';

/** The shared inspection input: one passage of text is traced (forward +
 * backward) and every inspector page reads from that same tape. */

import { useEffect } from 'react';
import { useTranslations } from 'next-intl';
import { useModelStore } from '@/stores/modelStore';
import { useTraceStore } from '@/stores/traceStore';

// Temporary diagnostics: reachable from the browser console.
if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__stores = {
    model: useModelStore,
    trace: useTraceStore,
  };
}

export function InspectBar() {
  const t = useTranslations('inspect');
  const model = useModelStore();
  const { text, status, error, loss, T, setText, trace } = useTraceStore();

  useEffect(() => {
    void model.load();
  }, [model.load]);

  // Auto-trace once the model is ready and nothing has been traced yet.
  useEffect(() => {
    if (model.status === 'ready' && status === 'idle') void trace();
  }, [model.status, status, trace]);

  return (
    <div className="space-y-2 rounded-lg border border-line bg-surface p-3">
      <div className="flex flex-wrap items-center gap-3">
        <input
          value={text.replace(/\n/g, '⏎')}
          onChange={(e) => setText(e.target.value.replace(/⏎/g, '\n'))}
          spellCheck={false}
          aria-label={t('inputLabel')}
          className="nums min-w-0 flex-1 rounded border border-line bg-bg px-3 py-1.5 text-sm text-ink outline-none focus:border-accent"
        />
        <button
          type="button"
          onClick={() => void trace()}
          disabled={status === 'tracing' || model.status !== 'ready'}
          className="rounded-full bg-fwd px-4 py-1.5 text-sm font-medium text-bg hover:opacity-90 disabled:opacity-30"
        >
          {status === 'tracing' ? t('tracing') : t('traceButton')}
        </button>
      </div>
      <p className="nums text-xs text-faint">
        {model.status === 'error' ? (
          <span className="text-danger">model: {model.error}</span>
        ) : status === 'error' ? (
          <span className="text-danger">{error}</span>
        ) : status === 'ready' && loss !== null ? (
          <>
            T = {T} · loss = <span className="text-ink">{loss.toFixed(4)}</span> ·{' '}
            {t('liveNote')}
          </>
        ) : (
          t('waiting')
        )}
      </p>
    </div>
  );
}
