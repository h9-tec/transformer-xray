'use client';

/** "Run on: CPU · your GPU". The GPU option is disabled, with the reason
 * shown, when the browser has no WebGPU adapter; timings from the GPU path
 * are labeled as WebGPU timings and never mixed with the CUDA bench. */

import { useTranslations } from 'next-intl';
import { useModelStore } from '@/stores/modelStore';

export function BackendSwitch({ lastMs }: { lastMs?: number | null }) {
  const t = useTranslations('backend');
  const { backend, backendInfo, gpuAvailable, setBackend } = useModelStore();
  const gpuDisabled = gpuAvailable === false;
  return (
    <div className="panel px-3.5 py-3 text-sm" data-testid="backend-switch">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted">{t('runOn')}</span>
        <div role="group" aria-label={t('runOn')} className="inline-flex overflow-hidden rounded-full border border-line">
          <button
            type="button"
            aria-pressed={backend === 'cpu'}
            onClick={() => void setBackend('cpu')}
            className={`px-3.5 py-1 text-[13px] ${backend === 'cpu' ? 'bg-accent font-medium text-bg' : 'text-muted hover:text-ink'}`}
          >
            CPU
          </button>
          <button
            type="button"
            aria-pressed={backend === 'gpu'}
            disabled={gpuDisabled}
            onClick={() => void setBackend('gpu')}
            title={gpuDisabled ? (backendInfo.reason ?? '') : undefined}
            className={`px-3.5 py-1 text-[13px] disabled:cursor-not-allowed disabled:opacity-40 ${
              backend === 'gpu' ? 'bg-accent font-medium text-bg' : 'text-muted hover:text-ink'
            }`}
          >
            {t('yourGpu')}
          </button>
        </div>
        <span className="nums text-xs text-faint" data-testid="backend-active">
          {backendInfo.active}
        </span>
      </div>
      <p className="nums m-0 mt-1.5 text-xs text-faint" dir="ltr">
        {backendInfo.active === 'gpu' && backendInfo.adapter ? (
          <>
            {backendInfo.adapter}
            {lastMs != null && (
              <span data-testid="backend-ms">
                {' '}
                {lastMs.toFixed(1)} ms {t('perStep')}
              </span>
            )}
          </>
        ) : backendInfo.reason ? (
          <span data-testid="backend-reason">{gpuDisabled ? t('unavailable') : t('fellBack')}: {backendInfo.reason}</span>
        ) : gpuAvailable === null ? (
          <span data-testid="backend-reason">{t('checking')}</span>
        ) : (
          <span>{t('cpuNote')}</span>
        )}
      </p>
    </div>
  );
}
