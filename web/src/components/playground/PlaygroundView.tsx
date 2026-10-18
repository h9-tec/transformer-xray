'use client';

/** The playground: prompt → live tokens → the distribution each one was
 * drawn from. Loads the model (7.4 MB of real weights) on first visit. */

import { useEffect } from 'react';
import { useTranslations } from 'next-intl';
import { useModelStore } from '@/stores/modelStore';
import { stepLogits, usePlaygroundStore } from '@/stores/playgroundStore';
import { LogitsBars } from './LogitsBars';
import { PromptInput } from './PromptInput';
import { SamplingControls } from './SamplingControls';
import { TokenStream } from './TokenStream';

export function PlaygroundView() {
  const t = useTranslations('playground');
  const { status, error, cfgJson, chars, load } = useModelStore();
  const steps = usePlaygroundStore((s) => s.steps);
  const selected = usePlaygroundStore((s) => s.selectedStep);
  const greedy = usePlaygroundStore((s) => s.greedy);

  useEffect(() => {
    void load();
  }, [load]);

  if (status === 'error') {
    return (
      <div role="alert" className="rounded-lg border border-danger/40 bg-surface p-6 text-sm">
        <p className="text-danger">{t('loadFailed')}</p>
        <p className="nums mt-2 text-muted">{error}</p>
      </div>
    );
  }

  if (status !== 'ready' || !cfgJson) {
    return (
      <div className="panel p-6 text-sm text-muted" aria-busy>
        {t('loading')}
      </div>
    );
  }

  const sel = selected !== null ? { step: usePlaygroundStore.getState().steps[selected], logits: stepLogits(selected) } : null;

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[minmax(16rem,20rem)_1fr]">
        <aside className="space-y-5">
          <PromptInput chars={chars} />
          <SamplingControls />
          <p className="nums text-xs text-faint">
            {cfgJson.n_params.toLocaleString()} params · val loss {cfgJson.val_loss.toFixed(3)} ·
            100% live in your browser
          </p>
        </aside>
        <section className="min-w-0 space-y-4">
          <TokenStream chars={chars} />
          {sel?.logits && sel.step ? (
            <LogitsBars
              logits={sel.logits}
              chars={chars}
              chosen={sel.step.token}
              temperature={sel.step.temperature}
              topK={sel.step.topK}
              greedy={greedy}
            />
          ) : (
            steps.length === 0 && <p className="text-sm text-faint">{t('hint')}</p>
          )}
        </section>
      </div>
    </div>
  );
}
