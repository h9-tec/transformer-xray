'use client';

/** The product's signature switch: flip the whole inspector between the
 * forward pass (green, data flows right) and the backward pass (magenta,
 * blame flows left). Every view swaps its tensors for their grad twins. */

import { useTranslations } from 'next-intl';
import { useUiStore } from '@/stores/uiStore';

export function PhaseToggle() {
  const t = useTranslations('inspect');
  const phase = useUiStore((s) => s.phase);
  const setPhase = useUiStore((s) => s.setPhase);

  return (
    <div
      role="group"
      aria-label={t('phaseLabel')}
      className="inline-flex overflow-hidden rounded-full border border-line text-sm"
    >
      <button
        type="button"
        aria-pressed={phase === 'forward'}
        onClick={() => setPhase('forward')}
        className={`px-4 py-1.5 transition-colors ${
          phase === 'forward' ? 'bg-fwd font-medium text-bg' : 'text-muted hover:text-ink'
        }`}
      >
        {t('phaseForward')} →
      </button>
      <button
        type="button"
        aria-pressed={phase === 'backward'}
        onClick={() => setPhase('backward')}
        className={`px-4 py-1.5 transition-colors ${
          phase === 'backward' ? 'bg-bwd font-medium text-bg' : 'text-muted hover:text-ink'
        }`}
      >
        ← {t('phaseBackward')}
      </button>
    </div>
  );
}
