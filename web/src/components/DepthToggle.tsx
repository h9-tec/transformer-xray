'use client';

import { useTranslations } from 'next-intl';
import { useEffect, useState } from 'react';

type Depth = 'junior' | 'senior';

export function DepthToggle() {
  const t = useTranslations('nav');
  const [depth, setDepth] = useState<Depth>('junior');

  useEffect(() => {
    setDepth((document.documentElement.dataset.depth as Depth) ?? 'junior');
  }, []);

  function apply(next: Depth) {
    setDepth(next);
    document.documentElement.dataset.depth = next;
    try {
      localStorage.setItem('xray-depth', next);
    } catch {
      /* private mode: toggle still works for this page */
    }
  }

  return (
    <div
      role="group"
      aria-label="Explanation depth"
      className="inline-flex overflow-hidden rounded-full border border-line text-sm"
    >
      {(['junior', 'senior'] as const).map((level) => (
        <button
          key={level}
          type="button"
          aria-pressed={depth === level}
          onClick={() => apply(level)}
          className={`px-2.5 py-1.5 text-[13px] transition-colors sm:px-4 sm:text-sm ${
            depth === level ? 'bg-accent font-medium text-bg' : 'text-muted hover:text-ink'
          }`}
        >
          {level === 'junior' ? t('depthJunior') : t('depthSenior')}
        </button>
      ))}
    </div>
  );
}
