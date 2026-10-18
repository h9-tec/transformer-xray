'use client';

import { useTranslations } from 'next-intl';
import { Link, usePathname } from '@/i18n/navigation';
import { FORWARD_STAGES, TRAIN_STAGES, type Stage } from '@/lib/stages';

function Group({ stages, accent }: { stages: Stage[]; accent: 'fwd' | 'bwd' }) {
  const t = useTranslations('pipeline');
  const pathname = usePathname();
  return (
    <>
      {stages.map((s) => {
        const href = `/inspect/${s.slug}`;
        const active = pathname === href;
        return (
          <Link
            key={s.slug}
            href={href}
            aria-current={active ? 'page' : undefined}
            className={`block rounded px-3 py-1.5 text-sm transition-colors ${
              active
                ? `border-s-2 bg-raised text-ink ${accent === 'fwd' ? 'border-fwd' : 'border-bwd'}`
                : 'text-muted hover:bg-surface hover:text-ink'
            }`}
          >
            {t(s.key)}
          </Link>
        );
      })}
    </>
  );
}

export function NavRail() {
  const t = useTranslations('pipeline');
  return (
    <nav aria-label={t('label')} className="space-y-1">
      <Group stages={FORWARD_STAGES} accent="fwd" />
      <p className="pt-3 text-xs text-faint">{t('trainingGroup')}</p>
      <Group stages={TRAIN_STAGES} accent="bwd" />
    </nav>
  );
}
