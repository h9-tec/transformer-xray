'use client';

import { useTranslations } from 'next-intl';
import { Link, usePathname } from '@/i18n/navigation';
import { FORWARD_STAGES } from '@/lib/stages';

export function NavRail() {
  const t = useTranslations('pipeline');
  const pathname = usePathname();

  return (
    <nav aria-label={t('label')} className="space-y-1">
      {FORWARD_STAGES.map((s) => {
        const href = `/inspect/${s.slug}`;
        const active = pathname === href;
        return (
          <Link
            key={s.slug}
            href={href}
            aria-current={active ? 'page' : undefined}
            className={`block rounded px-3 py-1.5 text-sm transition-colors ${
              active
                ? 'border-s-2 border-fwd bg-raised text-ink'
                : 'text-muted hover:bg-surface hover:text-ink'
            }`}
          >
            {t(s.key)}
          </Link>
        );
      })}
    </nav>
  );
}
