/** The forward pipeline as a clickable map — the navigation hub into the
 * inspectors. Server component; labels come from messages. */

import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';
import { FORWARD_STAGES as STAGES } from '@/lib/stages';

export function PipelineMap() {
  const t = useTranslations('pipeline');
  return (
    <nav aria-label={t('label')} className="flex flex-wrap items-center gap-y-2">
      {STAGES.map((s, i) => (
        <span key={s.slug} className="flex items-center">
          {i > 0 && (
            <span aria-hidden className="mx-1.5 inline-block h-px w-3 bg-line align-middle" />
          )}
          <Link
            href={`/inspect/${s.slug}`}
            className="rounded-full border border-line bg-surface px-3 py-1 text-xs text-muted transition-colors hover:border-fwd hover:text-fwd"
          >
            {t(s.key)}
          </Link>
        </span>
      ))}
    </nav>
  );
}
