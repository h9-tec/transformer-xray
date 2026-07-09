/** The forward pipeline as a clickable map — the navigation hub into the
 * inspectors. Server component; labels come from messages. */

import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';

const STAGES = [
  { slug: 'tokens', key: 'tokens' },
  { slug: 'embeddings', key: 'embeddings' },
  { slug: 'layernorm', key: 'layernorm' },
  { slug: 'attention', key: 'attention' },
  { slug: 'mlp', key: 'mlp' },
  { slug: 'residual', key: 'residual' },
  { slug: 'logits', key: 'logits' },
] as const;

export function PipelineMap() {
  const t = useTranslations('pipeline');
  return (
    <nav aria-label={t('label')} className="flex flex-wrap items-center gap-y-2">
      {STAGES.map((s, i) => (
        <span key={s.slug} className="flex items-center">
          {i > 0 && (
            <span aria-hidden className="mx-1.5 text-faint">
              →
            </span>
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
