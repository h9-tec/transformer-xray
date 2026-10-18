/** The inspector page shape: the story first (live tensors + the math),
 * the GPU kernel folded underneath. Server component — children carry the
 * client parts. */

import { useTranslations } from 'next-intl';

export function InspectorPanels({
  tensor,
  math,
  gpu,
}: {
  tensor: React.ReactNode;
  math: React.ReactNode;
  gpu?: React.ReactNode;
}) {
  const t = useTranslations('inspect');
  return (
    <div className="space-y-6">
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(20rem,26rem)]">
        <section aria-label="Live tensors" className="min-w-0 space-y-4">
          {tensor}
        </section>
        <aside aria-label="The math" className="prose-panel panel p-5">
          {math}
        </aside>
      </div>
      {gpu && (
        <details className="panel group open:border-accent/30">
          <summary className="flex cursor-pointer list-none items-center gap-3 px-5 py-4 font-display text-[17px] font-bold tracking-tight text-ink marker:hidden [&::-webkit-details-marker]:hidden">
            <span
              aria-hidden
              className="inline-block h-2 w-2 rounded-full bg-accent transition-transform group-open:scale-125"
            />
            {t('underTheHood')}
          </summary>
          <div className="border-t border-line p-5">{gpu}</div>
        </details>
      )}
    </div>
  );
}
