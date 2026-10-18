/** Lesson page frame. Server component: reads the real kernel source at
 * build time. Simulator, exercise and quiz are slots that fill in with the
 * interpreter release; until then they say so plainly. */

import { useTranslations } from 'next-intl';
import { CudaCode } from '@/components/gpu/CudaCode';
import { Link } from '@/i18n/navigation';
import { readKernel } from '@/lib/kernels';
import { DOCS } from '@/content/en/learn';
import { LESSONS, type Lesson } from '@/learn/lessons';
import { LessonRail } from './LessonRail';
import { RunPanel } from './RunPanel';
import { Simulator } from './Simulator';

function Slot({ title, body, tag }: { title: string; body: string; tag: string }) {
  return (
    <section className="mb-10">
      <h2 className="m-0 mb-3 flex items-center gap-3 font-display text-[26px] font-bold tracking-tight">
        {title}
        <span className="nums rounded-md border border-line bg-raised px-2 py-1 text-xs font-medium text-muted">
          {tag}
        </span>
      </h2>
      <div className="panel border-dashed px-5 py-4 text-[15px] text-muted">{body}</div>
    </section>
  );
}

export function LessonShell({ lesson }: { lesson: Lesson }) {
  const t = useTranslations('learn');
  const next = LESSONS.find((l) => l.n === lesson.n + 1);
  const kernel = lesson.kernelFile ? readKernel(lesson.kernelFile) : null;
  const Doc = DOCS[lesson.slug];

  return (
    <div className="grid gap-6 py-4 lg:grid-cols-[260px_minmax(0,1fr)] lg:gap-10">
      <LessonRail activeSlug={lesson.slug} />
      <main className="min-w-0">
        <header className="mb-9">
          <p className="m-0 mb-2.5 flex flex-wrap gap-3.5 text-[15px] text-muted">
            <b className="font-semibold text-accent">{t('lessonOf', { n: lesson.n })}</b>
            <span>{t('minutes', { n: lesson.minutes })}</span>
            {lesson.kernelFile && <span className="nums">{t('buildsToward', { file: lesson.kernelFile })}</span>}
          </p>
          <h1 lang="en" dir="ltr" className="m-0 mb-3.5 text-start font-display text-[clamp(34px,4.6vw,56px)] font-extrabold leading-[1] tracking-[-0.03em] text-balance">
            {lesson.title}
          </h1>
          <p lang="en" dir="ltr" className="m-0 max-w-[36em] text-start text-[18px] text-muted">{lesson.blurb}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            {lesson.goals.map((g) => (
              <span key={g} lang="en" dir="ltr" className="rounded-full border border-line bg-surface px-3 py-1.5 text-[13px] text-muted">
                {g}
              </span>
            ))}
          </div>
        </header>

        <section className="mb-10 max-w-[66ch]">
          <h2 className="m-0 mb-2 font-display text-[22px] font-bold tracking-tight">{t('youWillBuild')}</h2>
          <p lang="en" dir="ltr" className="m-0 text-start text-[17px] text-[#cfd6e6]">{lesson.builds}</p>
        </section>

        {Doc && (
          <section className="prose-lesson mb-12 max-w-[70ch]" lang="en" dir="ltr" data-testid="concept">
            <Doc components={{ h1: 'h2' }} />
          </section>
        )}

        <section className="mb-10" data-testid="simulator">
          <h2 className="m-0 mb-3 flex items-center gap-3 font-display text-[26px] font-bold tracking-tight">
            {t('simulator')}
            <span className="nums rounded-md border border-line bg-raised px-2 py-1 text-xs font-medium text-muted">simulator</span>
          </h2>
          <Simulator slug={lesson.slug} />
        </section>
        <section className="mb-10" data-testid="exercise">
          <h2 className="m-0 mb-3 flex items-center gap-3 font-display text-[26px] font-bold tracking-tight">
            {t('exercise')}
            <span className="nums rounded-md border border-line bg-raised px-2 py-1 text-xs font-medium text-muted">exercise</span>
          </h2>
          <RunPanel slug={lesson.slug} />
        </section>

        {kernel && (
          <section className="mb-10" data-testid="kernel-source">
            <h2 className="m-0 mb-3 flex items-center gap-3 font-display text-[26px] font-bold tracking-tight">
              {t('kernelTitle')}
              <span className="nums rounded-md border border-line bg-raised px-2 py-1 text-xs font-medium text-muted">
                kernels/{kernel.file}
              </span>
            </h2>
            <div className="panel overflow-hidden">
              <CudaCode code={kernel.code} />
            </div>
          </section>
        )}

        <Slot title={t('quiz')} tag="quiz" body={t('slotQuiz')} />

        <div className="flex flex-wrap items-center justify-between gap-4 rounded-[18px] border border-fwd/30 px-6 py-5" style={{ background: 'linear-gradient(135deg, rgba(116,199,255,.08), rgba(255,154,77,.06))' }}>
          <div className="flex flex-wrap gap-3">
            {lesson.inspectorHref && (
              <Link href={lesson.inspectorHref} className="btn-ghost">
                {t('inModel')}
              </Link>
            )}
          </div>
          {next && (
            <Link href={`/learn/${next.slug}`} className="btn-primary">
              <span>
                <span className="block font-display text-[15px]">{t('next')}: {next.title}</span>
                <span className="block text-xs font-normal opacity-80">{t('nextMeta', { n: next.n, min: next.minutes })}</span>
              </span>
            </Link>
          )}
        </div>
      </main>
    </div>
  );
}
