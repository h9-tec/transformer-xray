'use client';

/** Sticky course rail on lesson pages: progress bar + the twelve lessons
 * with status dots. */

import { useTranslations } from 'next-intl';
import { useEffect, useState } from 'react';
import { Link } from '@/i18n/navigation';
import { LESSONS } from '@/learn/lessons';
import { lessonStatus, useLearnStore } from '@/stores/learnStore';

export function LessonRail({ activeSlug }: { activeSlug: string }) {
  const t = useTranslations('learn');
  const passed = useLearnStore((s) => s.passed);
  const current = useLearnStore((s) => s.current);
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);
  const state = hydrated ? { passed, current } : { passed: [], current: '' };
  const k = state.passed.length;

  return (
    <aside className="lg:sticky lg:top-[84px] lg:self-start" aria-label={t('progress')}>
      <div className="panel mb-4 px-4 py-3.5">
        <b className="font-display text-[15px]">{t('passedOf', { k })}</b>
        <div className="mt-2.5 h-1.5 overflow-hidden rounded bg-white/[.06]">
          <i
            className="block h-full bg-gradient-to-r from-accent to-fwd"
            style={{ width: `${(k / LESSONS.length) * 100}%` }}
          />
        </div>
      </div>
      <ol className="m-0 flex list-none flex-row flex-wrap gap-0.5 p-0 lg:flex-col">
        {LESSONS.map((l) => {
          const st = lessonStatus(l.slug, state);
          const active = l.slug === activeSlug;
          return (
            <li key={l.slug}>
              <Link
                href={`/learn/${l.slug}`}
                aria-current={active ? 'page' : undefined}
                className={`grid grid-cols-[22px_1fr] items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm hover:bg-surface hover:text-ink ${
                  active ? 'bg-fwd/10 text-ink' : 'text-muted'
                }`}
              >
                <span
                  className={`nums grid h-[22px] w-[22px] place-items-center rounded-full border text-[11px] ${
                    st === 'passed'
                      ? 'border-accent/50 bg-accent/15 text-accent'
                      : st === 'current'
                        ? 'glow-fwd border-fwd bg-fwd text-[#06202f]'
                        : 'border-line text-faint'
                  }`}
                >
                  {l.n}
                </span>
                <span className="hidden lg:inline">{l.title}</span>
              </Link>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
