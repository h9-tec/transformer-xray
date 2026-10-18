'use client';

/** The course laid out like a die: twelve streaming multiprocessors, one per
 * lesson, lit by progress. Shared by the landing page and /learn. */

import { useTranslations } from 'next-intl';
import { useEffect, useState } from 'react';
import { Link } from '@/i18n/navigation';
import { LESSONS } from '@/learn/lessons';
import { lessonStatus, useLearnStore } from '@/stores/learnStore';

export function LessonMap() {
  const t = useTranslations('learn');
  const passed = useLearnStore((s) => s.passed);
  const current = useLearnStore((s) => s.current);
  // Progress comes from localStorage; render the neutral map on the server
  // and light it after hydration so markup matches.
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);
  const state = hydrated ? { passed, current } : { passed: [], current: '' };

  return (
    <div>
      <ul
        aria-label={t('mapLabel')}
        className="relative m-0 grid list-none grid-cols-1 gap-3 rounded-[22px] border border-line bg-surface p-[18px] sm:grid-cols-2 lg:grid-cols-4"
        style={{
          backgroundImage:
            'linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px)',
          backgroundSize: '100% 24px, 24px 100%',
          boxShadow: 'inset 0 0 0 1px rgba(116,199,255,.08), inset 0 0 60px rgba(116,199,255,.05)',
        }}
      >
        {LESSONS.map((l) => {
          const status = lessonStatus(l.slug, state);
          const lit = status === 'passed';
          const now = status === 'current';
          return (
            <li key={l.slug} className="min-w-0">
              <Link
                href={`/learn/${l.slug}`}
                data-status={status}
                className={`relative flex min-h-[150px] flex-col gap-2.5 rounded-xl border bg-raised px-3.5 pt-3.5 pb-3 transition-[transform,border-color,box-shadow] hover:-translate-y-0.5 hover:border-accent/45 ${
                  lit
                    ? 'border-accent/35 shadow-[0_0_0_1px_rgba(255,154,77,.15),0_0_28px_rgba(255,154,77,.15)]'
                    : now
                      ? 'border-fwd/60 shadow-[0_0_0_1px_rgba(116,199,255,.25),0_0_34px_rgba(116,199,255,.22)]'
                      : 'border-line'
                }`}
              >
                {now && (
                  <span className="absolute -top-2.5 end-3 rounded-full bg-fwd px-2 py-0.5 text-[11px] font-semibold text-[#06202f]">
                    {state.passed.length ? t('continue') : t('startHere')}
                  </span>
                )}
                <div className="nums flex justify-between text-xs text-faint">
                  <span>{String(l.n).padStart(2, '0')}</span>
                  <span>{t('minutes', { n: l.minutes })}</span>
                </div>
                <h3 lang="en" dir="ltr" className="m-0 text-start font-display text-[17px] font-bold leading-tight tracking-tight">
                  {l.title}
                </h3>
                <p lang="en" dir="ltr" className="m-0 flex-1 text-start text-[13.5px] text-muted">{l.blurb}</p>
                <div className="h-1 overflow-hidden rounded bg-white/[.06]">
                  <i
                    className={`block h-full ${
                      lit
                        ? 'w-full bg-gradient-to-r from-accent to-accent-deep shadow-[0_0_12px_rgba(255,154,77,.8)]'
                        : now
                          ? 'w-[40%] bg-gradient-to-r from-fwd to-fwd-deep shadow-[0_0_12px_rgba(116,199,255,.8)]'
                          : 'w-0'
                    }`}
                  />
                </div>
              </Link>
            </li>
          );
        })}
      </ul>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-faint">
        <span>
          <i className="me-1.5 inline-block h-2.5 w-2.5 rounded-[3px] bg-accent align-[-1px]" />
          {t('legendPassed')}
        </span>
        <span>
          <i className="me-1.5 inline-block h-2.5 w-2.5 rounded-[3px] bg-fwd align-[-1px]" />
          {t('legendCurrent')}
        </span>
        <span>
          <i className="me-1.5 inline-block h-2.5 w-2.5 rounded-[3px] border border-line bg-raised align-[-1px]" />
          {t('legendTodo')}
        </span>
        <span>{t('legendTotal')}</span>
      </div>
    </div>
  );
}
