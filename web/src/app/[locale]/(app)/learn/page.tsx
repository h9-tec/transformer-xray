import { setRequestLocale } from 'next-intl/server';
import { useTranslations } from 'next-intl';
import { use } from 'react';
import { LessonMap } from '@/components/learn/LessonMap';

export default function LearnIndex({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations('learn');
  return (
    <main className="py-6">
      <h1 className="m-0 mb-3 font-display text-[clamp(34px,4.6vw,56px)] font-extrabold leading-[1] tracking-[-0.03em] text-balance">
        {t('indexTitle')}
      </h1>
      <p className="m-0 mb-9 max-w-[40em] text-[18px] text-muted">{t('indexSub')}</p>
      <LessonMap />
    </main>
  );
}
