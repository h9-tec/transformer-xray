import { setRequestLocale } from 'next-intl/server';
import { useTranslations } from 'next-intl';
import { use } from 'react';
import { TrainView } from '@/components/training/TrainView';

export default function TrainPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations('train');

  return (
    <main className="space-y-6">
      <header className="space-y-1">
        <h1 className="font-display text-3xl font-light">{t('title')}</h1>
        <p className="max-w-2xl text-sm text-muted">{t('subtitle')}</p>
      </header>
      <TrainView />
    </main>
  );
}
