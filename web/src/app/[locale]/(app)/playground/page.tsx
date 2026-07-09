import { setRequestLocale } from 'next-intl/server';
import { useTranslations } from 'next-intl';
import { use } from 'react';
import { PipelineMap } from '@/components/PipelineMap';
import { PlaygroundView } from '@/components/playground/PlaygroundView';

export default function PlaygroundPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations('playground');

  return (
    <main className="space-y-6">
      <header className="space-y-1">
        <h1 className="font-display text-3xl font-light">{t('title')}</h1>
        <p className="text-sm text-muted">{t('subtitle')}</p>
      </header>
      <PlaygroundView />
      <footer className="border-t border-line pt-4">
        <PipelineMap />
      </footer>
    </main>
  );
}
