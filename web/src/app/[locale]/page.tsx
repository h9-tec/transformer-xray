import { Link } from '@/i18n/navigation';
import { setRequestLocale } from 'next-intl/server';
import { useTranslations } from 'next-intl';
import { use } from 'react';

export default function Home({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations('home');

  return (
    <main className="mx-auto flex min-h-dvh max-w-3xl flex-col items-start justify-center gap-6 px-6">
      <h1 className="font-display text-6xl font-extralight tracking-tight">
        Transformer{' '}
        <span className="font-black text-fwd" style={{ textShadow: '0 0 24px rgb(61 220 151 / 0.35)' }}>
          X-Ray
        </span>
      </h1>
      <p className="max-w-xl text-lg text-muted">{t('tagline')}</p>
      <Link
        href="/spike"
        className="rounded-full border border-accent px-6 py-2 text-accent transition-colors hover:bg-accent hover:text-bg"
      >
        {t('cta')} →
      </Link>
    </main>
  );
}
