import { setRequestLocale } from 'next-intl/server';
import { useTranslations } from 'next-intl';
import { use } from 'react';
import { HowItWorks } from '@/components/landing/HowItWorks';
import { Labs } from '@/components/landing/Labs';
import { WarpInstrument } from '@/components/landing/WarpInstrument';
import { LessonMap } from '@/components/learn/LessonMap';
import { Shell } from '@/components/Shell';
import { StoreBridge } from '@/components/StoreBridge';
import { Link } from '@/i18n/navigation';

export default function Home({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations('home');

  return (
    <Shell>
      <StoreBridge />
    <main className="relative">
      {/* hero */}
      <section className="relative overflow-hidden pt-8 pb-10 sm:pt-16">
        <div
          aria-hidden
          className="pointer-events-none absolute -inset-x-[10%] -top-[20%] h-[70%]"
          style={{
            background:
              'radial-gradient(600px 300px at 25% 30%, rgba(255,154,77,.18), transparent 70%), radial-gradient(700px 340px at 80% 20%, rgba(116,199,255,.16), transparent 70%)',
          }}
        />
        <div className="relative grid items-center gap-8 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:gap-14">
          <div>
            <p className="m-0 mb-3.5 text-[15px] text-muted">{t('kicker')}</p>
            <h1 className="m-0 mb-5 font-display text-[clamp(40px,6.2vw,78px)] font-extrabold leading-[.98] tracking-[-0.03em] text-balance">
              {t('h1a')}
              <span className="text-accent" style={{ textShadow: '0 0 34px rgba(255,154,77,.45)' }}>
                {t('h1b')}
              </span>
              {t('h1c')}
            </h1>
            <p className="m-0 mb-7 max-w-[36em] text-[19px] text-muted">{t('lede')}</p>
            <div className="flex flex-wrap items-center gap-3">
              <Link href="/learn/hello-thread" className="btn-primary">
                {t('ctaStart')} <span className="nums text-xs opacity-70">{t('ctaStartMeta')}</span>
              </Link>
              <Link href="/playground" className="btn-ghost">
                {t('ctaPlay')}
              </Link>
            </div>
            <div className="mt-7 flex flex-wrap gap-x-7 gap-y-4 text-sm text-faint">
              <div>
                <b className="font-semibold text-muted">{t('proof1a')}</b>
                {t('proof1b')}
              </div>
              <div>
                <b className="font-semibold text-muted">{t('proof2a')}</b>
                {t('proof2b')}
              </div>
              <div>
                <b className="font-semibold text-muted">{t('proof3a')}</b>
                {t('proof3b')}
              </div>
            </div>
          </div>
          <WarpInstrument />
        </div>
      </section>

      <section id="path" className="border-t border-line py-16 sm:py-20">
        <h2 className="m-0 mb-2.5 font-display text-3xl font-bold tracking-tight text-balance sm:text-4xl">
          {t('pathTitle')}
        </h2>
        <p className="m-0 mb-8 max-w-[40em] text-[17px] text-muted">{t('pathSub')}</p>
        <LessonMap />
      </section>

      <HowItWorks />
      <Labs />

      <footer className="flex flex-wrap justify-between gap-2.5 border-t border-line pt-7 pb-10 text-sm text-faint">
        <span>{t('footer1')}</span>
        <span className="nums">{t('footer2')}</span>
      </footer>
    </main>
    </Shell>
  );
}
