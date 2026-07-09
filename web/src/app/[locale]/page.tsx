import { setRequestLocale } from 'next-intl/server';
import { useTranslations } from 'next-intl';
import { use } from 'react';
import { Link } from '@/i18n/navigation';

const STATS = [
  { value: '1,841,472', key: 'params' },
  { value: '1.543', key: 'valLoss' },
  { value: '457', key: 'parityTests' },
  { value: '100%', key: 'inBrowser' },
] as const;

export default function Home({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  const t = useTranslations('home');

  const doors = [
    { href: '/playground', key: 'doorPlay', accent: 'fwd' },
    { href: '/train', key: 'doorTrain', accent: 'bwd' },
    { href: '/inspect/attention', key: 'doorInspect', accent: 'accent' },
  ] as const;

  return (
    <main className="relative mx-auto min-h-dvh max-w-5xl px-6">
      {/* scanline atmosphere */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'repeating-linear-gradient(0deg, transparent 0 3px, rgb(231 234 243 / 0.012) 3px 4px)',
        }}
      />

      <section className="flex min-h-[72dvh] flex-col justify-center gap-8 pt-16">
        <p className="reveal nums text-xs uppercase tracking-[0.3em] text-faint">{t('kicker')}</p>
        <h1 className="reveal font-display text-6xl font-extralight leading-[1.02] tracking-tight sm:text-8xl">
          Transformer
          <br />
          <span
            className="font-black text-fwd"
            style={{ textShadow: '0 0 42px rgb(61 220 151 / 0.4)' }}
          >
            X-Ray
          </span>
        </h1>
        <p className="reveal max-w-xl text-lg leading-relaxed text-muted">{t('tagline')}</p>

        <div className="reveal flex flex-wrap gap-3">
          {doors.map((d) => (
            <Link
              key={d.key}
              href={d.href}
              className={`rounded-full border px-6 py-2.5 transition-all hover:-translate-y-0.5 ${
                d.accent === 'fwd'
                  ? 'border-fwd text-fwd hover:bg-fwd hover:text-bg'
                  : d.accent === 'bwd'
                    ? 'border-bwd text-bwd hover:bg-bwd hover:text-bg'
                    : 'border-accent text-accent hover:bg-accent hover:text-bg'
              }`}
            >
              {t(d.key)} →
            </Link>
          ))}
        </div>
      </section>

      <section className="reveal grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
        {STATS.map((s) => (
          <div key={s.key} className="bg-surface p-5">
            <div className="nums text-2xl text-ink">{s.value}</div>
            <div className="mt-1 text-xs text-muted">{t(`stats.${s.key}`)}</div>
          </div>
        ))}
      </section>

      <section className="grid gap-10 py-16 md:grid-cols-2">
        <div className="space-y-3">
          <h2 className="font-display text-2xl font-light text-fwd">{t('realTitle')}</h2>
          <p className="text-sm leading-relaxed text-muted">{t('realBody')}</p>
        </div>
        <div className="space-y-3">
          <h2 className="font-display text-2xl font-light text-bwd">{t('deepTitle')}</h2>
          <p className="text-sm leading-relaxed text-muted">{t('deepBody')}</p>
        </div>
      </section>

      <footer className="border-t border-line py-8 text-xs text-faint">{t('footer')}</footer>
    </main>
  );
}
