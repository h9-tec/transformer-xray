import { useTranslations } from 'next-intl';

const ICONS = [
  <svg key="see" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
    <rect x="2.5" y="2.5" width="15" height="15" rx="2" />
    <path d="M6 10h8M10 6v8" />
  </svg>,
  <svg key="write" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
    <path d="M7 5 3 10l4 5M13 5l4 5-4 5" />
  </svg>,
  <svg key="find" width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
    <path d="M3 15 8 9l3 3 6-7" />
    <circle cx="17" cy="5" r="1.4" fill="currentColor" />
  </svg>,
];
const TONES = ['bg-fwd/10 text-fwd', 'bg-accent/12 text-accent', 'bg-ok/12 text-ok'];

export function HowItWorks() {
  const t = useTranslations('home');
  return (
    <section className="border-t border-line py-16 sm:py-20">
      <h2 className="m-0 mb-2.5 font-display text-3xl font-bold tracking-tight text-balance sm:text-4xl">
        {t('howTitle')}
      </h2>
      <p className="m-0 mb-3 max-w-[40em] text-[17px] text-muted">{t('howSub')}</p>
      <p className="m-0 mb-8 max-w-[40em] rounded-lg border border-dashed border-accent/40 bg-accent/5 px-3.5 py-2.5 text-[14px] text-muted">
        {t('howNote')}
      </p>
      <div className="grid gap-[18px] md:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <article key={i} className="panel min-w-0 px-[22px] pt-[22px] pb-5">
            <div className={`mb-3.5 grid h-[38px] w-[38px] place-items-center rounded-[10px] ${TONES[i - 1]}`}>
              {ICONS[i - 1]}
            </div>
            <h3 className="m-0 mb-2 font-display text-[19px] font-bold">{t(`how${i}Title`)}</h3>
            <p className="m-0 text-[15px] text-muted">{t(`how${i}Body`)}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
