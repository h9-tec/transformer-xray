import { useTranslations } from 'next-intl';
import { Link } from '@/i18n/navigation';

const PROBS = [1,2,1,3,2,1,4,2,1,1,2,30,3,2,1,1,2,6,3,2,1,12,2,1,1,2,1,3,2,1,1,2,1,2,3,1,2,1,2,5];

function BarsThumb() {
  return (
    <div className="absolute inset-x-3.5 bottom-3.5 flex h-[90px] items-end gap-[3px]" aria-hidden>
      {PROBS.map((p, i) => (
        <i
          key={i}
          className={`flex-1 rounded-t-sm ${p === 30 ? 'bg-accent shadow-[0_0_10px_rgba(255,154,77,.8)]' : 'bg-fwd opacity-45'}`}
          style={{ height: `${(p / 30) * 100}%` }}
        />
      ))}
    </div>
  );
}
function LossThumb() {
  return (
    <svg className="absolute inset-3.5 h-[calc(100%-28px)] w-[calc(100%-28px)]" viewBox="0 0 300 120" preserveAspectRatio="none" aria-hidden>
      <defs>
        <linearGradient id="lossfill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#ff9a4d" stopOpacity=".5" />
          <stop offset="1" stopColor="#ff9a4d" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d="M0 10 C40 20 60 60 100 70 S180 95 300 100 L300 120 L0 120Z" fill="url(#lossfill)" />
      <path d="M0 10 C40 20 60 60 100 70 S180 95 300 100" fill="none" stroke="#ff9a4d" strokeWidth="2" />
    </svg>
  );
}
function HeatThumb() {
  const cells: { o: number; hot: boolean }[] = [];
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 16; c++) {
      const v = c <= r * 2 + 1 ? Math.exp(-Math.abs(c - r * 2) / 3) : 0;
      cells.push({ o: 0.08 + v * 0.9, hot: v > 0.8 });
    }
  return (
    <div className="absolute inset-3.5 grid grid-cols-16 gap-[2px]" aria-hidden>
      {cells.map((x, i) => (
        <i key={i} className={`rounded-[1px] ${x.hot ? 'bg-accent' : 'bg-fwd'}`} style={{ opacity: x.o }} />
      ))}
    </div>
  );
}

export function Labs() {
  const t = useTranslations('home');
  const labs = [
    { href: '/playground', k: 'labPlay', thumb: <BarsThumb /> },
    { href: '/train', k: 'labTrain', thumb: <LossThumb /> },
    { href: '/inspect/attention', k: 'labInspect', thumb: <HeatThumb /> },
  ] as const;
  return (
    <section id="labs" className="border-t border-line py-16 sm:py-20">
      <h2 className="m-0 mb-2.5 font-display text-3xl font-bold tracking-tight text-balance sm:text-4xl">
        {t('labsTitle')}
      </h2>
      <p className="m-0 mb-8 max-w-[40em] text-[17px] text-muted">{t('labsSub')}</p>
      <div className="grid gap-[18px] md:grid-cols-3">
        {labs.map((l) => (
          <Link key={l.k} href={l.href} className="panel flex min-w-0 flex-col overflow-hidden !rounded-[18px]">
            <div className="relative h-[150px] border-b border-line bg-[#080b14]">{l.thumb}</div>
            <div className="px-5 pt-[18px] pb-5">
              <h3 className="m-0 mb-1.5 font-display text-[19px] font-bold">{t(`${l.k}Title`)}</h3>
              <p className="m-0 mb-3.5 text-[14.5px] text-muted">{t(`${l.k}Body`)}</p>
              <span className="text-sm font-semibold text-fwd">{t(`${l.k}Cta`)}</span>
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}
