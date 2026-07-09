/** App chrome: brand, primary nav, depth toggle, locale switch. */

import { useTranslations } from 'next-intl';
import { DepthToggle } from '@/components/DepthToggle';
import { Link } from '@/i18n/navigation';
import { LocaleSwitch } from './LocaleSwitch';

export function Shell({ children }: { children: React.ReactNode }) {
  const t = useTranslations('nav');
  return (
    <>
      <header className="sticky top-0 z-20 border-b border-line bg-bg/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
          <Link href="/" className="font-display text-lg font-extralight tracking-tight">
            Transformer <span className="font-black text-fwd">X-Ray</span>
          </Link>
          <nav className="flex gap-4 text-sm text-muted">
            <Link href="/playground" className="hover:text-ink">
              {t('playground')}
            </Link>
            <Link href="/train" className="hover:text-ink">
              {t('train')}
            </Link>
          </nav>
          <div className="ms-auto flex items-center gap-3">
            <DepthToggle />
            <LocaleSwitch />
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-6xl px-6 py-8">{children}</div>
    </>
  );
}
