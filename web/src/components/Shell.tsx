/** App chrome: brand, primary nav, depth toggle, locale switch. */

import { useTranslations } from 'next-intl';
import { DepthToggle } from '@/components/DepthToggle';
import { Link } from '@/i18n/navigation';
import { LocaleSwitch } from './LocaleSwitch';

export function Shell({ children }: { children: React.ReactNode }) {
  const t = useTranslations('nav');
  return (
    <>
      <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur-md">
        <div className="mx-auto flex h-[60px] max-w-6xl items-center gap-5 px-5">
          <Link href="/" className="flex items-center gap-2.5 font-display text-[17px] font-bold tracking-tight">
            <span aria-hidden className="glow-accent inline-block h-2.5 w-2.5 rounded-full bg-accent" />
            Transformer X-Ray
          </Link>
          <nav className="hidden items-center gap-1 text-sm text-muted sm:flex" aria-label="Primary">
            <Link href="/learn" className="rounded-full px-3 py-1.5 hover:bg-raised hover:text-ink">
              {t('learn')}
            </Link>
            <Link href="/playground" className="rounded-full px-3 py-1.5 hover:bg-raised hover:text-ink">
              {t('playground')}
            </Link>
            <Link href="/train" className="rounded-full px-3 py-1.5 hover:bg-raised hover:text-ink">
              {t('train')}
            </Link>
          </nav>
          <div className="ms-auto flex items-center gap-2.5">
            <DepthToggle />
            <LocaleSwitch />
            <Link href="/learn" className="btn-primary hidden !px-4 !py-1.5 text-sm md:inline-flex">
              {t('start')}
            </Link>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-6xl px-5 py-8">{children}</div>
    </>
  );
}
