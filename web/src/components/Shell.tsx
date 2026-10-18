/** App chrome: brand, primary nav, depth toggle, locale switch. Fits 360px:
 * the nav wraps to a second row on phones instead of disappearing. */

import { useTranslations } from 'next-intl';
import { DepthToggle } from '@/components/DepthToggle';
import { Link } from '@/i18n/navigation';
import { LocaleSwitch } from './LocaleSwitch';

export function Shell({ children }: { children: React.ReactNode }) {
  const t = useTranslations('nav');
  return (
    <>
      <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1200px] flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 sm:h-[60px] sm:flex-nowrap sm:gap-5 sm:px-5 sm:py-0">
          <Link
            href="/"
            className="flex shrink-0 items-center gap-2.5 sm:order-1 font-display text-[16px] font-bold tracking-tight sm:text-[17px]"
          >
            <span aria-hidden className="glow-accent inline-block h-2.5 w-2.5 rounded-full bg-accent" />
            Transformer X-Ray
          </Link>
          <div className="ms-auto flex shrink-0 items-center gap-2 sm:order-3">
            <DepthToggle />
            <LocaleSwitch />
            <Link href="/learn" className="btn-primary hidden !px-4 !py-1.5 text-sm lg:inline-flex">
              {t('start')}
            </Link>
          </div>
          <nav
            className="order-last flex w-full items-center gap-1 overflow-x-auto text-sm text-muted sm:order-2 sm:w-auto"
            aria-label="Primary"
          >
            <Link href="/learn" className="shrink-0 rounded-full px-3 py-1.5 hover:bg-raised hover:text-ink">
              {t('learn')}
            </Link>
            <Link href="/playground" className="shrink-0 rounded-full px-3 py-1.5 hover:bg-raised hover:text-ink">
              {t('playground')}
            </Link>
            <Link href="/train" className="shrink-0 rounded-full px-3 py-1.5 hover:bg-raised hover:text-ink">
              {t('train')}
            </Link>
          </nav>
        </div>
      </header>
      <div className="mx-auto max-w-[1200px] px-4 py-8 sm:px-5">{children}</div>
    </>
  );
}
