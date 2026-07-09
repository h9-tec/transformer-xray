'use client';

import { useLocale } from 'next-intl';
import { Link, usePathname } from '@/i18n/navigation';

export function LocaleSwitch() {
  const locale = useLocale();
  const pathname = usePathname();
  const other = locale === 'ar' ? 'en' : 'ar';
  return (
    <Link
      href={pathname}
      locale={other}
      className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:text-ink"
      aria-label={other === 'ar' ? 'التبديل إلى العربية' : 'Switch to English'}
    >
      {other === 'ar' ? 'عربي' : 'EN'}
    </Link>
  );
}
