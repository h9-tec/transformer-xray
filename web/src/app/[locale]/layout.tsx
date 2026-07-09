import type { Metadata } from 'next';
import { hasLocale, NextIntlClientProvider } from 'next-intl';
import { setRequestLocale } from 'next-intl/server';
import {
  Bricolage_Grotesque,
  IBM_Plex_Sans,
  IBM_Plex_Sans_Arabic,
  JetBrains_Mono,
} from 'next/font/google';
import { notFound } from 'next/navigation';
import { routing } from '@/i18n/routing';
import 'katex/dist/katex.min.css';
import '../globals.css';

const display = Bricolage_Grotesque({
  subsets: ['latin'],
  variable: '--font-bricolage',
});
const body = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-plex',
});
const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
});
const arabic = IBM_Plex_Sans_Arabic({
  subsets: ['arabic'],
  weight: ['400', '500', '700'],
  variable: '--font-plex-arabic',
});

export const metadata: Metadata = {
  title: 'Transformer X-Ray',
  description:
    'A real GPT thinking and learning live in your browser — every tensor, gradient, and CUDA mapping inspectable.',
};

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

// Restores the persisted depth level before paint (no flash, no hydration mismatch).
const depthInit = `try{var d=localStorage.getItem('xray-depth');if(d==='senior')document.documentElement.dataset.depth='senior'}catch(e){}`;

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  if (!hasLocale(routing.locales, locale)) notFound();
  setRequestLocale(locale);

  return (
    <html
      lang={locale}
      dir={locale === 'ar' ? 'rtl' : 'ltr'}
      data-depth="junior"
      suppressHydrationWarning
      className={`${display.variable} ${body.variable} ${mono.variable} ${arabic.variable}`}
    >
      <body>
        <script dangerouslySetInnerHTML={{ __html: depthInit }} />
        <NextIntlClientProvider>{children}</NextIntlClientProvider>
      </body>
    </html>
  );
}
