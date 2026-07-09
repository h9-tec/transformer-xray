import { setRequestLocale } from 'next-intl/server';
import { Shell } from '@/components/Shell';

export default async function AppLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <Shell>{children}</Shell>;
}
