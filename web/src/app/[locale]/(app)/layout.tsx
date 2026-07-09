import { setRequestLocale } from 'next-intl/server';
import { Shell } from '@/components/Shell';
import { StoreBridge } from '@/components/StoreBridge';

export default async function AppLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return (
    <Shell>
      <StoreBridge />
      {children}
    </Shell>
  );
}
