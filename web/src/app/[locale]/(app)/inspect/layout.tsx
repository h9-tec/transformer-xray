import { setRequestLocale } from 'next-intl/server';
import { InspectBar } from '@/components/inspect/InspectBar';
import { NavRail } from '@/components/inspect/NavRail';
import { PhaseToggle } from '@/components/inspect/PhaseToggle';

export default async function InspectLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  setRequestLocale(locale);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start gap-4">
        <div className="min-w-0 flex-1">
          <InspectBar />
        </div>
        <PhaseToggle />
      </div>
      <div className="grid gap-6 lg:grid-cols-[11rem_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-20 lg:self-start">
          <NavRail />
        </div>
        <div className="min-w-0">{children}</div>
      </div>
    </div>
  );
}
