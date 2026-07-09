import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { GpuPanel } from '@/components/gpu/GpuPanel';
import { InspectorPanels } from '@/components/inspect/Panels';
import { ResidualView } from '@/components/inspect/views/ResidualView';
import Doc from '@/content/en/residual.mdx';

export default function Page({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<ResidualView />} math={<Doc />} gpu={<GpuPanel stage="residual" />} />;
}
