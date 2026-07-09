import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { GpuPanel } from '@/components/gpu/GpuPanel';
import { InspectorPanels } from '@/components/inspect/Panels';
import { AttentionView } from '@/components/inspect/views/AttentionView';
import Doc from '@/content/en/attention.mdx';

export default function Page({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<AttentionView />} math={<Doc />} gpu={<GpuPanel stage="attention" />} />;
}
