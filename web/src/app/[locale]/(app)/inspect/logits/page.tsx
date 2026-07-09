import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { GpuPanel } from '@/components/gpu/GpuPanel';
import { InspectorPanels } from '@/components/inspect/Panels';
import { LogitsView } from '@/components/inspect/views/LogitsView';
import Doc from '@/content/en/logits.mdx';

export default function LogitsPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<LogitsView />} math={<Doc />} gpu={<GpuPanel stage="logits" />} />;
}
