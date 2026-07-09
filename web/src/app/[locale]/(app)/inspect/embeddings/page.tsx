import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { GpuPanel } from '@/components/gpu/GpuPanel';
import { InspectorPanels } from '@/components/inspect/Panels';
import { EmbeddingsView } from '@/components/inspect/views/EmbeddingsView';
import Doc from '@/content/en/embeddings.mdx';

export default function EmbeddingsPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<EmbeddingsView />} math={<Doc />} gpu={<GpuPanel stage="embeddings" />} />;
}
