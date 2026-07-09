import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { InspectorPanels } from '@/components/inspect/Panels';
import { LayerNormView } from '@/components/inspect/views/LayerNormView';
import Doc from '@/content/en/layernorm.mdx';

export default function Page({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<LayerNormView />} math={<Doc />} />;
}
