import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { InspectorPanels } from '@/components/inspect/Panels';
import { OptimizerView } from '@/components/inspect/views/OptimizerView';
import Doc from '@/content/en/optimizer.mdx';

export default function Page({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<OptimizerView />} math={<Doc />} />;
}
