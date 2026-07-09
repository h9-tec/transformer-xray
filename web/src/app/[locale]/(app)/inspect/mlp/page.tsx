import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { InspectorPanels } from '@/components/inspect/Panels';
import { MlpView } from '@/components/inspect/views/MlpView';
import Doc from '@/content/en/mlp.mdx';

export default function Page({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<MlpView />} math={<Doc />} />;
}
