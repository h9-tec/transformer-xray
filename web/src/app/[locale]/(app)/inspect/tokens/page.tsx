import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import { InspectorPanels } from '@/components/inspect/Panels';
import { TokensView } from '@/components/inspect/views/TokensView';
import Doc from '@/content/en/tokens.mdx';

export default function TokensPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return <InspectorPanels tensor={<TokensView />} math={<Doc />} />;
}
