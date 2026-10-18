import { setRequestLocale } from 'next-intl/server';
import { use } from 'react';
import Doc from '@/content/en/cuda-subset.mdx';
import { Link } from '@/i18n/navigation';

export default function SubsetPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);
  setRequestLocale(locale);
  return (
    <main className="py-4">
      <p className="m-0 mb-4 text-sm text-muted">
        <Link href="/learn" className="text-fwd hover:underline">
          Learn
        </Link>
      </p>
      <article className="prose-lesson max-w-[72ch]" lang="en" dir="ltr">
        <Doc />
      </article>
    </main>
  );
}
