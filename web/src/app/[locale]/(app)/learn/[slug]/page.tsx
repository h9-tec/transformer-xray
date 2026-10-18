import { setRequestLocale } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { LessonShell } from '@/components/learn/LessonShell';
import { routing } from '@/i18n/routing';
import { LESSONS, lessonBySlug } from '@/learn/lessons';

export function generateStaticParams() {
  return routing.locales.flatMap((locale) => LESSONS.map((l) => ({ locale, slug: l.slug })));
}

export default async function LessonPage({
  params,
}: {
  params: Promise<{ locale: string; slug: string }>;
}) {
  const { locale, slug } = await params;
  setRequestLocale(locale);
  const lesson = lessonBySlug(slug);
  if (!lesson) notFound();
  return <LessonShell lesson={lesson} />;
}
