import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { LESSONS, lessonBySlug } from '@/learn/lessons';

describe('LESSONS', () => {
  it('has 12 lessons numbered 1..12 with unique slugs', () => {
    expect(LESSONS.map((l) => l.n)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
    expect(new Set(LESSONS.map((l) => l.slug)).size).toBe(12);
  });
  it('every kernelFile names a real file in kernels/', () => {
    for (const l of LESSONS) {
      if (!l.kernelFile) continue;
      expect(fs.existsSync(path.join(process.cwd(), '..', 'kernels', l.kernelFile)), l.slug).toBe(true);
    }
  });
  it('every inspectorHref points at an existing inspector route', () => {
    for (const l of LESSONS) {
      if (!l.inspectorHref) continue;
      const p = path.join(process.cwd(), 'src/app/[locale]/(app)', l.inspectorHref, 'page.tsx');
      expect(fs.existsSync(p), l.slug).toBe(true);
    }
  });
  it('lessonBySlug finds tiled-matmul and misses unknown', () => {
    expect(lessonBySlug('tiled-matmul')?.n).toBe(8);
    expect(lessonBySlug('nope')).toBeUndefined();
  });
});
