'use client';

/** Course progress, persisted per browser. A throwing localStorage (private
 * mode, blocked site data) degrades to in-memory so the course still works. */

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { LESSONS } from '@/learn/lessons';

export type LessonStatus = 'passed' | 'current' | 'todo';

export interface LearnState {
  passed: string[];
  current: string;
  /** best quiz score per lesson */
  quiz: Record<string, number>;
  markPassed: (slug: string) => void;
  setCurrent: (slug: string) => void;
  setQuiz: (slug: string, score: number) => void;
  reset: () => void;
}

const FIRST = LESSONS[0].slug;

function nextSlug(slug: string): string {
  const i = LESSONS.findIndex((l) => l.slug === slug);
  return LESSONS[Math.min(i + 1, LESSONS.length - 1)]?.slug ?? slug;
}

function safeStorage(): Storage {
  const memory = new Map<string, string>();
  const fallback: Storage = {
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => void memory.set(k, v),
    removeItem: (k) => void memory.delete(k),
    clear: () => memory.clear(),
    key: () => null,
    length: 0,
  };
  // Each call is guarded: a storage that throws (private mode, blocked site
  // data) silently falls back to the in-memory map for that call.
  return {
    ...fallback,
    getItem: (k) => {
      try {
        return localStorage.getItem(k);
      } catch {
        return fallback.getItem(k);
      }
    },
    setItem: (k, v) => {
      try {
        localStorage.setItem(k, v);
      } catch {
        fallback.setItem(k, v);
      }
    },
    removeItem: (k) => {
      try {
        localStorage.removeItem(k);
      } catch {
        fallback.removeItem(k);
      }
    },
  };
}

export const useLearnStore = create<LearnState>()(
  persist(
    (set, get) => ({
      passed: [],
      current: FIRST,
      quiz: {},
      markPassed: (slug) => {
        const { passed, current } = get();
        set({
          passed: passed.includes(slug) ? passed : [...passed, slug],
          current: current === slug ? nextSlug(slug) : current,
        });
      },
      setCurrent: (slug) => set({ current: slug }),
      setQuiz: (slug, score) => set((s) => ({ quiz: { ...s.quiz, [slug]: Math.max(score, s.quiz[slug] ?? 0) } })),
      reset: () => set({ passed: [], current: FIRST, quiz: {} }),
    }),
    {
      name: 'xray-learn',
      storage: createJSONStorage(() => safeStorage()),
      partialize: (s) => ({ passed: s.passed, current: s.current, quiz: s.quiz }),
    },
  ),
);

export function lessonStatus(slug: string, s: Pick<LearnState, 'passed' | 'current'>): LessonStatus {
  if (s.passed.includes(slug)) return 'passed';
  if (s.current === slug) return 'current';
  return 'todo';
}
