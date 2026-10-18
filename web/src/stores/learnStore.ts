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
  markPassed: (slug: string) => void;
  setCurrent: (slug: string) => void;
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
  const wrap =
    (name: 'getItem' | 'setItem' | 'removeItem') =>
    (...args: [string, string]) => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return (localStorage as any)[name](...args);
      } catch {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return (fallback as any)[name](...args);
      }
    };
  return { ...fallback, getItem: wrap('getItem'), setItem: wrap('setItem'), removeItem: wrap('removeItem') };
}

export const useLearnStore = create<LearnState>()(
  persist(
    (set, get) => ({
      passed: [],
      current: FIRST,
      markPassed: (slug) => {
        const { passed, current } = get();
        set({
          passed: passed.includes(slug) ? passed : [...passed, slug],
          current: current === slug ? nextSlug(slug) : current,
        });
      },
      setCurrent: (slug) => set({ current: slug }),
      reset: () => set({ passed: [], current: FIRST }),
    }),
    {
      name: 'xray-learn',
      storage: createJSONStorage(() => safeStorage()),
      partialize: (s) => ({ passed: s.passed, current: s.current }),
    },
  ),
);

export function lessonStatus(slug: string, s: Pick<LearnState, 'passed' | 'current'>): LessonStatus {
  if (s.passed.includes(slug)) return 'passed';
  if (s.current === slug) return 'current';
  return 'todo';
}
