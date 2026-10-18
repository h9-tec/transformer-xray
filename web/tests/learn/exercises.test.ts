import { describe, expect, it } from 'vitest';
import { gradeExercise } from '@/learn/exercise';
import { EXERCISES } from '@/learn/exercises';
import { LESSONS } from '@/learn/lessons';

describe.each(Object.values(EXERCISES))('exercise $slug', (ex) => {
  it('solution passes every target', () => {
    const g = gradeExercise(ex, ex.solution);
    expect(g.targets.filter((t) => !t.pass)).toEqual([]);
    expect(g.ok).toBe(true);
  });
  it('starter does not pass', () => {
    let ok = true;
    try {
      ok = gradeExercise(ex, ex.starter).ok;
    } catch {
      ok = false;
    }
    expect(ok).toBe(false);
  });
  it('runs in under 1.5 s instrumented', () => {
    const t0 = performance.now();
    gradeExercise(ex, ex.solution);
    expect(performance.now() - t0).toBeLessThan(1500);
  });
});

it('every lesson has an exercise', () => {
  for (const l of LESSONS) expect(EXERCISES[l.slug], l.slug).toBeDefined();
});
