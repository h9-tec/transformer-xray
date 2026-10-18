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

it('reductions: dropping the lane-0 guard without shuffles still fails', () => {
  const ex = EXERCISES['reductions-shuffles'];
  const noGuard = ex.starter.replace(/if \(lane == 0\) \{/, '{');
  expect(noGuard).not.toBe(ex.starter);
  expect(gradeExercise(ex, noGuard).ok).toBe(false);
});

it('gradeFromWire round-trips Int32 inputs and matches gradeExercise', async () => {
  const { handle } = await import('@/cuda/worker/handler');
  const { gradeFromWire } = await import('@/learn/exercise');
  for (const slug of ['coalescing', 'atomics']) {
    const ex = EXERCISES[slug];
    const built = ex.build(1);
    const res = handle({
      id: 1,
      type: 'run',
      source: ex.solution,
      kernel: ex.kernel,
      grid: ex.launch.grid,
      block: ex.launch.block,
      sharedBytes: ex.launch.sharedBytes,
      instrument: true,
      args: built.args.map((a) =>
        a.kind === 'buffer'
          ? { kind: 'buffer' as const, name: a.name, dtype: a.data instanceof Int32Array ? ('i32' as const) : ('f32' as const), buffer: a.data.buffer.slice(0) as ArrayBuffer }
          : { kind: a.kind, value: a.value },
      ),
    });
    expect(res.type).toBe('done');
    if (res.type !== 'done') continue;
    const g = gradeFromWire(ex, built, res.buffers, res.result);
    expect(g.ok, slug).toBe(true);
  }
});
