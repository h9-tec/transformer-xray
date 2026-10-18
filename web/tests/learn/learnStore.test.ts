import { beforeEach, describe, expect, it, vi } from 'vitest';

function fakeStorage(opts: { throws?: boolean } = {}) {
  const m = new Map<string, string>();
  const boom = () => {
    throw new Error('SecurityError');
  };
  return {
    getItem: opts.throws ? boom : (k: string) => m.get(k) ?? null,
    setItem: opts.throws ? boom : (k: string, v: string) => void m.set(k, v),
    removeItem: opts.throws ? boom : (k: string) => void m.delete(k),
    clear: () => m.clear(),
    key: () => null,
    length: 0,
  } as unknown as Storage;
}

async function freshStore(storage: Storage) {
  vi.resetModules();
  vi.stubGlobal('localStorage', storage);
  return (await import('@/stores/learnStore'));
}

describe('learnStore', () => {
  beforeEach(() => vi.unstubAllGlobals());

  it('starts at hello-thread with nothing passed', async () => {
    const { useLearnStore } = await freshStore(fakeStorage());
    expect(useLearnStore.getState().passed).toEqual([]);
    expect(useLearnStore.getState().current).toBe('hello-thread');
  });

  it('markPassed records the slug and advances current', async () => {
    const { useLearnStore } = await freshStore(fakeStorage());
    useLearnStore.getState().markPassed('hello-thread');
    expect(useLearnStore.getState().passed).toContain('hello-thread');
    expect(useLearnStore.getState().current).toBe('bounds-grid-stride');
  });

  it('markPassed on a non-current lesson leaves current alone', async () => {
    const { useLearnStore } = await freshStore(fakeStorage());
    useLearnStore.getState().markPassed('warps');
    expect(useLearnStore.getState().current).toBe('hello-thread');
    expect(useLearnStore.getState().passed).toEqual(['warps']);
  });

  it('survives a localStorage that throws', async () => {
    const { useLearnStore } = await freshStore(fakeStorage({ throws: true }));
    useLearnStore.getState().markPassed('hello-thread');
    expect(useLearnStore.getState().passed).toEqual(['hello-thread']);
  });

  it('lessonStatus reports passed/current/todo', async () => {
    const { useLearnStore, lessonStatus } = await freshStore(fakeStorage());
    useLearnStore.getState().markPassed('hello-thread');
    const s = useLearnStore.getState();
    expect(lessonStatus('hello-thread', s)).toBe('passed');
    expect(lessonStatus('bounds-grid-stride', s)).toBe('current');
    expect(lessonStatus('warps', s)).toBe('todo');
  });

  it('setQuiz stores a score without touching passed', async () => {
    const { useLearnStore } = await freshStore(fakeStorage());
    useLearnStore.getState().setQuiz('hello-thread', 2);
    expect(useLearnStore.getState().quiz['hello-thread']).toBe(2);
    expect(useLearnStore.getState().passed).toEqual([]);
  });
});
