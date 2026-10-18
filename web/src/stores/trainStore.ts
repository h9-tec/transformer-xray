'use client';

/** Watch-it-learn state. Loss history is plain numbers, so it can live in
 * the store; the trained params live in the worker. */

import { withBase } from '@/lib/basePath';
import { create } from 'zustand';
import { engine } from './modelStore';
import { DEFAULT_TEXT, useTraceStore } from './traceStore';

export type TrainSource = 'scratch' | 'checkpoint';

interface TrainState {
  source: TrainSource;
  lr: number;
  seqLen: number;
  initialized: boolean;
  running: boolean;
  step: number;
  lossHistory: number[];
  sample: string;
  error: string | null;

  set: (p: Partial<Pick<TrainState, 'source' | 'lr' | 'seqLen'>>) => void;
  start: () => Promise<void>;
  stop: () => void;
  reset: () => void;
  inspectNow: () => Promise<void>;
}

async function fetchCorpus(): Promise<string> {
  const url = withBase('/model/corpus.txt');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed for ${url}: HTTP ${r.status}`);
  return r.text();
}

export const useTrainStore = create<TrainState>((set, get) => ({
  source: 'scratch',
  lr: 1e-3,
  seqLen: 64,
  initialized: false,
  running: false,
  step: 0,
  lossHistory: [],
  sample: '',
  error: null,

  set: (p) => set(p),

  async start() {
    const s = get();
    if (s.running) return;
    set({ running: true, error: null });
    try {
      if (!s.initialized) {
        const corpus = await fetchCorpus();
        await engine().trainInit(s.source === 'scratch', 1337, corpus);
        set({ initialized: true, step: 0, lossHistory: [], sample: '' });
      }
      await engine().trainRun(
        { steps: 400, seqLen: get().seqLen, lr: get().lr, sampleEvery: 25 },
        ({ step, loss, sample }) => {
          set((prev) => ({
            step,
            lossHistory: [...prev.lossHistory, loss],
            ...(sample !== undefined ? { sample } : {}),
          }));
        },
      );
      set({ running: false });
    } catch (e) {
      set({ running: false, error: e instanceof Error ? e.message : String(e) });
    }
  },

  stop() {
    engine().trainStop();
    set({ running: false });
  },

  reset() {
    engine().trainStop();
    engine().trainReset();
    set({ initialized: false, running: false, step: 0, lossHistory: [], sample: '', error: null });
  },

  /** Trace the live training params and push the result into the shared
   * trace store — every inspector then x-rays the learning model. */
  async inspectNow() {
    const x = DEFAULT_TEXT.slice(0, -1);
    const y = DEFAULT_TEXT.slice(1);
    const r = await engine().trace(x, y);
    useTraceStore.getState().applyExternal(r, DEFAULT_TEXT);
  },
}));
