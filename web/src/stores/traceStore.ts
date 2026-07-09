'use client';

/** One inspected forward+backward pass shared by every inspector page.
 * The TraceResult (with its Float32Arrays) lives at module level; the store
 * versions it for React. */

import { create } from 'zustand';
import type { TraceResult } from '@/engine/worker/client';
import { engine } from './modelStore';

let result: TraceResult | null = null;

export function currentTrace(): TraceResult | null {
  return result;
}

/** Default inspection text — the same passage the parity suite pins down. */
export const DEFAULT_TEXT = 'First Citizen:\nBefore we proceed any further, hear me speak.\nAll:\n';

interface TraceState {
  text: string;
  status: 'idle' | 'tracing' | 'ready' | 'error';
  error: string | null;
  loss: number | null;
  T: number;
  version: number;
  setText: (t: string) => void;
  trace: () => Promise<void>;
}

export const useTraceStore = create<TraceState>((set, get) => ({
  text: DEFAULT_TEXT,
  status: 'idle',
  error: null,
  loss: null,
  T: 0,
  version: 0,

  setText: (text) => set({ text }),

  async trace() {
    const { text, status } = get();
    if (status === 'tracing') return;
    if (text.length < 2) {
      set({ status: 'error', error: 'need at least 2 characters' });
      return;
    }
    set({ status: 'tracing', error: null });
    try {
      // Inspect x = text[:-1] predicting y = text[1:] — loss and gradients
      // come from real next-character targets.
      const x = text.slice(0, -1);
      const y = text.slice(1);
      result = await engine().trace(x, y);
      set((s) => ({
        status: 'ready',
        loss: result!.loss ?? null,
        T: x.length,
        version: s.version + 1,
      }));
    } catch (e) {
      set({ status: 'error', error: e instanceof Error ? e.message : String(e) });
    }
  },
}));
