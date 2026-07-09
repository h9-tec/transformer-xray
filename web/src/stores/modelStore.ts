'use client';

/** Model/worker lifecycle. The EngineClient (and every Float32Array) lives
 * outside Zustand — stores hold only serializable UI state. */

import { create } from 'zustand';
import { EngineClient } from '@/engine/worker/client';
import type { ModelConfigJson } from '@/engine/weights';

let client: EngineClient | null = null;

export function engine(): EngineClient {
  if (!client) client = new EngineClient();
  return client;
}

interface ModelState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  error: string | null;
  cfgJson: ModelConfigJson | null;
  chars: string;
  load: () => Promise<void>;
}

export const useModelStore = create<ModelState>((set, get) => ({
  status: 'idle',
  error: null,
  cfgJson: null,
  chars: '',

  async load() {
    if (get().status === 'loading' || get().status === 'ready') return;
    set({ status: 'loading', error: null });
    try {
      const { cfgJson, chars } = await engine().init();
      set({ status: 'ready', cfgJson, chars });
    } catch (e) {
      set({ status: 'error', error: e instanceof Error ? e.message : String(e) });
    }
  },
}));
